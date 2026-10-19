// Source is compiled after kv.metal; no query-by-key-by-head score tensor is materialized.
struct Dsv41Index {
    uint rows, heads, width, q_stride, weight_stride, position, ratio, keys, key_stride;
    uint tile_start, tile_width, topk, candidate_topk, block_size, use_candidates, make_candidates;
    float scale;
};
struct Dsv41Pick { float score; int position; };
inline bool dsv41_less(Dsv41Pick a, Dsv41Pick b) {
    return a.score < b.score || (a.score == b.score && a.position > b.position);
}
inline void dsv41_heap(device Dsv41Pick *h, uint size, Dsv41Pick v) {
    if (!dsv41_less(h[0], v)) return;
    uint at = 0;
    while (2 * at + 1 < size) {
        uint child = 2 * at + 1;
        if (child + 1 < size && dsv41_less(h[child + 1], h[child])) ++child;
        if (!dsv41_less(h[child], v)) break;
        h[at] = h[child]; at = child;
    }
    h[at] = v;
}
inline int dsv41_index_position(uint local, uint row, device const Dsv41Pick *candidates,
                                constant Dsv41Index &p) {
    uint slot = p.tile_start + local;
    if (!p.use_candidates) return int(slot);
    uint block = slot / p.block_size;
    if (block >= p.candidate_topk) return -1;
    int id = candidates[row * p.candidate_topk + block].position;
    return id < 0 ? -1 : id * int(p.block_size) + int(slot % p.block_size);
}
kernel void dsv41_index_score(device const ushort *q [[buffer(0)]], device const ushort *weight [[buffer(1)]],
    device const uchar *key [[buffer(2)]], device const Dsv41Pick *candidates [[buffer(3)]],
    device float *scores [[buffer(4)]], constant Dsv41Index &p [[buffer(5)]],
    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint local = g.x, row = g.y;
    if (local >= p.tile_width || row >= p.rows) return;
    int pos = dsv41_index_position(local, row, candidates, p);
    uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    if (pos < 0 || uint(pos) >= visible) {
        if (lane == 0) scores[row * p.tile_width + local] = -INFINITY;
        return;
    }
    device const uchar *k = key + uint(pos) * p.key_stride;
    float key_values[4];
    for (uint d = 0; d < p.width / 32; ++d) key_values[d] = dsv41_unpack(k, lane + 32 * d, p.width, 32, 2);
    float total = 0;
    for (uint head = 0; head < p.heads; ++head) {
        float dot = 0;
        for (uint d = 0; d < p.width / 32; ++d)
            dot += dsv41_bf(q[row * p.q_stride + head * p.width + lane + 32 * d]) * key_values[d];
        dot = dsv41_bf(dsv41_round(simd_sum(dot)));
        const float scaled_weight = dsv41_bf(dsv41_round(dsv41_bf(weight[row * p.weight_stride + head]) * p.scale));
        total += dsv41_bf(dsv41_round(max(dot, 0.0f) * scaled_weight));
    }
    if (lane == 0) scores[row * p.tile_width + local] = dsv41_bf(dsv41_round(total));
}
kernel void dsv41_index_clear(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    constant Dsv41Index &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    if (i.y >= p.rows) return;
    if (i.x < p.topk) top[i.y * p.topk + i.x] = {-INFINITY, -1};
    if (p.make_candidates && i.x < p.candidate_topk) candidate[i.y * p.candidate_topk + i.x] = {-INFINITY, -1};
}
// The serial merge and finish (a thread a row): the parallel ones' baseline in the tests and the kernel contract.
kernel void dsv41_index_merge_serial(device const float *scores [[buffer(0)]], device Dsv41Pick *top [[buffer(1)]],
    device Dsv41Pick *candidate [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    for (uint i = 0; i < p.tile_width; ++i) {
        float score = scores[row * p.tile_width + i];
        int pos = dsv41_index_position(i, row, candidate, p);
        if (isfinite(score)) dsv41_heap(heap, p.topk, {score, pos});
    }
    if (!p.make_candidates) return;
    uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
    device Dsv41Pick *blocks = candidate + row * p.candidate_topk;
    for (uint base = 0; base < p.tile_width; base += p.block_size) {
        float score = -INFINITY;
        for (uint j = 0; j < p.block_size; ++j) score = max(score, scores[row * p.tile_width + base + j]);
        uint block = (p.tile_start + base) / p.block_size;
        if (visible && block == (visible - 1) / p.block_size) score = INFINITY;
        if (score > -INFINITY) dsv41_heap(blocks, p.candidate_topk, {score, int(block)});
    }
}
inline void dsv41_sort_positions(device Dsv41Pick *p, uint size) {
    // Selection is score-ordered; attention sees the reference's ascending position order.
    for (uint i = 1; i < size; ++i) {
        Dsv41Pick value = p[i]; uint j = i;
        while (j > 0 && (p[j - 1].position < 0 ||
               (value.position >= 0 && p[j - 1].position > value.position))) {
            if (value.position < 0) break;
            p[j] = p[j - 1]; --j;
        }
        p[j] = value;
    }
}
kernel void dsv41_index_finish_serial(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    device int *positions [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device Dsv41Pick *heap = top + row * p.topk;
    dsv41_sort_positions(heap, p.topk);
    for (uint i = 0; i < p.topk; ++i) positions[row * p.topk + i] = heap[i].position;
    if (p.make_candidates) dsv41_sort_positions(candidate + row * p.candidate_topk, p.candidate_topk);
}
// A row's merge and finish on a threadgroup, the serial kernels' picks and bytes; a key ranks as dsv41_less does.
inline ulong dsv41_key(float score, int position) {
    uint u = as_type<uint>(score == 0.0f ? 0.0f : score);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (ulong(u) << 32) | ulong(~uint(position));
}
constant uint dsv41_lanes = 1024;
// The k-th largest of the group's keys (each thread holding N, `live` marking real ones): eight 8-bit passes.
template <uint N>
inline ulong dsv41_kth(thread const ulong (&keys)[N], thread const bool (&live)[N], uint k, uint t,
                       threadgroup atomic_uint *hist, threadgroup uint *pick) {
    ulong prefix = 0;
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (uint b = t; b < 256; b += dsv41_lanes) atomic_store_explicit(&hist[b], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const ulong high = shift == 56 ? 0 : prefix >> (shift + 8);
        for (uint i = 0; i < N; ++i)
            if (live[i] && (shift == 56 || (keys[i] >> (shift + 8)) == high))
                atomic_fetch_add_explicit(&hist[uint(keys[i] >> shift) & 255], 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint left = k, b = 255;
            for (;; --b) {
                const uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
                if (c >= left || b == 0) break;
                left -= c;
            }
            pick[0] = b;
            pick[1] = left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= ulong(pick[0]) << shift;
        k = pick[1];
    }
    return prefix;
}
// Keeps the best `k` of the group's live picks in out[0, k) in thread order, the rest of out as {-inf, -1}.
template <uint N>
inline void dsv41_keep(thread const ulong (&keys)[N], thread const bool (&live)[N], thread const Dsv41Pick (&picks)[N],
                       uint k, device Dsv41Pick *out, uint t, threadgroup atomic_uint *hist, threadgroup uint *pick,
                       threadgroup uint *sums) {
    for (uint b = t; b < 256; b += dsv41_lanes) atomic_store_explicit(&hist[b], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mine = 0;
    for (uint i = 0; i < N; ++i) mine += uint(live[i]);
    atomic_fetch_add_explicit(&hist[0], mine, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint count = atomic_load_explicit(&hist[0], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong least = count > k ? dsv41_kth(keys, live, k, t, hist, pick) : 0;
    // each thread's kept picks after those of the threads before it (out may be one of the inputs: all have read it)
    uint keep = 0;
    for (uint i = 0; i < N; ++i) keep += uint(live[i] && keys[i] >= least);
    const uint lane = t % 32, sg = t / 32;
    uint at = simd_prefix_exclusive_sum(keep);
    if (lane == 31) sums[sg] = at + keep;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (uint g = 0; g < sg; ++g) at += sums[g];
    for (uint i = 0; i < N; ++i)
        if (live[i] && keys[i] >= least) out[at++] = picks[i];
    uint kept = 0;
    for (uint g = 0; g < dsv41_lanes / 32; ++g) kept += sums[g];
    for (uint i = kept + t; i < k; i += dsv41_lanes) out[i] = {-INFINITY, -1};
}
kernel void dsv41_index_merge(device const float *scores [[buffer(0)]], device Dsv41Pick *top [[buffer(1)]],
    device Dsv41Pick *candidate [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint pick[2];
    threadgroup uint sums[dsv41_lanes / 32];
    if (row >= p.rows) return;
    {   // the row's 512 held picks and this tile's 4,096 scores (finite ones only, as the heap takes them)
        constexpr uint N = (512 + 4096 + dsv41_lanes - 1) / dsv41_lanes;
        ulong keys[N];
        bool live[N];
        Dsv41Pick picks[N];
        for (uint i = 0; i < N; ++i) {
            const uint at = t + i * dsv41_lanes;
            live[i] = false;
            if (at < p.topk) {
                picks[i] = top[row * p.topk + at];
                live[i] = isfinite(picks[i].score);
            } else if (at - p.topk < p.tile_width) {
                const uint local = at - p.topk;
                picks[i] = {scores[row * p.tile_width + local], dsv41_index_position(local, row, candidate, p)};
                live[i] = isfinite(picks[i].score);
            }
            keys[i] = live[i] ? dsv41_key(picks[i].score, picks[i].position) : 0;
        }
        dsv41_keep(keys, live, picks, p.topk, top + row * p.topk, t, hist, pick, sums);
    }
    if (!p.make_candidates) return;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    {   // the row's 2,048 held blocks and this tile's 512 block maxima, the newest visible block pinned
        constexpr uint N = (2048 + 512 + dsv41_lanes - 1) / dsv41_lanes;
        const uint visible = min(p.keys, (p.position + row + 1) / p.ratio);
        ulong keys[N];
        bool live[N];
        Dsv41Pick picks[N];
        for (uint i = 0; i < N; ++i) {
            const uint at = t + i * dsv41_lanes;
            live[i] = false;
            if (at < p.candidate_topk) {
                picks[i] = candidate[row * p.candidate_topk + at];
                live[i] = picks[i].score > -INFINITY;
            } else if ((at - p.candidate_topk) * p.block_size < p.tile_width) {
                const uint base = (at - p.candidate_topk) * p.block_size;
                float score = -INFINITY;
                for (uint j = 0; j < p.block_size; ++j) score = max(score, scores[row * p.tile_width + base + j]);
                const uint block = (p.tile_start + base) / p.block_size;
                if (visible && block == (visible - 1) / p.block_size) score = INFINITY;
                picks[i] = {score, int(block)};
                live[i] = score > -INFINITY;
            }
            keys[i] = live[i] ? dsv41_key(picks[i].score, picks[i].position) : 0;
        }
        dsv41_keep(keys, live, picks, p.candidate_topk, candidate + row * p.candidate_topk, t, hist, pick, sums);
    }
}
// Picks by position ascending, the -1 placeholders last (dsv41_sort_positions' order): a bitonic sort in groups.
inline void dsv41_sort_group(device Dsv41Pick *p, uint size, uint t, threadgroup Dsv41Pick *tmp) {
    for (uint i = t; i < size; i += dsv41_lanes) tmp[i] = p[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= size; k <<= 1)
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = t; i < size; i += dsv41_lanes) {
                const uint l = i ^ j;
                if (l > i) {
                    const uint a = uint(tmp[i].position), b = uint(tmp[l].position);  // -1 is the largest
                    if ((a > b) == ((i & k) == 0)) {
                        const Dsv41Pick x = tmp[i];
                        tmp[i] = tmp[l];
                        tmp[l] = x;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    for (uint i = t; i < size; i += dsv41_lanes) p[i] = tmp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
kernel void dsv41_index_finish(device Dsv41Pick *top [[buffer(0)]], device Dsv41Pick *candidate [[buffer(1)]],
    device int *positions [[buffer(2)]], constant Dsv41Index &p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
    threadgroup Dsv41Pick tmp[2048];
    if (row >= p.rows) return;
    dsv41_sort_group(top + row * p.topk, p.topk, t, tmp);
    for (uint i = t; i < p.topk; i += dsv41_lanes) positions[row * p.topk + i] = top[row * p.topk + i].position;
    if (p.make_candidates) dsv41_sort_group(candidate + row * p.candidate_topk, p.candidate_topk, t, tmp);
}
