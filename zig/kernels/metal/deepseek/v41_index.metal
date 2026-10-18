// DeepSeek-V4.1-Flash's indexer and top-k, compiled after v41.metal (kernel_sources joins the files).

// ---------------------------------------------------------------- indexer

struct ScoreArgs {
  int rows;
  int groups;     // index keys in the cache
  int ratio;
  int first_pos;  // row r's position is first_pos + r
  int stride;     // score row stride
};

// score = sum_h bf16(relu(bf16(q_h . k_t)) * w_h), -inf past the row's visible keys; a lane a head, in order.
static float index_score(const device bfloat* Q, const device bfloat* WT, const device bfloat* K, int r, int t, int h) {
  const device bfloat* k = K + size_t(t) * ID;
  const device bfloat* q = Q + (size_t(r) * IH + h) * ID;
  float dot = 0.0f;
  for (int i = 0; i < ID; i++) dot += float(q[i]) * float(k[i]);
  const float term = bf_round(max(bf_round(dot), 0.0f) * float(WT[r * IH + h]));
  float total = 0.0f;
  for (int j = 0; j < IH; j++) total += simd_shuffle(term, ushort(j));
  return bf_round(total);
}

[[kernel]] void ds_index_scores(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                const device bfloat* K [[buffer(2)]], device float* SCORE [[buffer(3)]],
                                constant ScoreArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const int t = int(g.x) / 32, r = int(g.y), h = int(lane);
  if (t >= a.groups) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  device float* out = SCORE + size_t(r) * a.stride;
  if (t >= visible) {
    if (lane == 0) out[t] = -INFINITY;
    return;
  }
  const float s = index_score(Q, WT, K, r, t, h);
  if (lane == 0) out[t] = s;
}

struct KeptArgs {
  int rows;
  int groups;
  int ratio;
  int first_pos;
  int block;
  int kept;  // KEEP's row width: the candidate blocks in ascending order, -1 after the last
};

// ds_index_scores over a row's candidate blocks: slot j is key keep[j / block] * block + j % block, in key order.
[[kernel]] void ds_index_scores_kept(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                     const device bfloat* K [[buffer(2)]], device float* SCORE [[buffer(3)]],
                                     const device int* KEEP [[buffer(4)]], constant KeptArgs& a [[buffer(5)]],
                                     uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const int j = int(g.x) / 32, r = int(g.y), h = int(lane);
  const int slots = a.kept * a.block;
  if (j >= slots) return;
  const int b = KEEP[size_t(r) * a.kept + j / a.block];
  const int t = b * a.block + j % a.block;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  device float* out = SCORE + size_t(r) * slots;
  if (b < 0 || t >= a.groups || t >= visible) {
    if (lane == 0) out[j] = -INFINITY;
    return;
  }
  const float s = index_score(Q, WT, K, r, t, h);
  if (lane == 0) out[j] = s;
}

// A row's picks over its candidate slots back to key positions (-1 stays).
[[kernel]] void ds_kept_keys(device int* PICKS [[buffer(0)]], const device int* KEEP [[buffer(1)]],
                             constant KeptArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int i = int(g.x), r = int(g.y);
  if (i >= IK || r >= a.rows) return;
  device int* p = PICKS + size_t(r) * IK + i;
  if (*p >= 0) *p = KEEP[size_t(r) * a.kept + *p / a.block] * a.block + *p % a.block;
}

struct BlockArgs {
  int rows;
  int groups;
  int block;
  int first_pos;
  int ratio;
  int stride;
};

// Block scores (the max of a block's positions; the row's newest block pinned at +inf) for the candidate pick.
[[kernel]] void ds_block_max(const device float* SCORE [[buffer(0)]], device float* BLK [[buffer(1)]],
                             constant BlockArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int b = int(g.x), r = int(g.y);
  const int nb = (a.groups + a.block - 1) / a.block;
  if (b >= nb) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  float m = -INFINITY;
  for (int i = 0; i < a.block; i++) {
    const int t = b * a.block + i;
    if (t < a.groups) m = max(m, SCORE[size_t(r) * a.stride + t]);
  }
  if (visible > 0 && b == (visible - 1) / a.block) m = FLT_MAX;  // pinned in (ds_topk takes finite scores only)
  BLK[size_t(r) * a.stride + b] = m;
}

struct TopkArgs {
  int n;       // candidates a row
  int k;       // picks a row
  int stride;  // score row stride
  int offset;  // added to each pick (the compressed keys' place behind the window)
};

inline uint order_key(float v) {
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A 1,024-thread group's sum of v, every thread getting it (unsigned integers: exact in any order).
static uint tg_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint s = simd_sum(v);
  if (lane == 0) sums[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total = simd_sum(sums[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

// Each thread's sum of v over the threads before it (unsigned integers: exact in any order).
static uint tg_exclusive_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint inside = simd_prefix_exclusive_sum(v);
  if (lane == 31) sums[sg] = inside + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lead = simd_prefix_exclusive_sum(sums[lane]);
  const uint before = simd_shuffle(lead, ushort(sg));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return before + inside;
}

// A row's k best finite scores (ties to the lower index) in index order + offset, -1 after: a radix select a row.
[[kernel]] void ds_topk(const device float* SCORE [[buffer(0)]], device int* OUT [[buffer(1)]],
                        constant TopkArgs& a [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_bits, want;
  threadgroup uint sums[32];
  const device float* s = SCORE + size_t(r) * a.stride;
  device int* out = OUT + size_t(r) * a.k;
  // finite candidates only: -inf marks keys that are not visible or not candidates
  if (t == 0) {
    prefix_bits = 0;
    want = uint(a.k);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint finite = 0;
  for (int i = int(t); i < a.n; i += 1024) finite += isfinite(s[i]) ? 1 : 0;
  const uint total = tg_sum(finite, t, lane, sg, sums);
  if (t == 0) want = min(want, total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = want;
  // the k-th largest key, 8 bits a pass from the top
  uint mask = 0;
  for (int shift = 24; shift >= 0 && k > 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < a.n; i += 1024) {
      if (!isfinite(s[i])) continue;
      const uint key = order_key(s[i]);
      if ((key & mask) == prefix_bits) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFF], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the bin holding the want-th largest: bins from the top, each thread's count of those above its own by a scan
    {
      const uint b = 255 - min(t, 255u);
      const uint hb = t < 256 ? atomic_load_explicit(&hist[b], memory_order_relaxed) : 0;
      const uint above = tg_exclusive_sum(hb, t, lane, sg, sums);
      const uint w = want;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t < 256 && above < w && above + hb >= w) {
        prefix_bits |= b << shift;
        want = w - above;
      }
    }
    mask |= 0xFFu << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint kth = prefix_bits;
  const uint ties = want;  // keys equal to kth to take, the lowest indices first
  // ordered compaction: each thread's span of indices, counted, then written in order
  const int span = (a.n + 1023) / 1024;
  const int lo = int(t) * span, hi = min(a.n, lo + span);
  uint above = 0, equal = 0;
  for (int i = lo; i < hi; i++) {
    if (!isfinite(s[i]) || k == 0) continue;
    const uint key = order_key(s[i]);
    above += key > kth ? 1 : 0;
    equal += key == kth ? 1 : 0;
  }
  const uint before_above = tg_exclusive_sum(above, t, lane, sg, sums);
  const uint before_equal = tg_exclusive_sum(equal, t, lane, sg, sums);
  uint taken_equal = before_equal;
  uint pos = before_above + min(before_equal, ties);
  for (int i = lo; i < hi; i++) {
    if (!isfinite(s[i]) || k == 0) continue;
    const uint key = order_key(s[i]);
    if (key > kth || (key == kth && taken_equal < ties)) {
      out[pos++] = i + a.offset;
    }
    if (key == kth) taken_equal++;
  }
  for (int i = int(k) + int(t); i < a.k; i += 1024) out[i] = -1;
}
