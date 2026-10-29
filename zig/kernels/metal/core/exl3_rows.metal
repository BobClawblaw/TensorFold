// EXL3 rows: a window's rows through resident mul1 projections, both rotations folded in, one dispatch a call.

// 128 output columns a threadgroup over one K range; the last range to arrive sums the partials in range order.
#include <metal_stdlib>
using namespace metal;

#define KS 1

// A resident matrix (Metal 3 argument-buffer pointers), placed at column `y_col` of the output rows.
struct TfExl3Job {
  const device uint* trellis;
  const device half* su;
  const device half* sv;
  ulong tile_k_stride, tile_n_stride; // in words
  uint halves, n, y_col, first_block;
};

struct TfExl3RowsArgs {
  uint rows, k, jobs, kinds;
  uint repeat, output_f32, experts, ranges;
  uint blocks;       // every job's 128-column blocks (a routed slot's: kinds * n / 128)
  uint routed_slots; // routed: slots from here take the shared expert (jobs after the routed ones), x row slot - shared_base
  uint shared_base, pad;
  ulong x_stride, y_stride;
};

// mul1: (1024 + byte sum of state * 0x83DCD12D) * 0x1EEE + 0xC931, exact before one FP16 round (an FP16 fma).
inline float tf_xr_mul1(uint state) {
  const uint x = state * 0x83DCD12Du;
  const uint pairs = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu); // two 16-bit lanes of byte sums
  const uint h = 1024 + ((pairs * 0x00010001u) >> 16);
  return float(fma(half(h), as_type<half>(ushort(0x1EEE)), as_type<half>(ushort(0xC931))));
}

// The codebook from a 16-bit state: 1024 + the byte sum of state * 0x83DCD12D, then the table of its 1,021 values.
inline float tf_xr_book(threadgroup const float* book, uint state) {
  const uint x = state * 0x83DCD12Du;
  const uint pairs = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
  return book[(pairs & 0xFFFFu) + (pairs >> 16)];
}

// The table: book[s] = tf_xr_mul1's value at byte sum s, filled by a threadgroup before its first lookup.
inline void tf_xr_fill(threadgroup float* book, uint t, uint threads) {
  for (uint s = t; s < 1024; s += threads) {
    const uint h = 1024 + min(s, 1020u);
    book[s] = float(fma(half(h), as_type<half>(ushort(0x1EEE)), as_type<half>(ushort(0xC931))));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Lane l holds indices l, l+32, l+64, l+96 of a 128-block; the butterfly order is fixed.
inline void tf_xr_hadamard(thread float (&v)[4], uint lane) {
  for (uint d = 1; d < 32; d *= 2) {
    for (uint j = 0; j < 4; ++j) {
      const float peer = simd_shuffle_xor(v[j], ushort(d));
      v[j] = (lane & d) ? peer - v[j] : v[j] + peer;
    }
  }
  const float a = v[0], b = v[1], c = v[2], d = v[3];
  v[0] = a + b; v[1] = a - b; v[2] = c + d; v[3] = c - d;
  const float e = v[0], f = v[1], g = v[2], h = v[3];
  v[0] = e + g; v[1] = f + h; v[2] = e - g; v[3] = f - h;
}

// Column block `cb` of job j: 128 sums of each live row's (x * su) rotated against the decoded weights, in order.
inline void tf_xr_finish(const device TfExl3Job& j, device uchar* y, constant TfExl3RowsArgs& a, uint cb,
                         uint dst, thread float (&v)[4], uint lane) {
  tf_xr_hadamard(v, lane);
  for (uint q = 0; q < 4; ++q) {
    const uint n = cb * 128 + lane + 32 * q;
    const float out = (v[q] * 0.08838834764831845f) * float(j.sv[n]);
    const ulong at = ulong(dst) * a.y_stride + j.y_col + n;
    if (a.output_f32) reinterpret_cast<device float*>(y)[at] = out;
    else reinterpret_cast<device bfloat*>(y)[at] = bfloat(out);
  }
}

// Row `row` of x (live or not) at K block kb rotated into `out`: x * su, the Hadamard, / sqrt(128), fp16.
inline void tf_xr_rotate(const device bfloat* x, const device TfExl3Job& j, constant TfExl3RowsArgs& a, uint row,
                         bool live, uint kb, uint lane, threadgroup float* out) {
  float v[4];
  for (uint q = 0; q < 4; ++q) {
    const uint k = kb + lane + 32 * q;
    v[q] = live ? float(x[ulong(row) * a.x_stride + k]) * float(j.su[k]) : 0.0f;
  }
  tf_xr_hadamard(v, lane);
  for (uint q = 0; q < 4; ++q) out[lane + 32 * q] = float(half(v[q] * 0.08838834764831845f));
}

// 16 bits of the MSB-first stream A:B:C from bit S (a constant): an extract in a word, extract and insert across.
template <uint S>
inline uint tf_xr_window(uint A, uint B, uint C) {
  if (S <= 16) return extract_bits(A, 16 - S, 16);
  if (S < 32) return insert_bits(B >> (48 - S), A, S - 16, 32 - S);
  if (S <= 48) return extract_bits(B, 48 - S, 16);
  return insert_bits(C >> (80 - S), B, S - 48, 64 - S);
}

// Rows through job j's block cb, K range `range`, K bits: lane l decodes tile weights 8l..8l+7 from one run.
template <uint R, uint K>
inline void tf_xr_block(const device bfloat* x, const device TfExl3Job& j, device uchar* y, device float* partial,
                        device atomic_uint* arrive, constant TfExl3RowsArgs& a, uint cb, uint gb, uint range,
                        thread const uint (&src)[R], thread const uint (&dst)[R], uint live,
                        uint lane, uint sg, threadgroup float* part, threadgroup const float* book, threadgroup float* xs) {
  const uint ct = sg % 8, ks = sg / 8;
  constexpr uint count = 8 * K, ring = 256 * K; // a tile's words and bits
  const uint start = ((8 * lane + 1) * K + ring - 16) % ring;
  const uint w0 = start / 32, off = start % 32, back = (32 - off) & 31u, keep = off != 0 ? 0xFFFFFFFFu : 0u;
  const uint i0 = w0, i1 = (w0 + 1) % count, i2 = (w0 + 2) % count, i3 = (w0 + 3) % count;
  const uint r0 = 2 * (lane % 4);
  float acc[R][2];
  for (uint i = 0; i < R; ++i) acc[i][0] = acc[i][1] = 0.0f;
  const device uint* col = j.trellis + ulong(cb * 8 + ct) * j.tile_n_stride;
  const uint step = KS * a.ranges * 128;
  // rows rotated once a block by simdgroup i (row i), the next block's while this one's decode: one barrier
  const uint kb0 = (range * KS + ks) * 128;
  if (sg < R && kb0 < a.k) tf_xr_rotate(x, j, a, src[sg], sg < live, kb0, lane, xs + sg * 128);
  uint buf = 0;
  for (uint kb = kb0; kb < a.k; kb += step) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < R && kb + step < a.k) tf_xr_rotate(x, j, a, src[sg], sg < live, kb + step, lane, xs + ((buf ^ 1u) * R + sg) * 128);
    threadgroup const float* rows = xs + buf * R * 128;
    buf ^= 1u;
    _Pragma("clang loop unroll_count(2)")
    for (uint t = 0; t < 8; ++t) {
      const device uint* tile = col + ulong(kb / 16 + t) * j.tile_k_stride;
      const uint u0 = tile[i0], u1 = tile[i1], u2 = tile[i2], u3 = tile[i3];
      // the lane's run from bit `start`, aligned: A:B:C
      const uint A = (u0 << off) | ((u1 >> back) & keep);
      const uint B = (u1 << off) | ((u2 >> back) & keep);
      const uint C = (u2 << off) | ((u3 >> back) & keep);
      float w[8];
      w[0] = tf_xr_book(book, tf_xr_window<0 * K>(A, B, C));
      w[1] = tf_xr_book(book, tf_xr_window<1 * K>(A, B, C));
      w[2] = tf_xr_book(book, tf_xr_window<2 * K>(A, B, C));
      w[3] = tf_xr_book(book, tf_xr_window<3 * K>(A, B, C));
      w[4] = tf_xr_book(book, tf_xr_window<4 * K>(A, B, C));
      w[5] = tf_xr_book(book, tf_xr_window<5 * K>(A, B, C));
      w[6] = tf_xr_book(book, tf_xr_window<6 * K>(A, B, C));
      w[7] = tf_xr_book(book, tf_xr_window<7 * K>(A, B, C));
      const uint from = 16 * t + r0;
      for (uint i = 0; i < R; ++i) {
        threadgroup const float* xr = rows + i * 128 + from;
        const float x0 = xr[0], x1 = xr[1], x8 = xr[8], x9 = xr[9];
        acc[i][0] = fma(x9, w[3], fma(x8, w[2], fma(x1, w[1], fma(x0, w[0], acc[i][0]))));
        acc[i][1] = fma(x9, w[7], fma(x8, w[6], fma(x1, w[5], fma(x0, w[4], acc[i][1]))));
      }
    }
  }
  // lanes 4m .. 4m + 3 hold columns m and m + 8
  for (uint i = 0; i < R; ++i) {
    for (uint h = 0; h < 2; ++h) {
      float v = acc[i][h];
      v += simd_shuffle_xor(v, ushort(1));
      v += simd_shuffle_xor(v, ushort(2));
      acc[i][h] = v;
    }
    if (lane % 4 == 0) {
      part[(ks * R + i) * 128 + ct * 16 + lane / 4] = acc[i][0];
      part[(ks * R + i) * 128 + ct * 16 + lane / 4 + 8] = acc[i][1];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < live) {
    float v[4];
    for (uint q = 0; q < 4; ++q) {
      const uint c = lane + 32 * q;
      float s = 0.0f;
      for (uint z = 0; z < KS; ++z) s += part[(z * R + sg) * 128 + c];
      v[q] = s;
    }
    if (a.ranges == 1) {
      tf_xr_finish(j, y, a, cb, dst[sg], v, lane);
    } else {
      for (uint q = 0; q < 4; ++q)
        partial[(ulong(range) * a.rows + dst[sg]) * (a.blocks * 128) + gb * 128 + lane + 32 * q] = v[q];
    }
  }
  if (a.ranges == 1) return;
  // the last range to arrive sums every range's partials, in range order, and finishes
  threadgroup uint* arrived = reinterpret_cast<threadgroup uint*>(part);
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (sg == 0 && lane == 0) *arrived = atomic_fetch_add_explicit(arrive, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (*arrived != a.ranges - 1) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (sg < live) {
    float v[4];
    for (uint q = 0; q < 4; ++q) {
      float s = 0.0f;
      for (uint z = 0; z < a.ranges; ++z) s += partial[(ulong(z) * a.rows + dst[sg]) * (a.blocks * 128) + gb * 128 + lane + 32 * q];
      v[q] = s;
    }
    tf_xr_finish(j, y, a, cb, dst[sg], v, lane);
  }
  if (sg == 0 && lane == 0) atomic_store_explicit(arrive, 0u, memory_order_relaxed);
}

// The block at a job's width: whole bits 2 to 8 (job() refuses the half steps).
#define TF_XR_BITS(R, halves, ARGS)                \
  switch (halves) {                                \
    case 4: tf_xr_block<R, 2> ARGS; break;         \
    case 6: tf_xr_block<R, 3> ARGS; break;         \
    case 8: tf_xr_block<R, 4> ARGS; break;         \
    case 10: tf_xr_block<R, 5> ARGS; break;        \
    case 12: tf_xr_block<R, 6> ARGS; break;        \
    default: tf_xr_block<R, 8> ARGS; break;        \
  }

inline uint tf_xr_job(const device TfExl3Job* jobs, constant TfExl3RowsArgs& a, uint block) {
  uint ji = 0;
  while (ji + 1 < a.jobs && jobs[ji + 1].first_block <= block) ++ji;
  return ji;
}

// Dense: grid.x every job's blocks, grid.y row groups of R, grid.z K ranges; counts zeroed per (group, block).
template <uint R>
kernel void tf_exl3_rows(const device bfloat* x [[buffer(0)]],
                         const device TfExl3Job* jobs [[buffer(1)]],
                         device uchar* y [[buffer(2)]],
                         constant TfExl3RowsArgs& a [[buffer(3)]],
                         device float* partial [[buffer(4)]],
                         device atomic_uint* counts [[buffer(5)]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint3 groups [[threadgroups_per_grid]]) {
  threadgroup float part[KS * R * 128];
  threadgroup float book[1024];
  threadgroup float xs[2 * R * 128];
  tf_xr_fill(book, sg * 32 + lane, KS * 8 * 32);
  const device TfExl3Job& j = jobs[tf_xr_job(jobs, a, tg.x)];
  const uint r0 = tg.y * R;
  const uint live = min(R, a.rows - r0);
  uint src[R], dst[R];
  for (uint i = 0; i < R; ++i) src[i] = dst[i] = r0 + min(i, live - 1);
  device atomic_uint* arrive = counts + tg.y * groups.x + tg.x;
  const uint cb = tg.x - j.first_block;
  TF_XR_BITS(R, j.halves, (x, j, y, partial, arrive, a, cb, tg.x, tg.z, src, dst, live, lane, sg, part, book, xs));
}

// Routed: grid.y a slot (expert routes[slot], from routed_slots the shared one), grid.x its blocks, grid.z ranges.
kernel void tf_exl3_rows_routed(const device bfloat* x [[buffer(0)]],
                                const device TfExl3Job* jobs [[buffer(1)]],
                                device uchar* y [[buffer(2)]],
                                constant TfExl3RowsArgs& a [[buffer(3)]],
                                device float* partial [[buffer(4)]],
                                device atomic_uint* counts [[buffer(5)]],
                                const device uint* routes [[buffer(6)]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float part[KS * 128];
  threadgroup float book[1024];
  threadgroup float xs[2 * 128];
  const bool shared = tg.y >= a.routed_slots;
  const uint e = shared ? a.experts : routes[tg.y];
  if (!shared && e >= a.experts) return;
  tf_xr_fill(book, sg * 32 + lane, KS * 8 * 32);
  const uint per = a.blocks / a.kinds;
  const device TfExl3Job& j = jobs[e * a.kinds + tg.x / per];
  uint src[1] = {shared ? tg.y - a.shared_base : tg.y / a.repeat}, dst[1] = {tg.y};
  device atomic_uint* arrive = counts + tg.y * a.blocks + tg.x;
  const uint cb = tg.x % per;
  TF_XR_BITS(1, j.halves, (x, j, y, partial, arrive, a, cb, tg.x, tg.z, src, dst, 1u, lane, sg, part, book, xs));
}

// A window's routed rows by expert (G, M: mm_route_groups'), the shared expert at y = shared_group, 8 rows a pass.

// Each expert's weights decode once for all its rows; every row's sums are tf_exl3_rows_routed's.
kernel void tf_exl3_rows_grouped(const device bfloat* x [[buffer(0)]],
                                 const device TfExl3Job* jobs [[buffer(1)]],
                                 device uchar* y [[buffer(2)]],
                                 constant TfExl3RowsArgs& a [[buffer(3)]],
                                 device float* partial [[buffer(4)]],
                                 device atomic_uint* counts [[buffer(5)]],
                                 const device uint* G [[buffer(6)]],
                                 const device uint* M [[buffer(7)]],
                                 constant uint& shared_group [[buffer(8)]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float part[KS * 8 * 128];
  threadgroup float book[1024];
  threadgroup float xs[2 * 8 * 128];
  const uint g = tg.y, groups = G[0];
  const bool shared = g == shared_group;
  if (!shared && g >= groups) return;
  // members [first, end): routed ones are M's slots, the shared group's are the rows after the routed slots
  const uint e = shared ? a.experts : G[1 + 2 * g];
  const uint first = shared ? 0u : G[2 + 2 * g];
  const uint end = shared ? a.rows - a.routed_slots : (g + 1 < groups ? G[2 + 2 * (g + 1)] : a.routed_slots);
  tf_xr_fill(book, sg * 32 + lane, KS * 8 * 32);
  const uint per = a.blocks / a.kinds;
  const device TfExl3Job& j = jobs[e * a.kinds + tg.x / per];
  const uint cb = tg.x % per;
  for (uint m = first; m < end; m += 8) {
    const uint live = min(8u, end - m);
    uint src[8], dst[8];
    for (uint i = 0; i < 8; ++i) {
      const uint at = m + min(i, live - 1);
      dst[i] = shared ? a.routed_slots + at : M[at];
      src[i] = shared ? (a.repeat == 1 ? dst[i] : at - a.shared_base + a.routed_slots) : dst[i] / a.repeat;
    }
    // one arrival counter a chunk: its first member's place (unique in M, or past it for the shared rows)
    device atomic_uint* arrive = counts + (shared ? a.routed_slots + m : m) * a.blocks + tg.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    TF_XR_BITS(8, j.halves, (x, j, y, partial, arrive, a, cb, tg.x, tg.z, src, dst, live, lane, sg, part, book, xs));
  }
}

template [[host_name("tf_exl3_rows_r1")]] kernel void tf_exl3_rows<1>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r2")]] kernel void tf_exl3_rows<2>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r4")]] kernel void tf_exl3_rows<4>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r8")]] kernel void tf_exl3_rows<8>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
