// EXL3 rows: a window's rows through resident mul1 projections, both rotations folded in, one dispatch a call.

// 128 output columns a threadgroup over one K range; the last range to arrive sums the partials in range order.
#include <metal_stdlib>
using namespace metal;

#define KS 2

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

// Rows src[0..R) through job j's column block cb over K range `range`; partials or finished rows out.
template <uint R>
inline void tf_xr_block(const device bfloat* x, const device TfExl3Job& j, device uchar* y, device float* partial,
                        device atomic_uint* arrive, constant TfExl3RowsArgs& a, uint cb, uint gb, uint range,
                        thread const uint (&src)[R], thread const uint (&dst)[R], uint live,
                        uint lane, uint sg, threadgroup float* part, threadgroup uint* words) {
  const uint ct = sg % 8, ks = sg / 8;
  const uint halves = j.halves, count = 4 * halves; // words a tile
  threadgroup uint* tw = words + sg * 64;
  // each weight's 16-bit window: words lo and hi of the tile, lo << off joined by hi >> rs (nothing when off is 0)
  ushort lo_at[8], hi_at[8];
  uchar off_at[8], rs_at[8];
  uint keep[8];
  for (uint q = 0; q < 8; ++q) {
    const uint p = lane + 32 * q, ring = 128 * halves;
    const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
    const uint start = (end + ring - 16) % ring;
    const uint wi = start / 32, off = start % 32;
    lo_at[q] = ushort(wi);
    hi_at[q] = ushort(wi + 1 == count ? 0 : wi + 1);
    off_at[q] = uchar(off);
    rs_at[q] = uchar((32 - off) & 31u);
    keep[q] = off != 0 ? 0xFFFFFFFFu : 0u;
  }
  // this lane's k row within a 16x16 tile; its eight weights are columns q + 8 * (lane bit 2)
  const uint r = 2 * (lane / 8) + (lane & 1u) + 8 * ((lane >> 1) & 1u);
  float acc[R][8];
  for (uint i = 0; i < R; ++i) for (uint q = 0; q < 8; ++q) acc[i][q] = 0.0f;
  const device uint* col = j.trellis + ulong(cb * 8 + ct) * j.tile_n_stride;
  const uint step = KS * a.ranges * 128;
  uint kb = (range * KS + ks) * 128;
  // the next tile's words, loaded a tile ahead
  uint w0 = 0, w1 = 0;
  if (kb < a.k) {
    const device uint* tile = col + ulong(kb / 16) * j.tile_k_stride;
    w0 = lane < count ? tile[lane] : 0;
    w1 = lane + 32 < count ? tile[lane + 32] : 0;
  }
  for (; kb < a.k; kb += step) {
    float xv[R][4];
    for (uint i = 0; i < R; ++i) {
      for (uint q = 0; q < 4; ++q) {
        const uint k = kb + lane + 32 * q;
        xv[i][q] = i < live ? float(x[ulong(src[i]) * a.x_stride + k]) * float(j.su[k]) : 0.0f;
      }
      tf_xr_hadamard(xv[i], lane);
      for (uint q = 0; q < 4; ++q) xv[i][q] = float(half(xv[i][q] * 0.08838834764831845f));
    }
    for (uint t = 0; t < 8; ++t) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
      tw[lane] = w0;
      tw[lane + 32] = w1;
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint nk = t < 7 ? kb + 16 * (t + 1) : kb + step;
      if (nk < a.k) {
        const device uint* tile = col + ulong(nk / 16) * j.tile_k_stride;
        w0 = lane < count ? tile[lane] : 0;
        w1 = lane + 32 < count ? tile[lane + 32] : 0;
      }
      float w[8];
      for (uint q = 0; q < 8; ++q) {
        const uint lo = tw[lo_at[q]], hi = tw[hi_at[q]];
        w[q] = tf_xr_mul1(((lo << off_at[q]) | ((hi >> rs_at[q]) & keep[q])) >> 16);
      }
      for (uint i = 0; i < R; ++i) {
        const float xr = simd_shuffle(xv[i][t / 2], ushort(16 * (t & 1u) + r));
        for (uint q = 0; q < 8; ++q) acc[i][q] = fma(xr, w[q], acc[i][q]);
      }
    }
  }
  // lanes with the same column half (lane bit 2) hold one column's partial sums
  for (uint i = 0; i < R; ++i) {
    for (uint q = 0; q < 8; ++q) {
      float v = acc[i][q];
      v += simd_shuffle_xor(v, ushort(1));
      v += simd_shuffle_xor(v, ushort(2));
      v += simd_shuffle_xor(v, ushort(8));
      v += simd_shuffle_xor(v, ushort(16));
      acc[i][q] = v;
    }
    if (lane == 0 || lane == 4)
      for (uint q = 0; q < 8; ++q) part[(ks * R + i) * 128 + ct * 16 + q + 2 * lane] = acc[i][q];
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
  threadgroup uint words[KS * 8 * 64];
  const device TfExl3Job& j = jobs[tf_xr_job(jobs, a, tg.x)];
  const uint r0 = tg.y * R;
  const uint live = min(R, a.rows - r0);
  uint src[R], dst[R];
  for (uint i = 0; i < R; ++i) src[i] = dst[i] = r0 + min(i, live - 1);
  tf_xr_block<R>(x, j, y, partial, counts + tg.y * groups.x + tg.x, a, tg.x - j.first_block, tg.x, tg.z, src, dst, live, lane, sg, part, words);
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
  threadgroup uint words[KS * 8 * 64];
  const bool shared = tg.y >= a.routed_slots;
  const uint e = shared ? a.experts : routes[tg.y];
  if (!shared && e >= a.experts) return;
  const uint per = a.blocks / a.kinds;
  const device TfExl3Job& j = jobs[e * a.kinds + tg.x / per];
  uint src[1] = {shared ? tg.y - a.shared_base : tg.y / a.repeat}, dst[1] = {tg.y};
  tf_xr_block<1>(x, j, y, partial, counts + tg.y * a.blocks + tg.x, a, tg.x % per, tg.x, tg.z, src, dst, 1, lane, sg, part, words);
}

template [[host_name("tf_exl3_rows_r1")]] kernel void tf_exl3_rows<1>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r2")]] kernel void tf_exl3_rows<2>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r4")]] kernel void tf_exl3_rows<4>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
template [[host_name("tf_exl3_rows_r8")]] kernel void tf_exl3_rows<8>(const device bfloat*, const device TfExl3Job*, device uchar*, constant TfExl3RowsArgs&, device float*, device atomic_uint*, uint, uint, uint3, uint3);
