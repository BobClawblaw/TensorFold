// EXL3 prompt chunks on the tensor units: 64 rows x 128 columns a threadgroup, a row's bits the same in any chunk.

// Rows rotated once a 128-deep block (x * su, Hadamard, fp16); each simdgroup decodes its two tiles into the operand.
#include "../nax.h"
using namespace tfp;

constant constexpr int BM = 64;
constant constexpr int XP = 128 + 8; // the rotated rows' pitch (fp16)
constant constexpr int SP = 128 + 4; // the epilogue's sums pitch (fp32)

struct TfExl3Job {
  const device uint* trellis;
  const device half* su;
  const device half* sv;
  ulong tile_k_stride, tile_n_stride; // in words
  uint halves, n, y_col, first_block;
};

struct TfExl3MmArgs {
  uint rows, k, jobs, kinds;
  uint experts, blocks, routed, scatter;
  ulong x_stride, y_stride;
};

inline float tf_xm_mul1(uint state) {
  const uint x = state * 0x83DCD12Du;
  const uint h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
  return float(half(fma(float(h), float(as_type<half>(ushort(0x1EEE))), float(as_type<half>(ushort(0xC931))))));
}

inline void tf_xm_hadamard(thread float (&v)[4], uint lane) {
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

// (lo | hi) += a * (b0 | b1) in fp16 with fp32 accumulation, full precision.
inline void tf_xm_mma(thread frag<float>& lo, thread frag<float>& hi, thread const frag<half>& a,
                      thread const frag<half>& b0, thread const frag<half>& b1) {
#ifndef TF_SIMD_FRAGS
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, float>();
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  TF_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
#else
  mma_16x32<false, false>(lo, hi, a, b0, b1);
#endif
}

// A tile's weight at fragment element e of this lane: its stream position's word index, shift and next word.
struct TfXmMap { ushort lo, off, hi; };

inline TfXmMap tf_xm_map(uint halves, uint r, uint c) {
  const uint l = (c & 7) * 4 + ((r >> 1) & 3);
  const uint e = (r & 1) | ((r >> 3) << 1) | ((c >> 3) << 2);
  const uint p = l * 8 + e, ring = 128 * halves, count = 4 * halves;
  const uint end = ((p + 1) * halves - ((halves & 1u) ? ((p + 1) & 1u) : 0u)) / 2;
  const uint start = (end + ring - 16) % ring;
  const uint wi = start / 32;
  return TfXmMap{ushort(wi), ushort(start % 32), ushort(wi + 1 == count ? 0 : wi + 1)};
}

inline half tf_xm_weight(const device uint* tile, TfXmMap m) {
  const uint lo = tile[m.lo], hi = tile[m.hi];
  const uint carry = (hi >> ((32 - m.off) & 31u)) & (0u - uint(m.off != 0));
  return half(tf_xm_mul1(((lo << m.off) | carry) >> 16));
}

// Rows [row0, row0 + live) of x through job j's column block cb; output row i at dst[i] (or row0 + i).
inline void tf_xm_block(const device bfloat* x, const device TfExl3Job& j, device bfloat* y,
                        constant TfExl3MmArgs& a, uint cb, uint row0, uint live, const device int* dst,
                        uint sg, uint lane, threadgroup half* xt, threadgroup float* sums) {
  const short2 home = frag_home(ushort(lane));
  TfXmMap maps[8];
  TF_UNROLL
  for (short e = 0; e < 8; e++) maps[e] = tf_xm_map(j.halves, uint(home.y + 8 * (e >> 2)), uint(home.x + TF_COL(e)));
  frag<float> acc[4][2];
  TF_UNROLL
  for (short m = 0; m < 4; m++) acc[m][0] = acc[m][1] = frag<float>(0);
  // this simdgroup's two tile columns
  const device uint* col = j.trellis + ulong(cb * 8 + 2 * sg) * j.tile_n_stride;
  for (uint kb = 0; kb < a.k; kb += 128) {
    // rows sg * 16 .. sg * 16 + 15: x * su, rotated, fp16
    for (uint i = 0; i < 16; ++i) {
      const uint r = sg * 16 + i;
      float v[4];
      for (uint q = 0; q < 4; ++q) {
        const uint k = kb + lane + 32 * q;
        v[q] = r < live ? float(x[ulong(row0 + r) * a.x_stride + k]) * float(j.su[k]) : 0.0f;
      }
      tf_xm_hadamard(v, lane);
      for (uint q = 0; q < 4; ++q) xt[r * XP + lane + 32 * q] = half(v[q] * 0.08838834764831845f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 8; ++t) {
      const device uint* t0 = col + ulong(kb / 16 + t) * j.tile_k_stride;
      const device uint* t1 = t0 + j.tile_n_stride;
      frag<half> b0, b1;
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        b0[e] = tf_xm_weight(t0, maps[e]);
        b1[e] = tf_xm_weight(t1, maps[e]);
      }
      TF_UNROLL
      for (short m = 0; m < 4; m++) {
        frag<half> am;
        frag_get(am, xt, XP, 16 * m, int(16 * t), home);
        tf_xm_mma(acc[m][0], acc[m][1], am, b0, b1);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // two halves of 32 rows: sums to threadgroup memory, then each row's 128 sums rotated back by one simdgroup
  for (uint half_ = 0; half_ < 2; ++half_) {
    TF_UNROLL
    for (short m = 0; m < 2; m++) {
      TF_UNROLL
      for (short e = 0; e < 8; e++) {
        const int rr = 16 * m + home.y + 8 * (e >> 2), cc = 32 * int(sg) + home.x + TF_COL(e);
        sums[rr * SP + cc] = acc[2 * half_ + m][0][e];
        sums[rr * SP + cc + 16] = acc[2 * half_ + m][1][e];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; ++i) {
      const uint r = sg * 8 + i, row = half_ * 32 + r;
      if (row >= live) break;
      float v[4];
      for (uint q = 0; q < 4; ++q) v[q] = sums[r * SP + lane + 32 * q];
      tf_xm_hadamard(v, lane);
      const ulong out = a.scatter ? ulong(dst[row0 + row]) : ulong(row0 + row);
      for (uint q = 0; q < 4; ++q) {
        const uint n = cb * 128 + lane + 32 * q;
        y[out * a.y_stride + j.y_col + n] = bfloat((v[q] * 0.08838834764831845f) * float(j.sv[n]));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Dense: grid.x every job's 128-column blocks in order, grid.y the 64-row tiles.
[[kernel]] void tf_exl3_mm(const device bfloat* x [[buffer(0)]], const device TfExl3Job* jobs [[buffer(1)]],
                           device bfloat* y [[buffer(2)]], constant TfExl3MmArgs& a [[buffer(3)]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half xt[BM * XP];
  threadgroup float* sums = reinterpret_cast<threadgroup float*>(xt);
  uint ji = 0;
  while (ji + 1 < a.jobs && jobs[ji + 1].first_block <= tg.x) ++ji;
  const uint row0 = tg.y * BM;
  tf_xm_block(x, jobs[ji], y, a, tg.x - jobs[ji].first_block, row0, min(uint(BM), a.rows - row0), nullptr, sg, lane, xt, sums);
}

// Expert-sorted rows: grid.y each expert's 64-row tiles (offsets), grid.x its blocks; scatter: row i to dst[i].
[[kernel]] void tf_exl3_mm_gather(const device bfloat* x [[buffer(0)]], const device TfExl3Job* jobs [[buffer(1)]],
                                  device bfloat* y [[buffer(2)]], constant TfExl3MmArgs& a [[buffer(3)]],
                                  const device int* offsets [[buffer(4)]], const device int* dst [[buffer(5)]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half xt[BM * XP];
  threadgroup float* sums = reinterpret_cast<threadgroup float*>(xt);
  // the tile's expert and rows: each expert's rows in tiles of BM, experts in order
  int before = 0, expert = -1, row = 0, rows = 0;
  const int tile = int(tg.y), total = int(a.rows), experts = int(a.experts);
  for (int base = 0; base < experts && expert < 0; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? offsets[e] : total, last = e + 1 < experts ? offsets[e + 1] : total;
    const int count = (last - first + BM - 1) / BM;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * BM;
      rows = min(BM, simd_shuffle(last, ushort(owner)) - row);
    }
    before = simd_shuffle(upto, ushort(31));
  }
  if (expert < 0 || rows <= 0) return;
  const uint per = a.blocks / a.kinds;
  tf_xm_block(x, jobs[uint(expert) * a.kinds + tg.x / per], y, a, tg.x % per, uint(row), uint(rows), dst, sg, lane, xt, sums);
}
