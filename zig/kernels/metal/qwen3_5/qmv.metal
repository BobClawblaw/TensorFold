// Qwen3.5 affine 4-bit projection on the FP32 ALUs for any row count (K, N prepended at compile time).
// A simdgroup owns NR output rows; lane l reads VPL consecutive inputs of each (16 bytes of nibbles), the 32 lanes
// covering 32 * VPL inputs an iteration, and the simdgroup tree-sums the lanes' partials at the end. Rows go in
// blocks of RB with the weights re-read per block, so a row's arithmetic is one fixed chain whatever the batch
// holds: the bytes of a row do not depend on the other rows. Prefill chunks beyond the decode widths stay on the
// matrix kernel.
#include <metal_stdlib>
using namespace metal;
#define NR 4
#define SGS 4
#define VPL 32
#define RB 4
#define SPAN (32 * VPL)
#define IT (K / SPAN)
#define G (K / 64)
inline float bfl(uint w) { return as_type<float>(w << 16); }
inline float bfh(uint w) { return as_type<float>(w & 0xFFFF0000u); }

template <int R_>
inline void rows_block(const device uint* X32, const device uint4* W4, const device bfloat* SC, const device bfloat* BI,
                       device bfloat* OUT, int R, int rb, int n0, uint lane) {
  float acc[NR][R_];
  int nn[NR];
  int rr[R_];
  for (int nr = 0; nr < NR; nr++) {
    nn[nr] = min(n0 + nr, N - 1);
    for (int r = 0; r < R_; r++) acc[nr][r] = 0.0f;
  }
  for (int r = 0; r < R_; r++) rr[r] = min(rb + r, R - 1);
  for (int it = 0; it < IT; it++) {
    const int k0 = it * SPAN + int(lane) * VPL;
    const int g = k0 / 64;
    uint4 wq[NR];
    float sc[NR], bi[NR];
    _Pragma("clang loop unroll(full)")
    for (int nr = 0; nr < NR; nr++) {
      wq[nr] = W4[(size_t(nn[nr]) * K + k0) / 32];
      sc[nr] = float(SC[size_t(nn[nr]) * G + g]);
      bi[nr] = float(BI[size_t(nn[nr]) * G + g]);
    }
    float p[NR][R_];
    float xs[R_];
    for (int nr = 0; nr < NR; nr++)
      for (int r = 0; r < R_; r++) p[nr][r] = 0.0f;
    for (int r = 0; r < R_; r++) xs[r] = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (int jj = 0; jj < VPL / 8; jj++) {
      float xv[R_][8];
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < R_; r++) {
        const uint4 v = ((const device uint4*)(X32 + (size_t(rr[r]) * K + k0) / 2))[jj];
        xv[r][0] = bfl(v.x); xv[r][1] = bfh(v.x); xv[r][2] = bfl(v.y); xv[r][3] = bfh(v.y);
        xv[r][4] = bfl(v.z); xv[r][5] = bfh(v.z); xv[r][6] = bfl(v.w); xv[r][7] = bfh(v.w);
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; j++) xs[r] = fma(xv[r][j], 1.0f, xs[r]);
      }
      _Pragma("clang loop unroll(full)")
      for (int nr = 0; nr < NR; nr++) {
        const uint w = wq[nr][jj];
        float wf[8];
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < 8; j++) wf[j] = float((w >> (4 * j)) & 0xFu);
        _Pragma("clang loop unroll(full)")
        for (int r = 0; r < R_; r++) {
          _Pragma("clang loop unroll(full)")
          for (int j = 0; j < 8; j++) p[nr][r] = fma(wf[j], xv[r][j], p[nr][r]);
        }
      }
    }
    _Pragma("clang loop unroll(full)")
    for (int nr = 0; nr < NR; nr++)
      for (int r = 0; r < R_; r++) acc[nr][r] = fma(bi[nr], xs[r], fma(sc[nr], p[nr][r], acc[nr][r]));
  }
  for (int nr = 0; nr < NR; nr++)
    for (int r = 0; r < R_; r++) {
      const float v = simd_sum(acc[nr][r]);
      if (lane == 0 && rb + r < R && n0 + nr < N) OUT[size_t(rb + r) * N + n0 + nr] = bfloat(v);
    }
}

[[kernel]] void qwen35_qmv(
  const device bfloat* X [[buffer(0)]],
  const constant int* X_shape [[buffer(1)]],
  const device uint32_t* W [[buffer(2)]],
  const device bfloat* SC [[buffer(3)]],
  const device bfloat* BI [[buffer(4)]],
  const constant float* ONE [[buffer(5)]],
  device bfloat* OUT [[buffer(6)]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = X_shape[0];
  const int n0 = (int(tg.x) * SGS + int(sg)) * NR;
  const device uint* X32 = (const device uint*)X;
  const device uint4* W4 = (const device uint4*)W;
  int rb = 0;
  for (; rb + RB <= R; rb += RB) rows_block<RB>(X32, W4, SC, BI, OUT, R, rb, n0, lane);
  switch (R - rb) {
    case 1: rows_block<1>(X32, W4, SC, BI, OUT, R, rb, n0, lane); break;
    case 2: rows_block<2>(X32, W4, SC, BI, OUT, R, rb, n0, lane); break;
    case 3: rows_block<3>(X32, W4, SC, BI, OUT, R, rb, n0, lane); break;
    default: break;
  }
}
