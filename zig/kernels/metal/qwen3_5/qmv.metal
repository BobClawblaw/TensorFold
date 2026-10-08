// Qwen3.5 affine 4-bit projection for 1 to RM rows on the FP32 ALUs (K, N, RM, VPL prepended at compile time).
// A simdgroup owns NR output rows; lane l reads VPL consecutive inputs of each (VPL nibbles of weights, a uint or a
// uint4), the 32 lanes covering 32 * VPL inputs an iteration. Each lane keeps a partial sum per (row, output) and
// the simdgroup tree-sums them at the end, so the bits are this path's own, not the matrix kernel's.
#include <metal_stdlib>
using namespace metal;
#define NR 4
#define SGS 4
inline float bfl(uint w) { return as_type<float>(w << 16); }
inline float bfh(uint w) { return as_type<float>(w & 0xFFFF0000u); }
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
  constexpr int G = K / 64;
  constexpr int SPAN = 32 * VPL;
  constexpr int IT = K / SPAN;
  const int n0 = (int(tg.x) * SGS + int(sg)) * NR;
  float acc[NR][RM];
  for (int nr = 0; nr < NR; nr++)
    for (int r = 0; r < RM; r++) acc[nr][r] = 0.0f;
  int nn[NR];
  for (int nr = 0; nr < NR; nr++) nn[nr] = min(n0 + nr, N - 1);
  const device uint* X32 = (const device uint*)X;
  for (int it = 0; it < IT; it++) {
    const int k0 = it * SPAN + int(lane) * VPL;
    const int g = k0 / 64;
    float xv[RM][VPL];
    float xs[RM];
    for (int r = 0; r < RM; r++) {
      const int rr = min(r, R - 1);
      const device uint* xp = X32 + (size_t(rr) * K + k0) / 2;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < VPL / 2; j++) { const uint w = xp[j]; xv[r][2 * j] = bfl(w); xv[r][2 * j + 1] = bfh(w); }
      float s = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < VPL; j++) s = fma(xv[r][j], 1.0f, s);
      xs[r] = s;
    }
    _Pragma("clang loop unroll(full)")
    for (int nr = 0; nr < NR; nr++) {
      const int n = nn[nr];
      uint wq[VPL / 8];
      if (VPL == 8) wq[0] = W[(size_t(n) * K + k0) / 8];
      else { const uint4 v = ((const device uint4*)W)[(size_t(n) * K + k0) / 32]; wq[0] = v.x; wq[1] = v.y; wq[2] = v.z; wq[3] = v.w; }
      const float sc = float(SC[size_t(n) * G + g]);
      const float bi = float(BI[size_t(n) * G + g]);
      float wf[VPL];
      _Pragma("clang loop unroll(full)")
      for (int j = 0; j < VPL; j++) wf[j] = float((wq[j / 8] >> (4 * (j % 8))) & 0xFu);
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RM; r++) {
        float p = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (int j = 0; j < VPL; j++) p = fma(wf[j], xv[r][j], p);
        acc[nr][r] = fma(bi, xs[r], fma(sc, p, acc[nr][r]));
      }
    }
  }
  for (int nr = 0; nr < NR; nr++)
    for (int r = 0; r < RM; r++) {
      const float v = simd_sum(acc[nr][r]);
      if (lane == 0 && r < R && n0 + nr < N) OUT[size_t(r) * N + n0 + nr] = bfloat(v);
    }
}
