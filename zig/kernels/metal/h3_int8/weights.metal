#include <metal_stdlib>
using namespace metal;

// ---- MiniMax H3 / FastH3: the block projections to int8 at load, with low-rank adapters (LoRAs) merged first.
// These kernels depend on no case, so the loader compiles them on their own and quantizes the checkpoint while the
// prompt's case is still being exported.

// A projection stored (N, K) as checkpoints keep it, to int8 (K, N) with one scale per output channel n.
// W: (N, K) bf16. P: K, N. W8: (K, N). WS: (N). Threadgroups [N, 1, 1] of [32, 1, 1].
[[kernel]] void h3_quant_weight_t(
  const device bfloat* W [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* W8 [[buffer(2)]],
  device float* WS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[0], N = P[1], n = int(tg.x);
  const device bfloat* w = W + long(n) * K;
  float top = 0.0f;
  for (int k = int(lane); k < K; k += 32) top = max(top, abs(float(w[k])));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) WS[n] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int k = int(lane); k < K; k += 32) W8[long(k) * N + n] = int8_t(clamp(int(rint(float(w[k]) * inverse)), -127, 127));
}

// M[n, k] = W[n, k] + sum over r of B[n, r] A[r, k], in float32: an adapter's update on a bf16 projection, as
// families/h3/lora.py `merge` adds it (the strengths are already folded into B).
// W: (N, K) bf16. A: (R, K) float. B: (N, R) float. P: K, N, R. M: (N, K) float. Threads [K, N, 1].
[[kernel]] void h3_lora_merge(
  const device bfloat* W [[buffer(0)]],
  const device float* A [[buffer(1)]],
  const device float* B [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  device float* M [[buffer(4)]],
  uint3 at [[thread_position_in_grid]]) {
  const int K = P[0], N = P[1], R = P[2];
  const int k = int(at.x), n = int(at.y);
  if (k >= K || n >= N) return;
  const device float* b = B + long(n) * R;
  float sum = 0.0f;
  for (int r = 0; r < R; r++) sum = fma(b[r], A[long(r) * K + k], sum);
  M[long(n) * K + k] = float(W[long(n) * K + k]) + sum;
}

// h3_quant_weight_t for a merged float32 projection: the update is rounded once, with the weight, to int8.
// M: (N, K) float. P: K, N. W8: (K, N). WS: (N). Threadgroups [N, 1, 1] of [32, 1, 1].
[[kernel]] void h3_quant_merged_t(
  const device float* M [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* W8 [[buffer(2)]],
  device float* WS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[0], N = P[1], n = int(tg.x);
  const device float* w = M + long(n) * K;
  float top = 0.0f;
  for (int k = int(lane); k < K; k += 32) top = max(top, abs(w[k]));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) WS[n] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int k = int(lane); k < K; k += 32) W8[long(k) * N + n] = int8_t(clamp(int(rint(w[k] * inverse)), -127, 127));
}
