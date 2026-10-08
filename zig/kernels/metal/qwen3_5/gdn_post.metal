// Qwen3.5 DeltaNet output norm and z gate (NV, DV, ZS, ZO); a geometry's constants are prepended at compile time.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_qwen35_gdn_post_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t(
  const device bfloat16_t* Y [[buffer(0)]],
  const device bfloat16_t* Z [[buffer(1)]],
  const device bfloat16_t* NW [[buffer(2)]],
  const constant float* eps [[buffer(3)]],
  device bfloat16_t* OUT [[buffer(4)]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {

  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    OUT[m * NV * DV + hv * DV + d] = bfloat(zf / (1.0f + metal::exp(-zf)) * x);
  }

}
