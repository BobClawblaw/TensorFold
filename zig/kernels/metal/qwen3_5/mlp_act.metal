// Qwen3.5 SiLU gate times up (N); a geometry's constants are prepended at compile time.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
typedef half float16_t;
[[kernel]] void custom_kernel_qwen35_mlp_act_bfloat16_t_bfloat16_t_bfloat16_t(
  const device bfloat16_t* GATE [[buffer(0)]],
  const device bfloat16_t* UP [[buffer(1)]],
  device bfloat16_t* HOUT [[buffer(2)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

  const uint i = thread_position_in_grid.x;
  const uint m = thread_position_in_grid.y;
  if (i >= uint(N)) return;
  const float gf = float(GATE[m * N + i]);
  HOUT[m * N + i] = bfloat(gf / (1.0f + metal::exp(-gf)) * float(UP[m * N + i]));

}
