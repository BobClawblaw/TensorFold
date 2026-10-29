// MiniMax-M3.1-Flash's EXL3 pack beside its EXL3 projections: bf16 embedding rows, fp16 index pair, bf16 DSpark rows.
#include <metal_stdlib>
using namespace metal;

// Row r of the fp32 residual R = embedding row ids[r] (bf16 [vocab, width]).
[[kernel]] void mm_embed_bf16(const device uint* ids [[buffer(0)]], const device bfloat* E [[buffer(1)]],
    constant uint2& a [[buffer(2)]], device float* R [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.x || p.y >= a.y) return;
  R[(size_t)p.y * a.x + p.x] = float(E[(size_t)ids[p.y] * a.x + p.x]);
}

// An fp16 [n, k] matrix at column y_col of the output rows.
struct mm_half_job { const device half* w; uint n, y_col, first; uint pad; };
struct mm_half_args { uint rows, k, jobs, x_stride, y_stride; };

// Eight rows of x against two columns a threadgroup, K in four simdgroup quarters summed in quarter order.
[[kernel]] void mm_half_rows(const device bfloat* x [[buffer(0)]], const device mm_half_job* jobs [[buffer(1)]],
    constant mm_half_args& a [[buffer(2)]], device bfloat* y [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  threadgroup float part[8][8];
  const uint col = tg.x * 2 + sg / 4, quarter = sg % 4;
  uint ji = 0;
  while (ji + 1 < a.jobs && jobs[ji + 1].first <= col) ++ji;
  const device mm_half_job& j = jobs[ji];
  const uint n = col - j.first;
  const uint r0 = tg.y * 8;
  const uint live = min(8u, a.rows - r0);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (n < j.n) {
    const device packed_half4* w = reinterpret_cast<const device packed_half4*>(j.w + (size_t)n * a.k);
    const uint k4 = a.k / 4, from = quarter * k4 / 4, to = (quarter + 1) * k4 / 4;
    for (uint q = from + lane; q < to; q += 32) {
      const float4 wv = float4(w[q]);
      for (uint i = 0; i < 8; ++i) {
        if (i >= live) break;
        const device bfloat* xr = x + (size_t)(r0 + i) * a.x_stride + 4 * q;
        acc[i] = fma(float(xr[0]), wv.x, fma(float(xr[1]), wv.y, fma(float(xr[2]), wv.z, fma(float(xr[3]), wv.w, acc[i]))));
      }
    }
  }
  for (uint i = 0; i < 8; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) part[sg][i] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (quarter == 0 && lane < live && n < j.n) {
    const uint b = sg;
    y[(size_t)(r0 + lane) * a.y_stride + j.y_col + n] = bfloat(((part[b][lane] + part[b + 1][lane]) + part[b + 2][lane]) + part[b + 3][lane]);
  }
}

struct mm_bf16_args { uint rows, n, k, x_stride, y_stride; };

// A bf16 [n, k] matrix (the pack's DSpark head, widened from MXFP8) on rows of x: eight rows, a column a simdgroup.
[[kernel]] void mm_bf16_rows(const device bfloat* x [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant mm_bf16_args& a [[buffer(2)]], device bfloat* y [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]]) {
  const uint n = tg.x * 8 + sg;
  if (n >= a.n) return;
  const uint r0 = tg.y * 8;
  const uint live = min(8u, a.rows - r0);
  const device packed_bfloat4* w = reinterpret_cast<const device packed_bfloat4*>(W + (size_t)n * a.k);
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (uint k4 = lane; k4 < a.k / 4; k4 += 32) {
    const float4 wv = float4(w[k4]);
    for (uint i = 0; i < 8; ++i) {
      if (i >= live) break;
      const device bfloat* xr = x + (size_t)(r0 + i) * a.x_stride + 4 * k4;
      acc[i] = fma(float(xr[0]), wv.x, fma(float(xr[1]), wv.y, fma(float(xr[2]), wv.z, fma(float(xr[3]), wv.w, acc[i]))));
    }
  }
  for (uint i = 0; i < live; ++i) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) y[(size_t)(r0 + i) * a.y_stride + n] = bfloat(s);
  }
}
