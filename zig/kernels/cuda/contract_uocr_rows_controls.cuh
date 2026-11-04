// Gate controls for Unlimited-OCR-2's rows kernels (uocr_vision.cu, uocr_decoder.cu), each breaking one of them.
#pragma once
#include "uocr_common.cuh"

namespace tf_ctl_uocr_rows {
using namespace tf_uocr;

// The controls image builds with fmad on: __fmul_rn keeps each product its own rounding, as the glue's --fmad=false.
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }

// decoder.cu's page2d angle: 63 token frequencies, then the page parity at frequency 1.
__device__ __forceinline__ float angle(const float* inv_tok, unsigned tok, unsigned page, int i) {
  const int j = i % 64;
  return j < 63 ? (float)tok * inv_tok[j] : (float)(page % 2);
}

// tf_uocr_add with its one rounding toward zero.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_add_rz(const bf16* __restrict__ a, const bf16* __restrict__ b,
                                                                          bf16* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16_rz(ld(a + i) + ld(b + i));
}

// tf_uocr_layer_norm with (x - mean) inv rounded to bf16 before the fp32 affine.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_layer_norm_xhat(const bf16* __restrict__ x,
                                                                                   const bf16* __restrict__ w,
                                                                                   const bf16* __restrict__ b,
                                                                                   bf16* __restrict__ y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float mean = block_sum(s, part) / (float)width;
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = ld(xr + i) - mean;
    v += mul(d, d);
  }
  const float inv = 1.0f / sqrtf(block_sum(v, part) / (float)width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to(mul(bf(mul(ld(xr + i) - mean, inv)), ld(w + i)) + ld(b + i));
}

// tf_uocr_layer_norm2d with bf16((x - u) / den) left in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_layer_norm2d_quot(const bf16* __restrict__ x,
                                                                                     const bf16* __restrict__ w,
                                                                                     const bf16* __restrict__ b,
                                                                                     bf16* __restrict__ y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  bf16* yr = y + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += ld(xr + i);
  const float u = bf(block_sum(s, part) / (float)width);
  float v = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(ld(xr + i) - u);
    v += bf(mul(d, d));
  }
  const float var = bf(block_sum(v, part) / (float)width);
  const float den = bf(sqrtf(bf(var + eps)));
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float d = bf(ld(xr + i) - u) / den;
    yr[i] = to(bf(mul(ld(w + i), d)) + ld(b + i));
  }
}

// tf_uocr_gelu with 1 + erf rounded to bf16 where the module keeps it in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_gelu_e1(bf16* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  x[i] = to(0.5f * v * bf(1.0f + erf_(v * 0.70710678118654752f)));
}

// tf_uocr_quick_gelu with bf16(sigmoid) left in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_quick_gelu_sig(bf16* __restrict__ x, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(x + i);
  const float z = bf(1.702f * v);
  const float s = 1.0f / (1.0f + expf(-z));
  x[i] = to(v * s);
}

// tf_uocr_rms_norm with bf16(x r) left in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_rms_norm_xr(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                               bf16* __restrict__ y, int width, float eps) {
  __shared__ float part[33];
  const bf16* xr = x + (size_t)blockIdx.x * width;
  float s = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const float v = ld(xr + i);
    s += mul(v, v);
  }
  s = block_sum(s, part);
  const float r = 1.0f / sqrtf(s / (float)width + eps);
  bf16* yr = y + (size_t)blockIdx.x * width;
  for (int i = threadIdx.x; i < width; i += blockDim.x) yr[i] = to(ld(w + i) * (ld(xr + i) * r));
}

// tf_uocr_rope with its two bf16 products left in fp32.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_rows_rope_prod(bf16* __restrict__ x, const unsigned* __restrict__ tok,
                                                                             const unsigned* __restrict__ page,
                                                                             const float* __restrict__ inv_tok, int stride) {
  const int d = threadIdx.x, row = blockIdx.x, head = blockIdx.y;
  bf16* h = x + (size_t)row * stride + head * 128;
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), s = bf(sinf(ang));
  const float rh = d < 64 ? -ld(h + d + 64) : ld(h + d - 64);
  const bf16 out = to(mul(ld(h + d), c) + mul(rh, s));
  __syncthreads();
  h[d] = out;
}

// tf_uocr_router_norm with the normed row kept in fp32 where the module rounds it to bf16.
extern "C" __global__ void __launch_bounds__(128) tf_ctl_uocr_rows_router_norm_row(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                                                   const unsigned* __restrict__ tok,
                                                                                   const unsigned* __restrict__ page,
                                                                                   const float* __restrict__ inv_tok,
                                                                                   float* __restrict__ y) {
  __shared__ float part[33];
  __shared__ float normed[128];
  const int row = blockIdx.x, d = threadIdx.x;
  float v = ld(x + (size_t)row * 128 + d);
  if (!isfinite(v)) v = 0.0f;
  const float s = block_sum(v * v, part);
  float n = clampf(v * (1.0f / sqrtf(s / 128.0f + 1e-6f)), -UD_CLAMP, UD_CLAMP);
  n = clampf(n * (1.0f + ld(w + d)), -UD_CLAMP, UD_CLAMP);
  normed[d] = n;
  __syncthreads();
  const float ang = angle(inv_tok, tok[row], page[row], d);
  const float c = bf(cosf(ang)), sn = bf(sinf(ang));
  const float rh = d < 64 ? -normed[d + 64] : normed[d - 64];
  float out = bf(bf(normed[d] * c) + bf(rh * sn));
  if (!isfinite(out)) out = 0.0f;
  y[(size_t)row * 128 + d] = out;
}

// tf_uocr_silu_mul with bf16(silu) left in fp32.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_rows_silu_mul_silu(const bf16* __restrict__ g, const bf16* __restrict__ u,
                                                                                 bf16* __restrict__ y, long long n) {
  const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
  if (i >= n) return;
  const float v = ld(g + i);
  y[i] = to(v / (1.0f + expf(-v)) * ld(u + i));
}

}  // namespace tf_ctl_uocr_rows
