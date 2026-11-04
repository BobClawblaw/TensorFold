// Gate controls for Unlimited-OCR-2's matvec kernels (uocr_vision.cu, uocr_decoder.cu), each breaking one of them.
#pragma once
#include <cuda_fp16.h>
#include "uocr_common.cuh"

namespace tf_ctl_uocr_matvec {
using namespace tf_uocr;

// tf_uocr_matvec's copy: Short stops before each row's last 8 inputs, Half rounds every partial sum to fp16.
template <bool Short, bool Half>
__device__ __forceinline__ void matvec(const bf16* __restrict__ x, const bf16* __restrict__ W, bf16* __restrict__ y, int n,
                                       int k) {
  const int sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= n) return;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; Short ? i + 8 < k : i < k; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      if (row0 + r >= n) break;
      float wv[8];
      load8(W + (size_t)(row0 + r) * k + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += xv[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += xv[j] * wv[j];
      acc[r] = Half ? __half2float(__float2half(acc[r] + (d0 + d1))) : acc[r] + (d0 + d1);
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float s = warp_sum(acc[r]);
    if (lane == 0 && row0 + r < n) y[row0 + r] = to(s);
  }
}

struct Expert {
  const bf16* gate;
  const bf16* up;
  const bf16* down;
};

// tf_uocr_expert_gu's copy: Slot0 runs every slot on slot 0's expert, NoPoint drops silu's bf16 rounding point.
template <bool Slot0, bool NoPoint>
__device__ __forceinline__ void expert_gu(const bf16* __restrict__ x, const Expert* __restrict__ tab,
                                          const unsigned* __restrict__ ids, bf16* __restrict__ act, int ffn) {
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const Expert e = tab[ids[Slot0 ? 0 : s]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= ffn) return;
  float g[4] = {0.0f, 0.0f, 0.0f, 0.0f}, u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < 1280; i += 256) {
    float xv[8];
    load8(x + i, xv);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wg[8], wu[8];
      load8(e.gate + (size_t)(row0 + r) * 1280 + i, wg);
      load8(e.up + (size_t)(row0 + r) * 1280 + i, wu);
      float g0 = 0.0f, g1 = 0.0f, u0 = 0.0f, u1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        g0 += xv[j] * wg[j];
        u0 += xv[j] * wu[j];
      }
#pragma unroll
      for (int j = 4; j < 8; j++) {
        g1 += xv[j] * wg[j];
        u1 += xv[j] * wu[j];
      }
      g[r] += g0 + g1;
      u[r] += u0 + u1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float gs = bf(warp_sum(g[r])), us = bf(warp_sum(u[r]));
    const float sl = gs / (1.0f + expf(-gs));
    if (lane == 0) act[(size_t)s * ffn + row0 + r] = to((NoPoint ? sl : bf(sl)) * us);
  }
}

// The greedy pick's breaks: a NaN read as +Inf, ties to the higher id, EOS never barred.
enum Pick { nan_as_inf, high_tie, eos_open };

// tf_uocr_argmax's copy with one of the breaks above.
template <Pick B>
__device__ __forceinline__ void argmax(const bf16* __restrict__ logits, int n, int bar_eos, unsigned* __restrict__ out) {
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int at = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += 1024) {
    if (B != eos_open && bar_eos && i == 1) continue;
    const float v0 = ld(logits + i);
    const float v = B == nan_as_inf && isnan(v0) ? INFINITY : v0;
    if (v > best) {
      best = v;
      at = i;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && (B == high_tie ? oi > at : oi < at))) {
      best = ov;
      at = oi;
    }
  }
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (lane == 0) {
    bv[w] = best;
    bi[w] = at;
  }
  __syncthreads();
  if (w != 0) return;
  best = bv[lane];
  at = bi[lane];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, at, o);
    if (ov > best || (ov == best && (B == high_tie ? oi > at : oi < at))) {
      best = ov;
      at = oi;
    }
  }
  if (lane == 0) out[0] = (unsigned)(at == 0x7fffffff ? 0 : at);
}

}  // namespace tf_ctl_uocr_matvec

// Gate 3: the last 8 inputs of a row dropped, which every k must show.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_short_k(const tf_uocr::bf16* x, const tf_uocr::bf16* W,
                                                                             tf_uocr::bf16* y, int n, int k) {
  tf_ctl_uocr_matvec::matvec<true, false>(x, W, y, n, k);
}

// Gate 4: the partial sums in fp16, which gate 4's integers overflow.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_half_sums(const tf_uocr::bf16* x, const tf_uocr::bf16* W,
                                                                               tf_uocr::bf16* y, int n, int k) {
  tf_ctl_uocr_matvec::matvec<false, true>(x, W, y, n, k);
}

// Gate 3: every slot on slot 0's expert.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_gu_slot0(const tf_uocr::bf16* x,
                                                                              const tf_ctl_uocr_matvec::Expert* tab,
                                                                              const unsigned* ids, tf_uocr::bf16* act, int ffn) {
  tf_ctl_uocr_matvec::expert_gu<true, false>(x, tab, ids, act, ffn);
}

// Gate 3: silu without its bf16 rounding point.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_gu_no_point(const tf_uocr::bf16* x,
                                                                                 const tf_ctl_uocr_matvec::Expert* tab,
                                                                                 const unsigned* ids, tf_uocr::bf16* act,
                                                                                 int ffn) {
  tf_ctl_uocr_matvec::expert_gu<false, true>(x, tab, ids, act, ffn);
}

// Gate 3: tf_uocr_expert_down's copy with every slot on slot 0's expert.
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_matvec_down_slot0(const tf_uocr::bf16* __restrict__ act,
                                                                                const tf_ctl_uocr_matvec::Expert* __restrict__ tab,
                                                                                const unsigned* __restrict__ ids,
                                                                                tf_uocr::bf16* __restrict__ slots, int ffn) {
  using namespace tf_uocr;
  const int s = blockIdx.y, sg = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const tf_ctl_uocr_matvec::Expert e = tab[ids[0]];
  const int row0 = (blockIdx.x * 8 + sg) * 4;
  if (row0 >= 1280) return;
  const bf16* a = act + (size_t)s * ffn;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = lane * 8; i < ffn; i += 256) {
    float av[8];
    load8(a + i, av);
#pragma unroll
    for (int r = 0; r < 4; r++) {
      float wv[8];
      load8(e.down + (size_t)(row0 + r) * ffn + i, wv);
      float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++) d0 += av[j] * wv[j];
#pragma unroll
      for (int j = 4; j < 8; j++) d1 += av[j] * wv[j];
      acc[r] += d0 + d1;
    }
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const float v = warp_sum(acc[r]);
    if (lane == 0) slots[(size_t)s * 1280 + row0 + r] = to(v);
  }
}

// Gate 1: a NaN logit read as +Inf, so it wins the pick.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_matvec_argmax_nan(const tf_uocr::bf16* logits, int n, int bar_eos,
                                                                                 unsigned* out) {
  tf_ctl_uocr_matvec::argmax<tf_ctl_uocr_matvec::nan_as_inf>(logits, n, bar_eos, out);
}

// Gate 3: equal maxima resolved to the higher id.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_matvec_argmax_tie(const tf_uocr::bf16* logits, int n, int bar_eos,
                                                                                 unsigned* out) {
  tf_ctl_uocr_matvec::argmax<tf_ctl_uocr_matvec::high_tie>(logits, n, bar_eos, out);
}

// Gate 3: the page guard's EOS bar ignored.
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_uocr_matvec_argmax_eos(const tf_uocr::bf16* logits, int n, int bar_eos,
                                                                                 unsigned* out) {
  tf_ctl_uocr_matvec::argmax<tf_ctl_uocr_matvec::eos_open>(logits, n, bar_eos, out);
}
