// Gate controls for Unlimited-OCR-2's route kernels (uocr_decoder.cu), each a copy that breaks one of them.
#pragma once
#include "uocr_common.cuh"

namespace tf_ctl_uocr_route {
using namespace tf_uocr;

__device__ __forceinline__ int warp_max_int(int v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// The planted breaks of tf_uocr_route, one a control.
enum Break { tie_high = 0, gap_short = 1, page_last = 2 };

// The page test: a block holds an image token of page qp (the page_last break leaves out a block's last page).
template <int B>
__device__ __forceinline__ bool in_page(const int* bmin, const int* bmax, int c, int qp) {
  return bmin[c] >= 0 && bmin[c] <= qp && (B == page_last ? qp < bmax[c] : qp <= bmax[c]);
}

// tf_uocr_route with break B: ties to the higher block, the local gap one short, or a block's last page left out.
template <int B>
__device__ __forceinline__ void route(const float* __restrict__ idx_q, const float* __restrict__ kbar,
                                      const int* __restrict__ bmin, const int* __restrict__ bmax,
                                      const int* __restrict__ qpage, const unsigned* __restrict__ qpos,
                                      int* __restrict__ sel, float* __restrict__ bias, int nblk, int complete,
                                      float scale, float idx_scale, float floor_, int round_bias) {
  __shared__ float x[4096];
  __shared__ unsigned char cand[4096];
  const int q = blockIdx.x, lane = threadIdx.x;
  const float* iq = idx_q + (size_t)q * 128;
  const int qp = qpage[q];
  const int last_remote = (int)(qpos[q] / 32) - (B == gap_short ? 3 : 4);
  bool any_page = false;
  if (qp >= 0) {
    for (int c = lane; c < nblk; c += 32) any_page = any_page || in_page<B>(bmin, bmax, c, qp);
    any_page = __any_sync(0xffffffffu, any_page);
  }
  bool any_cand = false;
  for (int c = lane; c < nblk; c += 32) {
    const bool remote = c <= last_remote && c < complete;
    const bool allow = !any_page || in_page<B>(bmin, bmax, c, qp);
    float z = UD_NEG;
    if (remote) {
      const float* kb = kbar + (size_t)c * 128;
      float s = 0.0f;
      for (int d = 0; d < 128; d++) s += iq[d] * kb[d];
      z = clampf(s * scale, -UD_CLAMP, UD_CLAMP);
    }
    x[c] = z;
    cand[c] = (unsigned char)(remote ? (allow ? 2 : 1) : 0);
    any_cand = any_cand || (remote && allow);
  }
  any_cand = __any_sync(0xffffffffu, any_cand);
  __syncwarp();
  for (int c = lane; c < nblk; c += 32) {
    const bool is = any_cand ? cand[c] == 2 : cand[c] >= 1;
    cand[c] = (unsigned char)is;
    float v = is ? x[c] * idx_scale : UD_ENTMAX_NEG;
    if (!isfinite(v)) v = UD_ENTMAX_NEG;
    x[c] = v;
  }
  __syncwarp();
  const int M = max(1, min(16, nblk));
  float xv[16];
  int xi[16];
  for (int r = 0; r < M; r++) {
    float best = -INFINITY;
    int at = B == tie_high ? -1 : 0x7fffffff;
    for (int c = lane; c < nblk; c += 32) {
      bool taken = false;
      for (int t = 0; t < r; t++) taken = taken || xi[t] == c;
      const bool better = B == tie_high ? x[c] >= best : (x[c] > best || (x[c] == best && c < at));
      if (!taken && better) {
        best = x[c];
        at = c;
      }
    }
    const float top = warp_max(best);
    const int idx = B == tie_high ? warp_max_int(best == top ? at : -1) : warp_min_int(best == top ? at : 0x7fffffff);
    xv[r] = top;
    xi[r] = idx;
  }
  if (lane != 0) return;
  const float mx = xv[0];
  float X[16], tau[16];
  float cs = 0.0f, cs2 = 0.0f;
  int support = 0;
  for (int k = 0; k < M; k++) {
    X[k] = (xv[k] - mx) / 2.0f;
    cs += X[k];
    cs2 += X[k] * X[k];
    const float rho = (float)(k + 1);
    const float mean = cs / rho, mean_sq = cs2 / rho;
    const float ss = rho * (mean_sq - mean * mean);
    const float delta = fmaxf((1.0f - ss) / rho, 0.0f);
    tau[k] = mean - sqrtf(delta);
    support += tau[k] <= X[k] ? 1 : 0;
  }
  const float tau_star = tau[max(support, 1) - 1];
  const int K = max(1, min(12, M));
  float logp[12];
  float total = 0.0f;
  int count = 0;
  for (int s = 0; s < K; s++) {
    const float d = fmaxf(X[s] - tau_star, 0.0f);
    logp[s] = logf(fmaxf(d * d, floor_));
    if (cand[xi[s]]) {
      total += logp[s];
      count++;
    }
  }
  const float mu = total / (float)max(count, 1);
  for (int s = 0; s < 12; s++) {
    const bool valid = s < K && cand[xi[s]];
    sel[q * 12 + s] = valid ? xi[s] : -1;
    const float b = valid ? logp[s] - mu : 0.0f;
    bias[q * 12 + s] = round_bias ? bf(b) : b;
  }
}

}  // namespace tf_ctl_uocr_route

// tf_uocr_block_keys writing a non-finite weighted sum as it is (the kernel writes 0 there).
extern "C" __global__ void __launch_bounds__(32) tf_ctl_uocr_route_keys_nonfinite(const float* __restrict__ idx_k,
                                                                                  const tf_uocr::bf16* __restrict__ head_cls,
                                                                                  float* __restrict__ kbar, int first, float scale) {
  using namespace tf_uocr;
  __shared__ float wts[32];
  const int c = first + blockIdx.x, j = threadIdx.x;
  const float* k = idx_k + ((size_t)c * 32 + j) * 128;
  float a = 0.0f;
  for (int d = 0; d < 128; d++) a += k[d] * ld(head_cls + d);
  a = clampf(a * scale, -UD_CLAMP, UD_CLAMP);
  const float m = warp_max(a);
  const float e = expf(a - m);
  const float total = fmaxf(warp_sum(e), 1e-6f);
  wts[j] = e / total;
  __syncwarp();
  for (int d = j; d < 128; d += 32) {
    float s = 0.0f;
    for (int t = 0; t < 32; t++) s += wts[t] * idx_k[((size_t)c * 32 + t) * 128 + d];
    kbar[(size_t)c * 128 + d] = s;
  }
}

#define TF_CTL_UOCR_ROUTE(name, brk)                                                                                     \
  extern "C" __global__ void __launch_bounds__(32)                                                                     \
      name(const float* __restrict__ idx_q, const float* __restrict__ kbar, const int* __restrict__ bmin,              \
           const int* __restrict__ bmax, const int* __restrict__ qpage, const unsigned* __restrict__ qpos,             \
           int* __restrict__ sel, float* __restrict__ bias, int nblk, int complete, float scale, float idx_scale,      \
           float floor_, int round_bias) {                                                                             \
    tf_ctl_uocr_route::route<tf_ctl_uocr_route::brk>(idx_q, kbar, bmin, bmax, qpage, qpos, sel, bias, nblk, complete, \
                                                     scale, idx_scale, floor_, round_bias);                            \
  }

TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_tie_high, tie_high)
TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_gap_short, gap_short)
TF_CTL_UOCR_ROUTE(tf_ctl_uocr_route_page_last, page_last)
#undef TF_CTL_UOCR_ROUTE

// tf_uocr_moe_gate with ties to the higher expert (>= where the kernel takes strict >).
extern "C" __global__ void __launch_bounds__(64) tf_ctl_uocr_route_gate_tie_high(const tf_uocr::bf16* __restrict__ x,
                                                                                 const tf_uocr::bf16* __restrict__ w,
                                                                                 unsigned* __restrict__ ids, float* __restrict__ wts) {
  using namespace tf_uocr;
  __shared__ float part[33];
  __shared__ float score[64];
  const int row = blockIdx.x, e = threadIdx.x;
  const bf16* xr = x + (size_t)row * 1280;
  const bf16* wr = w + (size_t)e * 1280;
  float s = 0.0f;
  for (int d = 0; d < 1280; d++) s += ld(xr + d) * ld(wr + d);
  float m = warp_max(s);
  if ((e & 31) == 0) part[e >> 5] = m;
  __syncthreads();
  m = fmaxf(part[0], part[1]);
  __syncthreads();
  const float ex = expf(s - m);
  const float total = block_sum(ex, part);
  score[e] = ex / total;
  __syncthreads();
  if (e != 0) return;
  bool taken[64];
  for (int i = 0; i < 64; i++) taken[i] = false;
  for (int k = 0; k < 6; k++) {
    int best = 0;
    float bv = -INFINITY;
    for (int i = 0; i < 64; i++) {
      if (!taken[i] && score[i] >= bv) {
        bv = score[i];
        best = i;
      }
    }
    taken[best] = true;
    ids[row * 6 + k] = best;
    wts[row * 6 + k] = bv;
  }
}

// tf_uocr_moe_combine without the routed sum's bf16 point: y = bf16(sum + shared).
extern "C" __global__ void __launch_bounds__(256) tf_ctl_uocr_route_combine_unrounded(const tf_uocr::bf16* __restrict__ slots,
                                                                                      const float* __restrict__ wts,
                                                                                      const tf_uocr::bf16* __restrict__ shared,
                                                                                      tf_uocr::bf16* __restrict__ y) {
  using namespace tf_uocr;
  const int c = blockIdx.y * 256 + threadIdx.x, row = blockIdx.x;
  if (c >= 1280) return;
  float s = 0.0f;
  for (int k = 0; k < 6; k++) s += ld(slots + ((size_t)row * 6 + k) * 1280 + c) * wts[row * 6 + k];
  const size_t i = (size_t)row * 1280 + c;
  y[i] = to(s + ld(shared + i));
}
