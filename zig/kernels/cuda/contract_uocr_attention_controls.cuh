// Gate controls for Unlimited-OCR-2's attention kernels (uocr_vision.cu, uocr_decoder.cu), each breaking one of them.
#pragma once
#include "uocr_common.cuh"

namespace tf_ctl_uocr_attention {
using namespace tf_uocr;

// The breaks: tower rel_w one column over, tower P x V without the last key; decoder's three below.
enum Break { rel_w_shift, last_key, past_row, slot_dropped, overlap_twice };

constexpr int tile = 64;
constexpr int pad = 66;

// tf_uocr_attention as uocr_vision.cu has it, but for break B.
template <int B>
__device__ __forceinline__ void tower(const bf16* __restrict__ qkv, const bf16* __restrict__ rel_h,
                                      const bf16* __restrict__ rel_w, bf16* __restrict__ o, int groups, int len, int width,
                                      int side, float scale) {
  __shared__ __align__(16) bf16 ks[tile * pad];
  __shared__ __align__(16) bf16 vs[tile * 64];
  const int tiles_per_group = (len + 7) / 8;
  const int g = blockIdx.x / tiles_per_group, h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qi = (blockIdx.x % tiles_per_group) * 8 + warp;
  const bool live = qi < len;
  const size_t stride = 3 * (size_t)width;
  const size_t row = (size_t)g * len + (live ? qi : 0);
  const bf16* kbase = qkv + (size_t)g * len * stride + width + h * 64;
  const bf16* vbase = kbase + width;
  float q[64];
  {
    const bf16* qr = qkv + row * stride + h * 64;
#pragma unroll
    for (int d = 0; d < 64; d += 8) load8(qr + d, q + d);
  }
  const size_t rel = (((size_t)h * groups + g) * len + (live ? qi : 0)) * (side > 0 ? side : 1);
  float m = -INFINITY, sum = 0.0f, a0 = 0.0f, a1 = 0.0f;
  for (int pass = 0; pass < 2; ++pass) {
    for (int t0 = 0; t0 < len; t0 += tile) {
      __syncthreads();
      for (int i = threadIdx.x; i < tile * 8; i += 256) {
        const int key = i >> 3, part = (i & 7) * 8;
        uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
        if (t0 + key < len) {
          kv = *reinterpret_cast<const uint4*>(kbase + (size_t)(t0 + key) * stride + part);
          if (pass == 1) vv = *reinterpret_cast<const uint4*>(vbase + (size_t)(t0 + key) * stride + part);
        }
        uint32_t* kd = reinterpret_cast<uint32_t*>(ks + key * pad + part);
        kd[0] = kv.x;
        kd[1] = kv.y;
        kd[2] = kv.z;
        kd[3] = kv.w;
        if (pass == 1) *reinterpret_cast<uint4*>(vs + key * 64 + part) = vv;
      }
      __syncthreads();
      if (!live) continue;
      float e[2];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int key = lane + 32 * half, j = t0 + key;
        float s = -INFINITY;
        if (j < len) {
          const __nv_bfloat162* kr = reinterpret_cast<const __nv_bfloat162*>(ks + key * pad);
          float acc = 0.0f;
#pragma unroll
          for (int d = 0; d < 32; ++d) {
            const float2 f = __bfloat1622float2(kr[d]);
            acc += q[2 * d] * f.x;
            acc += q[2 * d + 1] * f.y;
          }
          s = acc * scale;
          // the break: rel_w read one column over
          const int col = B == rel_w_shift ? (j + 1) % (side > 0 ? side : 1) : j % (side > 0 ? side : 1);
          if (side > 0) s += bf(ld(rel_h + rel + j / side) + ld(rel_w + rel + col));
        }
        e[half] = s;
      }
      if (pass == 0) {
        m = fmaxf(m, fmaxf(e[0], e[1]));
        continue;
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float x = t0 + lane + 32 * half < len ? expf(e[half] - m) : 0.0f;
        sum += x;
        e[half] = bf(x);
      }
      // the break: the last key's weight in the sum, not in P x V
      const int n = min(tile, len - t0) - (B == last_key && t0 + tile >= len ? 1 : 0);
      for (int k = 0; k < n; ++k) {
        const float w = __shfl_sync(0xffffffffu, e[k >> 5], k & 31);
        a0 += w * ld(vs + k * 64 + lane);
        a1 += w * ld(vs + k * 64 + lane + 32);
      }
    }
    if (pass == 0) m = warp_max(m);
  }
  if (!live) return;
  sum = warp_sum(sum);
  bf16* out = o + row * width + h * 64;
  out[lane] = to(a0 / sum);
  out[lane + 32] = to(a1 / sum);
}

// One more key than the kernel's 512, for the past-row break's extra window key.
constexpr int most = 520;

// tf_uocr_attend as uocr_decoder.cu has it, but for break B.
template <int B>
__device__ __forceinline__ void attend(const bf16* __restrict__ q, const bf16* __restrict__ kc, const bf16* __restrict__ vc,
                                       const int* __restrict__ sel, const float* __restrict__ bias,
                                       const unsigned* __restrict__ qpos, bf16* __restrict__ o, int width, float scale,
                                       int local) {
  __shared__ float score[most];
  __shared__ unsigned key[most];
  __shared__ unsigned seg_first[13], seg_start[14];
  __shared__ float seg_bias[13];
  __shared__ float part_max[8], part_sum[8];
  __shared__ float4 part_acc[8][32];
  const int row = blockIdx.x, h = blockIdx.y;
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  const unsigned pos = qpos[row];
  const unsigned lo = pos + 1 > (unsigned)local ? pos + 1 - local : 0;
  if (t == 0) {
    unsigned n = 0;
    for (int s = 0; s <= 12; s++) {
      unsigned first = 0, last = 0;
      float b = 0.0f;
      if (s < 12) {
        // the break: the 12th slot read as empty
        const int c = B == slot_dropped && s == 11 ? -1 : sel[row * 12 + s];
        if (c >= 0) {
          first = (unsigned)c * 32;
          // the break: a block's keys inside or past the window kept
          last = B == overlap_twice ? first + 32 : max(first, min(first + 32, lo));
          b = bias[row * 12 + s];
        }
      } else {
        first = lo;
        // the break: the window one key past the row
        last = B == past_row ? pos + 2 : pos + 1;
      }
      seg_first[s] = first;
      seg_start[s] = n;
      seg_bias[s] = b;
      n += last - first;
    }
    seg_start[13] = n;
  }
  __syncthreads();
  const unsigned n = seg_start[13];
  const bf16* qr = q + (size_t)row * width + h * 128 + lane * 4;
  const float4 qv = make_float4(ld(qr), ld(qr + 1), ld(qr + 2), ld(qr + 3));
  float m = -INFINITY;
  unsigned s = 0;
  for (unsigned i = sg; i < n; i += 8) {
    while (i >= seg_start[s + 1]) s++;
    const unsigned j = seg_first[s] + (i - seg_start[s]);
    const bf16* kr = kc + (size_t)j * width + h * 128 + lane * 4;
    const float dot = qv.x * ld(kr) + qv.y * ld(kr + 1) + qv.z * ld(kr + 2) + qv.w * ld(kr + 3);
    const float sc = warp_sum(dot) * scale + seg_bias[s];
    if (lane == 0) {
      score[i] = sc;
      key[i] = j;
    }
    m = fmaxf(m, sc);
  }
  if (lane == 0) part_max[sg] = m;
  __syncthreads();
  m = part_max[0];
  for (int g = 1; g < 8; g++) m = fmaxf(m, part_max[g]);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float total = 0.0f;
  for (unsigned i = sg; i < n; i += 8) {
    const float e = expf(score[i] - m);
    total += e;
    const bf16* vr = vc + (size_t)key[i] * width + h * 128 + lane * 4;
    acc.x += e * ld(vr);
    acc.y += e * ld(vr + 1);
    acc.z += e * ld(vr + 2);
    acc.w += e * ld(vr + 3);
  }
  part_acc[sg][lane] = acc;
  if (lane == 0) part_sum[sg] = total;
  __syncthreads();
  if (sg != 0) return;
  float4 sum = part_acc[0][lane];
  float tot = part_sum[0];
  for (int g = 1; g < 8; g++) {
    sum.x += part_acc[g][lane].x;
    sum.y += part_acc[g][lane].y;
    sum.z += part_acc[g][lane].z;
    sum.w += part_acc[g][lane].w;
    tot += part_sum[g];
  }
  bf16* out = o + (size_t)row * width + h * 128 + lane * 4;
  out[0] = to(sum.x / tot);
  out[1] = to(sum.y / tot);
  out[2] = to(sum.z / tot);
  out[3] = to(sum.w / tot);
}

}  // namespace tf_ctl_uocr_attention

#define TF_CTL_UOCR_TOWER(name, b)                                                                                       \
  extern "C" __global__ void __launch_bounds__(256)                                                                      \
      name(const tf_uocr::bf16* __restrict__ qkv, const tf_uocr::bf16* __restrict__ rel_h,                               \
           const tf_uocr::bf16* __restrict__ rel_w, tf_uocr::bf16* __restrict__ o, int groups, int len, int width,       \
           int side, float scale) {                                                                                      \
    tf_ctl_uocr_attention::tower<tf_ctl_uocr_attention::b>(qkv, rel_h, rel_w, o, groups, len, width, side, scale);       \
  }
TF_CTL_UOCR_TOWER(tf_ctl_uocr_attention_rel_w_shift, rel_w_shift)
TF_CTL_UOCR_TOWER(tf_ctl_uocr_attention_last_key, last_key)
#undef TF_CTL_UOCR_TOWER

#define TF_CTL_UOCR_ATTEND(name, b)                                                                                      \
  extern "C" __global__ void __launch_bounds__(256)                                                                      \
      name(const tf_uocr::bf16* __restrict__ q, const tf_uocr::bf16* __restrict__ kc,                                    \
           const tf_uocr::bf16* __restrict__ vc, const int* __restrict__ sel, const float* __restrict__ bias,            \
           const unsigned* __restrict__ qpos, tf_uocr::bf16* __restrict__ o, int width, float scale, int local) {        \
    tf_ctl_uocr_attention::attend<tf_ctl_uocr_attention::b>(q, kc, vc, sel, bias, qpos, o, width, scale, local);        \
  }
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attention_past_row, past_row)
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attention_slot_dropped, slot_dropped)
TF_CTL_UOCR_ATTEND(tf_ctl_uocr_attention_overlap_twice, overlap_twice)
#undef TF_CTL_UOCR_ATTEND
