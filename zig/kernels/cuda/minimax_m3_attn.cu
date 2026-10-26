// MiniMax-M3.1-Flash's block-sparse attention on CUDA: the lightning indexer's scores and picks, decode and prompts.

#include "minimax_m3.cuh"

namespace tf_mm3 {

// Index scores S[row][head][block]: the max over the block's keys <= pos + row of iq . ik (fp32); a thread a key
extern "C" __global__ void __launch_bounds__(128) tf_mm3_index_scores(const bf16* __restrict__ IQ,
                                                                      const bf16* __restrict__ IK, float* __restrict__ S, int rows, int pos, int heads, int blocks) {
  constexpr int RB = 16, D = 128;
  const int H = heads;
  extern __shared__ float sm[];
  float* q = sm;              // [RB * H][D] fp32
  float* part = sm + RB * H * D;  // [4][RB * H]
  const int j = blockIdx.x, r0 = blockIdx.y * RB;
  const int nr = min(RB, rows - r0);
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  for (int i = t; i < nr * H * D; i += 128) q[i] = ld(IQ + (size_t)r0 * H * D + i);
  __syncthreads();
  const int key = j * 128 + t;
  const bf16* k = IK + (size_t)key * D;
  float acc[RB * 4];
  for (int i = 0; i < RB * 4; ++i) acc[i] = 0.0f;
  const bool any = key <= pos + r0 + nr - 1;
  if (any) {
    for (int d = 0; d < D; d += 4) {
      float kv[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) kv[e] = ld(k + d + e);
      for (int r = 0; r < nr; ++r)
        for (int h = 0; h < H; ++h) {
          const float* qq = q + (r * H + h) * D + d;
          acc[r * H + h] += kv[0] * qq[0] + kv[1] * qq[1] + kv[2] * qq[2] + kv[3] * qq[3];
        }
    }
  }
  for (int r = 0; r < nr; ++r)
    for (int h = 0; h < H; ++h) {
      // fmaxf from -inf leaves a NaN dot out, as the tensor-unit kernel's running max does
      const float s = warp_max(key <= pos + r0 + r ? fmaxf(acc[r * H + h], -CUDART_INF_F) : -CUDART_INF_F);
      if (lane == 0) part[sg * RB * H + r * H + h] = s;
    }
  __syncthreads();
  if (t < nr * H) {
    const float s = fmaxf(fmaxf(part[t], part[RB * H + t]), fmaxf(part[2 * RB * H + t], part[3 * RB * H + t]));
    S[((size_t)(r0 + t / H) * H + t % H) * blocks + j] = s;
  }
}

// A (row, head)'s kept blocks ascending: every one up to topk, else its own and the best (ties low); and a bitmap
extern "C" __global__ void tf_mm3_index_select(const float* __restrict__ S, int* __restrict__ SEL,
                                               unsigned* __restrict__ BITS, int rows, int pos, int heads, int blocks,
                                               int topk, int words) {
  const int row = blockIdx.y, h = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows) return;
  const int nb = (pos + row) / 128 + 1;
  const float* s = S + ((size_t)row * heads + h) * blocks;
  int* out = SEL + ((size_t)row * heads + h) * topk;
  int list[32];
  int count = 0;
  if (nb <= topk) {
    for (int b = 0; b < nb; ++b) list[count++] = b;
  } else {
    list[count++] = nb - 1;
    float last_best = CUDART_INF_F;
    int last_id = nb - 1;
    for (int k = 1; k < topk; ++k) {
      float best = -CUDART_INF_F;
      int bid = 0x7fffffff;
      for (int b = lane; b < nb - 1; b += 32) {
        const float v = s[b];
        const bool after = v < last_best || (v == last_best && b > last_id);
        if (after && (v > best || (v == best && b < bid))) {
          best = v;
          bid = b;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bid, off);
        if (ob > best || (ob == best && oi < bid)) {
          best = ob;
          bid = oi;
        }
      }
      if (bid == 0x7fffffff || best == -CUDART_INF_F) break;
      list[count++] = bid;
      last_best = best;
      last_id = bid;
    }
  }
  for (int i = 1; i < count; ++i)
    for (int m = i; m > 0 && list[m] < list[m - 1]; --m) {
      const int tmp = list[m];
      list[m] = list[m - 1];
      list[m - 1] = tmp;
    }
  if (lane == 0)
    for (int i = 0; i < topk; ++i) out[i] = i < count ? list[i] : -1;
  if (words == 0) return;
  unsigned* bits = BITS + ((size_t)row * heads + h) * words;
  for (int wd = lane; wd < words; wd += 32) {
    unsigned m = 0;
    for (int i = 0; i < count; ++i)
      if (list[i] / 32 == wd) m |= 1u << (list[i] % 32);
    bits[wd] = m;
  }
}

// Decode attention partials (max, sum, output) a (key head, row, quarter of a kept block), its G query heads at once
extern "C" __global__ void tf_mm3_attn_decode(const bf16* __restrict__ Q, const bf16* __restrict__ KC,
                                              const bf16* __restrict__ VC, const int* __restrict__ SEL,
                                              float* __restrict__ PM, float* __restrict__ PL, float* __restrict__ PO,
                                              int rows, int pos, int hq, int hk, int topk, int all, float scale) {
  constexpr int D = 128;
  const int kvh = blockIdx.x, row = blockIdx.y, part = blockIdx.z;  // part: kept slot * 4 + quarter
  const int G = hq / hk;
  const int g = threadIdx.y, lane = threadIdx.x;
  const int head = kvh * G + g;
  const int quarters = topk * 4;
  const size_t pidx = ((size_t)row * hq + head) * quarters + part;
  const int last = pos + row;
  int block;
  if (all) {
    block = part / 4;
    if (block * 128 > last) {
      PM[pidx] = -CUDART_INF_F;
      PL[pidx] = 0.0f;
      return;
    }
  } else {
    block = SEL[((size_t)row * hk + kvh) * topk + part / 4];
    if (block < 0) {
      PM[pidx] = -CUDART_INF_F;
      PL[pidx] = 0.0f;
      return;
    }
  }
  const int key = block * 128 + (part % 4) * 32 + lane;
  const bool valid = key <= last;
  __shared__ float qs[32][D];
  const bf16* q = Q + ((size_t)row * hq + head) * D;
  for (int i = lane; i < D; i += 32) qs[g][i] = ld(q + i);
  __syncwarp();
  float s = -CUDART_INF_F;
  if (valid) {
    const bf16* k = KC + ((size_t)key * hk + kvh) * D;
    float acc = 0.0f;
    for (int d = 0; d < D; d += 2) {
      const __nv_bfloat162 kk = *reinterpret_cast<const __nv_bfloat162*>(k + d);
      acc = fmaf(qs[g][d], __low2float(kk), acc);
      acc = fmaf(qs[g][d + 1], __high2float(kk), acc);
    }
    s = acc * scale;
  }
  const float m = warp_max(s);
  if (m == -CUDART_INF_F) {
    if (lane == 0) {
      PM[pidx] = -CUDART_INF_F;
      PL[pidx] = 0.0f;
    }
    return;
  }
  const float p = valid ? expf(s - m) : 0.0f;
  const float l = warp_sum(p);
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int kk = 0; kk < 32; ++kk) {
    const float pk = __shfl_sync(0xffffffffu, p, kk);
    if (pk == 0.0f) continue;
    const int k2 = block * 128 + (part % 4) * 32 + kk;
    const bf16* v = VC + ((size_t)k2 * hk + kvh) * D;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = fmaf(pk, ld(v + lane + 32 * j), o[j]);
  }
  float* po = PO + pidx * D;
#pragma unroll
  for (int j = 0; j < 4; ++j) po[lane + 32 * j] = o[j];
  if (lane == 0) {
    PM[pidx] = m;
    PL[pidx] = l;
  }
}

// The merge of a (row, head)'s partials: softmax over the quarters' maxima, output rounded to bf16. A warp each.
extern "C" __global__ void tf_mm3_attn_merge(const float* __restrict__ PM, const float* __restrict__ PL,
                                             const float* __restrict__ PO, bf16* __restrict__ O, int rows, int hq,
                                             int quarters) {
  constexpr int D = 128;
  const int head = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
  if (row >= rows) return;
  const size_t base = ((size_t)row * hq + head) * quarters;
  float M = -CUDART_INF_F;
  for (int i = 0; i < quarters; ++i) M = fmaxf(M, PM[base + i]);
  float L = 0.0f;
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < quarters; ++i) {
    const float m = PM[base + i];
    if (m == -CUDART_INF_F) continue;
    const float f = expf(m - M);
    L += PL[base + i] * f;
    const float* po = PO + (base + i) * D;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = fmaf(f, po[lane + 32 * j], o[j]);
  }
  bf16* out = O + ((size_t)row * hq + head) * D;
  const float inv = L > 0.0f ? 1.0f / L : 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) st(out + lane + 32 * j, o[j] * inv);
}

// Prompt attention a (row, key head) over the row's kept blocks (bitmap) and keys <= pos + row; fp32 online softmax
extern "C" __global__ void __launch_bounds__(128) tf_mm3_attn_prompt(const bf16* __restrict__ Q, const bf16* __restrict__ KC,
                                              const bf16* __restrict__ VC, const unsigned* __restrict__ BITS,
                                              bf16* __restrict__ O, int rows, int pos, int hq, int hk, int words,
                                              float scale) {
  constexpr int D = 128, GMAX = 16;
  const int row = blockIdx.x, kvh = blockIdx.y;
  const int G = hq / hk;
  const int t = threadIdx.x, sg = t >> 5, lane = t & 31;
  if (row >= rows) return;
  const int last = pos + row;
  const int nb = last / 128 + 1;
  __shared__ float qs[GMAX][D];
  __shared__ float red[4][GMAX];
  __shared__ float pm[GMAX], pl[GMAX], pk[GMAX];
  for (int g = 0; g < G; ++g) qs[g][t] = ld(Q + ((size_t)row * hq + kvh * G + g) * D + t) * scale;
  if (t < G) {
    pm[t] = -CUDART_INF_F;
    pl[t] = 0.0f;
  }
  float o[GMAX];
  for (int g = 0; g < GMAX; ++g) o[g] = 0.0f;
  __syncthreads();
  const unsigned* bits = words > 0 ? BITS + ((size_t)row * hk + kvh) * words : nullptr;
  for (int b = 0; b < nb; ++b) {
    if (bits != nullptr && !((bits[b / 32] >> (b % 32)) & 1u)) continue;
    const int kend = min(127, last - b * 128);
    for (int kk = 0; kk <= kend; ++kk) {
      const int key = b * 128 + kk;
      const float kd = ld(KC + ((size_t)key * hk + kvh) * D + t);
      const float vd = ld(VC + ((size_t)key * hk + kvh) * D + t);
      float part[GMAX];
      for (int g = 0; g < G; ++g) part[g] = warp_sum(qs[g][t] * kd);
      if (lane == 0)
        for (int g = 0; g < G; ++g) red[sg][g] = part[g];
      __syncthreads();
      if (t < G) {
        const float s = red[0][t] + red[1][t] + red[2][t] + red[3][t];
        const float m = fmaxf(pm[t], s);
        const float f = expf(pm[t] - m);
        const float p = expf(s - m);
        pl[t] = pl[t] * f + p;
        pm[t] = m;
        pk[t] = p;
        red[0][t] = f;  // the rescale of the running output, read below
      }
      __syncthreads();
      for (int g = 0; g < G; ++g) o[g] = fmaf(pk[g], vd, o[g] * red[0][g]);
      __syncthreads();
    }
  }
  for (int g = 0; g < G; ++g) {
    const float inv = pl[g] > 0.0f ? 1.0f / pl[g] : 0.0f;
    st(O + ((size_t)row * hq + kvh * G + g) * D + t, o[g] * inv);
  }
}

// Prompt attention on the tensor units: 8 rows a CTA, warp w row r0 + w's 16 heads as one m16 tile, K and V staged
extern "C" __global__ void __launch_bounds__(256) tf_mm3_attn_prompt_tc(
    const bf16* __restrict__ Q, const bf16* __restrict__ KC, const bf16* __restrict__ VC,
    const unsigned* __restrict__ BITS, bf16* __restrict__ O, int rows, int pos, int hq, int hk, int words, float scale,
    int lo, int fixed) {
  constexpr int D = 128, KCH = 64, RPB = 8, KP = D + 8, VP = KCH + 8;
  __shared__ __align__(16) bf16 Ks[KCH][KP];
  __shared__ __align__(16) bf16 Vt[D][VP];
  const int kvh = blockIdx.y;
  const int r0 = blockIdx.x * RPB;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int row = r0 + w;
  const bool live = row < rows;
  // fixed >= 0: every row sees keys [lo, fixed] both ways (a draft block over its window); else causal from key 0
  const int last = fixed >= 0 ? fixed : pos + (live ? row : r0);
  const int maxlast = fixed >= 0 ? fixed : pos + min(rows, r0 + RPB) - 1;
  const float sl2 = scale * 1.4426950408889634f;
  // this warp's 16 heads' queries as A fragments, 8 k-steps of 16 dims
  uint32_t qa[8][4];
  {
    const bf16* q0 = Q + ((size_t)(live ? row : r0) * hq + kvh * 16 + g) * D;
    const bf16* q1 = q0 + 8 * D;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 2 * t);
      qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 2 * t);
      qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 8 + 2 * t);
      qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 8 + 2 * t);
    }
  }
  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.f, l1 = 0.f;
  const unsigned* bits = (words > 0 && live) ? BITS + ((size_t)row * hk + kvh) * words : nullptr;
  const int nb = maxlast / 128 + 1;
  for (int b = lo / 128; b < nb; ++b) {
    const bool mine = live && b * 128 <= last && (words == 0 || ((bits[b / 32] >> (b % 32)) & 1u));
    if (!__syncthreads_or(mine)) continue;
    for (int c = 0; c < 2; ++c) {
      const int base = b * 128 + c * KCH;
      if (base > maxlast) break;
      if (base + KCH <= lo) continue;  // every key below the window: block-uniform
      // stage the chunk: key kk = tid / 4, dims (tid % 4) * 32 ..; keys past the block's last row zeroed
      {
        const int kk = threadIdx.x >> 2, d0 = (threadIdx.x & 3) * 32;
        const int key = base + kk;
        const bool ok = key <= maxlast;
        const uint4* ks = reinterpret_cast<const uint4*>(KC + ((size_t)key * hk + kvh) * D + d0);
        const uint4* vs = reinterpret_cast<const uint4*>(VC + ((size_t)key * hk + kvh) * D + d0);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint4 kv = ok ? ks[i] : make_uint4(0, 0, 0, 0);
          *reinterpret_cast<uint4*>(&Ks[kk][d0 + 8 * i]) = kv;
          const uint4 vv = ok ? vs[i] : make_uint4(0, 0, 0, 0);
          const bf16* ve = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
          for (int e = 0; e < 8; ++e) Vt[d0 + 8 * i + e][kk] = ve[e];
        }
      }
      __syncthreads();
      if (mine && base <= last) {
        float sc[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#pragma unroll
          for (int s = 0; s < 8; ++s) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Ks[8 * j + g][16 * s + 2 * t]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Ks[8 * j + g][16 * s + 8 + 2 * t]);
            mma16816(sc[j], qa[s][0], qa[s][1], qa[s][2], qa[s][3], b0, b1);
          }
        }
        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int key = base + 8 * j + 2 * t + e;
            const bool ok = key <= last && key >= lo;
            sc[j][e] = ok ? sc[j][e] * sl2 : -CUDART_INF_F;
            sc[j][2 + e] = ok ? sc[j][2 + e] * sl2 : -CUDART_INF_F;
            mx0 = fmaxf(mx0, sc[j][e]);
            mx1 = fmaxf(mx1, sc[j][2 + e]);
          }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
          mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
          mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // masked keys add 0 and an unmoved max rescales by 1 (no -inf - -inf); a NaN score still gives NaN
        const float f0 = m0 == mx0 ? 1.f : exp2f(m0 - mx0), f1 = m1 == mx1 ? 1.f : exp2f(m1 - mx1);
        m0 = mx0;
        m1 = mx1;
        float s0 = 0.f, s1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            sc[j][e] = sc[j][e] == -CUDART_INF_F ? 0.f : exp2f(sc[j][e] - mx0);
            sc[j][2 + e] = sc[j][2 + e] == -CUDART_INF_F ? 0.f : exp2f(sc[j][2 + e] - mx1);
            s0 += sc[j][e];
            s1 += sc[j][2 + e];
          }
        l0 = l0 * f0 + s0;
        l1 = l1 * f1 + s1;
#pragma unroll
        for (int n = 0; n < 16; ++n) {
          o[n][0] *= f0;
          o[n][1] *= f0;
          o[n][2] *= f1;
          o[n][3] *= f1;
        }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t a0 = pack_bf16(sc[2 * s][0], sc[2 * s][1]);
          const uint32_t a1 = pack_bf16(sc[2 * s][2], sc[2 * s][3]);
          const uint32_t a2 = pack_bf16(sc[2 * s + 1][0], sc[2 * s + 1][1]);
          const uint32_t a3 = pack_bf16(sc[2 * s + 1][2], sc[2 * s + 1][3]);
#pragma unroll
          for (int n = 0; n < 16; ++n) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Vt[8 * n + g][16 * s + 2 * t]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Vt[8 * n + g][16 * s + 8 + 2 * t]);
            mma16816(o[n], a0, a1, a2, a3, b0, b1);
          }
        }
      }
      __syncthreads();
    }
  }
  if (!live) return;
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  bf16* out0 = O + ((size_t)row * hq + kvh * 16 + g) * D;
  bf16* out1 = out0 + 8 * D;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    *reinterpret_cast<uint32_t*>(out0 + 8 * n + 2 * t) = pack_bf16(o[n][0] * i0, o[n][1] * i0);
    *reinterpret_cast<uint32_t*>(out1 + 8 * n + 2 * t) = pack_bf16(o[n][2] * i1, o[n][3] * i1);
  }
}

// Index scores on the tensor units: 64 (row, head) pairs against one staged key block, max over keys <= pos + row
extern "C" __global__ void __launch_bounds__(128) tf_mm3_index_scores_tc(const bf16* __restrict__ IQ,
                                                                         const bf16* __restrict__ IK,
                                                                         float* __restrict__ S, int rows, int pos,
                                                                         int heads, int blocks) {
  constexpr int D = 128, KB = 128, KP = D + 8;
  __shared__ __align__(16) bf16 Ks[KB][KP];
  const int j = blockIdx.x;
  const int pairs = rows * heads;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int p0 = blockIdx.y * 64 + w * 16;
  // the block's keys (at most the last row's position; past it zero: never kept)
  const int maxlast = pos + rows - 1;
  for (int i = threadIdx.x; i < KB * (D / 8); i += blockDim.x) {
    const int kk = i / (D / 8), d0 = (i % (D / 8)) * 8;
    const int key = j * KB + kk;
    const uint4 v = key <= maxlast ? *reinterpret_cast<const uint4*>(IK + (size_t)key * D + d0) : make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(&Ks[kk][d0]) = v;
  }
  __syncthreads();
  if (p0 >= pairs) return;
  const int pa = min(p0 + g, pairs - 1), pb = min(p0 + g + 8, pairs - 1);
  const bf16* q0 = IQ + (size_t)pa * D;
  const bf16* q1 = IQ + (size_t)pb * D;
  uint32_t qa[8][4];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qa[s][0] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 2 * t);
    qa[s][1] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 2 * t);
    qa[s][2] = *reinterpret_cast<const uint32_t*>(q0 + 16 * s + 8 + 2 * t);
    qa[s][3] = *reinterpret_cast<const uint32_t*>(q1 + 16 * s + 8 + 2 * t);
  }
  const int last0 = pos + pa / heads, last1 = pos + pb / heads;
  float mx0 = -CUDART_INF_F, mx1 = -CUDART_INF_F;
#pragma unroll 4
  for (int n = 0; n < KB / 8; ++n) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&Ks[8 * n + g][16 * s + 2 * t]);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&Ks[8 * n + g][16 * s + 8 + 2 * t]);
      mma16816(c, qa[s][0], qa[s][1], qa[s][2], qa[s][3], b0, b1);
    }
    const int key = j * KB + 8 * n + 2 * t;
    if (key <= last0) mx0 = fmaxf(mx0, c[0]);
    if (key + 1 <= last0) mx0 = fmaxf(mx0, c[1]);
    if (key <= last1) mx1 = fmaxf(mx1, c[2]);
    if (key + 1 <= last1) mx1 = fmaxf(mx1, c[3]);
  }
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
  }
  if (t == 0) {
    if (p0 + g < pairs) S[(size_t)(p0 + g) * blocks + j] = mx0;
    if (p0 + g + 8 < pairs) S[(size_t)(p0 + g + 8) * blocks + j] = mx1;
  }
}

}  // namespace tf_mm3
