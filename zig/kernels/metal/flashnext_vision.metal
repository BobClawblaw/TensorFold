// Qwen3.8 Flash Next's image tower (27 pre-LayerNorm blocks of 1152, 16 heads of 72, axial 2-D rotary, a learned
// 48x48 position table resampled to each image, tanh-GELU MLPs and a 2x2 merger to the language model's 2560).
// Compiled after prefill/gemm_nax.metal, whose tfp::gemm_nax runs every projection; the rest rounds as transformers'
// Qwen4ExpVisionModel does in bf16 on the CPU: one rounding a fused op, fp32 inside.

#define FNV_GEMM(NAME, T, TA, TB, AM, AN)                                                                          \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                \
      const device T* A [[buffer(0)]], const device T* B [[buffer(1)]], const device int32_t* P [[buffer(2)]],     \
      device T* D [[buffer(3)]], uint sg [[simdgroup_index_in_threadgroup]],                                      \
      uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {                        \
    tfp::gemm_nax<T, TA, TB, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);                         \
  }
// x [M, K] w^T: every linear layer (bf16 in and out, fp32 sums)
FNV_GEMM(fnv_gemm_bf16_nt_00, bfloat16_t, false, true, false, false)
FNV_GEMM(fnv_gemm_bf16_nt_01, bfloat16_t, false, true, false, true)
FNV_GEMM(fnv_gemm_bf16_nt_10, bfloat16_t, false, true, true, false)
FNV_GEMM(fnv_gemm_bf16_nt_11, bfloat16_t, false, true, true, true)
// q k^T per head in fp32
FNV_GEMM(fnv_gemm_f32_nt_00, float, false, true, false, false)
FNV_GEMM(fnv_gemm_f32_nt_01, float, false, true, false, true)
FNV_GEMM(fnv_gemm_f32_nt_10, float, false, true, true, false)
FNV_GEMM(fnv_gemm_f32_nt_11, float, false, true, true, true)
// p v per head in fp32
FNV_GEMM(fnv_gemm_f32_nn_00, float, false, false, false, false)
FNV_GEMM(fnv_gemm_f32_nn_01, float, false, false, false, true)
FNV_GEMM(fnv_gemm_f32_nn_10, float, false, false, true, false)
FNV_GEMM(fnv_gemm_f32_nn_11, float, false, false, true, true)

struct FnvRows {
  uint rows;
  uint width;
  float eps;
};

// One threadgroup's sum of every thread's `v` (at most 1024 threads), returned to all of them.
inline float fnv_group_sum(float v, threadgroup float* part, uint n, uint lane, uint sg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float s = lane < (n + 31) / 32 ? part[lane] : 0.0f;
    s = simd_sum(s);
    if (lane == 0) part[32] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float out = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out;
}

// y[r, c] = y + b[c]: a linear layer's bias after its product
[[kernel]] void fnv_bias(device bfloat* y [[buffer(0)]], const device bfloat* b [[buffer(1)]],
                         constant uint& width [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const ulong i = ulong(at.y) * width + at.x;
  y[i] = bfloat(float(y[i]) + float(b[at.x]));
}

// x += y (a block's residual)
[[kernel]] void fnv_add(device bfloat* x [[buffer(0)]], const device bfloat* y [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  x[i] = bfloat(float(x[i]) + float(y[i]));
}

// nn.LayerNorm in bf16 as torch's CPU kernel runs it: fp32 mean and variance, one rounding of (x - m) * r * w + b
[[kernel]] void fnv_layer_norm(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                               const device bfloat* b [[buffer(2)]], device bfloat* y [[buffer(3)]],
                               constant FnvRows& p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint n [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  const device bfloat* xr = x + ulong(row) * p.width;
  device bfloat* yr = y + ulong(row) * p.width;
  float s = 0.0f;
  for (uint i = t; i < p.width; i += n) s += float(xr[i]);
  const float mean = fnv_group_sum(s, part, n, lane, sg) / float(p.width);
  float v = 0.0f;
  for (uint i = t; i < p.width; i += n) {
    const float d = float(xr[i]) - mean;
    v += d * d;
  }
  const float inv = metal::precise::rsqrt(fnv_group_sum(v, part, n, lane, sg) / float(p.width) + p.eps);
  for (uint i = t; i < p.width; i += n) yr[i] = bfloat((float(xr[i]) - mean) * inv * float(w[i]) + float(b[i]));
}

// x[p] += bf16(sum of four taps of the 48x48 position table times their fp32 weights): the resampled table, added
// after the patch projection as Qwen4ExpVisionModel adds it. idx and wt [P, 4] are the host's bilinear taps.
[[kernel]] void fnv_pos(device bfloat* x [[buffer(0)]], const device bfloat* table [[buffer(1)]],
                        const device uint* idx [[buffer(2)]], const device float* wt [[buffer(3)]],
                        uint2 at [[thread_position_in_grid]]) {
  const uint c = at.x, r = at.y;
  float s = 0.0f;
  for (uint k = 0; k < 4; k++) s += float(table[ulong(idx[r * 4 + k]) * 1152 + c]) * wt[r * 4 + k];
  const ulong i = ulong(r) * 1152 + c;
  x[i] = bfloat(float(x[i]) + float(bfloat(s)));
}

// One thread per (row, head, q|k|v, rotary pair j < 36) of qkv [L, 3, 16, 72]: q and k rotated by the row's angles
// (cos and sin [L, 72]: 18 height frequencies, 18 width, then the same again) in fp32 and rounded to bf16, as
// apply_rotary_pos_emb_vision does; written head-major in fp32 ([16, L, 72]); v as stored.
[[kernel]] void fnv_qkv(const device bfloat* qkv [[buffer(0)]], const device float* cs [[buffer(1)]],
                        const device float* sn [[buffer(2)]], device float* Q [[buffer(3)]],
                        device float* K [[buffer(4)]], device float* V [[buffer(5)]], constant uint& L [[buffer(6)]],
                        uint2 at [[thread_position_in_grid]]) {
  const uint j = at.x, g = at.y;
  const uint which = g % 3, h = (g / 3) % 16, r = g / 48;
  const device bfloat* src = qkv + ulong(r) * 3456 + which * 1152 + h * 72;
  const ulong out = (ulong(h) * L + r) * 72;
  const float a = float(src[j]), b = float(src[j + 36]);
  if (which == 2) {
    V[out + j] = a;
    V[out + j + 36] = b;
    return;
  }
  const device float* c = cs + ulong(r) * 72;
  const device float* s = sn + ulong(r) * 72;
  device float* dst = which == 0 ? Q : K;
  dst[out + j] = float(bfloat(a * c[j] - b * s[j]));
  dst[out + j + 36] = float(bfloat(b * c[j + 36] + a * s[j + 36]));
}

// Each row of S [rows, n] (fp32 scores) as softmax(scale * s), in place; one threadgroup a row.
[[kernel]] void fnv_softmax(device float* S [[buffer(0)]], constant uint& n [[buffer(1)]],
                            constant float& scale [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[33];
  device float* s = S + ulong(row) * n;
  float m = -INFINITY;
  for (uint i = t; i < n; i += tn) m = max(m, s[i] * scale);
  m = simd_max(m);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : -INFINITY;
    v = simd_max(v);
    if (lane == 0) part[32] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float z = 0.0f;
  for (uint i = t; i < n; i += tn) {
    const float e = metal::precise::exp(s[i] * scale - m);
    s[i] = e;
    z += e;
  }
  const float inv = 1.0f / fnv_group_sum(z, part, tn, lane, sg);
  for (uint i = t; i < n; i += tn) s[i] *= inv;
}

// O [16, rows, 72] (fp32, one chunk's attention) into rows [r0, r0 + rows) of out [L, 1152] (bf16, heads side by side)
[[kernel]] void fnv_heads(const device float* O [[buffer(0)]], device bfloat* out [[buffer(1)]],
                          constant uint2& p [[buffer(2)]], uint2 at [[thread_position_in_grid]]) {
  const uint rows = p.x, r0 = p.y, h = at.x / 72, d = at.x % 72, r = at.y;
  out[ulong(r0 + r) * 1152 + at.x] = bfloat(O[(ulong(h) * rows + r) * 72 + d]);
}

// gelu_pytorch_tanh in place, fp32 inside and one rounding: x / 2 * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 x^3)))
[[kernel]] void fnv_gelu_tanh(device bfloat* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const float v = float(x[i]);
  const float inner = 0.79788456080286535588f * (v + 0.044715f * v * v * v);
  x[i] = bfloat(0.5f * v * (1.0f + metal::precise::tanh(inner)));
}

// erf to fp32 precision (W. J. Cody's rational approximations, as libm's erff).
inline float fnv_erf(float x) {
  const float ax = fabs(x);
  if (ax < 0.84375f) {
    const float z = x * x;
    const float r = 1.28379167e-01f + z * (-3.25042108e-01f + z * (-2.84817495e-02f + z * (-5.77027029e-03f + z * -2.37630166e-05f)));
    const float s = 1.0f + z * (3.97917233e-01f + z * (6.50222499e-02f + z * (5.08130628e-03f + z * (1.32494738e-04f + z * -3.96022827e-06f))));
    return x + x * (r / s);
  }
  if (ax < 1.25f) {
    const float s = ax - 1.0f;
    const float P = -2.36211857e-03f + s * (4.14856106e-01f + s * (-3.72207481e-01f + s * (3.18346620e-01f + s * (-1.10894694e-01f + s * (3.54783031e-02f + s * -2.16637559e-03f)))));
    const float Q = 1.0f + s * (1.06420882e-01f + s * (5.40397942e-01f + s * (7.18286547e-02f + s * (1.26171216e-01f + s * (1.36370836e-02f + s * 1.19844998e-02f)))));
    const float e = 8.45062911e-01f + P / Q;
    return x >= 0 ? e : -e;
  }
  if (ax >= 6.0f) return x >= 0 ? 1.0f : -1.0f;
  const float s = 1.0f / (ax * ax);
  float R, S;
  if (ax < 1.0f / 0.35f) {
    R = -9.86494403e-03f + s * (-6.93858570e-01f + s * (-1.05586262e+01f + s * (-6.23753357e+01f + s * (-1.62396660e+02f + s * (-1.84605087e+02f + s * (-8.12874374e+01f + s * -9.81432934e+00f))))));
    S = 1.0f + s * (1.96512714e+01f + s * (1.37657754e+02f + s * (4.34565887e+02f + s * (6.45387268e+02f + s * (4.29008148e+02f + s * (1.08635002e+02f + s * (6.57024977e+00f + s * -6.04244141e-02f)))))));
  } else {
    R = -9.86494310e-03f + s * (-7.99283242e-01f + s * (-1.77579556e+01f + s * (-1.60636383e+02f + s * (-6.37566467e+02f + s * (-1.02509509e+03f + s * -4.83519196e+02f)))));
    S = 1.0f + s * (3.03380604e+01f + s * (3.25792511e+02f + s * (1.53672961e+03f + s * (3.19985820e+03f + s * (2.55305029e+03f + s * (4.74528534e+02f + s * -2.24409515e+01f))))));
  }
  const float zh = as_type<float>(as_type<uint>(ax) & 0xfffff000u);
  const float r = metal::precise::exp(-zh * zh - 0.5625f) * metal::precise::exp((zh - ax) * (zh + ax) + R / S);
  const float e = 1.0f - r / ax;
  return x >= 0 ? e : -e;
}

// nn.GELU() (exact) in place, fp32 inside and one rounding: x / 2 * (1 + erf(x / sqrt(2)))
[[kernel]] void fnv_gelu(device bfloat* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const float v = float(x[i]);
  x[i] = bfloat(0.5f * v * (1.0f + fnv_erf(v * 0.70710678118654752440f)));
}

// The language model's attention prep (q4_attn_prep, recorded) with three positions a row: Qwen's interleaved
// multimodal rotary over the 64 rotary dims, frequency i taking the row's h when i % 3 == 1 (i < 33), its w when
// i % 3 == 2 (i < 30), else its t. The rest is the recorded kernel's arithmetic, so a text row (t = h = w) keeps its
// bits. One threadgroup of 256 threads per (row, head): 24 query heads, 2 key heads, then 4 indexer query heads.
[[kernel]] void fnv_attn_prep_mrope(const device bfloat16_t* P [[buffer(0)]], const device int32_t* POS3 [[buffer(1)]],
                                    const device float* QW [[buffer(2)]], const device float* KW [[buffer(3)]],
                                    const device float* IW [[buffer(4)]], const constant float* eps [[buffer(5)]],
                                    const constant float* LOG2BASE [[buffer(6)]], device bfloat16_t* Q [[buffer(7)]],
                                    device bfloat16_t* Kout [[buffer(8)]], device bfloat16_t* IQ [[buffer(9)]],
                                    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
                                    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int NQ = 24;
  constexpr int NKV = 2;
  constexpr int HD = 256;
  constexpr int RD = 64;
  constexpr int PW = 13952;
  constexpr int NI = 4;
  constexpr int IHD = 128;
  const int d = int(thread_position_in_threadgroup.x);
  const int head = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  const bool isq = head < NQ;
  const bool isi = head >= NQ + NKV;
  const int width = isi ? IHD : HD;
  const bool live = d < width;
  int src;
  if (isq) src = r * PW + head * 2 * HD + d;
  else if (!isi) src = r * PW + NQ * 2 * HD + (head - NQ) * HD + d;
  else src = r * PW + NQ * 2 * HD + 2 * NKV * HD + (head - NQ - NKV) * IHD + d;
  threadgroup float part[HD / 32];
  threadgroup float normed[HD];
  const float x = live ? float(P[src]) : 0.0f;
  float ss = simd_sum(x * x);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int k = 0; k < width / 32; k++) total += part[k];
  const float inv = metal::rsqrt(total / float(width) + eps[0]);
  const float nw = live ? (isq ? QW[d] : (isi ? IW[d] : KW[d])) : 0.0f;
  normed[d] = float(bfloat((x * inv) * nw));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  float out = normed[d];
  if (d < RD) {
    const int hr = RD / 2;
    const int i = d % hr;
    const int axis = (i % 3 == 1 && i < 33) ? 1 : ((i % 3 == 2 && i < 30) ? 2 : 0);
    // LOG2BASE[1] != 0: YaRN (the frequencies at [3, 35), cos and sin scaled by [2]); else the plain rotary
    const float freq = LOG2BASE[1] != 0.0f ? LOG2BASE[3 + i] : metal::exp2(-(float(i) / float(hr)) * LOG2BASE[0]);
    const float angle = float(POS3[r * 3 + axis]) * freq;
    const float ys = LOG2BASE[1] != 0.0f ? LOG2BASE[2] : 1.0f;
    const float c = metal::fast::cos(angle) * ys, s = metal::fast::sin(angle) * ys;
    out = d < hr ? normed[d] * c - normed[d + hr] * s : normed[d - hr] * s + normed[d] * c;
  }
  if (isq) Q[(r * NQ + head) * HD + d] = bfloat(out);
  else if (isi) IQ[(r * NI + head - NQ - NKV) * IHD + d] = bfloat(out);
  else Kout[(r * NKV + head - NQ) * HD + d] = bfloat(out);
}
