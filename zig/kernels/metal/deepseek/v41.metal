// DeepSeek-V4.1-Flash's own operations, each the reference's (DeepSeek's inference/model.py) arithmetic:
// hyper-connection boundaries with the previous boundary's collapse, the quantize-dequantize steps the model was
// trained through (fp8 window keys, fp4 compressed and index keys), compressor pooling, the indexer's scores and
// top-k, sparse attention with sinks, routing, SwiGLU and the engram gate. Affine projections are core/row_projection.
#include <metal_stdlib>
using namespace metal;

constant constexpr int D = 5120;
constant constexpr int S = 4;
constant constexpr int F = S * D;
constant constexpr int MIX = (2 + S) * S;
constant constexpr int HD = 512;  // the attention head (one shared key/value latent)
constant constexpr int IH = 32;   // indexer heads
constant constexpr int ID = 128;  // indexer head dim

#pragma clang fp contract(off)

inline float bf_round(float v) { return float(bfloat(v)); }

// fp8 e4m3fn, round to nearest even, |v| <= 448 (callers clamp first).
inline float e4m3_round(float v) {
  const float a = fabs(v);
  if (a == 0.0f) return v;
  float q;
  if (a < 0.015625f) {  // subnormal: steps of 2^-9
    q = rint(a * 512.0f) / 512.0f;
  } else {
    const int e = ilogb(a);
    const float step = ldexp(1.0f, e - 3);
    q = rint(a / step) * step;
  }
  return copysign(q, v);
}

// fp4 e2m1, round to nearest even, |v| <= 6.
inline float e2m1_round(float v) {
  const float a = fabs(v);
  float q;
  if (a <= 0.25f) q = 0.0f;
  else if (a < 0.75f) q = 0.5f;
  else if (a <= 1.25f) q = 1.0f;
  else if (a < 1.75f) q = 1.5f;
  else if (a <= 2.5f) q = 2.0f;
  else if (a < 3.5f) q = 3.0f;
  else if (a <= 5.0f) q = 4.0f;
  else q = 6.0f;
  return copysign(q, v);
}

// 2^ceil(log2(x)) for x > 0, from the bits as the reference's fast_round_scale.
inline float pow2_ceil(float x) {
  const uint b = as_type<uint>(x);
  const int e = int((b >> 23) & 0xFF) - 127 + ((b & 0x7FFFFF) != 0 ? 1 : 0);
  return as_type<float>(uint(e + 127) << 23);
}

// ---------------------------------------------------------------- embedding and hyper-connections

// X [rows, 4, D]: every copy of the stream is the token's embedding.
[[kernel]] void ds_embed(const device uint* IDS [[buffer(0)]], const device bfloat* E [[buffer(1)]],
                         device bfloat* X [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  const bfloat v = E[size_t(IDS[r]) * D + d];
  for (int s = 0; s < S; s++) X[(size_t(r) * S + s) * D + d] = v;
}

// The MoE branch as ds_moe_combine rounds it: the routed experts' rows then the shared expert's, summed in f32.
static float combined(const device bfloat* Y, const device bfloat* SH, int topk, int r, int d) {
  float acc = 0.0f;
  for (int k = 0; k < topk; k++) acc += float(Y[(size_t(r) * topk + k) * D + d]);
  acc += float(SH[size_t(r) * D + d]);
  return float(bfloat(acc));
}

// The pending branch into the streams in place: x'[k] = post[k] * branch + sum_j comb[j][k] x[j]. With topk > 0 the
// branch is the MoE's, combined here from the routed experts' rows Y and the shared expert's SH.
[[kernel]] void ds_hc_expand(device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                             const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                             const device bfloat* Y [[buffer(4)]], const device bfloat* SH [[buffer(5)]],
                             constant int& topk [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  device bfloat* x = X + size_t(r) * F;
  const float b = topk > 0 ? combined(Y, SH, topk, int(r), int(d)) : float(BR[size_t(r) * D + d]);
  float xs[S];
  for (int j = 0; j < S; j++) xs[j] = float(x[j * D + d]);
  const device float* c = COMB + r * S * S;
  for (int k = 0; k < S; k++) {
    float mm = 0.0f;
    for (int j = 0; j < S; j++) mm += c[j * S + k] * xs[j];
    x[k * D + d] = bfloat(POST[r * S + k] * b + mm);
  }
}

struct HcArgs {
  float eps;      // the mixes' and the norm's (norm_eps)
  float hc_eps;   // the split's
  int iters;
};

constant constexpr int SLICES = F / 1024;

// The mixes' partial sums: a threadgroup of 256 a (1024-column slice, row), each slice's 24 mix products and its
// sum of squares into PART [rows, SLICES, MIX + 1].
[[kernel]] void ds_hc_mix_part(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                               device float* PART [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const device bfloat* x = X + size_t(r) * F + 1024 * j + 4 * int(t);
  float v[4];
  for (int i = 0; i < 4; i++) v[i] = float(x[i]);
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + 1024 * j + 4 * int(t);
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += red[k][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// ds_hc_expand then ds_hc_mix_part in one: the pending branch into the streams, read from X and written to XO (the
// other buffer: a threadgroup reads every stream at its columns while others write theirs), the slice's mix sums from
// the bf16 values written, as ds_hc_mix_part would read them back.
[[kernel]] void ds_hc_expand_mix(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                 const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                 const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                 device bfloat* XO [[buffer(6)]], const device bfloat* Y [[buffer(7)]],
                                 const device bfloat* SH [[buffer(8)]], constant int& topk [[buffer(9)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8][MIX + 1];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const int k = col / D, d0 = col % D;
  const device bfloat* x = X + size_t(r) * F;
  const device float* cm = COMB + r * S * S;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float b = topk > 0 ? combined(Y, SH, topk, r, d) : float(BR[size_t(r) * D + d]);
    float xs[S];
    for (int jj = 0; jj < S; jj++) xs[jj] = float(x[jj * D + d]);
    float mm = 0.0f;
    for (int jj = 0; jj < S; jj++) mm += cm[jj * S + k] * xs[jj];
    const bfloat o = bfloat(POST[r * S + k] * b + mm);
    XO[size_t(r) * F + col + i] = o;
    v[i] = float(o);
  }
  float acc[MIX + 1];
  acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int m = 0; m < MIX; m++) {
    const device float* f = FN + size_t(m) * F + col;
    acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
  for (int m = 0; m <= MIX; m++) {
    const float s = simd_sum(acc[m]);
    if (lane == 0) red[sg][m] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= MIX) {
    float s = 0.0f;
    for (int q = 0; q < 8; q++) s += red[q][t];
    PART[(size_t(r) * SLICES + j) * (MIX + 1) + t] = s;
  }
}

// One threadgroup of 1024 a row. MIX: the slices' partial sums (ds_hc_mix_part) added in order, normed by the
// stream's rms, split into this boundary's pre, post and Sinkhorn comb. Always: the streams collapsed with PRE_IN
// (the previous boundary's pre), rms-normed by NORMW.
template <bool MIXES>
[[kernel]] void ds_hc_boundary(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                               const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                               const device float* PRE_IN [[buffer(4)]], const device bfloat* NORMW [[buffer(5)]],
                               constant HcArgs& a [[buffer(6)]], device float* PRE_OUT [[buffer(7)]],
                               device float* POST_OUT [[buffer(8)]], device float* COMB_OUT [[buffer(9)]],
                               device bfloat* NORMED [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[32][MIX + 1];
  threadgroup float tot[MIX + 1];
  threadgroup float pre_in[S];
  const device bfloat* x = X + size_t(r) * F;
  // the mixes' totals beside the lagged collapse (bf16, as hc_pre returns) of the previous boundary's pre: the
  // collapse reads nothing this boundary computes, so the split below overlaps its rms reduction
  if (MIXES && int(t) <= MIX) {
    float s = 0.0f;
    for (int j = 0; j < SLICES; j++) s += PART[(size_t(r) * SLICES + j) * (MIX + 1) + t];
    tot[t] = s;
  }
  if (t < uint(S)) pre_in[t] = PRE_IN[r * S + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float ys[5];
  float ss = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    float y = 0.0f;
    for (int s = 0; s < S; s++) y += pre_in[s] * float(x[s * D + d]);
    ys[i] = bf_round(y);
    ss += ys[i] * ys[i];
  }
  const float s1 = simd_sum(ss);
  if (lane == 0) red[sg][0] = s1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the split on simdgroup 0: lanes 0-3 a pre and a post each, lanes 0-15 a cell of the 4x4 comb (j = lane / 4,
  // k = lane % 4), its row sums across lanes xor 1 and 2, its column sums across lanes xor 4 and 8
  if (MIXES && sg == 0) {
    const float rs = precise::rsqrt(tot[MIX] / float(F) + a.eps);
    if (lane < uint(S)) {
      PRE_OUT[r * S + lane] = 1.0f / (1.0f + precise::exp(-(tot[lane] * rs * SCALE[0] + BASE[lane]))) + a.hc_eps;
      POST_OUT[r * S + lane] = 2.0f / (1.0f + precise::exp(-(tot[S + lane] * rs * SCALE[1] + BASE[S + lane])));
    }
    const bool act = lane < uint(S * S);
    const uint cell = act ? lane : 0;
    const float v = tot[2 * S + cell] * rs * SCALE[2] + BASE[2 * S + cell];
    float mx = max(v, simd_shuffle_xor(v, ushort(1)));
    mx = max(mx, simd_shuffle_xor(mx, ushort(2)));
    const float e = precise::exp(v - mx);
    float row = e + simd_shuffle_xor(e, ushort(1));
    row = row + simd_shuffle_xor(row, ushort(2));
    float cm = act ? e / row + a.hc_eps : 0.0f;
    for (int it = 0; it < a.iters; it++) {
      if (it > 0) {
        float rsum = cm + simd_shuffle_xor(cm, ushort(1));
        rsum = rsum + simd_shuffle_xor(rsum, ushort(2));
        cm = act ? cm / (rsum + a.hc_eps) : 0.0f;
      }
      float csum = cm + simd_shuffle_xor(cm, ushort(4));
      csum = csum + simd_shuffle_xor(csum, ushort(8));
      cm = act ? cm / (csum + a.hc_eps) : 0.0f;
    }
    if (act) COMB_OUT[r * S * S + lane] = cm;
  }
  if (t == 32) {  // simdgroup 1, while simdgroup 0 runs the split
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += red[k][0];
    pre_in[0] = precise::rsqrt(s / float(D) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = pre_in[0];
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * D + d] = bfloat(float(NORMW[d]) * (ys[i] * inv));
  }
}

template [[host_name("ds_hc_boundary")]] [[kernel]] decltype(ds_hc_boundary<true>) ds_hc_boundary<true>;
template [[host_name("ds_hc_collapse")]] [[kernel]] decltype(ds_hc_boundary<false>) ds_hc_boundary<false>;

// ---------------------------------------------------------------- norms, projections, rope, quantize steps

struct RowsArgs {
  int n;         // row width
  int x_stride;  // values between rows
  int y_stride;
  float eps;
};

// y = norm(x) * w, rms over n (in place allowed); a threadgroup of 256 a row.
[[kernel]] void ds_rms(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                       device bfloat* Y [[buffer(2)]], constant RowsArgs& a [[buffer(3)]],
                       uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8];
  const device bfloat* x = X + size_t(r) * a.x_stride;
  float ss = 0.0f;
  for (int i = int(t); i < a.n; i += 256) {
    const float v = float(x[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int k = 0; k < 8; k++) tot += red[k];
  const float inv = precise::rsqrt(tot / float(a.n) + a.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device bfloat* y = Y + size_t(r) * a.y_stride;
  for (int i = int(t); i < a.n; i += 256) y[i] = bfloat(float(W[i]) * (float(x[i]) * inv));
}

struct GemvArgs {
  int rows;
  int n;
  int k;
  int group_n;   // outputs that share one x slice (n for an ordinary matrix; o_rank for wo_a's groups)
  int x_stride;  // values between x rows
  int y_stride;
};

// y[r, j] = sum_k x[r, (j / group_n) * k + kk] W[j, kk]: bf16 weights, fp32 sums; 8 simdgroups a group, one output
// column each, 8 rows a group.
template <typename XT, typename YT>
[[kernel]] void ds_gemv(const device XT* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                        device YT* Y [[buffer(2)]], constant GemvArgs& a [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  const int j = int(tg.x) * 8 + int(sg);
  if (j >= a.n) return;
  const int r0 = int(tg.y) * 8;
  const int nr = min(8, a.rows - r0);
  const device bfloat* w = W + size_t(j) * a.k;
  const int xo = (j / a.group_n) * a.k;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  // 8 consecutive inputs a lane a step (k is a multiple of 256 at every shape this family multiplies)
  for (int kk = int(lane) * 8; kk < a.k; kk += 256) {
    float wv[8];
    for (int u = 0; u < 8; u++) wv[u] = float(w[kk + u]);
    for (int i = 0; i < nr; i++) {
      const device XT* xr = X + size_t(r0 + i) * a.x_stride + xo + kk;
      float part = 0.0f;
      for (int u = 0; u < 8; u++) part += float(xr[u]) * wv[u];
      acc[i] += part;
    }
  }
  for (int i = 0; i < nr; i++) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) Y[size_t(r0 + i) * a.y_stride + j] = YT(s);
  }
}

template [[host_name("ds_gemv_bf_bf")]] [[kernel]] decltype(ds_gemv<bfloat, bfloat>) ds_gemv<bfloat, bfloat>;
template [[host_name("ds_gemv_bf_f32")]] [[kernel]] decltype(ds_gemv<bfloat, float>) ds_gemv<bfloat, float>;

// x = bf16(x * c) in place: a bf16 tensor times a Python float, as the reference scales the indexer's weights.
[[kernel]] void ds_scale(device bfloat* X [[buffer(0)]], constant float& c [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  X[i] = bfloat(float(X[i]) * c);
}

struct RopeArgs {
  int heads;
  int stride;   // values between heads (rows are heads * stride apart)
  int offset;   // the rotated dims' first, inside a head
  int inverse;
};

// Adjacent pairs of a head's last 64 dims rotated by position * freq (torch.polar's complex product), bf16 out.
[[kernel]] void ds_rope(device bfloat* X [[buffer(0)]], const device int* POS [[buffer(1)]],
                        const device float* FREQ [[buffer(2)]], constant RopeArgs& a [[buffer(3)]],
                        uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), h = int(g.y), r = int(g.z);
  device bfloat* x = X + (size_t(r) * a.heads + h) * a.stride + a.offset + 2 * i;
  const float ang = float(POS[r]) * FREQ[i];
  const float cs = precise::cos(ang);
  const float sn = a.inverse ? -precise::sin(ang) : precise::sin(ang);
  const float re = float(x[0]), im = float(x[1]);
  x[0] = bfloat(re * cs - im * sn);
  x[1] = bfloat(re * sn + im * cs);
}

// act_quant's fused quantize-dequantize: fp8 e4m3 with a power-of-two scale a block of 32, in place.
[[kernel]] void ds_fp8_qdq(device bfloat* X [[buffer(0)]], uint b [[thread_position_in_grid]]) {
  device bfloat* x = X + size_t(b) * 32;
  float amax = 0.0f;
  for (int i = 0; i < 32; i++) amax = max(amax, fabs(float(x[i])));
  const float s = pow2_ceil(max(amax, 1e-4f) / 448.0f);
  for (int i = 0; i < 32; i++) x[i] = bfloat(e4m3_round(clamp(float(x[i]) / s, -448.0f, 448.0f)) * s);
}

// fp4_act_quant's: e2m1 with a block of `block` (32 with an e8m0 scale, 16 with an e4m3 one), in place.
[[kernel]] void ds_fp4_qdq(device bfloat* X [[buffer(0)]], constant int& e4m3_scale [[buffer(1)]],
                           uint b [[thread_position_in_grid]]) {
  const int block = e4m3_scale ? 16 : 32;
  device bfloat* x = X + size_t(b) * block;
  float amax = 0.0f;
  for (int i = 0; i < block; i++) amax = max(amax, fabs(float(x[i])));
  const float s = e4m3_scale ? e4m3_round(max(amax, 6.0f * 0.001953125f) / 6.0f) : pow2_ceil(max(amax, 6.0f * 1.1754944e-38f) / 6.0f);
  for (int i = 0; i < block; i++) x[i] = bfloat(e2m1_round(clamp(float(x[i]) / s, -6.0f, 6.0f)) * s);
}

struct CopyArgs {
  int width;  // values a row
  int ring;   // destination rows (index = pos % ring); 0: index = pos
};

// Rows into a cache at their positions (a ring when `ring` > 0).
[[kernel]] void ds_put_rows(const device bfloat* SRC [[buffer(0)]], const device int* POS [[buffer(1)]],
                            device bfloat* DST [[buffer(2)]], constant CopyArgs& a [[buffer(3)]],
                            uint2 g [[thread_position_in_grid]]) {
  const int p = POS[g.y];
  const int at = a.ring > 0 ? p % a.ring : p;
  DST[size_t(at) * a.width + g.x] = SRC[size_t(g.y) * a.width + g.x];
}

// f32 rows into a ring at their positions.
[[kernel]] void ds_put_rows_f32(const device float* SRC [[buffer(0)]], const device int* POS [[buffer(1)]],
                                device float* DST [[buffer(2)]], constant CopyArgs& a [[buffer(3)]],
                                uint2 g [[thread_position_in_grid]]) {
  const int p = POS[g.y];
  DST[size_t(p % a.ring) * a.width + g.x] = SRC[size_t(g.y) * a.width + g.x];
}

// ---------------------------------------------------------------- compressor

struct PoolArgs {
  int first;  // the first group to pool
  int ring;   // the projection ring's rows
};

// Groups of 2 positions pooled by their learned softmax gate (fp32), bf16 out: OUT[g - first] for each group.
[[kernel]] void ds_pool2(const device float* KV [[buffer(0)]], const device float* SC [[buffer(1)]],
                         device bfloat* OUT [[buffer(2)]], constant PoolArgs& a [[buffer(3)]],
                         uint2 g [[thread_position_in_grid]]) {
  const int d = int(g.x), grp = a.first + int(g.y);
  const int p0 = (2 * grp) % a.ring, p1 = (2 * grp + 1) % a.ring;
  const float s0 = SC[size_t(p0) * HD + d], s1 = SC[size_t(p1) * HD + d];
  const float m = max(s0, s1);
  const float e0 = precise::exp(s0 - m), e1 = precise::exp(s1 - m);
  const float sum = e0 + e1;
  const float v = KV[size_t(p0) * HD + d] * (e0 / sum) + KV[size_t(p1) * HD + d] * (e1 / sum);
  OUT[size_t(g.y) * HD + d] = bfloat(v);
}

// ---------------------------------------------------------------- indexer

struct ScoreArgs {
  int rows;
  int groups;     // index keys in the cache
  int ratio;
  int first_pos;  // row r's position is first_pos + r
  int stride;     // score row stride
};

// score[r, t] = sum_h bf16(relu(bf16(q_h . k_t)) * w_h) (bf16 like the reference's einsum and product), -inf for
// keys the row cannot see yet ((pos + 1) / ratio of them).
// A simdgroup a key group, a lane a head (IH = 32): each head's dot in the order one thread took it, then lane 0 adds
// the heads' terms in head order (bit-identical to a thread a group).
[[kernel]] void ds_index_scores(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                const device bfloat* K [[buffer(2)]], device float* SCORE [[buffer(3)]],
                                constant ScoreArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const int t = int(g.x) / 32, r = int(g.y), h = int(lane);
  if (t >= a.groups) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  device float* out = SCORE + size_t(r) * a.stride;
  if (t >= visible) {
    if (lane == 0) out[t] = -INFINITY;
    return;
  }
  const device bfloat* k = K + size_t(t) * ID;
  const device bfloat* q = Q + (size_t(r) * IH + h) * ID;
  float dot = 0.0f;
  for (int i = 0; i < ID; i++) dot += float(q[i]) * float(k[i]);
  const float term = bf_round(max(bf_round(dot), 0.0f) * float(WT[r * IH + h]));
  float total = 0.0f;
  for (int j = 0; j < IH; j++) total += simd_shuffle(term, ushort(j));
  if (lane == 0) out[t] = bf_round(total);
}

struct MaskArgs {
  int groups;
  int stride;
  int block;
  int kept;  // KEEP's row width: the candidate blocks in ascending order, -1 after the last
};

// Scores of positions outside the row's candidate blocks to -inf (a binary search of the row's kept blocks).
[[kernel]] void ds_candidate_mask(device float* SCORE [[buffer(0)]], const device int* KEEP [[buffer(1)]],
                                  constant MaskArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int t = int(g.x), r = int(g.y);
  if (t >= a.groups) return;
  const device int* keep = KEEP + size_t(r) * a.kept;
  const int b = t / a.block;
  int lo = 0, hi = a.kept;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    const int v = keep[mid];
    if (v >= 0 && v < b) lo = mid + 1;
    else hi = mid;
  }
  if (lo >= a.kept || keep[lo] != b) SCORE[size_t(r) * a.stride + t] = -INFINITY;
}

struct BlockArgs {
  int rows;
  int groups;
  int block;
  int first_pos;
  int ratio;
  int stride;
};

// Block scores (the max of a block's positions; the row's newest block pinned at +inf) for the candidate pick.
[[kernel]] void ds_block_max(const device float* SCORE [[buffer(0)]], device float* BLK [[buffer(1)]],
                             constant BlockArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int b = int(g.x), r = int(g.y);
  const int nb = (a.groups + a.block - 1) / a.block;
  if (b >= nb) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  float m = -INFINITY;
  for (int i = 0; i < a.block; i++) {
    const int t = b * a.block + i;
    if (t < a.groups) m = max(m, SCORE[size_t(r) * a.stride + t]);
  }
  if (visible > 0 && b == (visible - 1) / a.block) m = FLT_MAX;  // pinned in (ds_topk takes finite scores only)
  BLK[size_t(r) * a.stride + b] = m;
}

struct TopkArgs {
  int n;       // candidates a row
  int k;       // picks a row
  int stride;  // score row stride
  int offset;  // added to each pick (the compressed keys' place behind the window)
};

inline uint order_key(float v) {
  const uint b = as_type<uint>(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// A row's k best finite scores (ties to the lower index), written in index order + offset; -1 fills the rest.
// One threadgroup of 1024 a row: a radix select of the k-th key, then an ordered compaction.
// A 1,024-thread group's sum of v (every thread gets it), and each thread's sum of v over the threads before it:
// unsigned integers, so exact in any order. `sums` is 32 words of threadgroup scratch.
static uint tg_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint s = simd_sum(v);
  if (lane == 0) sums[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint total = simd_sum(sums[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

static uint tg_exclusive_sum(uint v, uint t, uint lane, uint sg, threadgroup uint* sums) {
  const uint inside = simd_prefix_exclusive_sum(v);
  if (lane == 31) sums[sg] = inside + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint lead = simd_prefix_exclusive_sum(sums[lane]);
  const uint before = simd_shuffle(lead, ushort(sg));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return before + inside;
}

[[kernel]] void ds_topk(const device float* SCORE [[buffer(0)]], device int* OUT [[buffer(1)]],
                        constant TopkArgs& a [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_bits, want;
  threadgroup uint sums[32];
  const device float* s = SCORE + size_t(r) * a.stride;
  device int* out = OUT + size_t(r) * a.k;
  // finite candidates only: -inf marks keys that are not visible or not candidates
  if (t == 0) {
    prefix_bits = 0;
    want = uint(a.k);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint finite = 0;
  for (int i = int(t); i < a.n; i += 1024) finite += isfinite(s[i]) ? 1 : 0;
  const uint total = tg_sum(finite, t, lane, sg, sums);
  if (t == 0) want = min(want, total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint k = want;
  // the k-th largest key, 8 bits a pass from the top
  uint mask = 0;
  for (int shift = 24; shift >= 0 && k > 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < a.n; i += 1024) {
      if (!isfinite(s[i])) continue;
      const uint key = order_key(s[i]);
      if ((key & mask) == prefix_bits) atomic_fetch_add_explicit(&hist[(key >> shift) & 0xFF], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the bin holding the want-th largest: bins from the top (thread t reads bin 255 - t), each thread's count of
    // the bins above its own by a scan; the one bin where that count reaches want takes it
    {
      const uint b = 255 - min(t, 255u);
      const uint hb = t < 256 ? atomic_load_explicit(&hist[b], memory_order_relaxed) : 0;
      const uint above = tg_exclusive_sum(hb, t, lane, sg, sums);
      const uint w = want;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (t < 256 && above < w && above + hb >= w) {
        prefix_bits |= b << shift;
        want = w - above;
      }
    }
    mask |= 0xFFu << shift;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const uint kth = prefix_bits;
  const uint ties = want;  // keys equal to kth to take, the lowest indices first
  // ordered compaction: each thread's span of indices, counted, then written in order
  const int span = (a.n + 1023) / 1024;
  const int lo = int(t) * span, hi = min(a.n, lo + span);
  uint above = 0, equal = 0;
  for (int i = lo; i < hi; i++) {
    if (!isfinite(s[i]) || k == 0) continue;
    const uint key = order_key(s[i]);
    above += key > kth ? 1 : 0;
    equal += key == kth ? 1 : 0;
  }
  const uint before_above = tg_exclusive_sum(above, t, lane, sg, sums);
  const uint before_equal = tg_exclusive_sum(equal, t, lane, sg, sums);
  uint taken_equal = before_equal;
  uint pos = before_above + min(before_equal, ties);
  for (int i = lo; i < hi; i++) {
    if (!isfinite(s[i]) || k == 0) continue;
    const uint key = order_key(s[i]);
    if (key > kth || (key == kth && taken_equal < ties)) {
      out[pos++] = i + a.offset;
    }
    if (key == kth) taken_equal++;
  }
  for (int i = int(k) + int(t); i < a.k; i += 1024) out[i] = -1;
}

// ---------------------------------------------------------------- sparse attention

struct AttnArgs {
  int first_pos;  // row r's position is first_pos + r
  int ring;       // the window ring's rows
  int window;
  int picks;      // compressed picks a row (0: window only)
  float scale;
  int window_end; // >= 0: every row's window ends at this position (a draft block's), else at the row's own
};

// One threadgroup (256) a (row, head): the row's window of raw keys from the ring plus its compressed picks, keys
// equal values (one 512-wide latent), an exact fp32 softmax with the head's sink in the denominator.
[[kernel]] void ds_attend(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                          const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                          const device float* SINK [[buffer(4)]], device bfloat* O [[buffer(5)]],
                          constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h = int(tg.x), r = int(tg.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  threadgroup float sc[1024];
  threadgroup float qs[HD];
  threadgroup float red[8];
  const device bfloat* q = Q + (size_t(r) * 64 + h) * HD;
  for (int i = int(t); i < HD; i += 256) qs[i] = float(q[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // scores: simdgroup sg takes keys sg, sg + 8, ...
  for (int j = int(sg); j < nkeys; j += 8) {
    const device bfloat* key;
    bool valid = true;
    if (j < nwin) {
      key = RING + size_t((pos - nwin + 1 + j) % a.ring) * HD;
    } else {
      const int p = PICKS[size_t(r) * a.picks + (j - nwin)];
      valid = p >= 0;
      key = COMP + size_t(max(p, 0)) * HD;
    }
    float dot = 0.0f;
    for (int i = int(lane); i < HD; i += 32) dot += qs[i] * float(key[i]);
    dot = simd_sum(dot);
    if (lane == 0) sc[j] = valid ? dot * a.scale : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f;
  for (int j = int(t); j < nkeys; j += 256) m = max(m, sc[j]);
  m = simd_max(m);
  if (lane == 0) red[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = red[0];
  for (int k = 1; k < 8; k++) m = max(m, red[k]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float l = 0.0f;
  for (int j = int(t); j < nkeys; j += 256) {
    const float p = precise::exp(sc[j] - m);
    sc[j] = p;
    l += p;
  }
  l = simd_sum(l);
  if (lane == 0) red[sg] = l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = precise::exp(SINK[h] - m);
  for (int k = 0; k < 8; k++) denom += red[k];
  // outputs: thread t takes dims t and t + 256
  for (int i = int(t); i < HD; i += 256) {
    float acc = 0.0f;
    for (int j = 0; j < nkeys; j++) {
      if (sc[j] == 0.0f) continue;
      const device bfloat* key;
      if (j < nwin) {
        key = RING + size_t((pos - nwin + 1 + j) % a.ring) * HD;
      } else {
        key = COMP + size_t(max(PICKS[size_t(r) * a.picks + (j - nwin)], 0)) * HD;
      }
      acc += sc[j] * float(key[i]);
    }
    O[(size_t(r) * 64 + h) * HD + i] = bfloat(acc / denom);
  }
}

constant constexpr int AH = 8;     // heads a threadgroup
constant constexpr int AT = 8;     // keys a tile
constant constexpr int AMAX = 640; // keys a row at most: the window and the indexer's picks

// The row's j-th key: its window (positions pos - nwin + 1 .. pos, from the ring), then its compressed picks.
inline const device bfloat* attn_key(const device bfloat* RING, const device bfloat* COMP, const device int* picks,
                                     int pos, int nwin, int ring, int j) {
  if (j < nwin) return RING + size_t((pos - nwin + 1 + j) % ring) * HD;
  return COMP + size_t(max(picks[j - nwin], 0)) * HD;
}

// ds_attend's arithmetic for prompt chunks: one threadgroup (512 threads) a (row, 8 heads), each tile of 8 keys read
// once into threadgroup memory for the 8 heads (28 KB: the tile and the scores). A decode step has too few rows to
// fill the GPU this way (8 threadgroups), so it keeps ds_attend's threadgroup a head.
[[kernel]] void ds_attend_rows(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                               const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                               const device float* SINK [[buffer(4)]], device bfloat* O [[buffer(5)]],
                               constant AttnArgs& a [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h0 = int(tg.x) * AH, r = int(tg.y);
  const int pos = a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const device int* picks = PICKS + size_t(r) * a.picks;
  const device bfloat* q = Q + (size_t(r) * 64 + h0) * HD;
  threadgroup bfloat ks[AT][HD];
  threadgroup float sc[AH][AMAX];
  threadgroup float den[AH];
  for (int k0 = 0; k0 < nkeys; k0 += AT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < AT * HD; i += 512) {
      const int kk = i / HD;
      ks[kk][i % HD] = (k0 + kk < nkeys) ? attn_key(RING, COMP, picks, pos, nwin, a.ring, k0 + kk)[i % HD] : bfloat(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pr = int(sg); pr < AH * AT; pr += 16) {
      const int h = pr / AT, kk = pr % AT, j = k0 + kk;
      if (j >= nkeys) continue;
      float dot = 0.0f;
      for (int d = int(lane); d < HD; d += 32) dot += float(q[h * HD + d]) * float(ks[kk][d]);
      dot = simd_sum(dot);
      if (lane == 0) sc[h][j] = (j < nwin || picks[j - nwin] >= 0) ? dot * a.scale : -INFINITY;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(sg) < AH) {
    const int h = int(sg);
    float m = -1e30f;
    for (int j = int(lane); j < nkeys; j += 32) m = max(m, sc[h][j]);
    m = simd_max(m);
    float l = 0.0f;
    for (int j = int(lane); j < nkeys; j += 32) {
      const float p = precise::exp(sc[h][j] - m);
      sc[h][j] = p;
      l += p;
    }
    l = simd_sum(l);
    if (lane == 0) den[h] = l + precise::exp(SINK[h0 + h] - m);
  }
  float acc[AH];
  for (int h = 0; h < AH; h++) acc[h] = 0.0f;
  for (int k0 = 0; k0 < nkeys; k0 += AT) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(t); i < AT * HD; i += 512) {
      const int kk = i / HD;
      ks[kk][i % HD] = (k0 + kk < nkeys) ? attn_key(RING, COMP, picks, pos, nwin, a.ring, k0 + kk)[i % HD] : bfloat(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int nt = min(AT, nkeys - k0);
    for (int kk = 0; kk < nt; kk++) {
      const float v = float(ks[kk][t]);
      for (int h = 0; h < AH; h++) acc[h] += sc[h][k0 + kk] * v;
    }
  }
  for (int h = 0; h < AH; h++) O[(size_t(r) * 64 + h0 + h) * HD + t] = bfloat(acc[h] / den[h]);
}

constant constexpr int SPLIT = 16;  // keys a split threadgroup takes (held in threadgroup memory: 16 KiB)
constant constexpr int HG = 16;     // heads a split threadgroup takes (a simdgroup each), sharing its key tiles

// Decode and window rows' attention, split by keys: one threadgroup (HG simdgroups, a head each) a (HG heads, SPLIT-key
// chunk, row); the chunk's keys are read once into threadgroup memory for the HG heads. The queries' RoPE is
// applied as they load. Writes each head's chunk partial (its max score, its sum of exps, its unnormalized output);
// ds_attend_merge folds the chunks and the sink.
[[kernel]] void ds_attend_split(const device bfloat* Q [[buffer(0)]], const device bfloat* RING [[buffer(1)]],
                                const device bfloat* COMP [[buffer(2)]], const device int* PICKS [[buffer(3)]],
                                device float* PO [[buffer(4)]], device float2* PML [[buffer(5)]],
                                constant AttnArgs& a [[buffer(6)]], const device int* POS [[buffer(7)]],
                                const device float* FREQ [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],
                                uint3 tgn [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint3 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h = int(tg.x) * HG + int(sg), chunk = int(tg.y), r = int(tg.z), chunks = int(tgn.y);
  const int pos = a.window_end >= 0 ? a.window_end : a.first_pos + r;
  const int nwin = min(a.window, pos + 1);
  const int nkeys = nwin + a.picks;
  const int k0 = chunk * SPLIT, k1 = min(nkeys, k0 + SPLIT);
  const device int* picks = PICKS + size_t(r) * a.picks;
  threadgroup bfloat ks[SPLIT][HD];
  threadgroup float sc[HG][SPLIT];
  // the chunk's keys, once, 8 values a load
  for (int i = int(t); i < SPLIT * (HD / 8); i += HG * 32) {
    const int kk = i / (HD / 8), c8 = i % (HD / 8), j = k0 + kk;
    uint4 v = uint4(0);
    if (j < k1) v = ((const device uint4*)attn_key(RING, COMP, picks, pos, nwin, a.ring, j))[c8];
    ((threadgroup uint4*)ks[kk])[c8] = v;
  }
  float q[16];
  const device bfloat* qr = Q + (size_t(r) * 64 + h) * HD;
  for (int i = 0; i < 16; i++) q[i] = float(qr[int(lane) * 16 + i]);
  // the query's RoPE (its last 64 dims: lanes 28-31, pairs inside a lane), rounded to bf16 as ds_rope writes it
  if (lane >= uint((HD - 64) / 16)) {
    for (int p = 0; p < 8; p++) {
      const float ang = float(POS[r]) * FREQ[(int(lane) - (HD - 64) / 16) * 8 + p];
      const float cs = precise::cos(ang), sn = precise::sin(ang);
      const float re = q[2 * p], im = q[2 * p + 1];
      q[2 * p] = float(bfloat(re * cs - im * sn));
      q[2 * p + 1] = float(bfloat(re * sn + im * cs));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // scores: simdgroup sg's head against each key of the chunk (lane: 16 dims)
  for (int kk = 0; kk < k1 - k0; kk++) {
    float d = 0.0f;
    for (int i = 0; i < 16; i++) d += q[i] * float(ks[kk][int(lane) * 16 + i]);
    d = simd_sum(d);
    const int j = k0 + kk;
    const bool valid = j < nwin || picks[j - nwin] >= 0;
    if (lane == 0) sc[sg][kk] = valid ? d * a.scale : -INFINITY;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float m = -1e30f;
  for (int j = int(lane); j < k1 - k0; j += 32) m = max(m, sc[sg][j]);
  m = simd_max(m);
  float l = 0.0f;
  for (int j = int(lane); j < k1 - k0; j += 32) {
    const float p = precise::exp(sc[sg][j] - m);
    sc[sg][j] = p;
    l += p;
  }
  l = simd_sum(l);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  // outputs: the same keys, still in threadgroup memory (lane: 16 dims of its head)
  float o[16];
  for (int i = 0; i < 16; i++) o[i] = 0.0f;
  for (int kk = 0; kk < k1 - k0; kk++) {
    const float p = sc[sg][kk];
    for (int i = 0; i < 16; i++) o[i] += p * float(ks[kk][int(lane) * 16 + i]);
  }
  const size_t part = (size_t(r) * 64 + h) * chunks + chunk;
  for (int i = 0; i < 16; i++) PO[part * HD + int(lane) * 16 + i] = o[i];
  if (lane == 0) PML[part] = float2(m, l);
}

// The chunks' partials of a (row, head) folded with the head's sink: the softmax over every key and the sink; a thread
// a dim (a threadgroup of HD).
struct MergeArgs {
  int chunks;
  int finish;  // then as ds_rope (inverse) and ds_fp8_qdq would: the head's RoPE taken back, fp8-rounded
};

// A head's 64 RoPE dims (the last of its 512) rotated, in a threadgroup's bf16 row, as ds_rope does (32 threads).
static void rope_row(threadgroup bfloat* x, int pos, const device float* FREQ, bool inverse, uint t) {
  if (t < 32) {
    threadgroup bfloat* p = x + HD - 64 + 2 * int(t);
    const float ang = float(pos) * FREQ[t];
    const float cs = precise::cos(ang);
    const float sn = inverse ? -precise::sin(ang) : precise::sin(ang);
    const float re = float(p[0]), im = float(p[1]);
    p[0] = bfloat(re * cs - im * sn);
    p[1] = bfloat(re * sn + im * cs);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// A threadgroup's bf16 row of 512 fp8-rounded a block of 32, as ds_fp8_qdq does: a block a simdgroup (a value a lane;
// the block's max is the same in any order), `n` threads in the threadgroup.
static void fp8_row(threadgroup bfloat* x, uint t, uint n) {
  const uint lane = t % 32;
  for (uint b = t / 32; b < uint(HD / 32); b += n / 32) {
    threadgroup bfloat* p = x + b * 32;
    const float v = float(p[lane]);
    const float amax = simd_max(fabs(v));
    const float s = pow2_ceil(max(amax, 1e-4f) / 448.0f);
    p[lane] = bfloat(e4m3_round(clamp(v / s, -448.0f, 448.0f)) * s);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

[[kernel]] void ds_attend_merge(const device float* PO [[buffer(0)]], const device float2* PML [[buffer(1)]],
                                const device float* SINK [[buffer(2)]], device bfloat* O [[buffer(3)]],
                                constant MergeArgs& ma [[buffer(4)]], const device int* POS [[buffer(5)]],
                                const device float* FREQ [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint2 tpos [[thread_position_in_threadgroup]]) {
  const uint t = tpos.x;
  const int h = int(tg.x), r = int(tg.y), chunks = ma.chunks;
  threadgroup bfloat row[HD];
  threadgroup float scale[(128 + 512 + SPLIT - 1) / SPLIT];
  threadgroup float2 ml[(128 + 512 + SPLIT - 1) / SPLIT];
  const size_t base = (size_t(r) * 64 + h) * chunks;
  for (int c = int(t); c < chunks; c += HD) ml[c] = PML[base + c];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float m = SINK[h];
  for (int c = 0; c < chunks; c++) m = max(m, ml[c].x);
  for (int c = int(t); c < chunks; c += HD) scale[c] = precise::exp(ml[c].x - m);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float denom = precise::exp(SINK[h] - m);
  for (int c = 0; c < chunks; c++) denom += ml[c].y * scale[c];
  for (int d = int(t); d < HD; d += HD) {
    float acc = 0.0f;
    for (int c = 0; c < chunks; c++) acc += PO[(base + c) * HD + d] * scale[c];
    row[d] = bfloat(acc / denom);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ma.finish) {
    rope_row(row, POS[r], FREQ, true, t);
    fp8_row(row, t, HD);
  }
  for (int d = int(t); d < HD; d += HD) O[(size_t(r) * 64 + h) * HD + d] = row[d];
}

// A window key's tail in one threadgroup a row, as ds_rms (kv_norm), ds_rope, ds_fp8_qdq and ds_put_rows would run
// it: KV is normed, rotated and rounded in place, then put in the ring at its position (when ring > 0).
[[kernel]] void ds_kv_finish(device bfloat* KV [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                             const device int* POS [[buffer(2)]], const device float* FREQ [[buffer(3)]],
                             device bfloat* RING [[buffer(4)]], constant float2& a [[buffer(5)]],
                             uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float red[8];
  threadgroup bfloat row[HD];
  device bfloat* x = KV + size_t(r) * HD;
  float ss = 0.0f;
  for (int i = int(t); i < HD; i += 256) {
    const float v = float(x[i]);
    ss += v * v;
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.0f;
  for (int k = 0; k < 8; k++) tot += red[k];
  const float inv = precise::rsqrt(tot / float(HD) + a.x);
  for (int i = int(t); i < HD; i += 256) row[i] = bfloat(float(W[i]) * (float(x[i]) * inv));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rope_row(row, POS[r], FREQ, false, t);
  fp8_row(row, t, 256);
  const int ring = int(a.y);
  device bfloat* dst = RING + size_t(ring > 0 ? POS[r] % ring : 0) * HD;
  for (int i = int(t); i < HD; i += 256) {
    x[i] = row[i];
    if (ring > 0) dst[i] = row[i];
  }
}

// ---------------------------------------------------------------- MoE

struct RouteArgs {
  int experts;
  int topk;
  float scale;
};

// sqrt(softplus(logits)) scores; the top-k of score + bias (ties to the lower id) picked; their unbiased scores
// normalized and scaled; image-span rows (IMG) steer by bias_vl. Picks are written in expert order, the order the
// reference sums them in. One threadgroup
// (a thread an expert, `experts` <= 512) a row: the scores in parallel, then k rounds of a threadgroup argmax.
[[kernel]] void ds_route(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                         device uint* PICK [[buffer(2)]], device float* WEIGHT [[buffer(3)]],
                         constant RouteArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                         const device float* BIAS_VL [[buffer(6)]], uint r [[threadgroup_position_in_grid]],
                         uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float best_v[16];
  threadgroup int best_i[16];
  threadgroup int sel[16];
  threadgroup float w[16];
  const device float* z = LOGITS + size_t(r) * a.experts;
  float score = 0.0f, key = -INFINITY;
  if (int(t) < a.experts) {
    const float x = z[t];
    score = precise::sqrt(x > 20.0f ? x : precise::log(1.0f + precise::exp(x)));
    key = score + (IMG[r] ? BIAS_VL[t] : BIAS[t]);  // an image span's rows steer by their own bias
  }
  for (int k = 0; k < a.topk; k++) {
    // the simdgroup's best (value, then lower id), then the threadgroup's
    float v = key;
    int id = int(t) < a.experts ? int(t) : 0x7fffffff;
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_down(v, o);
      const int oi = simd_shuffle_down(id, o);
      if (ov > v || (ov == v && oi < id)) {
        v = ov;
        id = oi;
      }
    }
    if (lane == 0) {
      best_v[sg] = v;
      best_i[sg] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      float bv = best_v[0];
      int bi = best_i[0];
      for (uint g = 1; g < nsg; g++)
        if (best_v[g] > bv || (best_v[g] == bv && best_i[g] < bi)) {
          bv = best_v[g];
          bi = best_i[g];
        }
      sel[k] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) == sel[k]) {
      w[k] = score;
      key = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    int s2[16];
    float w2[16];
    float sum = 0.0f;
    for (int k = 0; k < a.topk; k++) {
      s2[k] = sel[k];
      w2[k] = w[k];
      sum += w[k];
    }
    for (int k = 0; k < a.topk; k++) w2[k] = w2[k] / (sum + 1e-20f) * a.scale;
    for (int i = 1; i < a.topk; i++)
      for (int j = i; j > 0 && s2[j - 1] > s2[j]; j--) {
        const int ts = s2[j];
        s2[j] = s2[j - 1];
        s2[j - 1] = ts;
        const float tw = w2[j];
        w2[j] = w2[j - 1];
        w2[j - 1] = tw;
      }
    for (int k = 0; k < a.topk; k++) {
      PICK[r * a.topk + k] = uint(s2[k]);
      WEIGHT[r * a.topk + k] = w2[k];
    }
  }
}

struct GluArgs {
  int width;    // the intermediate width
  int weighted; // multiply by WEIGHT[slot] (routed experts) or not (the shared expert)
  float limit;
};

// silu(min(gate, limit)) * clamp(up, +-limit) [* weight], bf16 out.
[[kernel]] void ds_swiglu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                          const device float* WEIGHT [[buffer(2)]], device bfloat* OUT [[buffer(3)]],
                          constant GluArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * a.width + g.x;
  const float gate = min(float(G[i]), a.limit);
  const float up = clamp(float(U[i]), -a.limit, a.limit);
  float v = gate / (1.0f + precise::exp(-gate)) * up;
  if (a.weighted) v = WEIGHT[g.y] * v;
  OUT[i] = bfloat(v);
}

// The routed experts' outputs (fp32 sum in slot order) plus the shared expert's, bf16.
[[kernel]] void ds_moe_combine(const device bfloat* Y [[buffer(0)]], const device bfloat* SH [[buffer(1)]],
                               device bfloat* OUT [[buffer(2)]], constant int& topk [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  float acc = 0.0f;
  for (int k = 0; k < topk; k++) acc += float(Y[(size_t(r) * topk + k) * D + d]);
  acc += float(SH[size_t(r) * D + d]);
  OUT[size_t(r) * D + d] = bfloat(acc);
}

struct SortArgs {
  int slots;
  int experts;
};

// A prompt chunk's routed slots sorted by expert (stable: slot order inside an expert), for core/affine_mm's
// gathers: OFFS each expert's first sorted slot, ORDER sorted -> slot, WS the slots' weights in sorted order.
// One thread: a chunk has at most a few thousand slots.
[[kernel]] void ds_sort_slots(const device uint* PICK [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                              device int* OFFS [[buffer(2)]], device int* ORDER [[buffer(3)]],
                              device float* WS [[buffer(4)]], constant SortArgs& a [[buffer(5)]],
                              uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  for (int e = 0; e < a.experts; e++) OFFS[e] = 0;
  for (int i = 0; i < a.slots; i++) OFFS[PICK[i]]++;
  int at = 0;
  for (int e = 0; e < a.experts; e++) {
    const int n = OFFS[e];
    OFFS[e] = at;
    at += n;
  }
  // place, then shift the offsets back (each expert's fill pointer ends at the next one's start)
  for (int i = 0; i < a.slots; i++) {
    const int at2 = OFFS[PICK[i]]++;
    ORDER[at2] = i;
    WS[at2] = WEIGHT[i];
  }
  for (int e = a.experts - 1; e > 0; e--) OFFS[e] = OFFS[e - 1];
  OFFS[0] = 0;
}

// Sorted slot i's input row: X[ORDER[i] / topk] (a token's row for each of its picks).
[[kernel]] void ds_gather_rows(const device bfloat* X [[buffer(0)]], const device int* ORDER [[buffer(1)]],
                               device bfloat* Y [[buffer(2)]], constant int2& wk [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  Y[size_t(g.y) * wk.x + g.x] = X[size_t(ORDER[g.y] / wk.y) * wk.x + g.x];
}

// The mean of a row's four streams (h.mean(dim=2), bf16) into Y at column `col` of rows `ld` wide: a DSpark
// target layer's attention input.
[[kernel]] void ds_stream_mean(const device bfloat* X [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                               constant int2& col_ld [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const device bfloat* x = X + size_t(g.y) * F;
  const float v = (float(x[g.x]) + float(x[D + g.x]) + float(x[2 * D + g.x]) + float(x[3 * D + g.x])) / 4.0f;
  Y[size_t(g.y) * col_ld.y + col_ld.x + g.x] = bfloat(v);
}

struct MarkovArgs {
  int vocab;
  int rank;
  int step;   // which draft row's logits, and IDS[step] the token whose Markov embedding biases them
};

// One draft step's logits plus the Markov head's bias from the previous token (embed[tok] . head[v], fp32), and
// the threadgroup's best (value, lower id) into PART: phase one of the step's argmax.
[[kernel]] void ds_markov(const device bfloat* LOGITS [[buffer(0)]], const device bfloat* EMB [[buffer(1)]],
                          const device bfloat* HEAD [[buffer(2)]], const device uint* IDS [[buffer(3)]],
                          device float2* PART [[buffer(4)]], constant MarkovArgs& a [[buffer(5)]],
                          uint tg [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float e[256];
  threadgroup float bv[8];
  threadgroup int bi[8];
  const uint tok = IDS[a.step];
  for (int j = int(t); j < a.rank; j += 256) e[j] = float(EMB[size_t(tok) * a.rank + j]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int v = int(tg) * 256 + int(t);
  float val = -INFINITY;
  int id = 0x7fffffff;
  if (v < a.vocab) {
    const device bfloat* h = HEAD + size_t(v) * a.rank;
    float bias = 0.0f;
    for (int j = 0; j < a.rank; j++) bias += e[j] * float(h[j]);
    val = float(LOGITS[size_t(a.step) * a.vocab + v]) + bias;
    id = v;
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_down(val, o);
    const int oi = simd_shuffle_down(id, o);
    if (ov > val || (ov == val && oi < id)) {
      val = ov;
      id = oi;
    }
  }
  if (lane == 0) {
    bv[sg] = val;
    bi[sg] = id;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = bv[0];
    int mi = bi[0];
    for (int k = 1; k < 8; k++)
      if (bv[k] > m || (bv[k] == m && bi[k] < mi)) {
        m = bv[k];
        mi = bi[k];
      }
    PART[tg] = float2(m, as_type<float>(mi));
  }
}

// Phase two: the step's token, the best of the threadgroups' bests, into IDS[step + 1].
[[kernel]] void ds_markov_pick(const device float2* PART [[buffer(0)]], device uint* IDS [[buffer(1)]],
                               constant int2& n_step [[buffer(2)]], uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  float m = -INFINITY;
  int mi = 0x7fffffff;
  for (int k = 0; k < n_step.x; k++) {
    const float v = PART[k].x;
    const int i = as_type<int>(PART[k].y);
    if (v > m || (v == m && i < mi)) {
      m = v;
      mi = i;
    }
  }
  IDS[n_step.y + 1] = uint(mi);
}

// ---------------------------------------------------------------- engram

struct EngramArgs {
  float eps;
};

// One threadgroup (256) a (row, copy): the gate sigmoid(signed sqrt(normalized <h, w * key>)) of the lookup's key
// against the copy, then h += gate * value.
[[kernel]] void ds_engram(device bfloat* X [[buffer(0)]], const device bfloat* KV [[buffer(1)]],
                          const device bfloat* QW [[buffer(2)]], const device bfloat* KW [[buffer(3)]],
                          constant EngramArgs& a [[buffer(4)]], const device int* IMG [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int c = int(tg.x), r = int(tg.y);
  if (IMG[r]) return;  // an image span's rows take no lookup (the reference shuts their gate)
  threadgroup float red[3][8];
  threadgroup float gate_s;
  device bfloat* x = X + (size_t(r) * S + c) * D;
  const device bfloat* key = KV + size_t(r) * (S + 1) * D + size_t(c) * D;
  const device bfloat* val = KV + size_t(r) * (S + 1) * D + size_t(S) * D;
  float hh = 0.0f, kk = 0.0f, hk = 0.0f;
  for (int d = int(t); d < D; d += 256) {
    const float h = float(x[d]), k = float(key[d]);
    const float w = float(QW[c * D + d]) * float(KW[c * D + d]);
    hh += h * h;
    kk += k * k;
    hk += h * w * k;
  }
  hh = simd_sum(hh);
  kk = simd_sum(kk);
  hk = simd_sum(hk);
  if (lane == 0) {
    red[0][sg] = hh;
    red[1][sg] = kk;
    red[2][sg] = hk;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float s0 = 0, s1 = 0, s2 = 0;
    for (int i = 0; i < 8; i++) {
      s0 += red[0][i];
      s1 += red[1][i];
      s2 += red[2][i];
    }
    const float rstd = precise::rsqrt(s0 / float(D) + a.eps) * precise::rsqrt(s1 / float(D) + a.eps);
    const float dot = s2 * rstd * precise::rsqrt(float(D));
    const float sq = precise::sqrt(max(fabs(dot), 1e-6f));
    gate_s = 1.0f / (1.0f + precise::exp(-copysign(sq, dot)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(t); d < D; d += 256) x[d] = bfloat(float(x[d]) + gate_s * float(val[d]));
}

// ---------------------------------------------------------------- the vision tower (vision.py) and aligner

constant constexpr int VD = 1024;  // the tower's width
constant constexpr int VH = 16;    // its heads
constant constexpr int VHD = 64;   // a head's width

struct LinArgs {
  int rows;
  int n;
  int k;
};

// y[r, j] = bf16(sum_k x[r, k] W[j, k] + b[j]), any k: the patch embedding (588 inputs). 8 simdgroups a group, one
// output column each, 8 rows a group.
[[kernel]] void ds_linear_bias(const device bfloat* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
                               const device bfloat* B [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                               constant LinArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  const int j = int(tg.x) * 8 + int(sg);
  if (j >= a.n) return;
  const int r0 = int(tg.y) * 8, nr = min(8, a.rows - r0);
  const device bfloat* w = W + size_t(j) * a.k;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int kk = int(lane); kk < a.k; kk += 32) {
    const float wv = float(w[kk]);
    for (int i = 0; i < nr; i++) acc[i] += float(X[size_t(r0 + i) * a.k + kk]) * wv;
  }
  for (int i = 0; i < nr; i++) {
    const float s = simd_sum(acc[i]);
    if (lane == 0) Y[size_t(r0 + i) * a.n + j] = bfloat(s + float(B[j]));
  }
}

// erf (Metal has none): Numerical Recipes' erfc, fractional error under 1.2e-7, far inside a bf16 step.
inline float ds_erf(float x) {
  const float z = fabs(x);
  const float t = 1.0f / (1.0f + 0.5f * z);
  const float r = t * precise::exp(-z * z - 1.26551223f +
                                   t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
                                   t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
  return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

// y = bf16(y + b) by column, and with `gelu` then GELU (erf form) on the sum: a Linear's bias, the aligner's w1.
[[kernel]] void ds_bias(device bfloat* Y [[buffer(0)]], const device bfloat* B [[buffer(1)]],
                        constant int2& n_gelu [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * n_gelu.x + g.x;
  float v = float(bfloat(float(Y[i]) + float(B[g.x])));
  if (n_gelu.y) v = 0.5f * v * (1.0f + ds_erf(v * 0.70710678118654752f));
  Y[i] = bfloat(v);
}

// x = bf16(x + y): a residual branch.
[[kernel]] void ds_add(device bfloat* X [[buffer(0)]], const device bfloat* Y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  X[i] = bfloat(float(X[i]) + float(Y[i]));
}

// The tower's 2D RoPE on q and k inside qkv [rows, 3 x 1024]: a patch's 32 frequencies are its row's then its
// column's, each half of a head rotated against the other (apply_rotary's chunk(2)).
[[kernel]] void ds_vit_rope(device bfloat* QKV [[buffer(0)]], const device float* FREQ [[buffer(1)]],
                            constant int& n_w [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  const int i = int(g.x), h = int(g.y), r = int(g.z);  // i < 32, h < 32 (q's 16 heads, then k's)
  const int f = i < 16 ? (r / n_w) : (r % n_w);
  const float ang = float(f) * FREQ[i % 16];
  const float cs = precise::cos(ang), sn = precise::sin(ang);
  device bfloat* x = QKV + size_t(r) * 3 * VD + size_t(h) * VHD;
  const float x1 = float(x[i]), x2 = float(x[i + 32]);
  x[i] = bfloat(x1 * cs - x2 * sn);
  x[i + 32] = bfloat(x2 * cs + x1 * sn);
}

// Bidirectional attention over the image's patches (scaled_dot_product_attention): one simdgroup a (patch, head),
// 32 keys a step (each lane one key's dot), an online fp32 softmax, each lane two output dims.
[[kernel]] void ds_vit_attend(const device bfloat* QKV [[buffer(0)]], device bfloat* O [[buffer(1)]],
                              constant int& n [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float qs[8][VHD];
  const int h = int(tg.x), r = int(tg.y) * 8 + int(sg);
  const bool live = r < n;
  const int rr = live ? r : n - 1;
  const device bfloat* q = QKV + size_t(rr) * 3 * VD + h * VHD;
  qs[sg][lane] = float(q[lane]);
  qs[sg][lane + 32] = float(q[lane + 32]);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = 0.125f;  // 1 / sqrt(64)
  float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (int k0 = 0; k0 < n; k0 += 32) {
    const int key = k0 + int(lane);
    float s = -INFINITY;
    if (key < n) {
      const device bfloat* kr = QKV + size_t(key) * 3 * VD + VD + h * VHD;
      float d = 0.0f;
      for (int i = 0; i < VHD; i++) d += qs[sg][i] * float(kr[i]);
      s = d * scale;
    }
    const float bm = simd_max(s);
    const float mn = max(m, bm);
    const float corr = precise::exp(m - mn);
    const float p = key < n ? precise::exp(s - mn) : 0.0f;
    l = l * corr + simd_sum(p);
    o0 *= corr;
    o1 *= corr;
    const int kn = min(32, n - k0);
    for (int j = 0; j < kn; j++) {
      const float pj = simd_shuffle(p, ushort(j));
      const device bfloat* v = QKV + size_t(k0 + j) * 3 * VD + 2 * VD + h * VHD;
      o0 += pj * float(v[lane]);
      o1 += pj * float(v[lane + 32]);
    }
    m = mn;
  }
  if (live) {
    O[size_t(r) * VD + h * VHD + lane] = bfloat(o0 / l);
    O[size_t(r) * VD + h * VHD + lane + 32] = bfloat(o1 / l);
  }
}

// The tower's MLP gate: [rows, 2 x 2816] -> bf16(bf16(silu(gate)) * up) [rows, 2816].
[[kernel]] void ds_vit_glu(const device bfloat* H [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                           constant int& inter [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const device bfloat* h = H + size_t(g.y) * 2 * inter;
  const float gt = float(h[g.x]);
  const float sl = float(bfloat(gt / (1.0f + precise::exp(-gt))));
  Y[size_t(g.y) * inter + g.x] = bfloat(sl * float(h[inter + g.x]));
}

struct UnfoldArgs {
  int n_h;
  int n_w;
  int out_w;  // ceil(n_w / 3)
};

// The aligner's 3x3 unfold of the patch grid (zero-padded right and bottom): token (oy, ox)'s input row is
// feature c x 9 + ky x 3 + kx = tower[(3 oy + ky) * n_w + 3 ox + kx][c].
[[kernel]] void ds_unfold(const device bfloat* T [[buffer(0)]], device bfloat* Y [[buffer(1)]],
                          constant UnfoldArgs& a [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  const int f = int(g.x), t = int(g.y);
  const int c = f / 9, ky = (f % 9) / 3, kx = f % 3;
  const int y = 3 * (t / a.out_w) + ky, x = 3 * (t % a.out_w) + kx;
  Y[size_t(t) * 9 * VD + f] = (y < a.n_h && x < a.n_w) ? T[size_t(y * a.n_w + x) * VD + c] : bfloat(0);
}

// An image span's rows into the streams: CODE[r] >= 0 the aligner row, -2 / -3 / -4 the span's start, end and
// line-break embeddings, -1 a text row (left as embedded). Every copy of the stream takes the row.
[[kernel]] void ds_put_image(device bfloat* X [[buffer(0)]], const device int* CODE [[buffer(1)]],
                             const device bfloat* ROWS [[buffer(2)]], const device bfloat* START [[buffer(3)]],
                             const device bfloat* END [[buffer(4)]], const device bfloat* NL [[buffer(5)]],
                             uint2 g [[thread_position_in_grid]]) {
  const int code = CODE[g.y];
  if (code == -1) return;
  const bfloat v = code >= 0 ? ROWS[size_t(code) * D + g.x] : (code == -2 ? START[g.x] : (code == -3 ? END[g.x] : NL[g.x]));
  for (int s = 0; s < S; s++) X[(size_t(g.y) * S + s) * D + g.x] = v;
}
