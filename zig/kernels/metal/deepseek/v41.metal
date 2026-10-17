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

// The pending branch into the streams in place: x'[k] = post[k] * branch + sum_j comb[j][k] x[j].
[[kernel]] void ds_hc_expand(device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                             const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  const uint d = g.x, r = g.y;
  device bfloat* x = X + size_t(r) * F;
  const float b = float(BR[size_t(r) * D + d]);
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
  if (MIXES) {
    if (int(t) <= MIX) {
      float s = 0.0f;
      for (int j = 0; j < SLICES; j++) s += PART[(size_t(r) * SLICES + j) * (MIX + 1) + t];
      tot[t] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      const float rs = precise::rsqrt(tot[MIX] / float(F) + a.eps);
      float mixes[MIX];
      for (int m = 0; m < MIX; m++) mixes[m] = tot[m] * rs;
      for (int j = 0; j < S; j++) {
        PRE_OUT[r * S + j] = 1.0f / (1.0f + precise::exp(-(mixes[j] * SCALE[0] + BASE[j]))) + a.hc_eps;
        POST_OUT[r * S + j] = 2.0f / (1.0f + precise::exp(-(mixes[j + S] * SCALE[1] + BASE[j + S])));
      }
      float cm[S][S];
      for (int j = 0; j < S; j++) {
        float mx = -INFINITY;
        for (int k = 0; k < S; k++) {
          cm[j][k] = mixes[2 * S + j * S + k] * SCALE[2] + BASE[2 * S + j * S + k];
          mx = max(mx, cm[j][k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < S; k++) {
          cm[j][k] = precise::exp(cm[j][k] - mx);
          sum += cm[j][k];
        }
        for (int k = 0; k < S; k++) cm[j][k] = cm[j][k] / sum + a.hc_eps;
      }
      for (int it = 0; it < a.iters; it++) {
        if (it > 0) {
          for (int j = 0; j < S; j++) {
            float rs2 = 0.0f;
            for (int k = 0; k < S; k++) rs2 += cm[j][k];
            for (int k = 0; k < S; k++) cm[j][k] = cm[j][k] / (rs2 + a.hc_eps);
          }
        }
        for (int k = 0; k < S; k++) {
          float cs = 0.0f;
          for (int j = 0; j < S; j++) cs += cm[j][k];
          for (int j = 0; j < S; j++) cm[j][k] = cm[j][k] / (cs + a.hc_eps);
        }
      }
      for (int j = 0; j < S; j++)
        for (int k = 0; k < S; k++) COMB_OUT[r * S * S + j * S + k] = cm[j][k];
    }
  }
  if (t < uint(S)) pre_in[t] = PRE_IN[r * S + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // collapse (bf16, as hc_pre returns) and the rms norm over D
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) red[sg][0] = s1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += red[k][0];
    tot[0] = precise::rsqrt(s / float(D) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * D + d] = bfloat(float(NORMW[d]) * (ys[i] * tot[0]));
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
[[kernel]] void ds_index_scores(const device bfloat* Q [[buffer(0)]], const device bfloat* WT [[buffer(1)]],
                                const device bfloat* K [[buffer(2)]], device float* SCORE [[buffer(3)]],
                                constant ScoreArgs& a [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  const int t = int(g.x), r = int(g.y);
  if (t >= a.groups) return;
  const int visible = (a.first_pos + r + 1) / a.ratio;
  device float* out = SCORE + size_t(r) * a.stride;
  if (t >= visible) {
    out[t] = -INFINITY;
    return;
  }
  const device bfloat* k = K + size_t(t) * ID;
  float total = 0.0f;
  for (int h = 0; h < IH; h++) {
    const device bfloat* q = Q + (size_t(r) * IH + h) * ID;
    float dot = 0.0f;
    for (int i = 0; i < ID; i++) dot += float(q[i]) * float(k[i]);
    total += bf_round(max(bf_round(dot), 0.0f) * float(WT[r * IH + h]));
  }
  out[t] = bf_round(total);
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
[[kernel]] void ds_topk(const device float* SCORE [[buffer(0)]], device int* OUT [[buffer(1)]],
                        constant TopkArgs& a [[buffer(2)]], uint r [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix_bits, want;
  threadgroup uint counts[1024];
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
  counts[t] = finite;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint total = 0;
    for (int i = 0; i < 1024; i++) total += counts[i];
    want = min(want, total);
  }
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
    if (t == 0) {
      uint above = 0;
      for (int b = 255; b >= 0; b--) {
        const uint hb = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (above + hb >= want) {
          prefix_bits |= uint(b) << shift;
          want -= above;
          break;
        }
        above += hb;
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
  counts[t] = (above << 16) | equal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before_above = 0, before_equal = 0;
  for (uint i = 0; i < t; i++) {
    before_above += counts[i] >> 16;
    before_equal += counts[i] & 0xFFFF;
  }
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

// ---------------------------------------------------------------- MoE

struct RouteArgs {
  int experts;
  int topk;
  float scale;
};

// sqrt(softplus(logits)) scores; the top-k of score + bias (ties to the lower id) picked; their unbiased scores
// normalized and scaled. Picks are written in expert order, the order the reference sums them in. One threadgroup
// (a thread an expert, `experts` <= 512) a row: the scores in parallel, then k rounds of a threadgroup argmax.
[[kernel]] void ds_route(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                         device uint* PICK [[buffer(2)]], device float* WEIGHT [[buffer(3)]],
                         constant RouteArgs& a [[buffer(4)]], uint r [[threadgroup_position_in_grid]],
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
    key = score + BIAS[t];
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
  int text_rows;  // rows past this many are not text (no update): unused, all rows text for now
};

// One threadgroup (256) a (row, copy): the gate sigmoid(signed sqrt(normalized <h, w * key>)) of the lookup's key
// against the copy, then h += gate * value.
[[kernel]] void ds_engram(device bfloat* X [[buffer(0)]], const device bfloat* KV [[buffer(1)]],
                          const device bfloat* QW [[buffer(2)]], const device bfloat* KW [[buffer(3)]],
                          constant EngramArgs& a [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  const uint t = tpos.x;
  const int c = int(tg.x), r = int(tg.y);
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
