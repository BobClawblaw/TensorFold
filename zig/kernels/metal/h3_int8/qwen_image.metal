// Qwen-Image-2.1 transformer kernels for the M5 tensor units: bf16 rows, fp32 arithmetic inside.
// The fragment layout and the 16x32x16 multiply-accumulate are ../nax.h's (its M5 branch), inlined so this file
// compiles alone with newLibraryWithSource; qi_gemm is ops/nax_gemm.metal's kernel under another name.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define QI_UNROLL _Pragma("clang loop unroll(full)")

template <typename T>
using qi_frag = vec<T, 8>;

inline short2 qi_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

template <typename T, typename S>
inline void qi_get(thread qi_frag<T>& f, const device S* p, long ld, int r, int c, short2 home, int nr, int nc) {
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    f[e] = (rr < nr && cc < nc) ? T(p[long(rr) * ld + cc]) : T(0);
  }
}

template <typename O>
inline void qi_put(thread const qi_frag<float>& f, device O* p, long ld, int r, int c, short2 home, int nr, int nc) {
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    const int rr = r + home.y + (e >> 2) * 8, cc = c + home.x + (e & 3);
    if (rr < nr && cc < nc) p[long(rr) * ld + cc] = O(f[e]);
  }
}

template <typename C, typename A, typename B>
inline void qi_mma(thread qi_frag<C>& lo, thread qi_frag<C>& hi, thread const qi_frag<A>& a,
                   thread const qi_frag<B>& b0, thread const qi_frag<B>& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<A, B, C>();
  auto right = op.template get_right_input_cooperative_tensor<A, B, C>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
                                                            metal::remove_addrspace_t<decltype(right)>, C>();
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  QI_UNROLL
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// D[z] = A[z] B[z]: A (M, K), B (K, N), D (M, N), row-major with leading dimensions lda, ldb, ldd.
// P: M, N, K. LD: lda, ldb, ldd, then the element strides of A, B and D from one z to the next.
// Threadgroups [ceil(N / 128), ceil(M / 64), batch] of [32, 4, 1].
[[kernel]] void qi_gemm(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* LD [[buffer(3)]],
  device bfloat* D [[buffer(4)]],
  uint sg [[simdgroup_index_in_threadgroup]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], N = P[1], K = P[2];
  const long lda = LD[0], ldb = LD[1], ldd = LD[2];
  const device bfloat* a = A + long(tg.z) * LD[3];
  const device bfloat* b = B + long(tg.z) * LD[4];
  device bfloat* d = D + long(tg.z) * LD[5];
  const int col0 = int(tg.x) * 128 + 32 * int(sg);
  if (col0 >= N) return;
  const short2 o = qi_home(ushort(lane));
  const int chunks = (K + 15) / 16;
  for (int row0 = int(tg.y) * 64; row0 < min(M, int(tg.y) * 64 + 64); row0 += 16) {
    qi_frag<float> c0 = 0.0f, c1 = 0.0f;
    for (int ch = 0; ch < chunks; ++ch) {
      const int k0 = 16 * ch;
      qi_frag<bfloat> fa, f0, f1;
      qi_get(fa, a, lda, row0, k0, o, M, K);
      qi_get(f0, b, ldb, k0, col0, o, K, N);
      qi_get(f1, b, ldb, k0, col0 + 16, o, K, N);
      qi_mma(c0, c1, fa, f0, f1);
    }
    qi_put(c0, d, ldd, row0, col0, o, M, N);
    qi_put(c1, d, ldd, row0, col0 + 16, o, M, N);
  }
}

// qi_gemm for whole 128 x 128 tiles: N and K multiples of 128, a's rows readable up to the next multiple of 128
// (what lies there is multiplied and dropped). One threadgroup of 256 threads owns one output tile and lets the
// tensor operation read its operands in place. Threadgroups [N / 128, ceil(M / 128), batch] of [256, 1, 1].
[[kernel]] void qi_tile_gemm(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant int64_t* LD [[buffer(3)]],
  device bfloat* D [[buffer(4)]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int T = 128;
  constexpr int CAP = T * T / 256;
  const int M = P[0], N = P[1], K = P[2];
  const int MP = (M + T - 1) / T * T;
  device bfloat* a = (device bfloat*)A + long(tg.z) * LD[3];
  device bfloat* b = (device bfloat*)B + long(tg.z) * LD[4];
  device bfloat* d = D + long(tg.z) * LD[5];
  const int n0 = int(tg.x) * T, r0 = int(tg.y) * T;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> x(a, dextents<int32_t, 2>(int(LD[0]), MP));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> w(b, dextents<int32_t, 2>(int(LD[1]), K));
  constexpr auto desc = matmul2d_descriptor(T, T, T, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<8>> op;
  auto a0 = x.slice<T, T>(0, r0);
  auto b0 = w.slice<T, T>(n0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) acc[i] = 0.0f;
  for (int k0 = 0; k0 < K; k0 += T) {
    auto at = x.slice<T, T>(k0, r0);
    auto bt = w.slice<T, T>(n0, k0);
    op.run(at, bt, acc);
  }
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    auto ids = acc.get_multidimensional_index(i);
    const int row = r0 + ids[1];
    if (row < M) d[long(row) * LD[2] + n0 + ids[0]] = bfloat(acc[i]);
  }
}

// Y = layer_norm(X) * S, one simdgroup a row. X, Y: (R, C). S: (C), one plus the modulation's scale. P: C. E: eps.
// Threadgroups [R, 1, 1] of [32, 1, 1].
[[kernel]] void qi_norm_scale(
  const device bfloat* X [[buffer(0)]],
  const device bfloat* S [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant float* E [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int C = P[0];
  const device bfloat* x = X + long(tg.x) * C;
  device bfloat* y = Y + long(tg.x) * C;
  float sum = 0.0f;
  for (int c = int(lane); c < C; c += 32) sum += float(x[c]);
  const float mean = simd_sum(sum) / float(C);
  float sq = 0.0f;
  for (int c = int(lane); c < C; c += 32) { const float v = float(x[c]) - mean; sq += v * v; }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  for (int c = int(lane); c < C; c += 32) y[c] = bfloat(float(bfloat((float(x[c]) - mean) * inv)) * float(S[c]));
}

// The projections' rows laid out for attention, one thread per (row, head): q and k take their head's RMSNorm
// and the rotary rotation of channel pairs (2c, 2c + 1); the row's keys and values land after the T text rows.
// QR, KR, VR: (R, H * 128). NQ, NK: (128). COS, SIN: (R, 64) float. P: R, H, T, N. E: eps.
// QH: (H, R, 128). KT: (H, 128, N). VH: (H, N, 128), N the keys' padded count. Threads [R, H, 1].
[[kernel]] void qi_heads(
  const device bfloat* QR [[buffer(0)]],
  const device bfloat* KR [[buffer(1)]],
  const device bfloat* VR [[buffer(2)]],
  const device bfloat* NQ [[buffer(3)]],
  const device bfloat* NK [[buffer(4)]],
  const device float* COS [[buffer(5)]],
  const device float* SIN [[buffer(6)]],
  const constant int32_t* P [[buffer(7)]],
  const constant float* E [[buffer(8)]],
  device bfloat* QH [[buffer(9)]],
  device bfloat* KT [[buffer(10)]],
  device bfloat* VH [[buffer(11)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1], T = P[2], N = P[3];
  const int row = int(at.x), head = int(at.y);
  if (row >= R || head >= H) return;
  const long from = (long(row) * H + head) * 128;
  const device float* cs = COS + long(row) * 64;
  const device float* sn = SIN + long(row) * 64;
  float q[128], k[128];
  float qq = 0.0f, kk = 0.0f;
  for (int c = 0; c < 128; c++) {
    q[c] = float(QR[from + c]);
    k[c] = float(KR[from + c]);
    qq += q[c] * q[c];
    kk += k[c] * k[c];
  }
  const float qi = rsqrt(qq / 128.0f + E[0]), ki = rsqrt(kk / 128.0f + E[0]);
  for (int c = 0; c < 128; c++) {
    q[c] = float(bfloat(q[c] * qi * float(NQ[c])));
    k[c] = float(bfloat(k[c] * ki * float(NK[c])));
  }
  device bfloat* qo = QH + (long(head) * R + row) * 128;
  device bfloat* ko = KT + long(head) * 128 * N + T + row;
  device bfloat* vo = VH + (long(head) * N + T + row) * 128;
  for (int c = 0; c < 64; c++) {
    const float co = cs[c], si = sn[c];
    qo[2 * c] = bfloat(q[2 * c] * co - q[2 * c + 1] * si);
    qo[2 * c + 1] = bfloat(q[2 * c] * si + q[2 * c + 1] * co);
    ko[long(2 * c) * N] = bfloat(k[2 * c] * co - k[2 * c + 1] * si);
    ko[long(2 * c + 1) * N] = bfloat(k[2 * c] * si + k[2 * c + 1] * co);
  }
  for (int c = 0; c < 128; c++) vo[c] = VR[from + c];
}

// softmax(S * scale) along each row in place, one simdgroup a row; columns from N to the leading dimension become 0.
// S: (rows, ld). P: N, ld. SC: scale. Threadgroups [rows, 1, 1] of [32, 1, 1].
[[kernel]] void qi_softmax(
  device bfloat* S [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  const constant float* SC [[buffer(2)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int N = P[0], ld = P[1];
  device bfloat* s = S + long(tg.x) * ld;
  float top = -3.0e38f;
  for (int c = int(lane); c < N; c += 32) top = max(top, float(s[c]));
  top = simd_max(top);
  float mass = 0.0f;
  for (int c = int(lane); c < N; c += 32) mass += exp((float(s[c]) - top) * SC[0]);
  const float inv = 1.0f / simd_sum(mass);
  for (int c = int(lane); c < ld; c += 32) s[c] = c < N ? bfloat(exp((float(s[c]) - top) * SC[0]) * inv) : bfloat(0.0f);
}

// (H, R, 128) to (R, H * 128), one thread per (row, head). P: R, H. Threads [R, H, 1].
[[kernel]] void qi_rows(
  const device bfloat* OH [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device bfloat* OR [[buffer(2)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], H = P[1];
  if (int(at.x) >= R || int(at.y) >= H) return;
  const device bfloat* from = OH + (long(at.y) * R + at.x) * 128;
  device bfloat* to = OR + (long(at.x) * H + at.y) * 128;
  for (int c = 0; c < 128; c++) to[c] = from[c];
}

// X += G * Y by column, the product rounded to bf16 first as the reference does. X, Y: (R, C). G: (C). P: R, C.
// Threads [C, R, 1].
[[kernel]] void qi_gate_add(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* G [[buffer(2)]],
  const constant int32_t* P [[buffer(3)]],
  uint3 at [[thread_position_in_grid]]) {
  const int R = P[0], C = P[1];
  if (int(at.x) >= C || int(at.y) >= R) return;
  const long i = long(at.y) * C + at.x;
  X[i] = bfloat(float(X[i]) + float(bfloat(float(G[at.x]) * float(Y[i]))));
}

// H = silu(A) * B, each stage rounded to bf16. P: count. Threads [1024, ceil(count / 1024), 1].
[[kernel]] void qi_swiglu(
  const device bfloat* A [[buffer(0)]],
  const device bfloat* B [[buffer(1)]],
  const constant int64_t* P [[buffer(2)]],
  device bfloat* H [[buffer(3)]],
  uint3 at [[thread_position_in_grid]]) {
  const long i = long(at.y) * 1024 + at.x;
  if (i >= P[0]) return;
  const float a = float(A[i]);
  const float gate = float(bfloat(a * float(bfloat(1.0f / (1.0f + exp(-a))))));
  H[i] = bfloat(gate * float(B[i]));
}

// ---- int8 path: activations take one scale per row (or per row and group of channels), weights one per output
// channel, and whole 128 x 128 x 128 int8 tiles accumulate in int32 on the tensor units. The scheme is antirez's
// h3.c (MIT), as in our MLX kernels; sizes are run-time values so any hidden size, head count or MLP width fits.

// The model's sizes are compile-time values: the loader writes these defines ahead of this source, which is worth
// a few percent on the products. The defaults are Qwen-Image-2.1 at 1344 x 768 with a 94-token prompt.
#ifndef QI_HIDDEN
#define QI_HIDDEN 4096
#define QI_MLP 12288
#define QI_WIDE_GROUP 1024
#define QI_HEADS 32
#define QI_ROWS 4032
#define QI_KEYS 4126
#endif

constant constexpr int QI_T = 128;
constant constexpr int QI_TM = 128;                    // rows a threadgroup of the int8 products owns
constant constexpr int QI_TN = 128;                    // output columns a threadgroup owns
constant constexpr int QI_TK = 128;                    // channels one tensor operation consumes
constant constexpr int QI_TQ = 64;                     // query rows one attention threadgroup owns
constant constexpr int QI_TKEYS = 64;                  // keys in one tile of scores
constant constexpr int QI_SG = 8;                      // simdgroups sharing one int8 product
constant constexpr int QI_THREADS = 32 * QI_SG;
constant constexpr int QI_GROUPS = 32;                 // activation scale groups a row may have

// A projection stored (K, N) to int8 with one scale per output channel n: the column's largest magnitude maps to 127.
// W: (K, N) bf16. P: K, N. W8: (K, N). WS: (N). Threadgroups [N, 1, 1] of [32, 1, 1].
[[kernel]] void qi_quant_weight(
  const device bfloat* W [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* W8 [[buffer(2)]],
  device float* WS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int K = P[0], N = P[1], n = int(tg.x);
  float top = 0.0f;
  for (int k = int(lane); k < K; k += 32) top = max(top, abs(float(W[long(k) * N + n])));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) WS[n] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int k = int(lane); k < K; k += 32)
    W8[long(k) * N + n] = int8_t(clamp(int(rint(float(W[long(k) * N + n]) * inverse)), -127, 127));
}

// Rows to int8, one scale per (row, group of G channels), one simdgroup each. X: (M, C) bf16. P: M, C, G.
// Q: (MP, C), rows from M zero. XS: (MP, C / G). Threadgroups [C / G, MP, 1] of [32, 1, 1].
[[kernel]] void qi_quant_rows(
  const device bfloat* X [[buffer(0)]],
  const constant int32_t* P [[buffer(1)]],
  device int8_t* Q [[buffer(2)]],
  device float* XS [[buffer(3)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], C = P[1], G = P[2];
  const int row = int(tg.y), g = int(tg.x);
  const long base = long(row) * C + g * G;
  if (row >= M) {
    for (int j = int(lane); j < G; j += 32) Q[base + j] = 0;
    if (lane == 0) XS[row * (C / G) + g] = 0.0f;
    return;
  }
  float top = 0.0f;
  for (int j = int(lane); j < G; j += 32) top = max(top, abs(float(X[base + j])));
  top = max(simd_max(top), 1e-12f);
  if (lane == 0) XS[row * (C / G) + g] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int j = int(lane); j < G; j += 32) Q[base + j] = int8_t(clamp(int(rint(float(X[base + j]) * inverse)), -127, 127));
}

// qi_norm_scale straight to int8 with one scale per row. X: (M, C) bf16. S: (C). P: M, C. E: eps.
// Q: (MP, C), rows from M zero. XS: (MP). Threadgroups [MP, 1, 1] of [32, 1, 1].
[[kernel]] void qi_norm_scale_q8(
  const device bfloat* X [[buffer(0)]],
  const device bfloat* S [[buffer(1)]],
  const constant int32_t* P [[buffer(2)]],
  const constant float* E [[buffer(3)]],
  device int8_t* Q [[buffer(4)]],
  device float* XS [[buffer(5)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], C = P[1], row = int(tg.x);
  device int8_t* q = Q + long(row) * C;
  if (row >= M) {
    for (int c = int(lane); c < C; c += 32) q[c] = 0;
    if (lane == 0) XS[row] = 0.0f;
    return;
  }
  const device bfloat* x = X + long(row) * C;
  float sum = 0.0f;
  for (int c = int(lane); c < C; c += 32) sum += float(x[c]);
  const float mean = simd_sum(sum) / float(C);
  // the variance and the largest scaled magnitude in one pass; the bound is a hair wide, for the bf16 rounding
  float sq = 0.0f, top = 0.0f;
  for (int c = int(lane); c < C; c += 32) {
    const float v = float(x[c]) - mean;
    sq += v * v;
    top = max(top, abs(v * float(S[c])));
  }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  top = max(simd_max(top) * inv * 1.004f, 1e-12f);
  if (lane == 0) XS[row] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int c = int(lane); c < C; c += 32)
    q[c] = int8_t(clamp(int(rint(float(bfloat((float(x[c]) - mean) * inv)) * float(S[c]) * inverse)), -127, 127));
}

// qi_gate_add then qi_norm_scale_q8 in one pass over each row: X += G * Y, and the new row normed, scaled by S
// and quantized. X, Y: (M, C) bf16. G, S: (C). P: M, C. E: eps. Q: (MP, C). XS: (MP).
// Threadgroups [MP, 1, 1] of [32, 1, 1].
[[kernel]] void qi_gate_norm_q8(
  device bfloat* X [[buffer(0)]],
  const device bfloat* Y [[buffer(1)]],
  const device bfloat* G [[buffer(2)]],
  const device bfloat* S [[buffer(3)]],
  const constant int32_t* P [[buffer(4)]],
  const constant float* E [[buffer(5)]],
  device int8_t* Q [[buffer(6)]],
  device float* XS [[buffer(7)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int M = P[0], C = P[1], row = int(tg.x);
  device int8_t* q = Q + long(row) * C;
  if (row >= M) {
    for (int c = int(lane); c < C; c += 32) q[c] = 0;
    if (lane == 0) XS[row] = 0.0f;
    return;
  }
  device bfloat* x = X + long(row) * C;
  const device bfloat* y = Y + long(row) * C;
  float sum = 0.0f;
  for (int c = int(lane); c < C; c += 32) {
    const bfloat v = bfloat(float(x[c]) + float(bfloat(float(G[c]) * float(y[c]))));
    x[c] = v;
    sum += float(v);
  }
  const float mean = simd_sum(sum) / float(C);
  // the variance and the largest scaled magnitude in one pass; the bound is a hair wide, for the bf16 rounding
  float sq = 0.0f, top = 0.0f;
  for (int c = int(lane); c < C; c += 32) {
    const float v = float(x[c]) - mean;
    sq += v * v;
    top = max(top, abs(v * float(S[c])));
  }
  const float inv = rsqrt(simd_sum(sq) / float(C) + E[0]);
  top = max(simd_max(top) * inv * 1.004f, 1e-12f);
  if (lane == 0) XS[row] = top / 127.0f;
  const float inverse = 127.0f / top;
  for (int c = int(lane); c < C; c += 32)
    q[c] = int8_t(clamp(int(rint(float(bfloat((float(x[c]) - mean) * inv)) * float(S[c]) * inverse)), -127, 127));
}

// Y[row, n] = (sum over groups of (int8 X . int8 W) * xscale[row, group]) * wscale[n].
// X: (MP, K) int8. XS: (MP, K / G). W: (K, N) int8. WS: (N). N, G and K / G are whole tiles' worth; M is QI_ROWS.
// Y: (M, N) bf16. Threadgroups [N / QI_TN, ceil(M / QI_TM), 1] of [32 * QI_SG, 1, 1].
template <int N, int K, int G>
inline void qi_i8_linear_body(const device int8_t* X, const device float* XS, const device int8_t* W,
                              const device float* WS, device bfloat* Y, uint tid, uint3 tg, threadgroup float* sc) {
  using namespace mpp::tensor_ops;
  constexpr int T = QI_TM, TN = QI_TN, TK = QI_TK;
  constexpr int CAP = T * TN / QI_THREADS;
  constexpr int M = QI_ROWS;
  constexpr int KG = K / G, KT = G / TK, MP = (M + T - 1) / T * T;
  const int n0 = int(tg.x) * TN, r0 = int(tg.y) * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> w((device int8_t*)W, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(T, TN, TK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<QI_SG>> op;
  auto a0 = x.slice<TK, T>(0, r0);
  auto b0 = w.slice<TN, TK>(n0, 0);
  for (int j = int(tid); j < T * KG; j += QI_THREADS) sc[j] = XS[r0 * KG + j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  float total[CAP];
  short erow[CAP], ecol[CAP];
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    total[i] = 0.0f;
    acc[i] = 0;
    auto ids = acc.get_multidimensional_index(i);
    ecol[i] = ids[0];
    erow[i] = ids[1];
  }
  for (int g = 0; g < KG; g++) {
    for (int t = 0; t < KT; t++) {
      auto a = x.slice<TK, T>((g * KT + t) * TK, r0);
      auto b = w.slice<TN, TK>(n0, (g * KT + t) * TK);
      op.run(a, b, acc);
    }
    QI_UNROLL
    for (ushort i = 0; i < CAP; i++) {
      total[i] = fma(float(acc[i]), sc[erow[i] * KG + g], total[i]);
      acc[i] = 0;
    }
  }
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    const int row = r0 + erow[i];
    if (row < M) Y[long(row) * N + n0 + ecol[i]] = bfloat(total[i] * WS[n0 + ecol[i]]);
  }
}

// The stream's width to itself (q, k, v and the attention output), one scale a row.
[[kernel]] void qi_i8_linear(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* W [[buffer(2)]],
  const device float* WS [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sc[QI_TM * QI_GROUPS];
  qi_i8_linear_body<QI_HIDDEN, QI_HIDDEN, QI_HIDDEN>(X, XS, W, WS, Y, tid, tg, sc);
}

// The MLP's wide rows back to the stream's width, a scale per QI_WIDE_GROUP channels.
[[kernel]] void qi_i8_linear_wide(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* W [[buffer(2)]],
  const device float* WS [[buffer(3)]],
  device bfloat* Y [[buffer(4)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float sc[QI_TM * QI_GROUPS];
  qi_i8_linear_body<QI_HIDDEN, QI_MLP, QI_WIDE_GROUP>(X, XS, W, WS, Y, tid, tg, sc);
}

// H[row, n] = silu(X . WG) * (X . WV): the SwiGLU's two projections in one pass, one activation scale per row.
// X: (MP, K) int8. XS: (MP). WG, WV: (K, N) int8 with scales SG, SV (N). H: (M, N) bf16; K is QI_HIDDEN, N QI_MLP.
// Threadgroups [N / QI_TN, ceil(M / QI_TM), 1] of [32 * QI_SG, 1, 1].
[[kernel]] void qi_i8_swiglu(
  const device int8_t* X [[buffer(0)]],
  const device float* XS [[buffer(1)]],
  const device int8_t* WG [[buffer(2)]],
  const device float* SG [[buffer(3)]],
  const device int8_t* WV [[buffer(4)]],
  const device float* SV [[buffer(5)]],
  device bfloat* H [[buffer(6)]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int T = QI_TM, TN = QI_TN, TK = QI_TK;
  constexpr int CAP = T * TN / QI_THREADS;
  constexpr int M = QI_ROWS, N = QI_MLP, K = QI_HIDDEN;
  constexpr int MP = (M + T - 1) / T * T;
  const int n0 = int(tg.x) * TN, r0 = int(tg.y) * T;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> x((device int8_t*)X, dextents<int32_t, 2>(K, MP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wg((device int8_t*)WG, dextents<int32_t, 2>(N, K));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> wv((device int8_t*)WV, dextents<int32_t, 2>(N, K));
  constexpr auto desc = matmul2d_descriptor(T, TN, TK, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<QI_SG>> op;
  auto a0 = x.slice<TK, T>(0, r0);
  auto b0 = wg.slice<TN, TK>(n0, 0);
  auto gate = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  auto value = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) { gate[i] = 0; value[i] = 0; }
  for (int k0 = 0; k0 < K; k0 += TK) {
    auto a = x.slice<TK, T>(k0, r0);
    auto bg = wg.slice<TN, TK>(n0, k0);
    auto bv = wv.slice<TN, TK>(n0, k0);
    op.run(a, bg, gate);
    op.run(a, bv, value);
  }
  QI_UNROLL
  for (ushort i = 0; i < CAP; i++) {
    auto ids = gate.get_multidimensional_index(i);
    const int row = r0 + ids[1], n = n0 + ids[0];
    if (row >= M) continue;
    const float xs = XS[row];
    const float g = float(gate[i]) * xs * SG[n];
    const float v = float(value[i]) * xs * SV[n];
    H[long(row) * N + n] = bfloat(g / (1.0f + exp(-g)) * v);
  }
}

// The projections' rows laid out for int8 attention, one simdgroup per (row, head), four channels a lane: q and k
// take their head's RMSNorm and the rotary rotation of channel pairs, then q, k and v each round to int8 with a
// scale of their own; the row's key and value land after the T prompt rows.
// QR, KR, VR: (R, H * 128) bf16. NQ, NK: (128). COS, SIN: (R, 64). P: R, H, T, RP, NP. E: eps.
// Q8: (H, RP, 128), QS: (H, RP). K8, V8: (H, NP, 128), KS, VS: (H, NP).
// Threadgroups [RP, H, 1] of [32, 1, 1]; query rows from R to RP become zero.
[[kernel]] void qi_heads_q8(
  const device bfloat* QR [[buffer(0)]],
  const device bfloat* KR [[buffer(1)]],
  const device bfloat* VR [[buffer(2)]],
  const device bfloat* NQ [[buffer(3)]],
  const device bfloat* NK [[buffer(4)]],
  const device float* COS [[buffer(5)]],
  const device float* SIN [[buffer(6)]],
  const constant int32_t* P [[buffer(7)]],
  const constant float* E [[buffer(8)]],
  device int8_t* Q8 [[buffer(9)]],
  device float* QS [[buffer(10)]],
  device int8_t* K8 [[buffer(11)]],
  device float* KS [[buffer(12)]],
  device int8_t* V8 [[buffer(13)]],
  device float* VS [[buffer(14)]],
  uint lane [[thread_index_in_simdgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  const int R = P[0], H = P[1], T = P[2], RP = P[3], NP = P[4];
  const int row = int(tg.x), head = int(tg.y);
  const int c0 = 4 * int(lane);
  device int8_t* qo = Q8 + (long(head) * RP + row) * 128 + c0;
  if (row >= R) {
    for (int j = 0; j < 4; j++) qo[j] = 0;
    if (lane == 0) QS[head * RP + row] = 0.0f;
    return;
  }
  const long from = (long(row) * H + head) * 128 + c0;
  float4 q, k, v;
  for (int j = 0; j < 4; j++) {
    q[j] = float(QR[from + j]);
    k[j] = float(KR[from + j]);
    v[j] = float(VR[from + j]);
  }
  const float qi = rsqrt(simd_sum(dot(q, q)) / 128.0f + E[0]), ki = rsqrt(simd_sum(dot(k, k)) / 128.0f + E[0]);
  for (int j = 0; j < 4; j++) {
    q[j] *= qi * float(NQ[c0 + j]);
    k[j] *= ki * float(NK[c0 + j]);
  }
  const device float* cs = COS + long(row) * 64 + 2 * lane;
  const device float* sn = SIN + long(row) * 64 + 2 * lane;
  const float4 co = float4(cs[0], cs[0], cs[1], cs[1]), si = float4(-sn[0], sn[0], -sn[1], sn[1]);
  q = q * co + q.yxwz * si;
  k = k * co + k.yxwz * si;
  const float4 aq = abs(q), ak = abs(k), av = abs(v);
  const float qtop = max(simd_max(max(max(aq.x, aq.y), max(aq.z, aq.w))), 1e-12f);
  const float ktop = max(simd_max(max(max(ak.x, ak.y), max(ak.z, ak.w))), 1e-12f);
  const float vtop = max(simd_max(max(max(av.x, av.y), max(av.z, av.w))), 1e-12f);
  const long key = long(head) * NP + T + row;
  if (lane == 0) {
    QS[head * RP + row] = qtop / 127.0f;
    KS[key] = ktop / 127.0f;
    VS[key] = vtop / 127.0f;
  }
  device int8_t* ko = K8 + key * 128 + c0;
  device int8_t* vo = V8 + key * 128 + c0;
  const float qv = 127.0f / qtop, kv = 127.0f / ktop, vv = 127.0f / vtop;
  for (int j = 0; j < 4; j++) {
    qo[j] = int8_t(clamp(int(rint(q[j] * qv)), -127, 127));
    ko[j] = int8_t(clamp(int(rint(k[j] * kv)), -127, 127));
    vo[j] = int8_t(clamp(int(rint(v[j] * vv)), -127, 127));
  }
}

// softmax(q k^T scale) v in one pass over key tiles, int8 scores and int8 values, as FlashAttention lays it out:
// a threadgroup owns QI_TQ query rows of one head, takes a QI_TKEYS-key tile of scores, updates each row's running maximum
// and sum, and adds the tile's probabilities times values, rescaling what it holds when the maximum moves.
// Q8: (H, RP, 128), QS: (H, RP). K8, V8: (H, NP, 128), KS, VS: (H, NP); R is QI_ROWS, N QI_KEYS, H QI_HEADS. SC: scale.
// Y: (R, H * 128) bf16. Threadgroups [ceil(R / QI_TQ), H, 1] of [256, 1, 1].
[[kernel]] void qi_attention_i8(
  const device int8_t* Q [[buffer(0)]],
  const device float* QS [[buffer(1)]],
  const device int8_t* K [[buffer(2)]],
  const device float* KS [[buffer(3)]],
  const device int8_t* V [[buffer(4)]],
  const device float* VS [[buffer(5)]],
  const constant float* SC [[buffer(6)]],
  device bfloat* Y [[buffer(7)]],
  uint tid [[thread_index_in_threadgroup]],
  uint3 tg [[threadgroup_position_in_grid]]) {
  using namespace mpp::tensor_ops;
  constexpr int D = 128, TQ = QI_TQ, KEYS = QI_TKEYS;
  constexpr int TPR = 256 / TQ;
  constexpr int C1 = TQ * KEYS / 256, C2 = TQ * D / 256;
  constexpr float LIFT = 1024.0f;
  constexpr int R = QI_ROWS, RP = (R + QI_T - 1) / QI_T * QI_T, N = QI_KEYS, NP = (N + QI_T - 1) / QI_T * QI_T, H = QI_HEADS;
  constexpr int LAST = (N + KEYS - 1) / KEYS * KEYS;      // key tiles past this hold padding only
  const int head = int(tg.y), r0 = int(tg.x) * TQ;
  const long qb = long(head) * RP, kb = long(head) * NP;
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> q((device int8_t*)Q + qb * D, dextents<int32_t, 2>(D, RP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> k((device int8_t*)K + kb * D, dextents<int32_t, 2>(D, NP));
  tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> v((device int8_t*)V + kb * D, dextents<int32_t, 2>(D, NP));
  threadgroup half pt[TQ * KEYS];
  threadgroup float tops[256];
  threadgroup float fac[TQ];
  threadgroup float qsc[TQ];
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> p((threadgroup half*)pt, dextents<int32_t, 2>(KEYS, TQ));
  constexpr auto d1 = matmul2d_descriptor(TQ, KEYS, D, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto d2 = matmul2d_descriptor(TQ, D, KEYS, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<d1, execution_simdgroups<8>> op1;
  matmul2d<d2, execution_simdgroups<8>> op2;
  auto a = q.slice<D, TQ>(0, r0);
  auto b0 = k.slice<D, KEYS>(0, 0);
  auto v0 = v.slice<D, KEYS>(0, 0);
  auto acc = op1.template get_destination_cooperative_tensor<decltype(a), decltype(b0), int32_t>();
  auto out = op2.template get_destination_cooperative_tensor<decltype(p), decltype(v0), float>();
  short srow[C1], scol[C1], orow[C2], ocol[C2];
  QI_UNROLL
  for (ushort i = 0; i < C1; i++) {
    auto ids = acc.get_multidimensional_index(i);
    scol[i] = ids[0];
    srow[i] = ids[1];
  }
  QI_UNROLL
  for (ushort i = 0; i < C2; i++) {
    auto ids = out.get_multidimensional_index(i);
    ocol[i] = ids[0];
    orow[i] = ids[1];
    out[i] = 0.0f;
  }
  for (int j = int(tid); j < TQ; j += 256) qsc[j] = QS[qb + r0 + j] * SC[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int row = int(tid) / TPR, first = (int(tid) % TPR) * C1;
  const int mine = row * KEYS + first;
  float top = -1e30f, total = 0.0f;
  float s[C1];
  for (int k0 = 0; k0 < LAST; k0 += KEYS) {
    QI_UNROLL
    for (ushort i = 0; i < C1; i++) acc[i] = 0;
    auto b = k.slice<D, KEYS>(0, k0);
    op1.run(a, b, acc);
    QI_UNROLL
    for (ushort i = 0; i < C1; i++) {
      const int key = k0 + scol[i];
      const float score = key < N ? float(acc[i]) * qsc[srow[i]] * KS[kb + key] : -60000.0f;
      pt[srow[i] * KEYS + scol[i]] = half(score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local = -1e30f;
    QI_UNROLL
    for (ushort j = 0; j < C1; j++) {
      s[j] = float(pt[mine + j]);
      local = max(local, s[j]);
    }
    tops[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float next = top;
    QI_UNROLL
    for (ushort j = 0; j < TPR; j++) next = max(next, tops[row * TPR + j]);
    const float shrink = exp(top - next);
    if (tid % TPR == 0) fac[row] = shrink;
    total *= shrink;
    top = next;
    QI_UNROLL
    for (ushort j = 0; j < C1; j++) {
      const float weight = exp(s[j] - next);
      total += weight;
      pt[mine + j] = half(weight * VS[kb + k0 + first + j] * LIFT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    QI_UNROLL
    for (ushort i = 0; i < C2; i++) out[i] *= fac[orow[i]];
    auto vb = v.slice<D, KEYS>(0, k0);
    op2.run(p, vb, out);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  tops[tid] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) {
    float sum = 0.0f;
    QI_UNROLL
    for (ushort j = 0; j < TPR; j++) sum += tops[tid + j];
    fac[row] = 1.0f / (sum * LIFT);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  QI_UNROLL
  for (ushort i = 0; i < C2; i++) {
    const int to = r0 + orow[i];
    if (to < R) Y[(long(to) * H + head) * D + ocol[i]] = bfloat(out[i] * fac[orow[i]]);
  }
}
