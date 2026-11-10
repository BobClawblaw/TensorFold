// Llama's prompt projections on the tensor units: bf16 x [M, K] by the release's W [N, K]; after gemm_nax.metal.

#define LL_GEMM(NAME, AM, AN)                                                                                       \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                  \
      const device bfloat16_t* A [[buffer(0)]], const device bfloat16_t* B [[buffer(1)]],                          \
      const device int32_t* P [[buffer(2)]], device bfloat16_t* D [[buffer(3)]],                                   \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    tfp::gemm_nax<bfloat16_t, false, true, AM, AN, false, 64, 128, 256, 2, 4>(A, B, D, P, tg, sg, lane);           \
  }
LL_GEMM(ll_gemm_00, false, false)
LL_GEMM(ll_gemm_01, false, true)
LL_GEMM(ll_gemm_10, true, false)
LL_GEMM(ll_gemm_11, true, true)

// gemm_nax's tiles with the fp32 sums stored unrounded: ll_bias_round adds a biased projection's bias and rounds once.
template <bool AM, bool AN>
inline void ll_gemm_f32_t(const device bfloat16_t* A, const device bfloat16_t* B, device float* D,
                          const device int* P, uint3 tg, uint sg, uint lane) {
  constexpr int BM = 64, BN = 128, BK = 256, WN = 4;
  constexpr bool TB = true;  // W [N, K] read transposed
  const int swz = P[8];
  const int ty = (int(tg.y) << swz) + (int(tg.x) & ((1 << swz) - 1)), tx = int(tg.x) >> swz;
  if (tx >= P[6] || ty >= P[7]) {
    return;
  }
  const int lda = P[3], ldb = P[4], ldd = P[5];
  const int row = ty * BM + 32 * int(sg / WN), col = tx * BN + 32 * int(sg % WN);
  A += long(P[13]) + long(P[10]) * tg.z + long(row) * lda;
  B += long(P[14]) + long(P[11]) * tg.z + long(col) * ldb;
  D += long(P[12]) * tg.z + long(row) * ldd + col;
  const int rows = AM ? 32 : min(32, P[0] - row), cols = AN ? 32 : min(32, P[1] - col);
  const bool full = (AM || rows == 32) && (AN || cols == 32);
  const short2 home = tfp::frag_home(ushort(lane));
  tfp::frag<float> acc[2][2];
  if (full) {
    tfp::gemm_block<bfloat16_t, false, TB, true, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    tfp::gemm_store<float, true>(acc, D, ldd, rows, cols, home);
  } else {
    tfp::gemm_block<bfloat16_t, false, TB, false, BK>(acc, A, B, lda, ldb, P[2], P[9], rows, cols, home);
    if (rows > 0 && cols > 0) {
      tfp::gemm_store<float, false>(acc, D, ldd, rows, cols, home);
    }
  }
}

#define LL_GEMM_F32(NAME, AM, AN)                                                                                   \
  [[max_total_threads_per_threadgroup(256)]] [[kernel]] void NAME(                                                  \
      const device bfloat16_t* A [[buffer(0)]], const device bfloat16_t* B [[buffer(1)]],                          \
      const device int32_t* P [[buffer(2)]], device float* D [[buffer(3)]],                                        \
      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],                        \
      uint3 tg [[threadgroup_position_in_grid]]) {                                                                 \
    ll_gemm_f32_t<AM, AN>(A, B, D, P, tg, sg, lane);                                                               \
  }
LL_GEMM_F32(ll_gemm_f32_00, false, false)
LL_GEMM_F32(ll_gemm_f32_01, false, true)
LL_GEMM_F32(ll_gemm_f32_10, true, false)
LL_GEMM_F32(ll_gemm_f32_11, true, true)
