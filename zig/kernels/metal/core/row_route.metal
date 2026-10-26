// The route folded into the indexed projection (after row_projection.metal's text): a slot picks its expert as it starts.

// The softmax router's shape: PICK, WEIGHT [rows, topk + 1] (the picks, then the shared expert at id `experts`); one launch writes them.
struct TfRpRoute { int experts, topk, ld, renorm, write; };

// Logits a lane holds: routers of up to 256 experts.
constant constexpr int TF_RP_ROUTE_PER = 8;

// A bf16 logit and its id as one key in the route's order (a greater logit, then the lower id); 0 is below every logit, -0 counts as 0.
inline uint tf_rp_route_key(bfloat v, int id) {
  ushort b = as_type<ushort>(v);
  if (b == 0x8000) b = 0;
  b = (b & 0x8000) ? ~b : (b | 0x8000);
  return (uint(b) << 16) | uint(0xffff - id);
}

// The logit a key holds.
inline float tf_rp_route_val(uint key) {
  const ushort b = ushort(key >> 16);
  return float(as_type<bfloat>(ushort((b & 0x8000) ? (b & 0x7fff) : ~b)));
}

// tf_row_projection_indexed_extra with slot y's expert taken from LOGITS [rows, ld] as q27_route takes it: top-k by logit (ties to the
// lower id), fp32 softmax rounded to bf16 for the weights; slot y of row y / repeat is the row's pick y % repeat, the last its shared
// expert. The writing launch's first threadgroup of each row (its first slot) also writes the row's PICK and WEIGHT, the same bits.
template <int RPS, int SG, bool RELU2, int BITS, bool BF16SUM>
[[kernel]] void tf_row_projection_indexed_route(const device bfloat* X [[buffer(0)]],
                                                const device uint32_t* W [[buffer(1)]],
                                                const device bfloat* S [[buffer(2)]],
                                                const device bfloat* B [[buffer(3)]],
                                                constant int* dims [[buffer(4)]],
                                                device bfloat* OUT [[buffer(5)]],
                                                device uint* PICK [[buffer(6)]],
                                                const device uint32_t* W2 [[buffer(7)]],
                                                const device bfloat* S2 [[buffer(8)]],
                                                const device bfloat* B2 [[buffer(9)]],
                                                const device bfloat* LOGITS [[buffer(10)]],
                                                device float* WEIGHT [[buffer(11)]],
                                                constant TfRpRoute& ra [[buffer(12)]],
                                                uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
                                                uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
                                                uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
  constexpr int PER = TF_RP_ROUTE_PER;
  const int K = dims[0], N = dims[1], GS = dims[2], experts = dims[4], repeat = dims[5];
  const uint lane = thread_index_in_simdgroup;
  const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  if (row0 >= N) return;
  const uint slot = threadgroup_position_in_grid.y;
  const int row = int(slot) / repeat, k = int(slot) % repeat, stride = ra.topk + 1;
  uint expert = uint(experts);
  if (k < ra.topk) {
    const bool writer = ra.write != 0 && k == 0 && threadgroup_position_in_grid.x == 0 && simdgroup_index_in_threadgroup == 0;
    const device bfloat* z = LOGITS + size_t(row) * ra.ld;
    uint key[PER];
    for (int j = 0; j < PER; j++) {
      const int id = 32 * j + int(lane);
      key[j] = id < experts ? tf_rp_route_key(z[id], id) : 0u;
    }
    float m = -INFINITY, s = 0.0f;
    if (writer) { // q27_route's softmax: the max, exp, the sum simdgroup by simdgroup in id order (each probability to bf16 at its pick)
      for (int j = 0; j < PER; j++)
        if (32 * j + int(lane) < experts) m = max(m, tf_rp_route_val(key[j]));
      m = simd_max(m);
      for (int j = 0; j < PER; j++) s += simd_sum(32 * j + int(lane) < experts ? precise::exp(tf_rp_route_val(key[j]) - m) : 0.0f);
    }
    const int rounds = writer ? ra.topk : k + 1;
    float sum = 0.0f;
    for (int t = 0; t < rounds; t++) {
      uint best = key[0];
      for (int j = 1; j < PER; j++) best = max(best, key[j]);
      best = simd_max(best);
      const int bid = 0xffff - int(best & 0xffffu);
      if (t == k) expert = uint(bid);
      if (writer) {
        float wv = 0.0f;
        for (int j = 0; j < PER; j++)
          if (32 * j + int(lane) == bid) wv = float(bfloat(precise::exp(tf_rp_route_val(key[j]) - m) / s));
        wv = simd_shuffle(wv, ushort(bid % 32));
        sum += wv;
        if (lane == 0) {
          PICK[size_t(row) * stride + t] = uint(bid);
          WEIGHT[size_t(row) * stride + t] = wv;
        }
      }
      for (int j = 0; j < PER; j++)
        if (32 * j + int(lane) == bid) key[j] = 0u;
    }
    if (writer && lane == 0) {
      const float norm = float(bfloat(sum));
      if (ra.renorm)
        for (int t = 0; t < ra.topk; t++) WEIGHT[size_t(row) * stride + t] = float(bfloat(WEIGHT[size_t(row) * stride + t] / norm));
      const float g = float(z[experts]);
      PICK[size_t(row) * stride + ra.topk] = uint(experts);
      WEIGHT[size_t(row) * stride + ra.topk] = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
    }
  }
  device bfloat* out = OUT + size_t(slot) * N + row0;
  // id `experts` is the extra matrix (W2, S2, B2: one more [N, K] of the same width), the shared expert's
  const bool extra = expert == uint(experts);
  const size_t at = (extra ? 0 : size_t(expert) * N) + size_t(row0);
  const device uint8_t* wr = (const device uint8_t*)(extra ? W2 : W) + at * (K * BITS / 8);
  const device bfloat* sr = (extra ? S2 : S) + at * (K / GS);
  const device bfloat* br = (extra ? B2 : B) + at * (K / GS);
  float acc[1][RPS];
  tf_rp_rows<RPS, 1, BITS, BF16SUM>(wr, sr, br, X + size_t(slot / repeat) * K, K, GS, lane, acc);
  if (lane == 0)
    for (int j = 0; j < RPS; j++) out[j] = tf_rp_out<RELU2>(acc[0][j]);
}

template [[host_name("tf_row_projection_indexed_route_q2_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 2, false>) tf_row_projection_indexed_route<4, 2, false, 2, false>;
template [[host_name("tf_row_projection_indexed_route_q2_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 2, true>) tf_row_projection_indexed_route<4, 2, false, 2, true>;
template [[host_name("tf_row_projection_indexed_route_q3_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 3, false>) tf_row_projection_indexed_route<4, 2, false, 3, false>;
template [[host_name("tf_row_projection_indexed_route_q3_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 3, true>) tf_row_projection_indexed_route<4, 2, false, 3, true>;
template [[host_name("tf_row_projection_indexed_route_q4_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 4, false>) tf_row_projection_indexed_route<4, 2, false, 4, false>;
template [[host_name("tf_row_projection_indexed_route_q4_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 4, true>) tf_row_projection_indexed_route<4, 2, false, 4, true>;
template [[host_name("tf_row_projection_indexed_route_q5_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 5, false>) tf_row_projection_indexed_route<4, 2, false, 5, false>;
template [[host_name("tf_row_projection_indexed_route_q5_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 5, true>) tf_row_projection_indexed_route<4, 2, false, 5, true>;
template [[host_name("tf_row_projection_indexed_route_q6_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 6, false>) tf_row_projection_indexed_route<4, 2, false, 6, false>;
template [[host_name("tf_row_projection_indexed_route_q6_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 6, true>) tf_row_projection_indexed_route<4, 2, false, 6, true>;
template [[host_name("tf_row_projection_indexed_route_q8_f32")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 8, false>) tf_row_projection_indexed_route<4, 2, false, 8, false>;
template [[host_name("tf_row_projection_indexed_route_q8_bf16")]] [[kernel]] decltype(tf_row_projection_indexed_route<4, 2, false, 8, true>) tf_row_projection_indexed_route<4, 2, false, 8, true>;
