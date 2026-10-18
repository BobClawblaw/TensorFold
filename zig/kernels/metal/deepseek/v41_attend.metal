// DeepSeek-V4.1-Flash's sparse attention and window-key kernels, compiled after v41_index.metal.

// ---------------------------------------------------------------- sparse attention

struct AttnArgs {
  int first_pos;  // row r's position is first_pos + r
  int ring;       // the window ring's rows
  int window;
  int picks;      // compressed picks a row (0: window only)
  float scale;
  int window_end; // >= 0: every row's window ends at this position (a draft block's), else at the row's own
};

constant constexpr int AH = 8;     // heads a threadgroup
constant constexpr int AT = 8;     // keys a tile
constant constexpr int AMAX = 640; // keys a row at most: the window and the indexer's picks

// The row's j-th key: its window (positions pos - nwin + 1 .. pos, from the ring), then its compressed picks.
inline const device bfloat* attn_key(const device bfloat* RING, const device bfloat* COMP, const device int* picks,
                                     int pos, int nwin, int ring, int j) {
  if (j < nwin) return RING + size_t((pos - nwin + 1 + j) % ring) * HD;
  return COMP + size_t(max(picks[j - nwin], 0)) * HD;
}

// Prompt chunks: a threadgroup (512) a (row, 8 heads) over its window and picks, each 8-key tile read once.
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

// Decode rows' attention split by keys: a threadgroup a (HG heads, chunk, row), RoPE as queries load; partials out.
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

// A (row, head)'s partials and sink as one softmax, then (as ds_rope inverse, ds_fp8_qdq) RoPE taken back, fp8-rounded.
struct MergeArgs {
  int chunks;
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

// A threadgroup's bf16 row of 512 fp8-rounded in blocks of 32 as ds_fp8_qdq does: a block a simdgroup.
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
  rope_row(row, POS[r], FREQ, true, t);
  fp8_row(row, t, HD);
  for (int d = int(t); d < HD; d += HD) O[(size_t(r) * 64 + h) * HD + d] = row[d];
}

// A window key's tail a threadgroup a row: ds_rms, ds_rope, ds_fp8_qdq in place, then into the ring (ring > 0).
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
