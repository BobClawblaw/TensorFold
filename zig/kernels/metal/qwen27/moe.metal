// The routed block: route, the sort for the tensor units, the experts' SwiGLU and the combine (bf16 as the reference).
#include <metal_stdlib>
using namespace metal;

struct RouteArgs { int experts, topk, ld, renorm; }; // PICK, WEIGHT [rows, topk + 1]: the picks, then the shared expert (id experts, its sigmoid)
struct SortArgs { int slots, experts; };
struct GluArgs { int width; };
struct CombineArgs { int hidden, topk, shared_in_out, pad; };

// A row a threadgroup: fp32 softmax (to bf16) over LOGITS[row, 0..experts), top-k by logit (lower id on ties).
[[kernel]] void q27_route(const device bfloat* LOGITS [[buffer(0)]], device uint* PICK [[buffer(1)]],
                          device float* WEIGHT [[buffer(2)]], constant RouteArgs& a [[buffer(3)]],
                          device atomic_uint* COUNTS [[buffer(4)]],
                          uint r [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float part[32];
  if (r == 0) { // the first threadgroup clears the experts' slot counts for q27_route_count
    if (int(t) < a.experts) atomic_store_explicit(COUNTS + t, 0u, memory_order_relaxed);
    if (t == 0) atomic_store_explicit(COUNTS + a.experts, 0u, memory_order_relaxed);
  }
  threadgroup float best_v[32];
  threadgroup int best_i[32];
  threadgroup int sel[16];
  threadgroup float w[16];
  const device bfloat* z = LOGITS + size_t(r) * a.ld;
  const bool mine = int(t) < a.experts;
  const float x = mine ? float(z[t]) : -INFINITY;
  // the row's max, then the sum of exp
  float m = simd_max(x);
  if (lane == 0) part[sg] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  m = part[0];
  for (uint g = 1; g < nsg; g++) m = max(m, part[g]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float ex = mine ? precise::exp(x - m) : 0.0f;
  float s = simd_sum(ex);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  s = 0.0f;
  for (uint g = 0; g < nsg; g++) s += part[g];
  const float p = float(bfloat(ex / s));
  float key = x;
  for (int k = 0; k < a.topk; k++) {
    float v = key;
    int id = mine ? int(t) : 0x7fffffff;
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
      w[k] = p;
      key = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    float sum = 0.0f;
    for (int k = 0; k < a.topk; k++) sum += w[k];
    const float norm = float(bfloat(sum));
    const int stride = a.topk + 1;
    for (int k = 0; k < a.topk; k++) {
      PICK[size_t(r) * stride + k] = uint(sel[k]);
      WEIGHT[size_t(r) * stride + k] = a.renorm ? float(bfloat(w[k] / norm)) : w[k];
    }
    const float g = float(z[a.experts]);
    PICK[size_t(r) * stride + a.topk] = uint(a.experts);
    WEIGHT[size_t(r) * stride + a.topk] = float(bfloat(1.0f / (1.0f + precise::exp(-g))));
  }
}

// Each routed slot (id < experts) counted to its expert; q27_route cleared the counts.
[[kernel]] void q27_route_count(const device uint* PICK [[buffer(0)]], device atomic_uint* COUNTS [[buffer(1)]],
                                constant SortArgs& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (int(i) < a.slots && PICK[i] < uint(a.experts)) atomic_fetch_add_explicit(COUNTS + PICK[i], 1u, memory_order_relaxed);
}

// A thread an expert: the routed slots sorted by expert (stable) for core/affine_mm: OFFS (starts, the end), ORDER, WS.
[[kernel]] void q27_route_place(const device uint* PICK [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                                const device uint* COUNTS [[buffer(2)]], device int* OFFS [[buffer(3)]],
                                device int* ORDER [[buffer(4)]], device float* WS [[buffer(5)]],
                                constant SortArgs& a [[buffer(6)]], uint e [[thread_position_in_grid]]) {
  if (int(e) >= a.experts) return;
  int at = 0;
  for (int j = 0; j < int(e); j++) at += int(COUNTS[j]);
  OFFS[e] = at;
  if (e == 0) {
    int total = 0;
    for (int j = 0; j < a.experts; j++) total += int(COUNTS[j]);
    OFFS[a.experts] = total;
  }
  for (int i = 0; i < a.slots; i++) {
    if (PICK[i] != e) continue;
    ORDER[at] = i;
    WS[at] = WEIGHT[i];
    at++;
  }
}

// Sorted slot i's input row: X[ORDER[i] / topk] (a row for each of its picks).
[[kernel]] void q27_route_gather(const device bfloat* X [[buffer(0)]], const device int* ORDER [[buffer(1)]],
                                 device bfloat* Y [[buffer(2)]], constant int2& wk [[buffer(3)]],
                                 uint2 g [[thread_position_in_grid]]) {
  Y[size_t(g.y) * wk.x + g.x] = X[size_t(ORDER[g.y] / wk.y) * wk.x + g.x];
}

// act = silu(gate) * up in fp32, rounded once, as the dense block's q27_mlp.
[[kernel]] void q27_expert_glu(const device bfloat* G [[buffer(0)]], const device bfloat* U [[buffer(1)]],
                               device bfloat* OUT [[buffer(2)]], constant GluArgs& a [[buffer(3)]],
                               uint2 g [[thread_position_in_grid]]) {
  const size_t i = size_t(g.y) * a.width + g.x;
  const float gate = float(G[i]);
  OUT[i] = bfloat(gate / (1.0f + exp(-gate)) * float(U[i]));
}

// OUT[r] = bf16(bf16(sum_k bf16(Y[k] * w_k)) + bf16(sigma * shared)); shared: OUT[r] (shared_in_out) or Y's slot topk.
[[kernel]] void q27_expert_combine(const device bfloat* Y [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                                   device bfloat* OUT [[buffer(2)]], constant CombineArgs& a [[buffer(3)]],
                                   uint2 g [[thread_position_in_grid]]) {
  const size_t r = g.y, j = g.x, stride = a.topk + 1;
  float routed = 0.0f;
  for (int k = 0; k < a.topk; k++)
    routed += float(bfloat(float(Y[(r * stride + k) * a.hidden + j]) * WEIGHT[r * stride + k]));
  const float sigma = WEIGHT[r * stride + a.topk];
  const float row = a.shared_in_out ? float(OUT[r * a.hidden + j]) : float(Y[(r * stride + a.topk) * a.hidden + j]);
  OUT[r * a.hidden + j] = bfloat(float(bfloat(routed)) + float(bfloat(sigma * row)));
}

// Eight bf16 at a 16-byte aligned address as floats.
inline void q27_load8(const device bfloat* at, thread float* out) {
  const uint4 v = *(const device uint4*)at;
  const uint words[4] = {v.x, v.y, v.z, v.w};
  for (int i = 0; i < 4; i++) {
    out[2 * i] = as_type<float>(words[i] << 16);
    out[2 * i + 1] = as_type<float>(words[i] & 0xffff0000u);
  }
}

// q27_expert_combine for a row a threadgroup of 1,024 threads (hidden / 1,024 columns each, into the threadgroup), then the next
// norm by the first hidden / 16 threads as norm_xs.metal computes it: HO = bf16(H + out), XO = bf16(Wt * (HO * rsqrt(mean(HO^2) +
// eps))) and XS its 64-value group sums in the lane projection's order. hidden at most 4,096.
[[kernel]] void q27_expert_combine_norm(const device bfloat* Y [[buffer(0)]], const device float* WEIGHT [[buffer(1)]],
                                        device bfloat* OUT [[buffer(2)]], constant CombineArgs& a [[buffer(3)]],
                                        const device bfloat* H [[buffer(4)]], const device bfloat* Wt [[buffer(5)]],
                                        const constant float* eps [[buffer(6)]], const device int* dims [[buffer(7)]],
                                        device bfloat* HO [[buffer(8)]], device bfloat* XO [[buffer(9)]], device float* XS [[buffer(10)]],
                                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                        uint3 tpos [[thread_position_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  const uint t = tpos.x, m = tg.y;
  const int M = dims[0], MP = dims[1], K = a.hidden;
  constexpr int E = 16;
  const int TPG = K / E;
  threadgroup float red[32];
  threadgroup bfloat outs[4096];
  if (int(m) >= M) {
    if (int(t) < TPG && (t & 3) == 0) XS[(t >> 2) * MP + m] = 0.0f;
    return;
  }
  const size_t r = m, stride = a.topk + 1;
  const float sigma = WEIGHT[r * stride + a.topk];
  for (int j = int(t); j < K; j += 1024) {
    float routed = 0.0f;
    for (int k = 0; k < a.topk; k++)
      routed += float(bfloat(float(Y[(r * stride + k) * K + j]) * WEIGHT[r * stride + k]));
    const float row = a.shared_in_out ? float(OUT[r * K + j]) : float(Y[(r * stride + a.topk) * K + j]);
    const bfloat out = bfloat(float(bfloat(routed)) + float(bfloat(sigma * row)));
    OUT[r * K + j] = out;
    outs[j] = out;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // the norm's threads (every thread reaches the barriers; the rest hold zeros)
  const bool norms = int(t) < TPG;
  const int base = int(m) * K + int(t) * E;
  float hin[E];
  if (norms) {
    q27_load8(H + base, hin);
    q27_load8(H + base + 8, hin + 8);
  }
  float hv[E];
  float ss = 0.0f;
  for (int i = 0; i < E; i++) {
    const bfloat h = norms ? bfloat(hin[i] + float(outs[int(t) * E + i])) : bfloat(0.0f);
    if (norms) HO[base + i] = h;
    hv[i] = float(h);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!norms) return;
  float total = 0.0f;
  for (int i = 0; i < TPG / 32; i++) total += red[i];
  const float inv = metal::rsqrt(total / float(K) + eps[0]);
  float wn[E];
  q27_load8(Wt + int(t) * E, wn);
  q27_load8(Wt + int(t) * E + 8, wn + 8);
  float xv[E];
  for (int i = 0; i < E; i++) {
    const bfloat x = bfloat(wn[i] * (hv[i] * inv));
    XO[base + i] = x;
    xv[i] = float(x);
  }
  // a group's 64 values summed in order from zero, as the lane projection's sums are: its four threads take turns
  float gs = 0.0f;
  for (uint turn = 0; turn < 4; turn++) {
    if ((t & 3) == turn) for (int i = 0; i < E; i++) gs += xv[i];
    gs = simd_shuffle(gs, ushort((lane & ~3u) | turn));
  }
  if ((t & 3) == 0) XS[(t >> 2) * MP + m] = gs;
}
