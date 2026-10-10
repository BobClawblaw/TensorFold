// Flash Next's Gated DeltaNet over a shared round: every stream's chain in one launch, and every stream's replay of
// its kept rows (all layers) in another. Ours: fn_gdn.cu's chain_kernel and replay_kernel (the Python engine's
// gdn.cu) with each block's stream read from a table, so a stream's rows give the bits of its own launch.
//
// A shared round used to launch the chain once per stream and layer, and the replay once per stream and layer at
// each keep: with 16 streams, 576 small launches each, every one of them using half the GPU's SMs.

#include "fn_gdn.cu"

namespace {

// One stream's chain in a layer: its projection rows, conv window, states and replay scratch, where its rows sit.
struct ChainSeg {
    const __nv_bfloat16* P;
    const __nv_bfloat16* cs;
    const float* state_in;
    float* state_out;
    float* k_save;
    __nv_bfloat16* v_save;
    float* g_save;
    float* b_save;
    long long rows;
    long long row0;
};

// One stream's replay in a layer: the states and the saved rows to keep (rows 0: nothing to replay).
struct ReplaySeg {
    const float* state_in;
    const float* k_save;
    const __nv_bfloat16* v_save;
    const float* g_save;
    const float* b_save;
    float* state_out;
    long long rows;
    long long pad;
};

// chain_kernel's body (AHEAD), block (hv, stream): the stream's table entry in place of the launch's arguments.
template <int NK, int NV>
__global__ void __launch_bounds__(1024) chain_multi_kernel(
        const ChainSeg* __restrict__ segs, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps,
        __nv_bfloat16* __restrict__ out_all, float* __restrict__ xs_all) {
    constexpr bool AHEAD = true;
    const ChainSeg sg = segs[blockIdx.y];
    const __nv_bfloat16* __restrict__ P = sg.P;
    const __nv_bfloat16* __restrict__ cs = sg.cs;
    const float* __restrict__ state_in = sg.state_in;
    float* __restrict__ state_out = sg.state_out;
    float* __restrict__ k_save = sg.k_save;
    __nv_bfloat16* __restrict__ v_save = sg.v_save;
    float* __restrict__ g_save = sg.g_save;
    float* __restrict__ b_save = sg.b_save;
    const int rows = (int)sg.rows;
    __nv_bfloat16* __restrict__ out = out_all + (size_t)sg.row0 * NV * DV;
    float* __restrict__ xs = xs_all + (size_t)sg.row0 * (NV * DV / 32);
    constexpr int C = 2 * NK * DK + NV * DV;          // conv channels: q | k | v
    constexpr int PW = C + NV * DV + 2 * NV;          // projection row: qkv | z | b | a
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV];
    __shared__ float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    float w[TAPS] = {}, win[TAPS - 1] = {};
    __nv_bfloat16 xin = {}, zin = {}, bin = {}, ain = {};
    const __nv_bfloat16 *pz = P + C + hv * DV + (t < DV ? t : 0), *pb = P + C + NV * DV + hv, *pa = pb + NV;
    if (AHEAD && rows > 0) {
        if (c >= 0) {
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[c * TAPS + tap]);
#pragma unroll
            for (int tap = 0; tap < TAPS - 1; ++tap) win[tap] = __bfloat162float(cs[tap * C + c]);
            xin = P[c];
        }
        if (t < DV) zin = pz[0];
        if (warp == 2 && lane == 0) { bin = pb[0]; ain = pa[0]; }
    }
    for (int r = 0; r < rows; ++r) {
        const __nv_bfloat16 xr = xin, zr = zin, br = bin, ar = ain;
        if (AHEAD && r + 1 < rows) {
            const size_t next = (size_t)(r + 1) * PW;
            if (c >= 0) xin = P[next + c];
            if (t < DV) zin = pz[next];
            if (warp == 2 && lane == 0) { bin = pb[next]; ain = pa[next]; }
        }
        if (c >= 0) {
            float acc = 0.0f;
            const float xn = __bfloat162float(xr);
#pragma unroll
            for (int tap = 0; tap < TAPS - 1; ++tap) acc = acc + w[tap] * win[tap];
            acc = acc + w[TAPS - 1] * xn;
            win[0] = win[1]; win[1] = win[2]; win[2] = xn;
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(br);
            const float a = __bfloat162float(ar);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        __syncthreads();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
        update(s, kk, vs, warp, g, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        if (k_save != nullptr) {
            if (t < DK && hv % (NV / NK) == 0) k_save[((size_t)r * NK + hk) * DK + t] = ks[t];
            if (t < DV) v_save[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(vs[t]);
            if (t == 0) { g_save[r * NV + hv] = g; b_save[r * NV + hv] = beta; }
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
            const float z = __bfloat162float(zr);
            const float o = bf(yn * sigmoidf_(z));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = warp_sum(o);
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

// replay_kernel's body, block (hv, stream, layer): ``segs`` is [layers][streams].
template <int NK, int NV>
__global__ void __launch_bounds__(1024) replay_multi_kernel(const ReplaySeg* __restrict__ segs, int streams) {
    const ReplaySeg sg = segs[(size_t)blockIdx.z * streams + blockIdx.y];
    const int rows = (int)sg.rows;
    if (rows <= 0) return;
    const float* __restrict__ state_in = sg.state_in;
    const float* __restrict__ k_save = sg.k_save;
    const __nv_bfloat16* __restrict__ v_save = sg.v_save;
    const float* __restrict__ g_save = sg.g_save;
    const float* __restrict__ b_save = sg.b_save;
    float* __restrict__ state_out = sg.state_out;
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        if (t < DV) vs[t] = __bfloat162float(v_save[((size_t)r * NV + hv) * DV + t]);
        __syncthreads();
        float kk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kk[i] = k_save[((size_t)r * NK + hk) * DK + lane * 4 + i];
        update(s, kk, vs, warp, g_save[r * NV + hv], b_save[r * NV + hv]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

}  // namespace

namespace {
template __global__ void chain_multi_kernel<(int)8, (int)24>(const ChainSeg*, const __nv_bfloat16*, const float*, const float*, const __nv_bfloat16*, float, __nv_bfloat16*, float*);
template __global__ void replay_multi_kernel<(int)8, (int)24>(const ReplaySeg*, int);
}  // namespace
