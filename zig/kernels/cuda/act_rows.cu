// Activation rows for every CUDA family (float64 references: core/act_ref.zig): SwiGLU, output gates, group sums.

// A thread 8 values with 16-byte loads; a row's bits depend on that row alone.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_act {

constexpr int NT = 256;

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __bfloat1622float2(h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

__device__ __forceinline__ uint4 pack8(const float* f) {
    uint4 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return u;
}

__device__ __forceinline__ float sigmoid(float x) { return __fdiv_rn(1.f, __fadd_rn(1.f, expf(-x))); }

// A thread's 8 stored values in order, then the group's 8 lanes in a fixed tree (every lane of the warp calls).
__device__ __forceinline__ float group_sum8(const uint4& packed, bool live) {
    float f[8];
    unpack8(packed, f);
    float s = 0.f;
    if (live)
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __fadd_rn(s, f[i]);
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 4));
    s = __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 2));
    return __fadd_rn(s, __shfl_xor_sync(0xFFFFFFFFu, s, 1));
}

}  // namespace tf_act

using namespace tf_act;

// out[r, i] = bf16(silu(gate[r * g_ld + i]) * up[r * u_ld + i]) for i < n (n % 64 == 0); xs (or null) out's sums.
extern "C" __global__ void __launch_bounds__(NT) tf_swiglu_rows(const __nv_bfloat16* __restrict__ gate, long long g_ld,
                                                                const __nv_bfloat16* __restrict__ up, long long u_ld,
                                                                __nv_bfloat16* __restrict__ out, float* __restrict__ xs,
                                                                int n) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * NT + threadIdx.x; // 8 values
    const bool live = 8 * c < n;
    float g[8], u[8], o[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(gate + r * g_ld + 8 * c), g);
        unpack8(*reinterpret_cast<const uint4*>(up + r * u_ld + 8 * c), u);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = live ? __fmul_rn(__fmul_rn(g[i], sigmoid(g[i])), u[i]) : 0.f;
    const uint4 op = pack8(o);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = op;
    if (xs != nullptr) {
        const float s = group_sum8(op, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// out[r, h * d + e] = bf16(o[r, h * d + e] * sigmoid(g[r * g_ld + h * g_hs + e])) (d % 64 == 0); xs (or null) its sums.
extern "C" __global__ void __launch_bounds__(NT) tf_gate_rows(const __nv_bfloat16* __restrict__ o,
                                                              const __nv_bfloat16* __restrict__ g, long long g_ld,
                                                              long long g_hs, __nv_bfloat16* __restrict__ out,
                                                              float* __restrict__ xs, int heads, int d) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * NT + threadIdx.x; // 8 values
    const int n = heads * d;
    const bool live = 8 * c < n;
    const int h = 8 * c / d, e = 8 * c % d;
    float ov[8], gv[8], y[8];
    if (live) {
        unpack8(*reinterpret_cast<const uint4*>(o + r * n + 8 * c), ov);
        unpack8(*reinterpret_cast<const uint4*>(g + r * g_ld + h * g_hs + e), gv);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = live ? __fmul_rn(ov[i], sigmoid(gv[i])) : 0.f;
    const uint4 yp = pack8(y);
    if (live) *reinterpret_cast<uint4*>(out + r * n + 8 * c) = yp;
    if (xs != nullptr) {
        const float s = group_sum8(yp, live);
        if (live && (c & 7) == 0) xs[r * (n / 64) + c / 8] = s;
    }
}

// xs[r, j] = the sum of x[r * ld + 64 j ..][0 .. 64] (k % 64 == 0), in group_sum8's order.
extern "C" __global__ void __launch_bounds__(NT) tf_group_sums_rows(const __nv_bfloat16* __restrict__ x, long long ld,
                                                                    float* __restrict__ xs, int k) {
    const int64_t r = blockIdx.x;
    const int c = blockIdx.y * NT + threadIdx.x;
    const bool live = 8 * c < k;
    const uint4 v = live ? *reinterpret_cast<const uint4*>(x + r * ld + 8 * c) : make_uint4(0, 0, 0, 0);
    const float s = group_sum8(v, live);
    if (live && (c & 7) == 0) xs[r * (k / 64) + c / 8] = s;
}
