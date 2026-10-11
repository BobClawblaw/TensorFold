// RMS norm rows for every CUDA family (float64 references: core/norms_ref.zig), a 256-thread block a row.

// A row's sums run in one fixed order, so its bits never depend on the row count; kinds 0 fp32, 1 fp16, 2 bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_norms {

constexpr int NT = 256;

template <int K>
__device__ __forceinline__ float ld(const void* p, int64_t i) {
    if constexpr (K == 0) return static_cast<const float*>(p)[i];
    else if constexpr (K == 1) return __half2float(static_cast<const __half*>(p)[i]);
    else return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

template <int K>
__device__ __forceinline__ void st(void* p, int64_t i, float v) {
    if constexpr (K == 0) static_cast<float*>(p)[i] = v;
    else if constexpr (K == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

__device__ __forceinline__ float load(const void* p, int kind, int64_t i) {
    if (kind == 0) return static_cast<const float*>(p)[i];
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

__device__ __forceinline__ void store(void* p, int kind, int64_t i, float v) {
    if (kind == 0) static_cast<float*>(p)[i] = v;
    else if (kind == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<__nv_bfloat16*>(p)[i] = __float2bfloat16_rn(v);
}

// rsqrt(sum / width + eps) of a block's per-thread partial sums, the same order for every row.
__device__ __forceinline__ float inv_rms(float part, int width, float eps) {
    __shared__ float warps[NT / 32];
    __shared__ float inv;
    for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x >> 5] = part;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int w = 0; w < NT / 32; ++w) s = __fadd_rn(s, warps[w]);
        inv = rsqrtf(__fadd_rn(__fdiv_rn(s, static_cast<float>(width)), eps));
    }
    __syncthreads();
    const float r = inv;
    __syncthreads(); // `warps` and `inv` are reused by a second norm in the same block
    return r;
}

}  // namespace tf_norms

using namespace tf_norms;

namespace tf_norms {

constexpr int CACHED = 32; // a thread keeps its row values in registers up to CACHED * NT = 8,192 wide

// rms rows; WIDE re-reads the row from memory (wider than CACHED * NT), the same arithmetic in the same order.
template <bool WIDE, int XK, int YK>
__device__ __forceinline__ void rms_row(const void* x, const float* w, void* y, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float v[WIDE ? 1 : CACHED];
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < (WIDE ? 1 : CACHED); ++j) v[j] = 0.f;
    if (WIDE) {
        for (int i = threadIdx.x; i < width; i += NT) {
            const float a = ld<XK>(x, at + i);
            part = __fmaf_rn(a, a, part);
        }
    } else {
#pragma unroll
        for (int j = 0; j < CACHED; ++j) {
            const int i = threadIdx.x + j * NT;
            if (i < width) {
                v[j] = ld<XK>(x, at + i);
                part = __fmaf_rn(v[j], v[j], part);
            }
        }
    }
    const float inv = inv_rms(part, width, eps);
    if (WIDE) {
        for (int i = threadIdx.x; i < width; i += NT) {
            float a = __fmul_rn(ld<XK>(x, at + i), inv);
            if (w != nullptr) a = __fmul_rn(a, w[i]);
            st<YK>(y, at + i, a);
        }
    } else {
#pragma unroll
        for (int j = 0; j < CACHED; ++j) {
            const int i = threadIdx.x + j * NT;
            if (i < width) {
                float a = __fmul_rn(v[j], inv);
                if (w != nullptr) a = __fmul_rn(a, w[i]);
                st<YK>(y, at + i, a);
            }
        }
    }
}

template <bool WIDE, int XK, int YK>
__device__ __forceinline__ void add_rms_row(const void* x, float* res, const float* w1, const float* w2, void* y, int width,
                                            float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    if (WIDE) {
        float inv1 = 1.f;
        if (w1 != nullptr) {
            float part = 0.f;
            for (int i = threadIdx.x; i < width; i += NT) {
                const float a = ld<XK>(x, at + i);
                part = __fmaf_rn(a, a, part);
            }
            inv1 = inv_rms(part, width, eps);
        }
        float part = 0.f;
        for (int i = threadIdx.x; i < width; i += NT) {
            float a = ld<XK>(x, at + i);
            if (w1 != nullptr) a = __fmul_rn(__fmul_rn(a, inv1), w1[i]);
            const float r = __fadd_rn(res[at + i], a);
            res[at + i] = r;
            part = __fmaf_rn(r, r, part);
        }
        const float inv2 = inv_rms(part, width, eps);
        for (int i = threadIdx.x; i < width; i += NT) st<YK>(y, at + i, __fmul_rn(__fmul_rn(res[at + i], inv2), w2[i]));
        return;
    }
    float v[CACHED];
    float part = 0.f;
#pragma unroll
    for (int j = 0; j < CACHED; ++j) {
        const int i = threadIdx.x + j * NT;
        v[j] = i < width ? ld<XK>(x, at + i) : 0.f;
        if (i < width) part = __fmaf_rn(v[j], v[j], part);
    }
    const float inv1 = w1 != nullptr ? inv_rms(part, width, eps) : 1.f;
    part = 0.f;
#pragma unroll
    for (int j = 0; j < CACHED; ++j) {
        const int i = threadIdx.x + j * NT;
        if (i < width) {
            float a = v[j];
            if (w1 != nullptr) a = __fmul_rn(__fmul_rn(a, inv1), w1[i]);
            v[j] = __fadd_rn(res[at + i], a);
            res[at + i] = v[j];
            part = __fmaf_rn(v[j], v[j], part);
        }
    }
    const float inv2 = inv_rms(part, width, eps);
#pragma unroll
    for (int j = 0; j < CACHED; ++j) {
        const int i = threadIdx.x + j * NT;
        if (i < width) st<YK>(y, at + i, __fmul_rn(__fmul_rn(v[j], inv2), w2[i]));
    }
}

}  // namespace tf_norms

// y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), in fp32, rounded once to y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_rms_rows(const void* x, int x_kind, const float* w, void* y,
                                                             int y_kind, int width, float eps) {
#define TF_RMS(XK, YK)                                                                                            \
    if (x_kind == XK && y_kind == YK) {                                                                           \
        if (width > CACHED * NT) rms_row<true, XK, YK>(x, w, y, width, eps);                                      \
        else rms_row<false, XK, YK>(x, w, y, width, eps);                                                         \
        return;                                                                                                   \
    }
    TF_RMS(0, 0) TF_RMS(0, 1) TF_RMS(0, 2) TF_RMS(1, 0) TF_RMS(1, 1) TF_RMS(1, 2) TF_RMS(2, 0) TF_RMS(2, 1) TF_RMS(2, 2)
#undef TF_RMS
}

// res (fp32, in place) += rms(x) * w1 (a sandwich norm; null: res += x), then y = rms(res) * w2 in y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_add_rms_rows(const void* x, int x_kind, float* res, const float* w1,
                                                                 const float* w2, void* y, int y_kind, int width,
                                                                 float eps) {
#define TF_ADD(XK, YK)                                                                                            \
    if (x_kind == XK && y_kind == YK) {                                                                           \
        if (width > CACHED * NT) add_rms_row<true, XK, YK>(x, res, w1, w2, y, width, eps);                        \
        else add_rms_row<false, XK, YK>(x, res, w1, w2, y, width, eps);                                           \
        return;                                                                                                   \
    }
    TF_ADD(0, 0) TF_ADD(0, 1) TF_ADD(0, 2) TF_ADD(1, 0) TF_ADD(1, 1) TF_ADD(1, 2) TF_ADD(2, 0) TF_ADD(2, 1) TF_ADD(2, 2)
#undef TF_ADD
}
