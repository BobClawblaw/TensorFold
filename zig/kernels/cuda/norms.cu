// RMS norm rows for every CUDA family (float64 references: core/norms_ref.zig), a 256-thread block a row.

// A row's sums run in one fixed order, so its bits never depend on the row count; kinds 0 fp32, 1 fp16, 2 bf16.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_norms {

constexpr int NT = 256;

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

// y = x * rsqrt(mean(x^2) + eps) * w (w null: no weight), in fp32, rounded once to y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_rms_rows(const void* x, int x_kind, const float* w, void* y,
                                                             int y_kind, int width, float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += NT) {
        const float v = load(x, x_kind, at + i);
        part = __fmaf_rn(v, v, part);
    }
    const float inv = inv_rms(part, width, eps);
    for (int i = threadIdx.x; i < width; i += NT) {
        float v = __fmul_rn(load(x, x_kind, at + i), inv);
        if (w != nullptr) v = __fmul_rn(v, w[i]);
        store(y, y_kind, at + i, v);
    }
}

// res (fp32, in place) += rms(x) * w1 (a sandwich norm; null: res += x), then y = rms(res) * w2 in y's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_add_rms_rows(const void* x, int x_kind, float* res, const float* w1,
                                                                 const float* w2, void* y, int y_kind, int width,
                                                                 float eps) {
    const int64_t at = static_cast<int64_t>(blockIdx.x) * width;
    float inv1 = 1.f;
    if (w1 != nullptr) {
        float part = 0.f;
        for (int i = threadIdx.x; i < width; i += NT) {
            const float v = load(x, x_kind, at + i);
            part = __fmaf_rn(v, v, part);
        }
        inv1 = inv_rms(part, width, eps);
    }
    float part = 0.f;
    for (int i = threadIdx.x; i < width; i += NT) {
        float v = load(x, x_kind, at + i);
        if (w1 != nullptr) v = __fmul_rn(__fmul_rn(v, inv1), w1[i]);
        const float r = __fadd_rn(res[at + i], v);
        res[at + i] = r;
        part = __fmaf_rn(r, r, part);
    }
    const float inv2 = inv_rms(part, width, eps);
    for (int i = threadIdx.x; i < width; i += NT) store(y, y_kind, at + i, __fmul_rn(__fmul_rn(res[at + i], inv2), w2[i]));
}
