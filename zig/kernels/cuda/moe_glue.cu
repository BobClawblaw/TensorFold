// MoE glue for every CUDA family (float64 references: core/moe_ref.zig): sigmoid top-k and the weighted combine.

// Every row's bits depend on that row alone; kinds 0 fp32, 1 fp16, 2 bf16 (as norms.cu).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

namespace tf_moe {

constexpr int NT = 256;
constexpr int MOST = 4096; // experts a row ranks

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

// (value, id) wins over (other value, other id): the larger value, the lower id among equals.
__device__ __forceinline__ bool beats(float v, int i, float w, int j) {
    return v > w || (v == w && i < j);
}

}  // namespace tf_moe

using namespace tf_moe;

// Top `topk` by logit + bias (ties: lower id), weights sigmoid(logit) [normalized] * scale; `shared`: slot topk = ne.
extern "C" __global__ void __launch_bounds__(NT) tf_moe_topk_sigmoid(const float* logits, const float* bias, int* pick,
                                                                     float* wts, int ne, int topk, int shared,
                                                                     int normalize, float scale) {
    __shared__ float v[MOST];
    __shared__ float best_v[NT / 32];
    __shared__ int best_i[NT / 32];
    __shared__ int picked[64];
    const int64_t r = blockIdx.x;
    const float* lg = logits + r * ne;
    for (int e = threadIdx.x; e < ne; e += NT) v[e] = __fadd_rn(lg[e], bias != nullptr ? bias[e] : 0.f);
    __syncthreads();
    for (int k = 0; k < topk; ++k) {
        float bv = -INFINITY;
        int bi = ne;
        for (int e = threadIdx.x; e < ne; e += NT)
            if (beats(v[e], e, bv, bi)) bv = v[e], bi = e;
        for (int m = 16; m > 0; m >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, m);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, m);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        if ((threadIdx.x & 31) == 0) best_v[threadIdx.x >> 5] = bv, best_i[threadIdx.x >> 5] = bi;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tv = best_v[0];
            int ti = best_i[0];
            for (int w = 1; w < NT / 32; ++w)
                if (beats(best_v[w], best_i[w], tv, ti)) tv = best_v[w], ti = best_i[w];
            picked[k] = ti;
            v[ti] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int slots = topk + (shared ? 1 : 0);
        float sum = 0.f;
        float w[64];
        for (int k = 0; k < topk; ++k) {
            w[k] = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-lg[picked[k]])));
            sum = __fadd_rn(sum, w[k]);
        }
        for (int k = 0; k < topk; ++k) {
            pick[r * slots + k] = picked[k];
            wts[r * slots + k] = __fmul_rn(normalize ? __fdiv_rn(w[k], sum) : w[k], scale);
        }
        if (shared) pick[r * slots + topk] = ne, wts[r * slots + topk] = 1.f;
    }
}

// out[r, d] = sum over the row's slots, in slot order, of w[r, k] * y[r, k, d] (fp32 fma), rounded once to out's kind.
extern "C" __global__ void __launch_bounds__(NT) tf_moe_combine(const void* y, int y_kind, const float* w, void* out,
                                                                int out_kind, int slots, int dims) {
    const int64_t r = blockIdx.x;
    const int d = blockIdx.y * NT + threadIdx.x;
    if (d >= dims) return;
    float acc = 0.f;
    for (int k = 0; k < slots; ++k) acc = __fmaf_rn(w[r * slots + k], load(y, y_kind, (r * slots + k) * dims + d), acc);
    store(out, out_kind, r * dims + d, acc);
}
