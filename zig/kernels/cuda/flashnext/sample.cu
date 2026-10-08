// Flash Next port: device-side greedy candidates. One block per bf16 logits row: the row's first maximum (lowest
// column among equal maxima, as argmax picks it), mapped to a global token id, and the row's log-sum-exp in fp32.
// out[row] = {id, max (fp32 bits), lse (fp32 bits), 0}. Build: nvcc -cubin -arch=sm_121 -O3 -o fn_sample.cubin fn_sample.cu
#include <cuda_bf16.h>
#include <math_constants.h>

struct Top {
    float m;      // running maximum
    int i;        // first column holding it
    float s;      // sum of exp(x - m)
};

__device__ __forceinline__ Top merge(Top a, Top b) {
    if (b.m > a.m || (b.m == a.m && b.i < a.i)) { Top t = a; a = b; b = t; }
    // a holds the larger maximum (or the earlier column at a tie)
    float s = a.s + (b.m == -CUDART_INF_F ? 0.f : b.s * __expf(b.m - a.m));
    return {a.m, a.i, s};
}

extern "C" __global__ void __launch_bounds__(1024) fn_rows_top(const __nv_bfloat16* __restrict__ logits, int cols,
        int stride, const int* __restrict__ id_map, int offset, int* __restrict__ out) {
    const int row = blockIdx.x;
    const __nv_bfloat16* x = logits + (size_t)row * stride;
    Top t = {-CUDART_INF_F, 0x7fffffff, 0.f};
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[c]);
        if (v > t.m) { t.s = t.s * __expf(t.m - v) + 1.f; t.m = v; t.i = c; }
        else t.s += __expf(v - t.m);
    }
    for (int o = 16; o > 0; o >>= 1) {
        Top u = {__shfl_xor_sync(0xffffffff, t.m, o), __shfl_xor_sync(0xffffffff, t.i, o), __shfl_xor_sync(0xffffffff, t.s, o)};
        t = merge(t, u);
    }
    __shared__ Top warps[32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) warps[w] = t;
    __syncthreads();
    if (w == 0) {
        t = lane < (blockDim.x >> 5) ? warps[lane] : Top{-CUDART_INF_F, 0x7fffffff, 0.f};
        for (int o = 16; o > 0; o >>= 1) {
            Top u = {__shfl_xor_sync(0xffffffff, t.m, o), __shfl_xor_sync(0xffffffff, t.i, o), __shfl_xor_sync(0xffffffff, t.s, o)};
            t = merge(t, u);
        }
        if (lane == 0) {
            int* o4 = out + 4 * row;
            o4[0] = id_map ? id_map[t.i] : t.i + offset;
            o4[1] = __float_as_int(t.m);
            o4[2] = __float_as_int(t.m + __logf(t.s));
            o4[3] = 0;
        }
    }
}
