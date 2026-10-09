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

// Sampling candidates: each row's 64 largest logits (by value, then the lower column at a tie), as global token ids
// and fp32 values, the row's log-sum-exp, and its log-sum-exp at the row's temperature (inv_t[row], x * inv_t); both
// ranks' sets are gathered and the keyed draw runs on the host over the merged 128 (whose first 64 are the vocabulary's
// top 64 exactly). Each thread keeps its columns' sorted top 64, then the block merges the threads' lists by repeated
// argmax over their heads (exact: no value of the row's top 64 can be missing from its thread's list).
// out[row * 132]: ids[64], values[64] (fp32 bits), lse, lse at the temperature (fp32 bits), 2 spare.
#define TOPK 64
#define TOPW (2 * TOPK + 4)
__device__ __forceinline__ bool above(float a, int ia, float b, int ib) { return a > b || (a == b && ia < ib); }

__device__ __forceinline__ void lse_merge(float& m, float& s, float m2, float s2) {
    const float mm = fmaxf(m, m2);
    if (mm == -CUDART_INF_F) return;
    s = (m == -CUDART_INF_F ? 0.f : s * __expf(m - mm)) + (m2 == -CUDART_INF_F ? 0.f : s2 * __expf(m2 - mm));
    m = mm;
}

extern "C" __global__ void __launch_bounds__(256) fn_rows_topk(const __nv_bfloat16* __restrict__ logits, int cols,
        int stride, int offset, const float* __restrict__ inv_t, int* __restrict__ out) {
    const int row = blockIdx.x;
    const __nv_bfloat16* x = logits + (size_t)row * stride;
    const float it = inv_t ? inv_t[row] : 1.f;
    float lv[TOPK];
    int li[TOPK];
    int n = 0;
    float m = -CUDART_INF_F, s = 0.f, mt = -CUDART_INF_F, st = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = __bfloat162float(x[c]);
        if (v > m) { s = s * __expf(m - v) + 1.f; m = v; } else s += __expf(v - m);
        const float vt = v * it;
        if (vt > mt) { st = st * __expf(mt - vt) + 1.f; mt = vt; } else st += __expf(vt - mt);
        if (n < TOPK || above(v, c, lv[TOPK - 1], li[TOPK - 1])) {
            int j = n < TOPK ? n++ : TOPK - 1;
            while (j > 0 && above(v, c, lv[j - 1], li[j - 1])) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; --j; }
            lv[j] = v;
            li[j] = c;
        }
    }
    __shared__ float sm[8], ss[8], smt[8], sst[8];
    __shared__ float hv[256];
    __shared__ int hi[256], winner;
    for (int o = 16; o > 0; o >>= 1) {
        lse_merge(m, s, __shfl_xor_sync(0xffffffff, m, o), __shfl_xor_sync(0xffffffff, s, o));
        lse_merge(mt, st, __shfl_xor_sync(0xffffffff, mt, o), __shfl_xor_sync(0xffffffff, st, o));
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) { sm[w] = m; ss[w] = s; smt[w] = mt; sst[w] = st; }
    __syncthreads();
    int* o = out + (size_t)row * TOPW;
    if (threadIdx.x == 0) {
        float M = -CUDART_INF_F, S = 0.f, MT = -CUDART_INF_F, ST = 0.f;
        for (int i = 0; i < nw; ++i) { lse_merge(M, S, sm[i], ss[i]); lse_merge(MT, ST, smt[i], sst[i]); }
        o[2 * TOPK] = __float_as_int(M + __logf(S));
        o[2 * TOPK + 1] = __float_as_int(MT + __logf(ST));
    }
    int head = 0;
    for (int k = 0; k < TOPK; ++k) {
        hv[threadIdx.x] = head < n ? lv[head] : -CUDART_INF_F;
        hi[threadIdx.x] = head < n ? li[head] : 0x7fffffff;
        __syncthreads();
        if (threadIdx.x == 0) {
            int b = 0;
            for (int t = 1; t < (int)blockDim.x; ++t) if (above(hv[t], hi[t], hv[b], hi[b])) b = t;
            winner = b;
            o[k] = hi[b] == 0x7fffffff ? -1 : hi[b] + offset;
            o[TOPK + k] = __float_as_int(hv[b]);
        }
        __syncthreads();
        if (threadIdx.x == winner) ++head;
        __syncthreads();
    }
}
