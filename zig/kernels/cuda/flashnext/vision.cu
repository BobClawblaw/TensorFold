// The Flash Next vision tower's elementwise and row kernels (ours): the tower's linears run on cuBLASLt (fp32
// accumulate), these do what torch does around them, rounding to bf16 where torch's bf16 tensors round: a Linear's
// bias in its fp32 epilogue, GELU on the rounded output, LayerNorm and the 2-D rotary in fp32, softmax in fp32.
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
__device__ __forceinline__ float f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 b(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float round_bf(float x) { return f(b(x)); }

__device__ __forceinline__ float gelu_tanh(float x) {   // torch's gelu(approximate="tanh") in fp32
    const float k0 = 0.7978845608028654f, k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) s += red[i];
    return s;
}
__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int w = threadIdx.x / 32, l = threadIdx.x % 32;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int i = 0; i < (int)(blockDim.x + 31) / 32; ++i) m = fmaxf(m, red[i]);
    return m;
}
}  // namespace

// out[r, c] = act(bf16(acc[r, c] + bias[c])): a Linear's fp32 product, its bias, then 0 none / 1 gelu-tanh / 2 gelu-erf
extern "C" __global__ void fn_vis_bias_act(const float* __restrict__ acc, const __nv_bfloat16* __restrict__ bias,
                                           __nv_bfloat16* __restrict__ out, int64_t n, int cols, int act) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        float t = round_bf(acc[i] + f(bias[i % cols]));
        if (act == 1) t = gelu_tanh(t);
        else if (act == 2) t = gelu_erf(t);
        out[i] = b(t);
    }
}

// x[i] = bf16(x[i] + y[i]) (a residual)
extern "C" __global__ void fn_vis_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] = b(f(x[i]) + f(y[i]));
}

// fp32 patches to bf16 (the patch embedding's input cast)
extern "C" __global__ void fn_vis_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        y[i] = b(x[i]);
}

// LayerNorm over each row of ``cols`` (fp32 mean and variance, then weight and bias): one block a row
extern "C" __global__ void fn_vis_layernorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                            const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y,
                                            int cols, float eps) {
    __shared__ float red[32];
    const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * cols;
    __nv_bfloat16* yr = y + (int64_t)blockIdx.x * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += f(xr[c]);
    const float mean = block_sum(s, red) / cols;
    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = f(xr[c]) - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum(v, red) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = b((f(xr[c]) - mean) * rstd * f(w[c]) + f(bias[c]));
}

// the position table interpolated (four taps a patch, fp32 weights) and added: x = bf16(x + bf16(sum_j w_j table[i_j]))
extern "C" __global__ void fn_vis_pos_embed(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ table,
                                            const int* __restrict__ idx, const float* __restrict__ wts, int rows, int cols) {
    const int64_t n = (int64_t)rows * cols;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / cols), c = (int)(i % cols);
        float p = 0.0f;
        for (int j = 0; j < 4; ++j) p += f(table[(int64_t)idx[r * 4 + j] * cols + c]) * wts[r * 4 + j];
        x[i] = b(f(x[i]) + round_bf(p));
    }
}

// qkv [N, 3, H, D] -> q, k [H, N, D] rotated by the 2-D rotary (row then column angles, duplicated over the head) in
// fp32, and v transposed [H, D, N] (the PV product's weight layout)
extern "C" __global__ void fn_vis_rope_split(const __nv_bfloat16* __restrict__ qkv, const int* __restrict__ pos,
                                             const float* __restrict__ inv_freq, __nv_bfloat16* __restrict__ q,
                                             __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ vt, int N, int H,
                                             int D) {
    const int half = D / 2, quarter = D / 4;
    const int64_t n = (int64_t)N * H * D;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int d = (int)(i % D), h = (int)((i / D) % H), r = (int)(i / ((int64_t)D * H));
        const int j = d % half;
        const float angle = (float)pos[r * 2 + (j < quarter ? 0 : 1)] * inv_freq[j % quarter];
        const float c = cosf(angle), s = sinf(angle);
        const __nv_bfloat16* row = qkv + (int64_t)r * 3 * H * D;
        const int pair = d < half ? d + half : d - half;
        const float sign = d < half ? -1.0f : 1.0f;
        const float qv = f(row[h * D + d]), qp = f(row[h * D + pair]);
        const float kv = f(row[(H + h) * D + d]), kp = f(row[(H + h) * D + pair]);
        const int64_t o = ((int64_t)h * N + r) * D + d;
        q[o] = b(qv * c + sign * qp * s);
        k[o] = b(kv * c + sign * kp * s);
        vt[((int64_t)h * D + d) * N + r] = row[(2 * H + h) * D + d];
    }
}

// a row of scores as flash attention keeps them: p = bf16(exp(s * scale - max)) (unnormalized, the values product's
// input) and the row's fp32 sum of the unrounded exponentials (the output is divided by it after the product)
extern "C" __global__ void fn_vis_softmax(const float* __restrict__ s, __nv_bfloat16* __restrict__ p, float* __restrict__ sums,
                                          int cols, float scale) {
    __shared__ float red[32];
    const float* sr = s + (int64_t)blockIdx.x * cols;
    __nv_bfloat16* pr = p + (int64_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, sr[c] * scale);
    m = block_max(m, red);
    float t = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float e = expf(sr[c] * scale - m);
        t += e;
        pr[c] = b(e);
    }
    t = block_sum(t, red);
    if (threadIdx.x == 0) sums[blockIdx.x] = t;
}

// a head's output [rows, D] (the fp32 values product of unnormalized probabilities) divided by each row's sum and
// rounded into the merged [rows, H * D] at head h
extern "C" __global__ void fn_vis_head_out(const float* __restrict__ o, const float* __restrict__ sums,
                                           __nv_bfloat16* __restrict__ out, int rows, int H, int D, int h) {
    const int64_t n = (int64_t)rows * D;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / D), d = (int)(i % D);
        out[(int64_t)r * H * D + h * D + d] = b(o[i] / sums[r]);
    }
}

// image features into a prompt piece's hidden rows: for each (row, feature) pair, the feature in every one of the
// row's ``copies`` hyper-connection streams (the Python engine's index_copy of features.repeat(1, streams))
extern "C" __global__ void fn_vis_splice(__nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ feats,
                                         const int* __restrict__ pairs, int width, int copies) {
    const int row = pairs[blockIdx.x * 2], src = pairs[blockIdx.x * 2 + 1];
    const __nv_bfloat16* f = feats + (int64_t)src * width;
    __nv_bfloat16* dst = h + (int64_t)row * width * copies;
    for (int i = threadIdx.x; i < width * copies; i += blockDim.x) dst[i] = f[i % width];
}

// the fp32 residual stream (the tower's precise variant): x += y (bf16 branch output), LayerNorm reading fp32 rows,
// and the casts in and out
extern "C" __global__ void fn_vis_add32(float* __restrict__ x, const __nv_bfloat16* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] += f(y[i]);
}
extern "C" __global__ void fn_vis_layernorm32(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                              const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ y,
                                              int cols, float eps) {
    __shared__ float red[32];
    const float* xr = x + (int64_t)blockIdx.x * cols;
    __nv_bfloat16* yr = y + (int64_t)blockIdx.x * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += xr[c];
    const float mean = block_sum(s, red) / cols;
    float v = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = xr[c] - mean;
        v += d * d;
    }
    const float rstd = rsqrtf(block_sum(v, red) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) yr[c] = b((xr[c] - mean) * rstd * f(w[c]) + f(bias[c]));
}
extern "C" __global__ void fn_vis_to_f32(const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        y[i] = f(x[i]);
}
