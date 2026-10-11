// Attention inputs for every CUDA family (float64 references: core/qk_ref.zig): QK norm, RoPE and the cache write.

// A block a (row, head), a thread an element; a head's norm sums in one fixed order, never depending on the row count.

#include <cuda_bf16.h>
#include <stdint.h>

namespace tf_qk {

constexpr int MOST = 256; // head_dim at most

}  // namespace tf_qk

using namespace tf_qk;

// Packed rows -> q, k, v (bf16); q, k normed by nw, rotate-half RoPE on `rot` dims; k, v also into kc/vc (or not).
extern "C" __global__ void __launch_bounds__(MOST) tf_qk_rows(const __nv_bfloat16* qkv, const float* nw, const float* cosv,
                                                              const float* sinv, int rot, const int* pos,
                                                              const int* slot, __nv_bfloat16* q, __nv_bfloat16* ko,
                                                              __nv_bfloat16* vo, __nv_bfloat16* kc, __nv_bfloat16* vc,
                                                              long long len, int h, int hk, float eps) {
    __shared__ float warps[MOST / 32];
    const int d = threadIdx.x;
    const int dim = blockDim.x;
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y;
    const __nv_bfloat16* src = qkv + (r * (h + 2 * hk) + hd) * dim;
    const float x = __bfloat162float(src[d]);
    const int p = pos[r];
    const int64_t at = slot != nullptr ? (static_cast<int64_t>(slot[r]) * len + p % len) : p % len;
    if (hd >= h + hk) { // v: as is
        const int k = hd - h - hk;
        vo[(r * hk + k) * dim + d] = src[d];
        if (vc != nullptr) vc[(at * hk + k) * dim + d] = src[d];
        return;
    }
    float y = x;
    float yp = 0.f;
    const int half = rot / 2;
    const int partner = d < half ? d + half : d - half;
    if (nw != nullptr) {
        float part = __fmul_rn(x, x);
        for (int m = 16; m > 0; m >>= 1) part = __fadd_rn(part, __shfl_xor_sync(0xFFFFFFFFu, part, m));
        if ((d & 31) == 0) warps[d >> 5] = part;
        __syncthreads();
        float s = 0.f;
        for (int w = 0; w < dim / 32; ++w) s = __fadd_rn(s, warps[w]);
        const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(s, static_cast<float>(dim)), eps));
        const float* wrow = nw + (hd >= h ? dim : 0);
        y = __fmul_rn(__fmul_rn(x, rstd), wrow[d]);
        if (d < rot) yp = __fmul_rn(__fmul_rn(__bfloat162float(src[partner]), rstd), wrow[partner]);
    } else if (d < rot) {
        yp = __bfloat162float(src[partner]);
    }
    if (cosv != nullptr && d < rot) {
        const float c = cosv[static_cast<int64_t>(p) * rot + d];
        const float sn = sinv[static_cast<int64_t>(p) * rot + d];
        y = __fadd_rn(__fmul_rn(y, c), __fmul_rn(d < half ? -yp : yp, sn));
    }
    const __nv_bfloat16 out = __float2bfloat16_rn(y);
    if (hd < h) {
        q[(r * h + hd) * dim + d] = out;
    } else {
        const int k = hd - h;
        ko[(r * hk + k) * dim + d] = out;
        if (kc != nullptr) kc[(at * hk + k) * dim + d] = out;
    }
}
