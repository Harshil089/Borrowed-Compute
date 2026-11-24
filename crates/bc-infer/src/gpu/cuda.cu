// CUDA entry points for kernels.h, compiled ahead of time to PTX (kernels.ptx, by
// build-ptx.sh) with clang alone: no CUDA Toolkit, no CUDA headers. The NVIDIA driver
// compiles that PTX for whatever card it finds, so users need only the driver.

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long ulong;

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define BX __nvvm_read_ptx_sreg_ctaid_x()
#define BY __nvvm_read_ptx_sreg_ctaid_y()
#define TX __nvvm_read_ptx_sreg_tid_x()
#define BDX __nvvm_read_ptx_sreg_ntid_x()

struct float4 { float x, y, z, w; } __attribute__((aligned(16)));

__device__ static inline float4 f4(float a, float b, float c, float d) {
    float4 v;
    v.x = a;
    v.y = b;
    v.z = c;
    v.w = d;
    return v;
}
#define F4(a, b, c, d) f4((a), (b), (c), (d))

#define GLOBAL
#define LOCAL
#define PRIV
#define DEVFN __device__ static inline
#define BITS2F(b) __builtin_bit_cast(float, (uint)(b))
#define NEG_INF (-__builtin_inff())

__device__ static inline float h2f(uint h) {
    float f;
    unsigned short s = (unsigned short)h;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(s));
    return f;
}
#define H2F(h) h2f(h)
__device__ static inline ushort f2h(float f) {
    unsigned short s;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(s) : "f"(f));
    return s;
}
#define F2H(x) f2h(x)
#define POPC(x) __builtin_popcount(x)

__device__ static inline float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, o, 0x1f);
    return v;
}
__device__ static inline float wmax(float v) {
    for (int o = 16; o > 0; o >>= 1) {
        float w = __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, o, 0x1f);
        v = v > w ? v : w;
    }
    return v;
}
#define WSUM(v) wsum(v)
#define WMAX(v) wmax(v)
#define BARRIER() __nvvm_barrier_sync(0)

// Without CUDA's math library: sqrt is the hardware's correctly rounded one; e^x is
// 2^(x·log2 e) on the hardware's 2^x unit (about 2 ulp); sin and cos first reduce the
// angle to [−π, π] with 2π in three parts (Cody and Waite), since RoPE angles reach
// tens of thousands of radians and the hardware's sine is only accurate near zero.
#define SQRT(x) __nvvm_sqrt_rn_f(x)
#define EXP(x) __nvvm_ex2_approx_f((x) * 1.4426950408889634f)
__device__ static inline float reduce(float t) {
    float k = __builtin_rintf(t * 0.15915494309189535f);
    return ((t - k * 6.28125f) - k * 1.9353071795864769e-3f) - k * 2.5410146811e-11f;
}
#define SIN(x) __nvvm_sin_approx_f(reduce(x))
#define COS(x) __nvvm_cos_approx_f(reduce(x))
#define LOG(x) (__nvvm_lg2_approx_f(x) * 0.6931471805599453f)

#include "kernels.h"

#define IX (BX * BDX + TX)
extern "C" __global__ void k_matmul(const uchar* w, const float* x, float* out, uint ty, uint rows, uint cols, uint n, uint rb) {
    matmul(w, x, out, ty, rows, cols, n, rb, BX * ROWS_PER_GROUP + TX / 32, BY * NB, TX % 32);
}
extern "C" __global__ void k_rms_norm(const float* x, const float* w, float* out, uint cols, float eps) { rms_norm(x, w, out, cols, eps, BX, TX); }
extern "C" __global__ void k_rope(float* x, const float* f, const uint* positions, uint heads, uint hd, uint rd, uint neox, uint n) { if (IX < n) rope(x, f, positions, heads, hd, rd, neox, IX); }
extern "C" __global__ void k_attention(const float* q, const ushort* kc, const ushort* vc, float* out, const uint* masks, uint tree, uint nh, uint nkv, uint hd, uint pos, float scale) {
    __shared__ static float sh[TILE];
    attention(q, kc, vc, out, sh, masks, tree, nh, nkv, hd, pos, scale, BX, BY, TX);
}
extern "C" __global__ void k_bias_add(float* x, const float* b, uint n, uint dim) { bias_add(x, b, n, dim, IX); }
extern "C" __global__ void k_silu_mul(float* g, const float* u, uint n) { silu_mul(g, u, n, IX); }
extern "C" __global__ void k_add(float* x, const float* y, uint n) { add(x, y, n, IX); }
extern "C" __global__ void k_move(float* b, uint from, uint to, uint n) { move(b, from, to, n, IX); }
extern "C" __global__ void k_copy(const float* s, float* d, uint off, uint n) { copy(s, d, off, n, IX); }
extern "C" __global__ void k_copy_h(const float* s, ushort* d, uint off, uint n) { copy_h(s, d, off, n, IX); }
extern "C" __global__ void k_copy16(const ushort* s, ushort* d, uint off, uint n) { copy16(s, d, off, n, IX); }
extern "C" __global__ void k_move16(ushort* b, uint from, uint to, uint n) { move16(b, from, to, n, IX); }
extern "C" __global__ void k_conv_silu(const float* in, uint iw, const float* w, uint taps, float* win, float* out, uint cd, uint rows) { conv_silu(in, iw, w, taps, win, out, cd, rows, IX); }
extern "C" __global__ void k_l2_heads(float* x, uint stride, uint heads, uint hd, float eps) { l2_heads(x, stride, heads, hd, eps, BX, TX); }
extern "C" __global__ void k_delta(const float* x, const float* ba, uint ba_w, const float* al, uint a_w, uint a_off, const float* dt, const float* a, float* state, uint soff,
    float* out, uint rows, uint cd, uint key_dim, uint k_heads, uint hd, uint vd, float scale) {
    delta_rule(x, ba, ba_w, al, a_w, a_off, dt, a, state, soff, out, rows, cd, key_dim, k_heads, hd, vd, scale, BX, TX);
}
extern "C" __global__ void k_silu_gate(float* x, const float* z, uint zw, uint zoff, uint vd, uint n) { silu_gate(x, z, zw, zoff, vd, n, IX); }
extern "C" __global__ void k_split_qgate(const float* full, float* q, float* g, uint hd, uint n) { split_qgate(full, q, g, hd, n, IX); }
extern "C" __global__ void k_sigmoid_mul(float* x, const float* g, uint n) { sigmoid_mul(x, g, n, IX); }
extern "C" __global__ void k_route(const float* logits, uint* ids, float* wts, uint E, uint k, float scale, uint rows) { route(logits, ids, wts, E, k, scale, rows, IX); }
extern "C" __global__ void k_matmul_id(const uchar* w, const float* x, float* out, const uint* ids, uint ty, uint rows, uint cols, uint rb, uint xdiv, uint estride, uint roff) {
    matmul_id(w, x, out, ids, ty, rows, cols, rb, xdiv, estride, roff, BX * ROWS_PER_GROUP + TX / 32, BY, TX % 32);
}
extern "C" __global__ void k_moe_sum(float* x, const float* d, const float* wts, uint k, uint e, uint n) { moe_sum(x, d, wts, k, e, n, IX); }
extern "C" __global__ void k_gated_add(float* x, const float* y, const float* g, uint e, uint n) { gated_add(x, y, g, e, n, IX); }
