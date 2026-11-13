//! Apple GPUs through Metal. Memory is unified, so weights are not copied: each Metal
//! buffer wraps the very pages the CPU reads (`newBufferWithBytesNoCopy`), and the GPU
//! gets the whole of the model at no extra memory.

use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{anyhow, bail, Result};
use metal::{CompileOptions, ComputePipelineState, MTLResourceOptions, MTLSize};

use super::{Arg, Cmd, K};
use crate::quant::Bytes;

pub struct Buf(metal::Buffer);

pub struct Backend {
    device: metal::Device,
    queue: metal::CommandQueue,
    kernels: HashMap<K, ComputePipelineState>,
}

const PRELUDE: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define GLOBAL device
#define LOCAL threadgroup
#define PRIV thread
#define DEVFN static inline
#define BITS2F(b) as_type<float>((uint)(b))
#define H2F(h) ((float)as_type<half>((ushort)(h)))
#define F4(a, b, c, d) float4((a), (b), (c), (d))
#define POPC(x) popcount(x)
#define WSUM(v) simd_sum(v)
#define WMAX(v) simd_max(v)
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup)
#define NEG_INF (-INFINITY)
#define SQRT(x) precise::sqrt(x)
#define EXP(x) precise::exp(x)
#define SIN(x) precise::sin(x)
#define COS(x) precise::cos(x)
"#;

const ENTRIES: &str = r#"
#define G uint3 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
#define I uint3 gi [[thread_position_in_grid]]
kernel void k_matmul(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& ty [[buffer(3)]], constant uint& rows [[buffer(4)]], constant uint& cols [[buffer(5)]], constant uint& n [[buffer(6)]],
    constant uint& rb [[buffer(7)]], G) { matmul(w, x, out, ty, rows, cols, n, rb, g.x * ROWS_PER_GROUP + sg, g.y * NB, lane); }
// One input row (decoding a token), the product every token waits on. A SIMD group takes
// VR weight rows at once, so each thread loads its 32 inputs once for all of them, and a
// thread group is 4 SIMD groups. Lane l takes groups l, l + 32, ... as in matmul.
// K-quant weights are q·ds − dm with one ds, dm per 32: the − dm part is dm · Σx, summed
// once per group for every row, which leaves one multiply-add per weight.
#define VR 4u
#define XK(k) float4 xv##k = x4(xp, k);
#define QK5(k) s += dot(bytes4(((q[k] >> sh) & 0x0f0f0f0fu) | (((qh[k] >> (hi + 2u * c)) & 0x01010101u) << 4), 0.0f), xv##k);
#define YK(k) float4 yv##k = x4(xp + 32, k);
#define PK(k) { uint v = q[k]; s += dot(bytes4(v & 0x0f0f0f0fu, 0.0f), xv##k); u += dot(bytes4((v >> 4) & 0x0f0f0f0fu, 0.0f), yv##k); }
#define Y6(k) float4 yv##k = x4(xp + 64, k);
#define P6(k, s, u) { uint lo = ql[2 * k] | ((uint)ql[2 * k + 1] << 16), h = qh[2 * k] | ((uint)qh[2 * k + 1] << 16); \
    s += dot(bytes4((lo & 0x0f0f0f0fu) | (((h >> (2u * e)) & 0x03030303u) << 4), 32.0f), xv##k); \
    u += dot(bytes4(((lo >> 4) & 0x0f0f0f0fu) | (((h >> (2u * e + 4u)) & 0x03030303u) << 4), 32.0f), yv##k); }
#define WK(k) s += dot(wv[k], xv##k);
kernel void k_matvec(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& ty [[buffer(3)]], constant uint& rows [[buffer(4)]], constant uint& cols [[buffer(5)]], constant uint& n [[buffer(6)]],
    constant uint& rb [[buffer(7)]], G) {
    uint r0 = (g.x * 4u + sg) * VR;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    bool q5 = ty == T_Q5K;
    // Q4K, what most published models are in: its 32 bytes hold one group in their low
    // halves and the next in their high halves, so a lane takes both groups from one load.
    for (uint pr = lane; ty == T_Q4K && pr < cols / 64u; pr += 32u) {
        device const float* xp = x + pr * 64u;
        XK(0) XK(1) XK(2) XK(3) XK(4) XK(5) XK(6) XK(7)
        YK(0) YK(1) YK(2) YK(3) YK(4) YK(5) YK(6) YK(7)
        float4 xs = ((xv0 + xv1) + (xv2 + xv3)) + ((xv4 + xv5) + (xv6 + xv7));
        float4 ys = ((yv0 + yv1) + (yv2 + yv3)) + ((yv4 + yv5) + (yv6 + yv7));
        float sumx = (xs.x + xs.y) + (xs.z + xs.w), sumy = (ys.x + ys.y) + (ys.z + ys.w);
        uint c = pr % 4u, j0 = (2u * c) & 3u;
#define PROW(j, a) if (r0 + j < rows) { \
            device const uchar* b = w + (ulong)(r0 + j) * rb + (pr / 4u) * 144u; \
            /* Scales and mins of groups 2c and 2c + 1 (ggml's get_scale_min_k4). */ \
            uint qa = b[4u + j0], qb = b[8u + j0], qc = b[12u + j0], ra = b[5u + j0], rb2 = b[9u + j0], rc = b[13u + j0]; \
            float sc0 = (float)(c < 2u ? qa & 63u : (qc & 15u) | ((qa >> 6) << 4)), mn0 = (float)(c < 2u ? qb & 63u : (qc >> 4) | ((qb >> 6) << 4)); \
            float sc1 = (float)(c < 2u ? ra & 63u : (rc & 15u) | ((ra >> 6) << 4)), mn1 = (float)(c < 2u ? rb2 & 63u : (rc >> 4) | ((rb2 >> 6) << 4)); \
            device const uint* q = (device const uint*)(b + 16u + 32u * c); \
            float s = 0.0f, u = 0.0f; \
            PK(0) PK(1) PK(2) PK(3) PK(4) PK(5) PK(6) PK(7) \
            a += f16at(b, 0) * (sc0 * s + sc1 * u) - f16at(b, 2) * (mn0 * sumx + mn1 * sumy); }
        PROW(0u, a0) PROW(1u, a1) PROW(2u, a2) PROW(3u, a3)
#undef PROW
    }
    // Q6K the same way: a block's low-bit bytes hold groups e and e + 2 of a half in their
    // halves, and its high-bit bytes two bits of each. A scale covers 16 weights.
    for (uint pr = lane; ty == T_Q6K && pr < cols / 64u; pr += 32u) {
        uint hh = (pr % 4u) / 2u, e = pr % 2u;
        device const float* xp = x + ((pr / 4u) * 8u + 4u * hh + e) * 32u;
        XK(0) XK(1) XK(2) XK(3) XK(4) XK(5) XK(6) XK(7)
        Y6(0) Y6(1) Y6(2) Y6(3) Y6(4) Y6(5) Y6(6) Y6(7)
#define PROW(j, a) if (r0 + j < rows) { \
            device const uchar* b = w + (ulong)(r0 + j) * rb + (pr / 4u) * 210u; \
            device const ushort* ql = (device const ushort*)(b + 64u * hh + 32u * e); \
            device const ushort* qh = (device const ushort*)(b + 128u + 32u * hh); \
            device const uchar* sc = b + 192u + 8u * hh + 2u * e; \
            float s0 = 0.0f, s1 = 0.0f, u0 = 0.0f, u1 = 0.0f; \
            P6(0, s0, u0) P6(1, s0, u0) P6(2, s0, u0) P6(3, s0, u0) P6(4, s1, u1) P6(5, s1, u1) P6(6, s1, u1) P6(7, s1, u1) \
            a += f16at(b, 208) * (((float)i8of(sc[0]) * s0 + (float)i8of(sc[1]) * s1) + ((float)i8of(sc[4]) * u0 + (float)i8of(sc[5]) * u1)); }
        PROW(0u, a0) PROW(1u, a1) PROW(2u, a2) PROW(3u, a3)
#undef PROW
    }
    for (uint gr = lane; ty != T_Q4K && ty != T_Q6K && gr < cols / 32u; gr += 32u) {
        device const float* xp = x + gr * 32u;
        XK(0) XK(1) XK(2) XK(3) XK(4) XK(5) XK(6) XK(7)
        float4 xs = ((xv0 + xv1) + (xv2 + xv3)) + ((xv4 + xv5) + (xv6 + xv7));
        float sumx = (xs.x + xs.y) + (xs.z + xs.w);
        uint sub = gr % 8u, c = sub / 2u, hi = sub % 2u, sh = 4u * hi;
#define VROW(j, a) if (r0 + j < rows) { \
            device const uchar* row = w + (ulong)(r0 + j) * rb; \
            float s = 0.0f; \
            if (q5) { \
                device const uchar* b = row + (gr / 8u) * 176u; \
                uint qa = b[4u + (sub & 3u)], qb = b[8u + (sub & 3u)], qc = b[12u + (sub & 3u)]; \
                float sc = (float)(sub < 4u ? qa & 63u : (qc & 15u) | ((qa >> 6) << 4)); \
                float mn = (float)(sub < 4u ? qb & 63u : (qc >> 4) | ((qb >> 6) << 4)); \
                device const uint* q = (device const uint*)(b + 48u + 32u * c); \
                device const uint* qh = (device const uint*)(b + 16); \
                QK5(0) QK5(1) QK5(2) QK5(3) QK5(4) QK5(5) QK5(6) QK5(7) \
                a += f16at(b, 0) * sc * s - f16at(b, 2) * mn * sumx; \
            } else { \
                float4 wv[8]; \
                group4(row, ty, gr, wv); \
                WK(0) WK(1) WK(2) WK(3) WK(4) WK(5) WK(6) WK(7) \
                a += s; \
            } }
        VROW(0u, a0) VROW(1u, a1) VROW(2u, a2) VROW(3u, a3)
#undef VROW
    }
#define VOUT(j, a) { float s = simd_sum(a); if (lane == 0u && r0 + j < rows) out[r0 + j] = s; }
    VOUT(0u, a0) VOUT(1u, a1) VOUT(2u, a2) VOUT(3u, a3)
#undef VOUT
}
// Batched product on the matrix units, for batches of a few rows to 64 (a tree of
// guesses). A thread group of 4 SIMD groups computes BM weight rows x BN input rows, 32
// columns (one weight group) at a time: the 128 threads decode the weights and load the
// inputs into threadgroup memory, then each SIMD group multiplies its BM / 4 rows by the
// BN inputs with 8 x 8 matrix products. Out of range rows read zeros. Loops over the
// accumulators are unrolled: an array indexed in a loop that is not lives in slow
// per-thread memory.
#define UNROLL _Pragma("unroll")
// Tile row stride: 32 columns plus 4, so the threads writing one column each of
// 32 rows hit different memory banks.
#define SA 36u
#define MM_KERNEL(NAME, BM, BN)                                                                                      \
kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* out [[buffer(2)]], \
    constant uint& ty [[buffer(3)]], constant uint& rows [[buffer(4)]], constant uint& cols [[buffer(5)]], constant uint& n [[buffer(6)]], \
    constant uint& rb [[buffer(7)]], uint3 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    const uint MR = BM / 32u, MB = BN / 8u, WT = BM * 32u / 128u;                                                    \
    threadgroup float sa[BM * SA], sb[BN * SA];                                                                   \
    uint r0 = g.x * BM, i0 = g.y * BN;                                                                               \
    uint tr = t / (32u / WT), tc = (t % (32u / WT)) * WT;                                                            \
    uint wr = r0 + tr;                                                                                               \
    device const uchar* row = w + (ulong)(wr < rows ? wr : 0u) * rb;                                                 \
    uint mr = sg * (BM / 4u);                                                                                        \
    simdgroup_float8x8 c[MR][MB];                                                                                    \
    UNROLL for (uint a = 0; a < MR; a++) { UNROLL for (uint b = 0; b < MB; b++) c[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); } \
    for (uint gr = 0; gr < cols / 32u; gr++) {                                                                       \
        if (WT == 32u) {                                                                                             \
            float4 wv[8];                                                                                            \
            if (wr < rows) group4(row, ty, gr, wv); else { UNROLL for (uint k = 0; k < 8u; k++) wv[k] = float4(0.0f); } \
            UNROLL for (uint k = 0; k < 8u; k++) *(threadgroup float4*)(sa + tr * SA + k * 4u) = wv[k];             \
        } else {                                                                                                     \
            UNROLL for (uint q = 0; q < WT; q += 4u)                                                                  \
                *(threadgroup float4*)(sa + tr * SA + tc + q) = wr < rows ? four(row, ty, gr, (tc + q) / 4u) : float4(0.0f); \
        }                                                                                                            \
        UNROLL for (uint e = t; e < BN * 8u; e += 128u) {                                                            \
            uint i = e / 8u, q = (e % 8u) * 4u;                                                                      \
            *(threadgroup float4*)(sb + i * SA + q) = i0 + i < n ? *(device const float4*)(x + (ulong)(i0 + i) * cols + gr * 32u + q) : float4(0.0f); \
        }                                                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                            \
        UNROLL for (uint kk = 0; kk < 32u; kk += 8u) {                                                               \
            simdgroup_float8x8 ma[MR], mb[MB];                                                                       \
            UNROLL for (uint a = 0; a < MR; a++) simdgroup_load(ma[a], sa, SA, ulong2(kk, mr + a * 8u));             \
            UNROLL for (uint b = 0; b < MB; b++) simdgroup_load(mb[b], sb, SA, ulong2(kk, b * 8u), true);            \
            UNROLL for (uint a = 0; a < MR; a++) { UNROLL for (uint b = 0; b < MB; b++) simdgroup_multiply_accumulate(c[a][b], ma[a], mb[b], c[a][b]); } \
        }                                                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                            \
    }                                                                                                                \
    /* Each SIMD group stores its corner (BM / 4 x BN) through sa, so edge tiles write only what is in range. */   \
    threadgroup float* sc = sa + sg * (BM / 4u) * BN;                                                                \
    UNROLL for (uint a = 0; a < MR; a++) { UNROLL for (uint b = 0; b < MB; b++) simdgroup_store(c[a][b], sc, BN, ulong2(b * 8u, a * 8u)); } \
    simdgroup_barrier(mem_flags::mem_threadgroup);                                                                  \
    for (uint e = t % 32u; e < (BM / 4u) * BN; e += 32u) {                                                           \
        uint r = mr + e / BN, i = e % BN;                                                                            \
        if (r0 + r < rows && i0 + i < n) out[(ulong)(i0 + i) * rows + r0 + r] = sc[e];                               \
    }                                                                                                                \
}
MM_KERNEL(k_matmul_mm, 128u, 16u)
kernel void k_rms_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], device float* out [[buffer(2)]],
    constant uint& cols [[buffer(3)]], constant float& eps [[buffer(4)]], G) { rms_norm(x, w, out, cols, eps, g.x, lane); }
kernel void k_rope(device float* x [[buffer(0)]], device const float* f [[buffer(1)]], device const uint* positions [[buffer(2)]],
    constant uint& heads [[buffer(3)]], constant uint& hd [[buffer(4)]], constant uint& rd [[buffer(5)]], constant uint& neox [[buffer(6)]],
    constant uint& n [[buffer(7)]], I) { if (gi.x < n) rope(x, f, positions, heads, hd, rd, neox, gi.x); }
kernel void k_attention(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]],
    device float* out [[buffer(3)]], device const uint* masks [[buffer(4)]], constant uint& tree [[buffer(5)]], constant uint& nh [[buffer(6)]],
    constant uint& nkv [[buffer(7)]], constant uint& hd [[buffer(8)]], constant uint& pos [[buffer(9)]], constant float& scale [[buffer(10)]], G) {
    threadgroup float sh[TILE];
    attention(q, kc, vc, out, sh, masks, tree, nh, nkv, hd, pos, scale, g.x, g.y, lane);
}
// The same attention on 128 threads a head: a GPU thread is slow in a row, and at a few
// hundred tokens of context 32 threads a head took a sixth of a token's time. Threads
// share the positions when scoring (four values a load) and take one dimension each when
// summing values; the four SIMD groups' maxima and sums meet in threadgroup memory. The
// head dimension must be a multiple of 4.
kernel void k_attention_w(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]], device const float* vc [[buffer(2)]],
    device float* out [[buffer(3)]], device const uint* masks [[buffer(4)]], constant uint& tree [[buffer(5)]], constant uint& nh [[buffer(6)]],
    constant uint& nkv [[buffer(7)]], constant uint& hd [[buffer(8)]], constant uint& pos [[buffer(9)]], constant float& scale [[buffer(10)]],
    uint3 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sh[TILE];
    threadgroup float red[8];
    uint h = g.x, r = g.y, kvd = nkv * hd, kvh = h / (nh / nkv);
    uint lo = tree != 0u ? masks[2u * r] : 0u, hi = tree != 0u ? masks[2u * r + 1u] : 0u;
    uint upto = tree != 0u ? pos + popcount(lo) + popcount(hi) : pos + r + 1u;
#define SLOT(k) ((tree == 0u || (k) < pos) ? (k) : pos + nth_bit(lo, hi, (k) - pos))
    device const packed_float4* qh = (device const packed_float4*)(q + ((ulong)r * nh + h) * hd);
    float acc0 = 0.0f, acc1 = 0.0f, m = -INFINITY, l = 0.0f;
    for (uint start = 0; start < upto; start += TILE) {
        uint end = min(start + TILE, upto);
        float tmax = -INFINITY;
        for (uint p = start + t; p < end; p += 128u) {
            device const packed_float4* kt = (device const packed_float4*)(kc + (ulong)SLOT(p) * kvd + kvh * hd);
            float4 s4 = float4(0.0f);
            for (uint d = 0; d < hd / 4u; d++) s4 += float4(qh[d]) * float4(kt[d]);
            float s = ((s4.x + s4.y) + (s4.z + s4.w)) * scale;
            sh[p - start] = s;
            tmax = max(tmax, s);
        }
        tmax = simd_max(tmax);
        if (lane == 0u) red[sg] = tmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mn = max(m, max(max(red[0], red[1]), max(red[2], red[3])));
        float corr = mn == -INFINITY ? 1.0f : precise::exp(m - mn);
        float part = 0.0f;
        for (uint p = start + t; p < end; p += 128u) {
            float e = precise::exp(sh[p - start] - mn);
            sh[p - start] = e;
            part += e;
        }
        part = simd_sum(part);
        if (lane == 0u) red[4u + sg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l = l * corr + ((red[4] + red[5]) + (red[6] + red[7]));
        acc0 *= corr;
        acc1 *= corr;
        if (t < hd) {
            for (uint p = start; p < end; p++) {
                device const float* vt = vc + (ulong)SLOT(p) * kvd + kvh * hd;
                acc0 += sh[p - start] * vt[t];
                if (t + 128u < hd) acc1 += sh[p - start] * vt[t + 128u];
            }
        }
        m = mn;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#undef SLOT
    device float* o = out + ((ulong)r * nh + h) * hd;
    if (t < hd) o[t] = acc0 / l;
    if (t + 128u < hd) o[t + 128u] = acc1 / l;
}
kernel void k_bias_add(device float* x [[buffer(0)]], device const float* b [[buffer(1)]], constant uint& n [[buffer(2)]],
    constant uint& dim [[buffer(3)]], I) { bias_add(x, b, n, dim, gi.x); }
kernel void k_silu_mul(device float* g [[buffer(0)]], device const float* u [[buffer(1)]], constant uint& n [[buffer(2)]], I) { silu_mul(g, u, n, gi.x); }
kernel void k_add(device float* x [[buffer(0)]], device const float* y [[buffer(1)]], constant uint& n [[buffer(2)]], I) { add(x, y, n, gi.x); }
kernel void k_move(device float* b [[buffer(0)]], constant uint& from [[buffer(1)]], constant uint& to [[buffer(2)]],
    constant uint& n [[buffer(3)]], I) { move(b, from, to, n, gi.x); }
kernel void k_copy(device const float* s [[buffer(0)]], device float* d [[buffer(1)]], constant uint& off [[buffer(2)]],
    constant uint& n [[buffer(3)]], I) { copy(s, d, off, n, gi.x); }
"#;

impl Backend {
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or_else(|| anyhow!("no Metal device"))?;
        let opts = CompileOptions::new();
        // Exact IEEE arithmetic: the same bits on every run and every Apple GPU.
        opts.set_fast_math_enabled(false);
        let src = format!("{PRELUDE}{}{ENTRIES}", include_str!("kernels.h"));
        let lib = device.new_library_with_source(&src, &opts).map_err(|e| anyhow!("compiling the Metal kernels: {e}"))?;
        let mut kernels = HashMap::new();
        for k in K::ALL.into_iter().chain([K::MatmulMm, K::Matvec, K::AttentionW]) {
            let f = lib.get_function(k.name(), None).map_err(|e| anyhow!("{}: {e}", k.name()))?;
            let p = device.new_compute_pipeline_state_with_function(&f).map_err(|e| anyhow!("{}: {e}", k.name()))?;
            kernels.insert(k, p);
        }
        let queue = device.new_command_queue();
        Ok(Backend { device, queue, kernels })
    }

    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    /// Unified memory: the GPU uses the RAM the weights are already in.
    pub fn memory(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }

    pub fn unified(&self) -> bool {
        self.device.has_unified_memory()
    }

    pub fn weights(&self, data: &Bytes) -> Result<Buf> {
        // SAFETY: `data` is page aligned and padded to whole pages (quant::Bytes); the
        // caller keeps its tensor alive as long as the buffer.
        let buf = self.device.new_buffer_with_bytes_no_copy(data.as_ptr() as *const c_void, data.padded_len() as u64, MTLResourceOptions::StorageModeShared, None);
        Ok(Buf(buf))
    }

    pub fn upload(&self, data: &[f32]) -> Result<Buf> {
        let bytes = std::mem::size_of_val(data).max(4) as u64;
        Ok(Buf(self.device.new_buffer_with_data(data.as_ptr() as *const c_void, bytes, MTLResourceOptions::StorageModeShared)))
    }

    pub fn zeros(&self, floats: usize) -> Result<Buf> {
        Ok(Buf(self.device.new_buffer((floats.max(1) * 4) as u64, MTLResourceOptions::StorageModeShared)))
    }

    pub fn read(&self, b: &Buf, floats: usize) -> Result<Vec<f32>> {
        if (b.0.length() as usize) < floats * 4 {
            bail!("reading {floats} floats from a {}-byte buffer", b.0.length());
        }
        // SAFETY: shared storage, the length was checked, and no command is running.
        Ok(unsafe { std::slice::from_raw_parts(b.0.contents() as *const f32, floats) }.to_vec())
    }

    pub fn run(&self, cmds: &[Cmd]) -> Result<()> {
        metal::objc::rc::autoreleasepool(|| {
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for c in cmds {
                enc.set_compute_pipeline_state(&self.kernels[&c.k]);
                for (i, a) in c.args.iter().enumerate() {
                    match a {
                        Arg::B(b) => enc.set_buffer(i as u64, Some(&b.0), 0),
                        Arg::U(v) => enc.set_bytes(i as u64, 4, v as *const u32 as *const c_void),
                        Arg::F(v) => enc.set_bytes(i as u64, 4, v as *const f32 as *const c_void),
                    }
                }
                enc.dispatch_thread_groups(MTLSize::new(c.grid.0 as u64, c.grid.1 as u64, 1), MTLSize::new(c.threads as u64, 1, 1));
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            match cb.status() {
                metal::MTLCommandBufferStatus::Completed => Ok(()),
                s => bail!("the GPU stopped with status {s:?}"),
            }
        })
    }
}
