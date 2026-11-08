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
        for k in K::ALL.into_iter().chain([K::MatmulMm]) {
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
