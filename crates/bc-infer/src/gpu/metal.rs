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
        for k in K::ALL {
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
