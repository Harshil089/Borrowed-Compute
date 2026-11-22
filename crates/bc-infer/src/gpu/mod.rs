//! The engine on a graphics card: Metal on a Mac, CUDA on an NVIDIA card elsewhere.
//!
//! Decoding reads every weight once per token, so its speed is memory bandwidth, and a
//! GPU has several times a CPU's (an M2's GPU reaches about 90 GB/s of its 100, a desktop
//! RTX card 500 to 1000). So a part runs its layers on the GPU whenever one is there and
//! has room, all of a layer at once (norms, products, RoPE, attention), with the key/value
//! cache on the card: one round trip to the GPU per pass, not one per product. Layers
//! that do not fit the card's memory run on the CPU after the ones that do.
//!
//! Every product keeps the weights in their ggml block format; the kernels are one C
//! source (`kernels.h`) both backends compile: Metal at startup, CUDA ahead of time
//! into `kernels.ptx`. `BC_INFER_GPU=0` turns the GPU off.

use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Result};

use crate::model::Config;
use crate::quant::QTensor;

#[cfg(target_os = "macos")]
mod metal;
#[cfg(target_os = "macos")]
use metal as backend;
#[cfg(not(target_os = "macos"))]
mod cuda;
#[cfg(not(target_os = "macos"))]
use cuda as backend;

pub use backend::Buf;

/// Rows per matrix-product thread group (`NB` in kernels.h).
const NB: u32 = 8;
/// Weight rows per matrix-product thread group (`ROWS_PER_GROUP`).
const ROWS_PER_GROUP: u32 = 4;
const MAX_HEAD: usize = 256;
/// Input rows from which a product runs on the matrix units (`k_matmul_mm`): 3 to 4
/// times faster for batches than `k_matmul`, but it sums in a different order, so a
/// batch row is within rounding of a one-row pass, not bit for bit. On a Mac one row runs
/// `k_matvec`: on an M2, 70 GB/s of Q4K or Q6K weights where `k_matmul` read 21 and 37.
const MM_FROM: usize = 2;
/// Weight rows per `k_matmul_mm` thread group (`BM`).
const MM_ROWS: u32 = 128;
/// Weight rows per `k_matvec` thread group (4 SIMD groups of `VR`).
const MV_ROWS: u32 = 16;
/// Input rows per `k_matmul_mm` thread group (`BN`).
const MM_BATCH: u32 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum K {
    Matmul,
    /// The batched product on Apple's matrix units (Metal only).
    MatmulMm,
    /// The one-row product, several weight rows a thread (Metal only).
    Matvec,
    RmsNorm,
    Rope,
    Attention,
    /// Attention on 128 threads a head (Metal only).
    AttentionW,
    BiasAdd,
    SiluMul,
    Add,
    Copy,
    Move,
    ConvSilu,
    L2Heads,
    Delta,
    SiluGate,
    SplitQGate,
    SigmoidMul,
    Route,
    MatmulId,
    MoeSum,
    GatedAdd,
}

impl K {
    const ALL: [K; 19] = [
        K::Matmul,
        K::RmsNorm,
        K::Rope,
        K::Attention,
        K::BiasAdd,
        K::SiluMul,
        K::Add,
        K::Copy,
        K::Move,
        K::ConvSilu,
        K::L2Heads,
        K::Delta,
        K::SiluGate,
        K::SplitQGate,
        K::SigmoidMul,
        K::Route,
        K::MatmulId,
        K::MoeSum,
        K::GatedAdd,
    ];

    fn name(self) -> &'static str {
        match self {
            K::Matmul => "k_matmul",
            K::MatmulMm => "k_matmul_mm",
            K::Matvec => "k_matvec",
            K::RmsNorm => "k_rms_norm",
            K::Rope => "k_rope",
            K::Attention => "k_attention",
            K::AttentionW => "k_attention_w",
            K::BiasAdd => "k_bias_add",
            K::SiluMul => "k_silu_mul",
            K::Add => "k_add",
            K::Copy => "k_copy",
            K::Move => "k_move",
            K::ConvSilu => "k_conv_silu",
            K::L2Heads => "k_l2_heads",
            K::Delta => "k_delta",
            K::SiluGate => "k_silu_gate",
            K::SplitQGate => "k_split_qgate",
            K::SigmoidMul => "k_sigmoid_mul",
            K::Route => "k_route",
            K::MatmulId => "k_matmul_id",
            K::MoeSum => "k_moe_sum",
            K::GatedAdd => "k_gated_add",
        }
    }
}

pub enum Arg<'a> {
    B(&'a Buf),
    U(u32),
    F(f32),
}

pub struct Cmd<'a> {
    k: K,
    grid: (u32, u32),
    threads: u32,
    args: Vec<Arg<'a>>,
}

/// The device's GPU, if it has one the engine can use.
pub struct Gpu {
    backend: backend::Backend,
    /// One pass at a time: the GPU runs them in turn anyway, and CUDA's one stream must
    /// not interleave two passes' waits.
    lock: Mutex<()>,
    pub name: String,
}

/// The GPU, set up and checked against the CPU once; `None` without one (or with
/// `BC_INFER_GPU=0`, or when its kernels do not compute what the CPU does).
pub fn get() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| {
        if std::env::var("BC_INFER_GPU").is_ok_and(|v| v == "0") {
            return None;
        }
        let gpu = match backend::Backend::new() {
            Ok(b) => Gpu { name: b.name(), backend: b, lock: Mutex::new(()) },
            Err(e) => {
                if std::env::var("BC_INFER_GPU").is_ok_and(|v| v == "debug") {
                    eprintln!("bc-infer: no GPU: {e:#}");
                }
                return None;
            }
        };
        match gpu.self_check() {
            Ok(()) => Some(gpu),
            Err(e) => {
                eprintln!("bc-infer: not using the GPU ({}): {e:#}", gpu.name);
                None
            }
        }
    })
    .as_ref()
}

impl Gpu {
    /// Bytes of weights the card can take: free memory on a discrete card, the GPU's
    /// working-set size on unified memory, where weights cost nothing extra.
    pub fn memory(&self) -> u64 {
        self.backend.memory()
    }

    pub fn unified(&self) -> bool {
        self.backend.unified()
    }

    fn run(&self, cmds: &[Cmd]) -> Result<()> {
        let _one = self.lock.lock().unwrap();
        self.backend.run(cmds)
    }

    fn weights(&self, t: &Arc<QTensor>) -> Result<W> {
        Ok(W { buf: self.backend.weights(&t.data)?, ty: t.ty.id(), rows: t.rows as u32, cols: t.cols as u32, row_bytes: t.row_bytes() as u32, _keep: t.clone() })
    }

    fn matmul<'a>(&self, w: &'a W, x: &'a Buf, out: &'a Buf, n: usize) -> Cmd<'a> {
        self.matmul_rows(w, x, out, n, w.rows)
    }

    /// [`Gpu::matmul`] against the first `rows` weight rows only.
    fn matmul_rows<'a>(&self, w: &'a W, x: &'a Buf, out: &'a Buf, n: usize, rows: u32) -> Cmd<'a> {
        let args = vec![Arg::B(&w.buf), Arg::B(x), Arg::B(out), Arg::U(w.ty), Arg::U(rows), Arg::U(w.cols), Arg::U(n as u32), Arg::U(w.row_bytes)];
        if cfg!(target_os = "macos") && n >= MM_FROM {
            return Cmd { k: K::MatmulMm, grid: (rows.div_ceil(MM_ROWS), (n as u32).div_ceil(MM_BATCH)), threads: 128, args };
        }
        if cfg!(target_os = "macos") {
            return Cmd { k: K::Matvec, grid: (rows.div_ceil(MV_ROWS), 1), threads: 128, args };
        }
        Cmd {
            k: K::Matmul,
            grid: (rows.div_ceil(ROWS_PER_GROUP), (n as u32).div_ceil(NB)),
            threads: 32 * ROWS_PER_GROUP,
            args,
        }
    }

    /// The GPU's product against the CPU's on every weight format, and its measured speed.
    fn self_check(&self) -> Result<()> {
        use crate::quant::GgmlType::*;
        for ty in [F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4K, Q5K, Q6K] {
            let (rows, cols) = (37, 512);
            let t = Arc::new(crate::fixtures::random_tensor(ty, rows, cols, 7));
            // One row and a batch run different kernels.
            for n in [1, 3] {
                let x: Vec<f32> = (0..n * cols).map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0).collect();
                let w = self.weights(&t)?;
                let (bx, out) = (self.backend.upload(&x)?, self.backend.zeros(n * rows)?);
                self.run(&[self.matmul(&w, &bx, &out, n)])?;
                let got = self.backend.read(&out, n * rows)?;
                let wf = t.to_f32();
                for i in 0..n {
                    for r in 0..rows {
                        let want: f64 = (0..cols).map(|c| wf[r * cols + c] as f64 * x[i * cols + c] as f64).sum();
                        let g = got[i * rows + r] as f64;
                        if (g - want).abs() > 1e-3 * (1.0 + want.abs()) {
                            bail!("{ty:?} row {r}: GPU {g}, CPU {want}");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[doc(hidden)]
    /// Bytes a second of a plain copy: the most any kernel could read at.
    pub fn copy_bandwidth(&self) -> Result<f64> {
        let n = 64 << 20;
        let (a, b) = (self.backend.zeros(n)?, self.backend.zeros(n)?);
        let cmd = || Cmd { k: K::Copy, grid: ((n as u32).div_ceil(256), 1), threads: 256, args: vec![Arg::B(&a), Arg::B(&b), Arg::U(0), Arg::U(n as u32)] };
        self.run(&[cmd()])?;
        let start = std::time::Instant::now();
        self.run(&(0..8).map(|_| cmd()).collect::<Vec<_>>())?;
        Ok((n * 8 * 8) as f64 / start.elapsed().as_secs_f64())
    }

    /// Bytes a second the GPU reads weights at in a decoding product (for the planner).
    pub fn bandwidth(&self) -> Result<f64> {
        self.product_rate(crate::quant::GgmlType::Q4K, 1)
    }

    #[doc(hidden)]
    /// Bytes of weights of format `ty` a second in a product with `n` input rows.
    pub fn product_rate(&self, ty: crate::quant::GgmlType, n: usize) -> Result<f64> {
        let (rows, cols) = (8192, 8192);
        let t = Arc::new(crate::fixtures::random_tensor(ty, rows, cols, 1));
        let w = self.weights(&t)?;
        let (bx, out) = (self.backend.upload(&vec![0.5; cols * n])?, self.backend.zeros(rows * n)?);
        self.run(&[self.matmul(&w, &bx, &out, n)])?;
        let reps = 8;
        let cmds: Vec<Cmd> = (0..reps).map(|_| self.matmul(&w, &bx, &out, n)).collect();
        // The best of three: another program's burst must not make a device look slow.
        let mut best = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            self.run(&cmds)?;
            best = best.min(start.elapsed().as_secs_f64());
        }
        Ok(t.data.len() as f64 * reps as f64 / best)
    }
}

/// A weight matrix on the GPU, keeping its tensor alive (Metal reads its very memory).
struct W {
    buf: Buf,
    ty: u32,
    rows: u32,
    cols: u32,
    row_bytes: u32,
    _keep: Arc<QTensor>,
}

struct Layer {
    attn_norm: Buf,
    mix: Mix,
    ffn_norm: Buf,
    ffn: Ffn,
}

enum Ffn {
    Dense { gate: W, up: W, down: W },
    Moe { router: W, gate: Option<W>, up: W, down: W, ff: usize, shared: Option<[W; 4]> },
}

enum Mix {
    Attn {
        wq: W,
        wk: W,
        wv: W,
        wo: W,
        bias: [Option<Buf>; 3],
        q_norm: Option<Buf>,
        k_norm: Option<Buf>,
        gated: bool,
    },
    Delta {
        qkv: W,
        z: Option<W>,
        beta: W,
        alpha: Option<W>,
        conv: Buf,
        dt: Buf,
        a: Buf,
        norm: Buf,
        out: W,
    },
}

/// The weights a [`crate::Part`] keeps on the GPU: its first layers, and its output head
/// when everything fits.
pub struct Layers {
    gpu: &'static Gpu,
    layers: Vec<Layer>,
    head: Option<(Buf, W)>,
    inv_freq: Buf,
    /// Working buffers for a pass, kept between passes: allocating eight buffers on the
    /// card for every token costs time a token cannot spare. Grows to the largest batch.
    scratch: std::sync::Mutex<Option<Scratch>>,
    /// Floats per row of the DeltaNet buffers: [mixed input, convolved, gate z, β(α),
    /// separate α, output]; zeros without DeltaNet layers.
    delta_w: [usize; 6],
    /// Widest dense or shared-expert feed-forward, and [experts, used, expert width] for
    /// a mixture of experts (zeros without).
    ff: usize,
    moe_w: [usize; 3],
}

/// Room for `rows` rows of every intermediate result of a layer.
struct Scratch {
    rows: usize,
    n: Buf,
    q: Buf,
    k: Buf,
    v: Buf,
    att: Buf,
    o: Buf,
    gate: Buf,
    up: Buf,
    /// Qwen3.5 attention: W_q's queries with their gates, and the gates.
    qg: Buf,
    g: Buf,
    /// DeltaNet: see `Layers::delta_w`.
    d: [Buf; 6],
    /// Experts: router scores, chosen ids, their weights, gate and up per (row, slot),
    /// each slot's output, the shared expert's gate.
    m: [Buf; 7],
}

/// One conversation's keys and values on the GPU, per layer, with room for `cap` tokens;
/// a DeltaNet layer's running state instead (convolution window, then state matrices).
pub struct Cache {
    gpu: &'static Gpu,
    kvd: usize,
    k: Vec<Buf>,
    v: Vec<Buf>,
    /// Per layer: its DeltaNet state, `None` for attention layers.
    state: Vec<Option<Buf>>,
    cap: usize,
    bytes: usize,
}

/// What [`Layers::new`] needs of one layer, from the CPU part.
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub mix: MixWeights<'a>,
    pub ffn_norm: &'a [f32],
    pub ffn: FfnWeights<'a>,
}

pub enum FfnWeights<'a> {
    Dense { gate: &'a Arc<QTensor>, up: &'a Arc<QTensor>, down: &'a Arc<QTensor> },
    /// See `model::Experts`; `shared` is [gate scale, gate, up, down].
    Moe { router: &'a Arc<QTensor>, gate: Option<&'a Arc<QTensor>>, up: &'a Arc<QTensor>, down: &'a Arc<QTensor>, ff: usize, shared: Option<[&'a Arc<QTensor>; 4]> },
}

pub enum MixWeights<'a> {
    Attn {
        wq: &'a Arc<QTensor>,
        wk: &'a Arc<QTensor>,
        wv: &'a Arc<QTensor>,
        wo: &'a Arc<QTensor>,
        bias: [Option<&'a [f32]>; 3],
        q_norm: Option<&'a [f32]>,
        k_norm: Option<&'a [f32]>,
        gated: bool,
    },
    Delta {
        qkv: &'a Arc<QTensor>,
        z: Option<&'a Arc<QTensor>>,
        beta: &'a Arc<QTensor>,
        alpha: Option<&'a Arc<QTensor>>,
        conv: &'a [f32],
        dt: &'a [f32],
        a: &'a [f32],
        norm: &'a [f32],
        out: &'a Arc<QTensor>,
    },
}

impl Cache {
    /// Bytes the cache holds on the GPU.
    /// See [`crate::Kv::keep`]: moves the accepted rows' keys and values down to
    /// consecutive slots, in order, every layer in one trip to the GPU.
    pub fn keep(&mut self, base: usize, rows: &[u32]) -> Result<()> {
        if self.state.iter().any(Option::is_some) {
            bail!("this model keeps a running state and cannot drop guessed tokens");
        }
        let n = self.kvd as u32;
        let mut cmds = Vec::new();
        for buf in self.k.iter().chain(&self.v) {
            for (i, &r) in rows.iter().enumerate() {
                if r as usize != i {
                    let (from, to) = (((base + r as usize) * self.kvd) as u32, ((base + i) * self.kvd) as u32);
                    cmds.push(Cmd { k: K::Move, grid: (n.div_ceil(256), 1), threads: 256, args: vec![Arg::B(buf), Arg::U(from), Arg::U(to), Arg::U(n)] });
                }
            }
        }
        if !cmds.is_empty() {
            self.gpu.run(&cmds)?;
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Layers {
    /// Puts as many of `layers` (in order) on the GPU as `budget` bytes hold, and the head
    /// if all of them fit. `None` if not even one layer does or the model's shape is one
    /// the kernels do not take.
    pub fn new(gpu: &'static Gpu, cfg: &Config, layers: &[LayerWeights], head: Option<(&[f32], &Arc<QTensor>)>, inv_freq: &[f32], budget: u64) -> Result<Option<Self>> {
        if cfg.head_dim > MAX_HEAD || !cfg.n_embd.is_multiple_of(32) || cfg.n_head_kv == 0 || !cfg.n_head.is_multiple_of(cfg.n_head_kv) {
            return Ok(None);
        }
        // The delta rule runs a thread per state column, the convolution keeps 8 inputs.
        if cfg.delta.as_ref().is_some_and(|d| d.head > 1024 || d.conv > 9 || d.conv < 2) {
            return Ok(None);
        }
        let b = &gpu.backend;
        let mut left = budget;
        let mut on = Vec::new();
        let mut delta_w = [0usize; 6];
        let (mut ff, mut moe_w) = (0usize, [0usize; 3]);
        for l in layers {
            let mats: Vec<&Arc<QTensor>> = match &l.mix {
                MixWeights::Attn { wq, wk, wv, wo, .. } => vec![wq, wk, wv, wo],
                MixWeights::Delta { qkv, z, beta, alpha, out, .. } => [Some(*qkv), *z, Some(*beta), *alpha, Some(*out)].into_iter().flatten().collect(),
            };
            let ffn_mats: Vec<&Arc<QTensor>> = match &l.ffn {
                FfnWeights::Dense { gate, up, down } => vec![gate, up, down],
                FfnWeights::Moe { router, gate, up, down, shared, .. } => [Some(*router), *gate, Some(*up), Some(*down)].into_iter().flatten().chain(shared.iter().flatten().copied()).collect(),
            };
            let mats: Vec<&Arc<QTensor>> = mats.into_iter().chain(ffn_mats).collect();
            let bytes: u64 = mats.iter().map(|t| t.data.len() as u64).sum();
            if bytes > left || mats.iter().any(|t| !t.cols.is_multiple_of(32)) {
                break;
            }
            left -= bytes;
            let mix = match &l.mix {
                MixWeights::Attn { wq, wk, wv, wo, bias, q_norm, k_norm, gated } => {
                    let [bq, bk, bv] = bias.map(|v| v.map(|v| b.upload(v)).transpose());
                    Mix::Attn {
                        wq: gpu.weights(wq)?,
                        wk: gpu.weights(wk)?,
                        wv: gpu.weights(wv)?,
                        wo: gpu.weights(wo)?,
                        bias: [bq?, bk?, bv?],
                        q_norm: q_norm.map(|w| b.upload(w)).transpose()?,
                        k_norm: k_norm.map(|w| b.upload(w)).transpose()?,
                        gated: *gated,
                    }
                }
                MixWeights::Delta { qkv, z, beta, alpha, conv, dt, a, norm, out } => {
                    let d = cfg.delta.as_ref().expect("DeltaNet weights in a model with them");
                    let (cd, vd) = (2 * d.k_heads * d.head + d.v_heads * d.head, d.v_heads * d.head);
                    let widths = [qkv.rows, cd, vd, beta.rows, d.v_heads, vd];
                    for (w, n) in delta_w.iter_mut().zip(widths) {
                        *w = (*w).max(n);
                    }
                    Mix::Delta {
                        qkv: gpu.weights(qkv)?,
                        z: z.map(|z| gpu.weights(z)).transpose()?,
                        beta: gpu.weights(beta)?,
                        alpha: alpha.map(|a| gpu.weights(a)).transpose()?,
                        conv: b.upload(conv)?,
                        dt: b.upload(dt)?,
                        a: b.upload(a)?,
                        norm: b.upload(norm)?,
                        out: gpu.weights(out)?,
                    }
                }
            };
            let ffn = match &l.ffn {
                FfnWeights::Dense { gate, up, down } => {
                    ff = ff.max(gate.rows);
                    Ffn::Dense { gate: gpu.weights(gate)?, up: gpu.weights(up)?, down: gpu.weights(down)? }
                }
                FfnWeights::Moe { router, gate, up, down, ff: width, shared } => {
                    let m = cfg.moe.as_ref().expect("experts in a model with them");
                    moe_w = [moe_w[0].max(m.experts), moe_w[1].max(m.used), moe_w[2].max(*width)];
                    if let Some([_, g, _, _]) = shared {
                        ff = ff.max(g.rows);
                    }
                    Ffn::Moe {
                        router: gpu.weights(router)?,
                        gate: gate.map(|g| gpu.weights(g)).transpose()?,
                        up: gpu.weights(up)?,
                        down: gpu.weights(down)?,
                        ff: *width,
                        shared: match shared {
                            Some([w, g, u, d]) => Some([gpu.weights(w)?, gpu.weights(g)?, gpu.weights(u)?, gpu.weights(d)?]),
                            None => None,
                        },
                    }
                }
            };
            on.push(Layer { attn_norm: b.upload(l.attn_norm)?, mix, ffn_norm: b.upload(l.ffn_norm)?, ffn });
        }
        if on.is_empty() {
            return Ok(None);
        }
        let head = match head {
            Some((norm, out)) if on.len() == layers.len() && (out.data.len() as u64) <= left => Some((b.upload(norm)?, gpu.weights(out)?)),
            _ => None,
        };
        Ok(Some(Layers { gpu, layers: on, head, inv_freq: b.upload(inv_freq)?, scratch: std::sync::Mutex::new(None), delta_w, ff, moe_w }))
    }

    /// Layers on the GPU, from the start of the part.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    pub fn new_cache(&self) -> Cache {
        Cache { gpu: self.gpu, kvd: 0, k: Vec::new(), v: Vec::new(), state: Vec::new(), cap: 0, bytes: 0 }
    }

    /// Floats of a DeltaNet layer's state: the convolution window, then the matrices.
    fn state_floats(cfg: &Config) -> (usize, usize) {
        let d = cfg.delta.as_ref().expect("a model with DeltaNet layers");
        let cd = 2 * d.k_heads * d.head + d.v_heads * d.head;
        ((d.conv - 1) * cd, d.v_heads * d.head * d.head)
    }

    /// Grows `c` to hold `need` tokens, keeping its first `keep`; at `keep` 0 every
    /// DeltaNet state starts over.
    fn reserve(&self, cfg: &Config, c: &mut Cache, need: usize, keep: usize) -> Result<()> {
        let b = &self.gpu.backend;
        if keep == 0 || c.state.len() != self.layers.len() {
            c.state = self
                .layers
                .iter()
                .map(|l| match l.mix {
                    Mix::Delta { .. } => {
                        let (w, m) = Self::state_floats(cfg);
                        b.zeros(w + m).map(Some)
                    }
                    Mix::Attn { .. } => Ok(None),
                })
                .collect::<Result<_>>()?;
        }
        let state_bytes = if c.state.iter().any(Option::is_some) {
            let (w, m) = Self::state_floats(cfg);
            c.state.iter().flatten().count() * (w + m) * 4
        } else {
            0
        };
        if need <= c.cap {
            return Ok(());
        }
        let kvd = cfg.n_head_kv * cfg.head_dim;
        let cap = need.max(c.cap * 2).max(256).min(cfg.context.max(need));
        // DeltaNet layers keep no keys or values: a float each, never read.
        let fresh = |i: usize| b.zeros(if matches!(self.layers[i].mix, Mix::Attn { .. }) { cap * kvd } else { 1 });
        let (k, v): (Vec<Buf>, Vec<Buf>) = (0..self.layers.len()).map(|i| Ok((fresh(i)?, fresh(i)?))).collect::<Result<Vec<_>>>()?.into_iter().unzip();
        if keep > 0 && !c.k.is_empty() {
            let n = (keep * kvd) as u32;
            let mut cmds = Vec::new();
            for (i, (old, new)) in c.k.iter().zip(&k).chain(c.v.iter().zip(&v)).enumerate() {
                if matches!(self.layers[i % self.layers.len()].mix, Mix::Attn { .. }) {
                    cmds.push(Cmd { k: K::Copy, grid: (n.div_ceil(256), 1), threads: 256, args: vec![Arg::B(old), Arg::B(new), Arg::U(0), Arg::U(n)] });
                }
            }
            self.gpu.run(&cmds)?;
        }
        let attn = self.layers.iter().filter(|l| matches!(l.mix, Mix::Attn { .. })).count();
        let state = std::mem::take(&mut c.state);
        *c = Cache { gpu: self.gpu, kvd, bytes: attn * 2 * cap * kvd * 4 + state_bytes, k, v, state, cap };
        Ok(())
    }

    /// Runs the GPU's layers on `rows` hidden states at positions `pos..`, in place.
    /// One row per entry of `positions`, row i at `positions[i]`; with `masks`, a tree
    /// (see `Part::forward_tree`) whose slots start at the first of the pair.
    pub fn forward(&self, cfg: &Config, x: &mut [f32], pos: usize, positions: &[u32], masks: Option<(usize, &[u64])>, cache: &mut Cache) -> Result<()> {
        let rows = positions.len();
        self.reserve(cfg, cache, pos + rows, pos)?;
        let b = &self.gpu.backend;
        let (e, hd, nh, nkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let ff = self.ff;
        let bx = b.upload(x)?;
        // Integers ride in float buffers bit for bit; the kernels read them as uint.
        let bpos = b.upload(&positions.iter().map(|&p| f32::from_bits(p)).collect::<Vec<_>>())?;
        let bits: Vec<f32> = masks.map_or(&[0][..], |m| m.1).iter().flat_map(|&m| [f32::from_bits(m as u32), f32::from_bits((m >> 32) as u32)]).collect();
        let bmask = b.upload(&bits)?;
        let tree = masks.is_some() as usize;
        // For a tree the attention kernel's position is where the tree's slots start.
        let seen = masks.map_or(pos, |m| m.0);
        // Every kernel writes all of its output, so reused buffers need no clearing.
        let mut held = self.scratch.lock().unwrap();
        if held.as_ref().is_none_or(|s| s.rows < rows) {
            let r = rows.max(held.as_ref().map_or(0, |s| s.rows));
            let dw = self.delta_w;
            *held = Some(Scratch {
                rows: r,
                n: b.zeros(r * e)?,
                q: b.zeros(r * qd)?,
                k: b.zeros(r * kvd)?,
                v: b.zeros(r * kvd)?,
                att: b.zeros(r * qd)?,
                o: b.zeros(r * e)?,
                gate: b.zeros(r * ff)?,
                up: b.zeros(r * ff)?,
                qg: b.zeros(r * 2 * qd)?,
                g: b.zeros(r * qd)?,
                d: [b.zeros(r * dw[0])?, b.zeros(r * dw[1])?, b.zeros(r * dw[2])?, b.zeros(r * dw[3])?, b.zeros(r * dw[4])?, b.zeros(r * dw[5])?],
                m: {
                    let [ex, k, w] = self.moe_w;
                    [b.zeros(r * ex)?, b.zeros(r * k)?, b.zeros(r * k)?, b.zeros(r * k * w)?, b.zeros(r * k * w)?, b.zeros(r * k * e)?, b.zeros(r)?]
                },
            });
        }
        let s = held.as_ref().unwrap();
        let (bn, bq, bk, bv, ba, bo, bg, bu) = (&s.n, &s.q, &s.k, &s.v, &s.att, &s.o, &s.gate, &s.up);
        let ew = |n: usize| ((n as u32).div_ceil(256), 1);
        let u = |v: usize| Arg::U(v as u32);
        // RMSNorm of `n` rows of `cols`, weights `w`.
        fn norm<'a>(x: &'a Buf, w: &'a Buf, out: &'a Buf, n: usize, cols: usize, eps: f32) -> Cmd<'a> {
            Cmd { k: K::RmsNorm, grid: (n as u32, 1), threads: 32, args: vec![Arg::B(x), Arg::B(w), Arg::B(out), Arg::U(cols as u32), Arg::F(eps)] }
        }
        let mut cmds = Vec::new();
        for ((l, (kc, vc)), state) in self.layers.iter().zip(cache.k.iter().zip(&cache.v)).zip(&cache.state) {
            cmds.push(norm(&bx, &l.attn_norm, bn, rows, e, cfg.eps));
            match &l.mix {
                Mix::Attn { wq, wk, wv, wo, bias, q_norm, k_norm, gated } => {
                    if *gated {
                        cmds.push(self.gpu.matmul(wq, bn, &s.qg, rows));
                        cmds.push(Cmd { k: K::SplitQGate, grid: ew(rows * qd), threads: 256, args: vec![Arg::B(&s.qg), Arg::B(bq), Arg::B(&s.g), u(hd), u(rows * qd)] });
                    } else {
                        cmds.push(self.gpu.matmul(wq, bn, bq, rows));
                    }
                    cmds.push(self.gpu.matmul(wk, bn, bk, rows));
                    cmds.push(self.gpu.matmul(wv, bn, bv, rows));
                    for ((bias, out), dim) in bias.iter().zip([bq, bk, bv]).zip([qd, kvd, kvd]) {
                        if let Some(bias) = bias {
                            cmds.push(Cmd { k: K::BiasAdd, grid: ew(rows * dim), threads: 256, args: vec![Arg::B(out), Arg::B(bias), u(rows * dim), u(dim)] });
                        }
                    }
                    // Per head, in place: each lane reads its values before writing them.
                    for (w, t, heads) in [(q_norm, bq, nh), (k_norm, bk, nkv)] {
                        if let Some(w) = w {
                            cmds.push(norm(t, w, t, rows * heads, hd, cfg.eps));
                        }
                    }
                    for (t, heads) in [(bq, nh), (bk, nkv)] {
                        let n = rows * heads * (cfg.rope_dim / 2);
                        cmds.push(Cmd {
                            k: K::Rope,
                            grid: ew(n),
                            threads: 256,
                            args: vec![Arg::B(t), Arg::B(&self.inv_freq), Arg::B(&bpos), u(heads), u(hd), u(cfg.rope_dim), u(cfg.rope_neox as usize), u(n)],
                        });
                    }
                    for (src, dst) in [(bk, kc), (bv, vc)] {
                        cmds.push(Cmd { k: K::Copy, grid: ew(rows * kvd), threads: 256, args: vec![Arg::B(src), Arg::B(dst), u(pos * kvd), u(rows * kvd)] });
                    }
                    let wide = cfg!(target_os = "macos") && hd.is_multiple_of(4);
                    cmds.push(Cmd {
                        k: if wide { K::AttentionW } else { K::Attention },
                        grid: (nh as u32, rows as u32),
                        threads: if wide { 128 } else { 32 },
                        args: vec![Arg::B(bq), Arg::B(kc), Arg::B(vc), Arg::B(ba), Arg::B(&bmask), u(tree), u(nh), u(nkv), u(hd), u(seen), Arg::F(1.0 / (hd as f32).sqrt())],
                    });
                    if *gated {
                        cmds.push(Cmd { k: K::SigmoidMul, grid: ew(rows * qd), threads: 256, args: vec![Arg::B(ba), Arg::B(&s.g), u(rows * qd)] });
                    }
                    cmds.push(self.gpu.matmul(wo, ba, bo, rows));
                }
                Mix::Delta { qkv, z, beta, alpha, conv, dt, a, norm: nw, out } => {
                    let d = cfg.delta.as_ref().expect("DeltaNet layers in a model with them");
                    let state = state.as_ref().expect("a DeltaNet layer's state");
                    let (kd, vd, sh) = (d.k_heads * d.head, d.v_heads * d.head, d.head);
                    let cd = 2 * kd + vd;
                    let [mixed, convd, zb, bab, alb, dout] = &s.d;
                    cmds.push(self.gpu.matmul(qkv, bn, mixed, rows));
                    if let Some(z) = z {
                        cmds.push(self.gpu.matmul(z, bn, zb, rows));
                    }
                    cmds.push(self.gpu.matmul(beta, bn, bab, rows));
                    if let Some(alpha) = alpha {
                        cmds.push(self.gpu.matmul(alpha, bn, alb, rows));
                    }
                    let (taps, qkv_w) = (d.conv, qkv.rows as usize);
                    cmds.push(Cmd { k: K::ConvSilu, grid: ew(cd), threads: 256, args: vec![Arg::B(mixed), u(qkv_w), Arg::B(conv), u(taps), Arg::B(state), Arg::B(convd), u(cd), u(rows)] });
                    cmds.push(Cmd { k: K::L2Heads, grid: ((rows * 2 * d.k_heads) as u32, 1), threads: 32, args: vec![Arg::B(convd), u(cd), u(2 * d.k_heads), u(sh), Arg::F(cfg.eps)] });
                    let ba_w = beta.rows as usize;
                    let (al, a_w, a_off) = if alpha.is_some() { (alb, d.v_heads, 0) } else { (bab, ba_w, d.v_heads) };
                    let soff = (taps - 1) * cd;
                    cmds.push(Cmd {
                        k: K::Delta,
                        grid: (d.v_heads as u32, 1),
                        threads: sh as u32,
                        args: vec![
                            Arg::B(convd), Arg::B(bab), u(ba_w), Arg::B(al), u(a_w), u(a_off), Arg::B(dt), Arg::B(a), Arg::B(state), u(soff), Arg::B(dout),
                            u(rows), u(cd), u(kd), u(d.k_heads), u(sh), u(vd), Arg::F(1.0 / (sh as f32).sqrt()),
                        ],
                    });
                    cmds.push(norm(dout, nw, dout, rows * d.v_heads, sh, cfg.eps));
                    let (zsrc, zw, zoff) = if z.is_some() { (zb, vd, 0) } else { (mixed, qkv_w, cd) };
                    cmds.push(Cmd { k: K::SiluGate, grid: ew(rows * vd), threads: 256, args: vec![Arg::B(dout), Arg::B(zsrc), u(zw), u(zoff), u(vd), u(rows * vd)] });
                    cmds.push(self.gpu.matmul(out, dout, bo, rows));
                }
            }
            cmds.push(Cmd { k: K::Add, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(bo), u(rows * e)] });
            cmds.push(norm(&bx, &l.ffn_norm, bn, rows, e, cfg.eps));
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    let ff = gate.rows as usize;
                    cmds.push(self.gpu.matmul(gate, bn, bg, rows));
                    cmds.push(self.gpu.matmul(up, bn, bu, rows));
                    cmds.push(Cmd { k: K::SiluMul, grid: ew(rows * ff), threads: 256, args: vec![Arg::B(bg), Arg::B(bu), u(rows * ff)] });
                    cmds.push(self.gpu.matmul(down, bg, bo, rows));
                    cmds.push(Cmd { k: K::Add, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(bo), u(rows * e)] });
                }
                Ffn::Moe { router, gate, up, down, ff: width, shared } => {
                    let m = cfg.moe.as_ref().expect("experts in a model with them");
                    let (k, w) = (m.used, *width);
                    let [logits, ids, wts, gb, ub, db, sg] = &s.m;
                    // Every slot (row, chosen expert) is its own product, by the expert's id.
                    // Every slot (row, chosen expert) is its own product, by the expert's id.
                    #[allow(clippy::too_many_arguments)]
                    fn id<'a>(wt: &'a W, x: &'a Buf, out: &'a Buf, ids: &'a Buf, rows_out: usize, slots: usize, xdiv: usize, estride: usize, roff: usize) -> Cmd<'a> {
                        Cmd {
                            k: K::MatmulId,
                            grid: ((rows_out as u32).div_ceil(ROWS_PER_GROUP), slots as u32),
                            threads: 32 * ROWS_PER_GROUP,
                            args: vec![Arg::B(&wt.buf), Arg::B(x), Arg::B(out), Arg::B(ids), Arg::U(wt.ty), Arg::U(rows_out as u32), Arg::U(wt.cols), Arg::U(wt.row_bytes), Arg::U(xdiv as u32), Arg::U(estride as u32), Arg::U(roff as u32)],
                        }
                    }
                    let slots = rows * k;
                    cmds.push(self.gpu.matmul(router, bn, logits, rows));
                    cmds.push(Cmd { k: K::Route, grid: ew(rows), threads: 256, args: vec![Arg::B(logits), Arg::B(ids), Arg::B(wts), u(m.experts), u(k), Arg::F(m.scale), u(rows)] });
                    match gate {
                        Some(gate) => {
                            cmds.push(id(gate, bn, gb, ids, w, slots, k, w, 0));
                            cmds.push(id(up, bn, ub, ids, w, slots, k, w, 0));
                        }
                        None => {
                            cmds.push(id(up, bn, gb, ids, w, slots, k, 2 * w, 0));
                            cmds.push(id(up, bn, ub, ids, w, slots, k, 2 * w, w));
                        }
                    }
                    cmds.push(Cmd { k: K::SiluMul, grid: ew(rows * k * w), threads: 256, args: vec![Arg::B(gb), Arg::B(ub), u(rows * k * w)] });
                    cmds.push(id(down, gb, db, ids, e, slots, 1, e, 0));
                    cmds.push(Cmd { k: K::MoeSum, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(db), Arg::B(wts), u(k), u(e), u(rows * e)] });
                    if let Some([sw, sgate, sup, sdown]) = shared {
                        let sff = sgate.rows as usize;
                        cmds.push(self.gpu.matmul(sw, bn, sg, rows));
                        cmds.push(self.gpu.matmul(sgate, bn, bg, rows));
                        cmds.push(self.gpu.matmul(sup, bn, bu, rows));
                        cmds.push(Cmd { k: K::SiluMul, grid: ew(rows * sff), threads: 256, args: vec![Arg::B(bg), Arg::B(bu), u(rows * sff)] });
                        cmds.push(self.gpu.matmul(sdown, bg, bo, rows));
                        cmds.push(Cmd { k: K::GatedAdd, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(bo), Arg::B(sg), u(e), u(rows * e)] });
                    }
                }
            }
        }
        self.gpu.run(&cmds)?;
        x.copy_from_slice(&b.read(&bx, rows * e)?);
        Ok(())
    }

    /// Next-token scores for the first `upto` tokens (at most the vocabulary), if the
    /// head is on the GPU.
    pub fn logits(&self, cfg: &Config, x: &[f32], rows: usize, upto: usize) -> Result<Option<Vec<f32>>> {
        let Some((norm, out)) = &self.head else { return Ok(None) };
        let b = &self.gpu.backend;
        let (bx, bn, bl) = (b.upload(x)?, b.zeros(rows * cfg.n_embd)?, b.zeros(rows * upto)?);
        self.gpu.run(&[
            Cmd { k: K::RmsNorm, grid: (rows as u32, 1), threads: 32, args: vec![Arg::B(&bx), Arg::B(norm), Arg::B(&bn), Arg::U(cfg.n_embd as u32), Arg::F(cfg.eps)] },
            self.gpu.matmul_rows(out, &bn, &bl, rows, upto.min(out.rows as usize) as u32),
        ])?;
        Ok(Some(b.read(&bl, rows * upto.min(out.rows as usize))?))
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    #[test]
    fn the_cuda_kernels_are_built_from_the_current_source() {
        let source = [include_str!("cuda.cu"), include_str!("kernels.h")].concat();
        let hash = hex::encode(Sha256::digest(source.as_bytes()));
        let ptx = include_str!("kernels.ptx");
        assert!(ptx.starts_with(&format!("// source sha256 {hash} ")), "kernels.ptx is stale: run crates/bc-infer/src/gpu/build-ptx.sh");
        assert_eq!(ptx.matches(".entry").count(), super::K::ALL.len(), "one PTX entry per kernel");
    }
}
