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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum K {
    Matmul,
    RmsNorm,
    Rope,
    Attention,
    BiasAdd,
    SiluMul,
    Add,
    Copy,
}

impl K {
    const ALL: [K; 8] = [K::Matmul, K::RmsNorm, K::Rope, K::Attention, K::BiasAdd, K::SiluMul, K::Add, K::Copy];

    fn name(self) -> &'static str {
        match self {
            K::Matmul => "k_matmul",
            K::RmsNorm => "k_rms_norm",
            K::Rope => "k_rope",
            K::Attention => "k_attention",
            K::BiasAdd => "k_bias_add",
            K::SiluMul => "k_silu_mul",
            K::Add => "k_add",
            K::Copy => "k_copy",
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
        Cmd {
            k: K::Matmul,
            grid: (w.rows.div_ceil(ROWS_PER_GROUP), (n as u32).div_ceil(NB)),
            threads: 32 * ROWS_PER_GROUP,
            args: vec![Arg::B(&w.buf), Arg::B(x), Arg::B(out), Arg::U(w.ty), Arg::U(w.rows), Arg::U(w.cols), Arg::U(n as u32), Arg::U(w.row_bytes)],
        }
    }

    /// The GPU's product against the CPU's on every weight format, and its measured speed.
    fn self_check(&self) -> Result<()> {
        use crate::quant::GgmlType::*;
        for ty in [F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4K, Q5K, Q6K] {
            let (rows, cols, n) = (37, 512, 3);
            let t = Arc::new(crate::fixtures::random_tensor(ty, rows, cols, 7));
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
        let (rows, cols) = (8192, 8192);
        let t = Arc::new(crate::fixtures::random_tensor(crate::quant::GgmlType::Q4_0, rows, cols, 1));
        let w = self.weights(&t)?;
        let (bx, out) = (self.backend.upload(&vec![0.5; cols])?, self.backend.zeros(rows)?);
        self.run(&[self.matmul(&w, &bx, &out, 1)])?;
        let reps = 8;
        let start = std::time::Instant::now();
        let cmds: Vec<Cmd> = (0..reps).map(|_| self.matmul(&w, &bx, &out, 1)).collect();
        self.run(&cmds)?;
        Ok(t.data.len() as f64 * reps as f64 / start.elapsed().as_secs_f64())
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
    wq: W,
    wk: W,
    wv: W,
    wo: W,
    bias: [Option<Buf>; 3],
    ffn_norm: Buf,
    gate: W,
    up: W,
    down: W,
}

/// The weights a [`crate::Part`] keeps on the GPU: its first layers, and its output head
/// when everything fits.
pub struct Layers {
    gpu: &'static Gpu,
    layers: Vec<Layer>,
    head: Option<(Buf, W)>,
    inv_freq: Buf,
}

/// One conversation's keys and values on the GPU, per layer, with room for `cap` tokens.
pub struct Cache {
    k: Vec<Buf>,
    v: Vec<Buf>,
    cap: usize,
    bytes: usize,
}

/// What [`Layers::new`] needs of one layer, from the CPU part.
pub struct LayerWeights<'a> {
    pub attn_norm: &'a [f32],
    pub wq: &'a Arc<QTensor>,
    pub wk: &'a Arc<QTensor>,
    pub wv: &'a Arc<QTensor>,
    pub wo: &'a Arc<QTensor>,
    pub bias: [Option<&'a [f32]>; 3],
    pub ffn_norm: &'a [f32],
    pub gate: &'a Arc<QTensor>,
    pub up: &'a Arc<QTensor>,
    pub down: &'a Arc<QTensor>,
}

impl Cache {
    /// Bytes the cache holds on the GPU.
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
        let b = &gpu.backend;
        let mut left = budget;
        let mut on = Vec::new();
        for l in layers {
            let bytes: u64 = [l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down].iter().map(|t| t.data.len() as u64).sum();
            if bytes > left || [l.wq, l.wk, l.wv, l.wo, l.gate, l.up, l.down].iter().any(|t| !t.cols.is_multiple_of(32)) {
                break;
            }
            left -= bytes;
            let [bq, bk, bv] = l.bias.map(|v| v.map(|v| b.upload(v)).transpose());
            on.push(Layer {
                attn_norm: b.upload(l.attn_norm)?,
                wq: gpu.weights(l.wq)?,
                wk: gpu.weights(l.wk)?,
                wv: gpu.weights(l.wv)?,
                wo: gpu.weights(l.wo)?,
                bias: [bq?, bk?, bv?],
                ffn_norm: b.upload(l.ffn_norm)?,
                gate: gpu.weights(l.gate)?,
                up: gpu.weights(l.up)?,
                down: gpu.weights(l.down)?,
            });
        }
        if on.is_empty() {
            return Ok(None);
        }
        let head = match head {
            Some((norm, out)) if on.len() == layers.len() && (out.data.len() as u64) <= left => Some((b.upload(norm)?, gpu.weights(out)?)),
            _ => None,
        };
        Ok(Some(Layers { gpu, layers: on, head, inv_freq: b.upload(inv_freq)? }))
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
        Cache { k: Vec::new(), v: Vec::new(), cap: 0, bytes: 0 }
    }


    /// Grows `c` to hold `need` tokens, keeping its first `keep`.
    fn reserve(&self, cfg: &Config, c: &mut Cache, need: usize, keep: usize) -> Result<()> {
        if need <= c.cap {
            return Ok(());
        }
        let kvd = cfg.n_head_kv * cfg.head_dim;
        let cap = need.max(c.cap * 2).max(256).min(cfg.context.max(need));
        let b = &self.gpu.backend;
        let fresh = |_: usize| b.zeros(cap * kvd);
        let (k, v): (Vec<Buf>, Vec<Buf>) = (0..self.layers.len()).map(|i| Ok((fresh(i)?, fresh(i)?))).collect::<Result<Vec<_>>>()?.into_iter().unzip();
        if keep > 0 && !c.k.is_empty() {
            let n = (keep * kvd) as u32;
            let mut cmds = Vec::new();
            for (old, new) in c.k.iter().zip(&k).chain(c.v.iter().zip(&v)) {
                cmds.push(Cmd { k: K::Copy, grid: (n.div_ceil(256), 1), threads: 256, args: vec![Arg::B(old), Arg::B(new), Arg::U(0), Arg::U(n)] });
            }
            self.gpu.run(&cmds)?;
        }
        *c = Cache { bytes: k.len() * 2 * cap * kvd * 4, k, v, cap };
        Ok(())
    }

    /// Runs the GPU's layers on `rows` hidden states at positions `pos..`, in place.
    pub fn forward(&self, cfg: &Config, x: &mut [f32], rows: usize, pos: usize, cache: &mut Cache) -> Result<()> {
        self.reserve(cfg, cache, pos + rows, pos)?;
        let b = &self.gpu.backend;
        let (e, hd, nh, nkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let ff = self.layers[0].gate.rows as usize;
        let bx = b.upload(x)?;
        let (bn, bq, bk, bv, ba, bo) = (b.zeros(rows * e)?, b.zeros(rows * qd)?, b.zeros(rows * kvd)?, b.zeros(rows * kvd)?, b.zeros(rows * qd)?, b.zeros(rows * e)?);
        let (bg, bu) = (b.zeros(rows * ff)?, b.zeros(rows * ff)?);
        let ew = |n: usize| ((n as u32).div_ceil(256), 1);
        let u = |v: usize| Arg::U(v as u32);
        let mut cmds = Vec::new();
        for (l, (kc, vc)) in self.layers.iter().zip(cache.k.iter().zip(&cache.v)) {
            cmds.push(Cmd { k: K::RmsNorm, grid: (rows as u32, 1), threads: 32, args: vec![Arg::B(&bx), Arg::B(&l.attn_norm), Arg::B(&bn), u(e), Arg::F(cfg.eps)] });
            cmds.push(self.gpu.matmul(&l.wq, &bn, &bq, rows));
            cmds.push(self.gpu.matmul(&l.wk, &bn, &bk, rows));
            cmds.push(self.gpu.matmul(&l.wv, &bn, &bv, rows));
            for ((bias, out), dim) in l.bias.iter().zip([&bq, &bk, &bv]).zip([qd, kvd, kvd]) {
                if let Some(bias) = bias {
                    cmds.push(Cmd { k: K::BiasAdd, grid: ew(rows * dim), threads: 256, args: vec![Arg::B(out), Arg::B(bias), u(rows * dim), u(dim)] });
                }
            }
            for (t, heads) in [(&bq, nh), (&bk, nkv)] {
                let n = rows * heads * (cfg.rope_dim / 2);
                cmds.push(Cmd {
                    k: K::Rope,
                    grid: ew(n),
                    threads: 256,
                    args: vec![Arg::B(t), Arg::B(&self.inv_freq), u(heads), u(hd), u(cfg.rope_dim), u(pos), u(cfg.rope_neox as usize), u(n)],
                });
            }
            for (src, dst) in [(&bk, kc), (&bv, vc)] {
                cmds.push(Cmd { k: K::Copy, grid: ew(rows * kvd), threads: 256, args: vec![Arg::B(src), Arg::B(dst), u(pos * kvd), u(rows * kvd)] });
            }
            cmds.push(Cmd {
                k: K::Attention,
                grid: (nh as u32, rows as u32),
                threads: 32,
                args: vec![Arg::B(&bq), Arg::B(kc), Arg::B(vc), Arg::B(&ba), u(nh), u(nkv), u(hd), u(pos), Arg::F(1.0 / (hd as f32).sqrt())],
            });
            cmds.push(self.gpu.matmul(&l.wo, &ba, &bo, rows));
            cmds.push(Cmd { k: K::Add, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(&bo), u(rows * e)] });
            cmds.push(Cmd { k: K::RmsNorm, grid: (rows as u32, 1), threads: 32, args: vec![Arg::B(&bx), Arg::B(&l.ffn_norm), Arg::B(&bn), u(e), Arg::F(cfg.eps)] });
            cmds.push(self.gpu.matmul(&l.gate, &bn, &bg, rows));
            cmds.push(self.gpu.matmul(&l.up, &bn, &bu, rows));
            cmds.push(Cmd { k: K::SiluMul, grid: ew(rows * ff), threads: 256, args: vec![Arg::B(&bg), Arg::B(&bu), u(rows * ff)] });
            cmds.push(self.gpu.matmul(&l.down, &bg, &bo, rows));
            cmds.push(Cmd { k: K::Add, grid: ew(rows * e), threads: 256, args: vec![Arg::B(&bx), Arg::B(&bo), u(rows * e)] });
        }
        self.gpu.run(&cmds)?;
        x.copy_from_slice(&b.read(&bx, rows * e)?);
        Ok(())
    }

    /// Next-token scores for each row, if the head is on the GPU.
    pub fn logits(&self, cfg: &Config, x: &[f32], rows: usize) -> Result<Option<Vec<f32>>> {
        let Some((norm, out)) = &self.head else { return Ok(None) };
        let b = &self.gpu.backend;
        let (bx, bn, bl) = (b.upload(x)?, b.zeros(rows * cfg.n_embd)?, b.zeros(rows * out.rows as usize)?);
        self.gpu.run(&[
            Cmd { k: K::RmsNorm, grid: (rows as u32, 1), threads: 32, args: vec![Arg::B(&bx), Arg::B(norm), Arg::B(&bn), Arg::U(cfg.n_embd as u32), Arg::F(cfg.eps)] },
            self.gpu.matmul(out, &bn, &bl, rows),
        ])?;
        Ok(Some(b.read(&bl, rows * out.rows as usize)?))
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
