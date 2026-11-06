//! A decoder-only transformer (Llama and Qwen2 families), runnable a layer range at a time.
//!
//! Per layer, for hidden state x (one row per token):
//! - h = x + W_o · Attention(RoPE(W_q·n(x) + b_q), RoPE(W_k·n(x) + b_k), W_v·n(x) + b_v)
//! - x' = h + W_down · (SiLU(W_gate·n(h)) ⊙ W_up·n(h))
//!
//! with n(x) = x / √(mean(x²) + ε) ⊙ w (RMSNorm), SiLU(z) = z/(1+e⁻ᶻ), and attention
//! softmax(q·kᵀ/√d)·v over every earlier position. Several query heads share one key/value
//! head (grouped-query attention). RoPE rotates each pair of a head's dimensions by the
//! angle p·θᵢ, θᵢ = base^(−2i/d): Llama pairs neighbours (2i, 2i+1), Qwen2 pairs i with
//! i + d/2 ("NeoX" order).
//!
//! A [`Part`] holds only its layers, plus the embeddings if it is first and the output
//! head if it is last. Its key/value cache lives in a [`Kv`], one per conversation, so a
//! stage keeps each conversation's history and only new tokens ever cross the network.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::gguf::{Header, TensorInfo};
use crate::quant::{dot, QTensor};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub arch: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// Qwen2 rotates (i, i + d/2); Llama rotates (2i, 2i + 1).
    pub rope_neox: bool,
    pub eps: f32,
    pub vocab: usize,
    pub context: usize,
}

impl Config {
    pub fn from_header(h: &Header) -> Result<Self> {
        let arch = h.arch().to_string();
        if !matches!(arch.as_str(), "llama" | "qwen2" | "mistral") {
            bail!("the {arch} architecture is not supported yet (llama and qwen2 are)");
        }
        let n_embd = h.arch_u64("embedding_length").context("no embedding_length")? as usize;
        let n_head = h.arch_u64("attention.head_count").context("no head_count")? as usize;
        let head_dim = h.arch_u64("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head.max(1));
        let vocab = h
            .tensor("token_embd.weight")
            .map(|t| t.shape2().1)
            .context("no token_embd.weight")?;
        Ok(Config {
            n_embd,
            n_layer: h.arch_u64("block_count").context("no block_count")? as usize,
            n_head,
            n_head_kv: h.arch_u64("attention.head_count_kv").map(|v| v as usize).unwrap_or(n_head),
            head_dim,
            rope_dim: h.arch_u64("rope.dimension_count").map(|v| v as usize).unwrap_or(head_dim),
            rope_base: h.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            rope_neox: arch == "qwen2",
            eps: h.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            vocab,
            context: h.arch_u64("context_length").unwrap_or(4096) as usize,
            arch,
        })
    }

    fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// Bytes of key/value cache one token takes in one layer (f32).
    pub fn kv_bytes_per_token_layer(&self) -> u64 {
        (2 * self.kv_dim() * 4) as u64
    }
}

/// Tensors a part needs: its layers, the embeddings if `first`, the output head if `last`.
pub fn needed<'a>(h: &'a Header, layers: &Range<u32>, first: bool, last: bool) -> Vec<&'a TensorInfo> {
    let has_output = h.tensor("output.weight").is_some();
    h.tensors
        .iter()
        .filter(|t| {
            if let Some(rest) = t.name.strip_prefix("blk.") {
                let layer: u32 = rest.split('.').next().and_then(|n| n.parse().ok()).unwrap_or(u32::MAX);
                return layers.contains(&layer);
            }
            match t.name.as_str() {
                "token_embd.weight" => first || (last && !has_output),
                "output_norm.weight" | "output.weight" => last,
                // Llama 3.1+ long-context frequency factors, used by every layer.
                "rope_freqs.weight" => !layers.is_empty(),
                _ => false,
            }
        })
        .collect()
}

struct Block {
    attn_norm: Vec<f32>,
    wq: Arc<QTensor>,
    wk: Arc<QTensor>,
    wv: Arc<QTensor>,
    wo: Arc<QTensor>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

pub struct Part {
    pub cfg: Config,
    pub layers: Range<u32>,
    blocks: Vec<Block>,
    embed: Option<Arc<QTensor>>,
    head: Option<(Vec<f32>, Arc<QTensor>)>,
    /// θᵢ per rotated pair: base^(−2i/d), divided by the model's frequency factor if it
    /// has one (Llama 3.1+ scales its low frequencies for long context).
    inv_freq: Vec<f32>,
    /// The first layers (and maybe the head) on this device's GPU, when it has one.
    gpu: Option<crate::gpu::Layers>,
}

/// One conversation's keys and values for a part's layers.
pub struct Kv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    gpu: Option<crate::gpu::Cache>,
    pub len: usize,
}

impl Kv {
    /// Bytes held, in RAM and on the GPU.
    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).map(|x| x.len() * 4).sum::<usize>() + self.gpu.as_ref().map_or(0, |g| g.bytes())
    }
}

impl Part {
    /// Builds a part from tensors already in memory, by name.
    pub fn new(cfg: Config, layers: Range<u32>, first: bool, last: bool, tensors: &HashMap<String, Arc<QTensor>>) -> Result<Self> {
        let get = |n: &str| tensors.get(n).cloned().with_context(|| format!("missing tensor {n}"));
        let vec = |n: &str| -> Result<Vec<f32>> { Ok(get(n)?.to_f32()) };
        let opt = |n: &str| tensors.get(n).map(|t| t.to_f32());
        let mut blocks = Vec::new();
        for l in layers.clone() {
            let p = |s: &str| format!("blk.{l}.{s}");
            blocks.push(Block {
                attn_norm: vec(&p("attn_norm.weight"))?,
                wq: get(&p("attn_q.weight"))?,
                wk: get(&p("attn_k.weight"))?,
                wv: get(&p("attn_v.weight"))?,
                wo: get(&p("attn_output.weight"))?,
                bq: opt(&p("attn_q.bias")),
                bk: opt(&p("attn_k.bias")),
                bv: opt(&p("attn_v.bias")),
                ffn_norm: vec(&p("ffn_norm.weight"))?,
                gate: get(&p("ffn_gate.weight"))?,
                up: get(&p("ffn_up.weight"))?,
                down: get(&p("ffn_down.weight"))?,
            });
        }
        let embed = if first { Some(get("token_embd.weight")?) } else { None };
        let head = if last {
            let out = tensors.get("output.weight").cloned().map_or_else(|| get("token_embd.weight"), Ok)?;
            Some((vec("output_norm.weight")?, out))
        } else {
            None
        };
        if layers.end as usize > cfg.n_layer {
            bail!("layers {layers:?} are past the model's {}", cfg.n_layer);
        }
        let factors = tensors.get("rope_freqs.weight").map(|t| t.to_f32());
        let inv_freq: Vec<f32> = (0..cfg.rope_dim / 2)
            .map(|i| cfg.rope_base.powf(-2.0 * i as f32 / cfg.rope_dim as f32) / factors.as_ref().and_then(|f| f.get(i)).copied().unwrap_or(1.0))
            .collect();
        let gpu = match crate::gpu::get() {
            Some(g) => Self::on_gpu(g, &cfg, &blocks, head.as_ref(), &inv_freq)?,
            None => None,
        };
        Ok(Part { cfg, layers, blocks, embed, head, inv_freq, gpu })
    }

    /// Layers the GPU's memory holds, leaving room for key/value caches: on unified memory
    /// the weights are shared with the CPU and cost nothing more, so all of them go.
    fn on_gpu(g: &'static crate::gpu::Gpu, cfg: &Config, blocks: &[Block], head: Option<&(Vec<f32>, Arc<QTensor>)>, inv_freq: &[f32]) -> Result<Option<crate::gpu::Layers>> {
        let budget = if g.unified() { u64::MAX } else { (g.memory() as f64 * 0.85) as u64 };
        let weights: Vec<crate::gpu::LayerWeights> = blocks
            .iter()
            .map(|b| crate::gpu::LayerWeights {
                attn_norm: &b.attn_norm,
                wq: &b.wq,
                wk: &b.wk,
                wv: &b.wv,
                wo: &b.wo,
                bias: [b.bq.as_deref(), b.bk.as_deref(), b.bv.as_deref()],
                ffn_norm: &b.ffn_norm,
                gate: &b.gate,
                up: &b.up,
                down: &b.down,
            })
            .collect();
        crate::gpu::Layers::new(g, cfg, &weights, head.map(|(n, o)| (n.as_slice(), o)), inv_freq, budget)
    }

    /// The same part on the CPU only (for checking the GPU against it).
    pub fn without_gpu(mut self) -> Self {
        self.gpu = None;
        self
    }

    /// Where this part runs: layers on the GPU (from the first) and whether the head is.
    pub fn gpu_layers(&self) -> (usize, bool) {
        self.gpu.as_ref().map_or((0, false), |g| (g.len(), g.has_head()))
    }

    pub fn new_kv(&self) -> Kv {
        Kv { k: vec![Vec::new(); self.blocks.len()], v: vec![Vec::new(); self.blocks.len()], gpu: self.gpu.as_ref().map(|g| g.new_cache()), len: 0 }
    }

    pub fn is_first(&self) -> bool {
        self.embed.is_some()
    }
    pub fn is_last(&self) -> bool {
        self.head.is_some()
    }

    /// Hidden states for token ids (n × n_embd).
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let e = self.embed.as_ref().context("this part has no embeddings")?;
        let mut out = vec![0.0; tokens.len() * self.cfg.n_embd];
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= e.rows {
                bail!("token {t} is outside the vocabulary");
            }
            e.dequant_row(t as usize, &mut out[i * self.cfg.n_embd..(i + 1) * self.cfg.n_embd]);
        }
        Ok(out)
    }

    /// Runs this part's layers on `rows` hidden states at positions `pos..`, in place.
    /// The cache is cut back to `pos` first, which drops tokens a speculative pass
    /// proposed and the coordinator then rejected.
    pub fn forward(&self, x: &mut [f32], rows: usize, pos: usize, kv: &mut Kv) -> Result<()> {
        let c = &self.cfg;
        if x.len() != rows * c.n_embd {
            bail!("{} values are not {rows} rows of {}", x.len(), c.n_embd);
        }
        if pos > kv.len {
            bail!("position {pos} skips past the {} cached tokens", kv.len);
        }
        if pos + rows > c.context {
            bail!("{} tokens are past the model's context of {}", pos + rows, c.context);
        }
        let kvd = c.kv_dim();
        let on_gpu = match (&self.gpu, kv.gpu.as_mut()) {
            (Some(g), Some(cache)) => {
                g.forward(c, x, rows, pos, cache)?;
                g.len()
            }
            _ => 0,
        };
        for (b, (kc, vc)) in self.blocks.iter().zip(kv.k.iter_mut().zip(kv.v.iter_mut())).skip(on_gpu) {
            kc.truncate(pos * kvd);
            vc.truncate(pos * kvd);
            let n = rms_norm(x, &b.attn_norm, c.eps);
            let mut q = b.wq.matmul(&n, rows);
            let mut k = b.wk.matmul(&n, rows);
            let mut v = b.wv.matmul(&n, rows);
            add_bias(&mut q, b.bq.as_deref());
            add_bias(&mut k, b.bk.as_deref());
            add_bias(&mut v, b.bv.as_deref());
            for i in 0..rows {
                let p = pos + i;
                rope(&mut q[i * c.n_head * c.head_dim..(i + 1) * c.n_head * c.head_dim], c, p, &self.inv_freq);
                rope(&mut k[i * kvd..(i + 1) * kvd], c, p, &self.inv_freq);
            }
            kc.extend_from_slice(&k);
            vc.extend_from_slice(&v);
            let att = attention(c, &q, kc, vc, rows, pos);
            let o = b.wo.matmul(&att, rows);
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            let n = rms_norm(x, &b.ffn_norm, c.eps);
            let g = b.gate.matmul(&n, rows);
            let u = b.up.matmul(&n, rows);
            let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
            let d = b.down.matmul(&h, rows);
            x.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        }
        kv.len = pos + rows;
        Ok(())
    }

    /// Next-token scores for each row (rows × vocab).
    pub fn logits(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (norm, out) = self.head.as_ref().context("this part has no output head")?;
        if let Some(scores) = self.gpu.as_ref().map(|g| g.logits(&self.cfg, x, rows)).transpose()?.flatten() {
            return Ok(scores);
        }
        let n = rms_norm(x, norm, self.cfg.eps);
        debug_assert_eq!(n.len(), rows * self.cfg.n_embd);
        Ok(out.matmul(&n, rows))
    }
}

fn add_bias(x: &mut [f32], bias: Option<&[f32]>) {
    if let Some(b) = bias {
        for row in x.chunks_exact_mut(b.len()) {
            row.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        }
    }
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(w.len()) {
        let ms = dot(row, row) / row.len() as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(v, w)| v * scale * w));
    }
    out
}

fn rope(x: &mut [f32], c: &Config, pos: usize, inv_freq: &[f32]) {
    let half = c.rope_dim / 2;
    for head in x.chunks_exact_mut(c.head_dim) {
        for (i, f) in inv_freq.iter().enumerate().take(half) {
            let theta = pos as f32 * f;
            let (sin, cos) = theta.sin_cos();
            let (a, b) = if c.rope_neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let (x0, x1) = (head[a], head[b]);
            head[a] = x0 * cos - x1 * sin;
            head[b] = x0 * sin + x1 * cos;
        }
    }
}

/// Causal grouped-query attention for `rows` new tokens at `pos..`, against the cache.
fn attention(c: &Config, q: &[f32], kc: &[f32], vc: &[f32], rows: usize, pos: usize) -> Vec<f32> {
    let (hd, kvd) = (c.head_dim, c.kv_dim());
    let group = c.n_head / c.n_head_kv.max(1);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; rows * c.n_head * hd];
    let mut scores = Vec::new();
    for i in 0..rows {
        let upto = pos + i + 1;
        for h in 0..c.n_head {
            let qh = &q[(i * c.n_head + h) * hd..(i * c.n_head + h + 1) * hd];
            let kvh = h / group.max(1);
            scores.clear();
            scores.extend((0..upto).map(|t| dot(qh, &kc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd]) * scale));
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let o = &mut out[(i * c.n_head + h) * hd..(i * c.n_head + h + 1) * hd];
            for (t, p) in scores.iter().enumerate() {
                let vt = &vc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd];
                let w = p / sum;
                o.iter_mut().zip(vt).for_each(|(a, b)| *a += w * b);
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub use crate::fixtures::tiny_model;
    use crate::quant::GgmlType;

    pub fn load_all(file: &[u8], layers: Range<u32>, first: bool, last: bool) -> Part {
        let h = crate::gguf::parse(file).unwrap();
        let cfg = Config::from_header(&h).unwrap();
        let mut map = HashMap::new();
        for t in needed(&h, &layers, first, last) {
            let (cols, rows) = t.shape2();
            let data = file[t.range.start as usize..t.range.end as usize].to_vec();
            map.insert(t.name.clone(), Arc::new(QTensor::new(t.ty, rows, cols, data).unwrap()));
        }
        Part::new(cfg, layers, first, last, &map).unwrap()
    }

    fn run(parts: &[&Part], kvs: &mut [Kv], tokens: &[u32], pos: usize) -> Vec<f32> {
        let mut x = parts[0].embed(tokens).unwrap();
        for (p, kv) in parts.iter().zip(kvs.iter_mut()) {
            p.forward(&mut x, tokens.len(), pos, kv).unwrap();
        }
        parts.last().unwrap().logits(&x, tokens.len()).unwrap()
    }

    #[test]
    fn a_split_model_computes_what_the_whole_one_does() {
        for neox in [false, true] {
            let file = tiny_model(4, neox);
            let whole = load_all(&file, 0..4, true, true);
            let a = load_all(&file, 0..1, true, false);
            let b = load_all(&file, 1..3, false, false);
            let c = load_all(&file, 3..4, false, true);
            assert!(needed(&crate::gguf::parse(&file).unwrap(), &(1..3), false, false).iter().all(|t| t.name.starts_with("blk.1") || t.name.starts_with("blk.2")));
            let mut kv1 = [whole.new_kv()];
            let mut kv3 = [a.new_kv(), b.new_kv(), c.new_kv()];
            // A prompt, then one token at a time: the cache carries the history.
            let one = run(&[&whole], &mut kv1, &[5, 9, 13, 2], 0);
            let three = run(&[&a, &b, &c], &mut kv3, &[5, 9, 13, 2], 0);
            assert_eq!(one, three);
            let one = run(&[&whole], &mut kv1, &[40], 4);
            let three = run(&[&a, &b, &c], &mut kv3, &[40], 4);
            assert_eq!(one, three);
            // Prompt then step equals the whole sequence at once (the cache is exact).
            let mut fresh = [whole.new_kv()];
            let all = run(&[&whole], &mut fresh, &[5, 9, 13, 2, 40], 0);
            let last = &all[4 * whole.cfg.vocab..];
            assert!(last.iter().zip(&one).all(|(a, b)| (a - b).abs() < 1e-4));
            // Going back drops later tokens: rerunning position 4 gives the same answer.
            let again = run(&[&whole], &mut kv1, &[40], 4);
            assert_eq!(again, one);
        }
    }

    #[test]
    fn the_gpu_computes_what_the_cpu_does() {
        if crate::gpu::get().is_none() {
            eprintln!("no GPU here: skipped");
            return;
        }
        for neox in [false, true] {
            let file = tiny_model(4, neox);
            let gpu = load_all(&file, 0..4, true, true);
            assert_eq!(gpu.gpu_layers(), (4, true), "all of a small model goes on the GPU");
            let cpu = load_all(&file, 0..4, true, true).without_gpu();
            let (mut kg, mut kc) = ([gpu.new_kv()], [cpu.new_kv()]);
            let prompt = [5, 9, 13, 2, 7, 1];
            let close = |a: &[f32], b: &[f32]| {
                let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let top = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!(err <= 0.02 * (1.0 + top), "max error {err} against scores up to {top}");
            };
            close(&run(&[&gpu], &mut kg, &prompt, 0), &run(&[&cpu], &mut kc, &prompt, 0));
            close(&run(&[&gpu], &mut kg, &[40], 6), &run(&[&cpu], &mut kc, &[40], 6));
            // On the GPU itself, results are exact: a batch scores each row as a
            // one-token pass would, and a split model matches the whole one bit for bit.
            let mut fresh = [gpu.new_kv()];
            let all = run(&[&gpu], &mut fresh, &[5, 9, 13, 2, 7, 1, 40], 0);
            let mut steps = [gpu.new_kv()];
            run(&[&gpu], &mut steps, &prompt, 0);
            let one = run(&[&gpu], &mut steps, &[40], 6);
            assert_eq!(&all[6 * gpu.cfg.vocab..], &one[..]);
            let a = load_all(&file, 0..1, true, false);
            let b = load_all(&file, 1..4, false, true);
            let mut ks = [a.new_kv(), b.new_kv()];
            assert_eq!(run(&[&a, &b], &mut ks, &[5, 9, 13, 2, 7, 1, 40], 0), all);
            // Going back cuts the GPU cache too.
            assert_eq!(run(&[&gpu], &mut steps, &[40], 6), one);
        }
    }

    #[test]
    fn a_long_conversation_grows_the_gpu_cache() {
        if crate::gpu::get().is_none() {
            return;
        }
        let file = tiny_model(2, true);
        let gpu = load_all(&file, 0..2, true, true);
        let cpu = load_all(&file, 0..2, true, true).without_gpu();
        // Past the first 256-token allocation and past one attention tile of 1024.
        let prompt: Vec<u32> = (0..1100).map(|i| (i * 7 % 90) as u32 + 1).collect();
        let (mut kg, mut kc) = ([gpu.new_kv()], [cpu.new_kv()]);
        let mut g = Vec::new();
        let mut c = Vec::new();
        for (i, chunk) in prompt.chunks(300).enumerate() {
            g = run(&[&gpu], &mut kg, chunk, i * 300);
            c = run(&[&cpu], &mut kc, chunk, i * 300);
        }
        let v = gpu.cfg.vocab;
        let (g, c) = (&g[g.len() - v..], &c[c.len() - v..]);
        let err = g.iter().zip(c).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(err < 0.05, "{err}");
        assert!(kg[0].bytes() >= 1100 * 2 * 2 * 32 * 4);
    }

    #[test]
    fn rope_rotates_pairs_by_position() {
        let c = Config { arch: "llama".into(), n_embd: 4, n_layer: 1, n_head: 1, n_head_kv: 1, head_dim: 4, rope_dim: 4, rope_base: 10000.0, rope_neox: false, eps: 1e-5, vocab: 1, context: 8 };
        let inv: Vec<f32> = (0..2).map(|i| 10000f32.powf(-2.0 * i as f32 / 4.0)).collect();
        let mut x = [1.0, 0.0, 1.0, 0.0];
        rope(&mut x, &c, 0, &inv);
        assert_eq!(x, [1.0, 0.0, 1.0, 0.0], "position 0 is the identity");
        rope(&mut x, &c, 1, &inv);
        assert!((x[0] - 1f32.cos()).abs() < 1e-6 && (x[1] - 1f32.sin()).abs() < 1e-6, "{x:?}");
        let norm = (x[2] * x[2] + x[3] * x[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "rotations keep length");
        let _ = GgmlType::F32;
    }
}
