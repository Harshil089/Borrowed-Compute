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
//! Qwen3.5 (`qwen35`) differs in three ways:
//! - Three layers in four replace attention with a gated DeltaNet: a short causal
//!   convolution over W·n(x), then per head a state matrix S that decays by e^g, learns
//!   k → v with rate β (S += k (β(v − Sᵀk))ᵀ) and is read with q; the result is RMS-normed
//!   per head, gated by SiLU(z) and projected back. Its memory is S and the last few
//!   convolution inputs, not one key and value per token.
//! - Attention layers normalise q and k per head, rotate only the first quarter of each
//!   head, and gate their output with σ(gate), the gate coming from W_q as well.
//! - The norm before the feed-forward is called `post_attention_norm`.
//!
//! Qwen3.5 MoE (`qwen35moe`) has the same layers with a mixture of experts for the
//! feed-forward: a router scores every expert, the `used` likeliest run (softmax, top-k,
//! weights renormalised to sum to one) and their outputs are summed by weight, plus a
//! shared expert scaled by σ(w·n(h)). Each token reads only its experts' weights.
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
    /// Qwen3.5's gated DeltaNet layers, `None` for models with attention in every layer.
    pub delta: Option<Delta>,
    /// Experts per layer and how many each token uses, for a mixture of experts.
    pub moe: Option<Moe>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Moe {
    pub experts: usize,
    pub used: usize,
    /// Factor on the chosen experts' weights; 0 or 1 leaves them as they are.
    pub scale: f32,
}

/// Shape of the gated DeltaNet layers (`qwen35`).
#[derive(Debug, Clone, PartialEq)]
pub struct Delta {
    /// Which layers are DeltaNet layers; the others are gated attention.
    pub layers: Vec<bool>,
    /// Convolution taps, the current input included.
    pub conv: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    /// Dimension of each key, query and value head (one size for all three).
    pub head: usize,
}

impl Delta {
    fn key_dim(&self) -> usize {
        self.k_heads * self.head
    }
    fn value_dim(&self) -> usize {
        self.v_heads * self.head
    }
    /// Channels of the convolution: queries, keys, then values.
    fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

impl Config {
    pub fn from_header(h: &Header) -> Result<Self> {
        let arch = h.arch().to_string();
        if !matches!(arch.as_str(), "llama" | "qwen2" | "mistral" | "qwen35" | "qwen35moe") {
            bail!("the {arch} architecture is not supported yet (llama, qwen2, qwen35 and qwen35moe are)");
        }
        let qwen35 = matches!(arch.as_str(), "qwen35" | "qwen35moe");
        // Multi-token-prediction blocks come after the model's own layers and are not run.
        let n_layer = (h.arch_u64("block_count").context("no block_count")? - h.arch_u64("nextn_predict_layers").unwrap_or(0)) as usize;
        let moe = match h.arch_u64("expert_count") {
            Some(experts) if experts > 1 => {
                let used = h.arch_u64("expert_used_count").context("no expert_used_count")?;
                if used == 0 || used > experts {
                    bail!("{used} of {experts} experts a token");
                }
                Some(Moe { experts: experts as usize, used: used as usize, scale: h.arch_f64("expert_weights_scale").unwrap_or(0.0) as f32 })
            }
            _ => None,
        };
        let delta = if qwen35 {
            let need = |k: &str| h.arch_u64(k).with_context(|| format!("no {k}")).map(|v| v as usize);
            let (k_heads, v_heads, head) = (need("ssm.group_count")?, need("ssm.time_step_rank")?, need("ssm.state_size")?);
            if need("ssm.inner_size")? != v_heads * head || k_heads == 0 || v_heads % k_heads != 0 || need("ssm.conv_kernel")? < 2 {
                bail!("a DeltaNet shape this engine does not take");
            }
            let every = h.arch_u64("full_attention_interval").unwrap_or(4).max(1) as usize;
            let layers = match h.get(&format!("{arch}.attention.recurrent_layers")).and_then(crate::gguf::Value::as_array) {
                Some(a) => a.iter().take(n_layer).map(|v| matches!(v, crate::gguf::Value::Bool(true)) || v.as_u64() == Some(1)).collect(),
                None => (0..n_layer).map(|i| (i + 1) % every != 0).collect(),
            };
            Some(Delta { layers, conv: need("ssm.conv_kernel")?, k_heads, v_heads, head })
        } else {
            None
        };
        let n_embd = h.arch_u64("embedding_length").context("no embedding_length")? as usize;
        let n_head = h.arch_u64("attention.head_count").context("no head_count")? as usize;
        let head_dim = h.arch_u64("attention.key_length").map(|v| v as usize).unwrap_or(n_embd / n_head.max(1));
        let vocab = h
            .tensor("token_embd.weight")
            .map(|t| t.shape2().1)
            .context("no token_embd.weight")?;
        Ok(Config {
            n_embd,
            n_layer,
            n_head,
            n_head_kv: h.arch_u64("attention.head_count_kv").map(|v| v as usize).unwrap_or(n_head),
            head_dim,
            rope_dim: h.arch_u64("rope.dimension_count").map(|v| v as usize).unwrap_or(head_dim),
            rope_base: h.arch_f64("rope.freq_base").unwrap_or(10000.0) as f32,
            rope_neox: arch == "qwen2" || qwen35,
            eps: h.arch_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32,
            vocab,
            context: h.arch_u64("context_length").unwrap_or(4096) as usize,
            arch,
            delta,
            moe,
        })
    }

    /// True when some layers keep a running state instead of every token's keys: such a
    /// model's cache cannot go back to an earlier position, so it takes no guessing.
    pub fn recurrent(&self) -> bool {
        self.delta.as_ref().is_some_and(|d| d.layers.iter().any(|&r| r))
    }

    fn is_delta(&self, layer: usize) -> bool {
        self.delta.as_ref().is_some_and(|d| d.layers.get(layer).copied().unwrap_or(false))
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
    mix: Mix,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

/// The feed-forward: SiLU(W_gate·n) ⊙ W_up·n through W_down, once or per expert.
enum Ffn {
    Dense { gate: Arc<QTensor>, up: Arc<QTensor>, down: Arc<QTensor> },
    Moe(Experts),
}

struct Experts {
    /// Scores every expert (rows: experts).
    router: Arc<QTensor>,
    /// Every expert's W_gate and W_up stacked (rows: expert e's at e·ff..), or both in one
    /// stack (e's gate rows then its up rows, 2·ff a expert).
    gate: Option<Arc<QTensor>>,
    up: Arc<QTensor>,
    /// Every expert's W_down (rows: e·n_embd..).
    down: Arc<QTensor>,
    ff: usize,
    /// The shared expert, and the one-row W that scales it by σ(w·n).
    shared: Option<[Arc<QTensor>; 4]>,
}

/// What mixes tokens in a layer.
enum Mix {
    Attn(Attn),
    Delta(DeltaNet),
}

struct Attn {
    wq: Arc<QTensor>,
    wk: Arc<QTensor>,
    wv: Arc<QTensor>,
    wo: Arc<QTensor>,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    /// Qwen3.5: per-head RMSNorm weights for q and k, and W_q giving each head's output
    /// gate after its query.
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gated: bool,
}

struct DeltaNet {
    /// q, k, v (the convolution's input), then z, the output gate: one matrix or two.
    qkv: Arc<QTensor>,
    z: Option<Arc<QTensor>>,
    /// β, then α, per value head: one matrix or two.
    beta: Arc<QTensor>,
    alpha: Option<Arc<QTensor>>,
    /// Convolution taps, `conv` per channel, oldest input first.
    conv: Vec<f32>,
    dt: Vec<f32>,
    /// −e^A per value head: with softplus(α + dt) it gives the log decay g.
    a: Vec<f32>,
    norm: Vec<f32>,
    out: Arc<QTensor>,
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
    /// Microseconds the last pass spent on the GPU's layers and on the CPU's, so a
    /// lending device can show where its time goes.
    split_us: [std::sync::atomic::AtomicU64; 2],
}

/// One conversation's keys and values for a part's layers.
pub struct Kv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// Per DeltaNet layer: the last `conv - 1` convolution inputs, then the state matrices
    /// (each stored transposed, row j = column j of S). Empty for attention layers.
    state: Vec<Vec<f32>>,
    gpu: Option<crate::gpu::Cache>,
    pub len: usize,
}

impl Kv {
    /// Keeps, of the rows a tree pass placed at slots `base..`, only `rows` (ascending,
    /// the accepted path), moved to `base, base + 1, …`, and drops everything after them:
    /// the cache then holds exactly the accepted text, as if it had been fed in a line.
    pub fn keep(&mut self, base: usize, rows: &[u32]) -> Result<()> {
        if rows.windows(2).any(|w| w[0] >= w[1]) || rows.iter().any(|&r| base + r as usize >= self.len) {
            bail!("rows {rows:?} at {base} are not an ascending path within the {} cached tokens", self.len);
        }
        // A running state holds every token fed; it can only keep all of them.
        if self.state.iter().any(|s| !s.is_empty()) && (base + rows.len() != self.len || rows.iter().enumerate().any(|(i, &r)| r as usize != i)) {
            bail!("this model keeps a running state and cannot drop guessed tokens");
        }
        if let Some(g) = self.gpu.as_mut() {
            g.keep(base, rows)?;
        }
        for c in self.k.iter_mut().chain(self.v.iter_mut()) {
            if c.is_empty() {
                continue;
            }
            let width = c.len() / self.len;
            for (n, &r) in rows.iter().enumerate() {
                let (from, to) = ((base + r as usize) * width, (base + n) * width);
                c.copy_within(from..from + width, to);
            }
            c.truncate((base + rows.len()) * width);
        }
        self.len = base + rows.len();
        Ok(())
    }

    /// Bytes held, in RAM and on the GPU.
    pub fn bytes(&self) -> usize {
        self.k.iter().chain(&self.v).chain(&self.state).map(|x| x.len() * 4).sum::<usize>() + self.gpu.as_ref().map_or(0, |g| g.bytes())
    }
}

impl Part {
    /// Builds a part from tensors already in memory, by name.
    pub fn new(cfg: Config, layers: Range<u32>, first: bool, last: bool, tensors: &HashMap<String, Arc<QTensor>>) -> Result<Self> {
        let get = |n: &str| tensors.get(n).cloned().with_context(|| format!("missing tensor {n}"));
        let vec = |n: &str| -> Result<Vec<f32>> { Ok(get(n)?.to_f32()) };
        let opt = |n: &str| tensors.get(n).map(|t| t.to_f32());
        let mut blocks = Vec::new();
        let qwen35 = cfg.delta.is_some();
        let e = cfg.n_embd;
        for l in layers.clone() {
            let p = |s: &str| format!("blk.{l}.{s}");
            let mix = if cfg.is_delta(l as usize) {
                let (qkv, z) = match tensors.get(&p("attn_qkv_gate.weight")) {
                    Some(t) => (t.clone(), None),
                    None => (get(&p("attn_qkv.weight"))?, Some(get(&p("attn_gate.weight"))?)),
                };
                let (beta, alpha) = match tensors.get(&p("ssm_beta_alpha.weight")) {
                    Some(t) => (t.clone(), None),
                    None => (get(&p("ssm_beta.weight"))?, Some(get(&p("ssm_alpha.weight"))?)),
                };
                Mix::Delta(DeltaNet {
                    qkv,
                    z,
                    beta,
                    alpha,
                    conv: vec(&p("ssm_conv1d.weight"))?,
                    dt: vec(&p("ssm_dt.bias"))?,
                    a: vec(&p("ssm_a"))?,
                    norm: vec(&p("ssm_norm.weight"))?,
                    out: get(&p("ssm_out.weight"))?,
                })
            } else {
                Mix::Attn(Attn {
                    wq: get(&p("attn_q.weight"))?,
                    wk: get(&p("attn_k.weight"))?,
                    wv: get(&p("attn_v.weight"))?,
                    wo: get(&p("attn_output.weight"))?,
                    bq: opt(&p("attn_q.bias")),
                    bk: opt(&p("attn_k.bias")),
                    bv: opt(&p("attn_v.bias")),
                    q_norm: opt(&p("attn_q_norm.weight")),
                    k_norm: opt(&p("attn_k_norm.weight")),
                    gated: qwen35,
                })
            };
            blocks.push(Block {
                attn_norm: vec(&p("attn_norm.weight"))?,
                mix,
                ffn_norm: vec(&p(if qwen35 { "post_attention_norm.weight" } else { "ffn_norm.weight" }))?,
                ffn: match &cfg.moe {
                    None => Ffn::Dense { gate: get(&p("ffn_gate.weight"))?, up: get(&p("ffn_up.weight"))?, down: get(&p("ffn_down.weight"))? },
                    Some(m) => {
                        let (gate, up) = match tensors.get(&p("ffn_gate_up_exps.weight")) {
                            Some(t) => (None, t.clone()),
                            None => (Some(get(&p("ffn_gate_exps.weight"))?), get(&p("ffn_up_exps.weight"))?),
                        };
                        let down = get(&p("ffn_down_exps.weight"))?;
                        let router = get(&p("ffn_gate_inp.weight"))?;
                        let ff = down.cols;
                        let stack = if gate.is_some() { ff } else { 2 * ff };
                        if router.rows != m.experts || down.rows != m.experts * e || up.rows != m.experts * stack || gate.as_ref().is_some_and(|g| g.rows != up.rows) {
                            bail!("layer {l}'s experts do not match the model's {} of width {e}", m.experts);
                        }
                        let shared = match tensors.get(&p("ffn_up_shexp.weight")) {
                            Some(up) => Some([get(&p("ffn_gate_inp_shexp.weight"))?, get(&p("ffn_gate_shexp.weight"))?, up.clone(), get(&p("ffn_down_shexp.weight"))?]),
                            None => None,
                        };
                        Ffn::Moe(Experts { router, gate, up, down, ff, shared })
                    }
                },
            });
        }
        if let Some(d) = &cfg.delta {
            for b in &blocks {
                if let Mix::Delta(n) = &b.mix {
                    let fused = n.z.is_none() as usize * d.value_dim();
                    let ba = if n.alpha.is_none() { 2 * d.v_heads } else { d.v_heads };
                    if n.qkv.rows != d.conv_dim() + fused || n.beta.rows != ba || n.conv.len() != d.conv_dim() * d.conv || n.dt.len() != d.v_heads || n.a.len() != d.v_heads || n.norm.len() != d.head {
                        bail!("a DeltaNet layer's tensors do not match the model's shape");
                    }
                }
            }
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
        Ok(Part { cfg, layers, blocks, embed, head, inv_freq, gpu, split_us: Default::default() })
    }

    /// Layers the GPU's memory holds, leaving room for key/value caches: on unified memory
    /// the weights are shared with the CPU and cost nothing more, so all of them go.
    fn on_gpu(g: &'static crate::gpu::Gpu, cfg: &Config, blocks: &[Block], head: Option<&(Vec<f32>, Arc<QTensor>)>, inv_freq: &[f32]) -> Result<Option<crate::gpu::Layers>> {
        let budget = if g.unified() { u64::MAX } else { (g.memory() as f64 * 0.85) as u64 };
        let weights: Vec<crate::gpu::LayerWeights> = blocks
            .iter()
            .map(|b| crate::gpu::LayerWeights {
                attn_norm: &b.attn_norm,
                mix: match &b.mix {
                    Mix::Attn(a) => crate::gpu::MixWeights::Attn {
                        wq: &a.wq,
                        wk: &a.wk,
                        wv: &a.wv,
                        wo: &a.wo,
                        bias: [a.bq.as_deref(), a.bk.as_deref(), a.bv.as_deref()],
                        q_norm: a.q_norm.as_deref(),
                        k_norm: a.k_norm.as_deref(),
                        gated: a.gated,
                    },
                    Mix::Delta(d) => crate::gpu::MixWeights::Delta {
                        qkv: &d.qkv,
                        z: d.z.as_ref(),
                        beta: &d.beta,
                        alpha: d.alpha.as_ref(),
                        conv: &d.conv,
                        dt: &d.dt,
                        a: &d.a,
                        norm: &d.norm,
                        out: &d.out,
                    },
                },
                ffn_norm: &b.ffn_norm,
                ffn: match &b.ffn {
                    Ffn::Dense { gate, up, down } => crate::gpu::FfnWeights::Dense { gate, up, down },
                    Ffn::Moe(x) => crate::gpu::FfnWeights::Moe {
                        router: &x.router,
                        gate: x.gate.as_ref(),
                        up: &x.up,
                        down: &x.down,
                        ff: x.ff,
                        shared: x.shared.as_ref().map(|[w, g, u, d]| [w, g, u, d]),
                    },
                },
            })
            .collect();
        crate::gpu::Layers::new(g, cfg, &weights, head.map(|(n, o)| (n.as_slice(), o)), inv_freq, budget)
    }

    /// Milliseconds the last [`Part::forward`] spent on the GPU's layers and on the CPU's.
    pub fn last_split_ms(&self) -> (f32, f32) {
        use std::sync::atomic::Ordering::Relaxed;
        (self.split_us[0].load(Relaxed) as f32 / 1000.0, self.split_us[1].load(Relaxed) as f32 / 1000.0)
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
        Kv {
            k: vec![Vec::new(); self.blocks.len()],
            v: vec![Vec::new(); self.blocks.len()],
            state: self.blocks.iter().map(|b| self.fresh_state(b)).collect(),
            gpu: self.gpu.as_ref().map(|g| g.new_cache()),
            len: 0,
        }
    }

    /// A DeltaNet layer's state before any token: zeros.
    fn fresh_state(&self, b: &Block) -> Vec<f32> {
        match (&b.mix, &self.cfg.delta) {
            (Mix::Delta(_), Some(d)) => vec![0.0; (d.conv - 1) * d.conv_dim() + d.v_heads * d.head * d.head],
            _ => Vec::new(),
        }
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
        self.forward_tree(x, rows, pos, None, kv)
    }

    /// [`Part::forward`] for a tree of rows: `tree[i]` is node i's parent node, or [`ROOT`]
    /// for a node that follows the cached text directly. The tree's nodes take slots
    /// `base..pos + rows`, `base = pos + rows - tree.len()`: the batch is its last `rows`
    /// nodes, the ones before it were run by earlier calls with the same tree so far (a
    /// tree grown a level at a time). Node i sits at position `base + depth(i)` and sees
    /// the cache before `base` plus its own ancestors, so each row is scored exactly as if
    /// its path alone had been fed. [`Kv::keep`] then keeps the accepted path. `None` is a
    /// line.
    pub fn forward_tree(&self, x: &mut [f32], rows: usize, pos: usize, tree: Option<&[u32]>, kv: &mut Kv) -> Result<()> {
        let c = &self.cfg;
        let shape = Shape::new(rows, pos, tree)?;
        if x.len() != rows * c.n_embd {
            bail!("{} values are not {rows} rows of {}", x.len(), c.n_embd);
        }
        if pos > kv.len {
            bail!("position {pos} skips past the {} cached tokens", kv.len);
        }
        if pos + rows > c.context {
            bail!("{} tokens are past the model's context of {}", pos + rows, c.context);
        }
        if kv.state.iter().any(|s| !s.is_empty()) {
            if tree.is_some() {
                bail!("this model keeps a running state and takes no tree of guesses");
            }
            // Starting over is the only way back: the state holds every token fed.
            if pos == 0 {
                for (s, b) in kv.state.iter_mut().zip(&self.blocks) {
                    *s = self.fresh_state(b);
                }
            } else if pos != kv.len {
                bail!("this model keeps a running state: it continues at {} or starts over, not at {pos}", kv.len);
            }
        }
        let kvd = c.kv_dim();
        let started = std::time::Instant::now();
        let on_gpu = match (&self.gpu, kv.gpu.as_mut()) {
            (Some(g), Some(cache)) => {
                let positions: Vec<u32> = shape.positions.iter().map(|&p| p as u32).collect();
                g.forward(c, x, pos, &positions, shape.visible.as_deref().map(|v| (shape.base, v)), cache)?;
                g.len()
            }
            _ => 0,
        };
        let gpu_done = std::time::Instant::now();
        for (b, ((kc, vc), state)) in self.blocks.iter().zip(kv.k.iter_mut().zip(kv.v.iter_mut()).zip(kv.state.iter_mut())).skip(on_gpu) {
            let n = rms_norm(x, &b.attn_norm, c.eps);
            let o = match &b.mix {
                Mix::Attn(a) => {
                    kc.truncate(pos * kvd);
                    vc.truncate(pos * kvd);
                    self.attend(a, &n, rows, pos, &shape, kc, vc)
                }
                Mix::Delta(d) => self.delta_net(d, &n, rows, state),
            };
            x.iter_mut().zip(&o).for_each(|(a, b)| *a += b);

            let n = rms_norm(x, &b.ffn_norm, c.eps);
            let d = match &b.ffn {
                Ffn::Dense { gate, up, down } => swiglu(gate, up, down, &n, rows),
                Ffn::Moe(m) => self.experts(m, &n, rows),
            };
            x.iter_mut().zip(&d).for_each(|(a, b)| *a += b);
        }
        kv.len = pos + rows;
        use std::sync::atomic::Ordering::Relaxed;
        self.split_us[0].store((gpu_done - started).as_micros() as u64, Relaxed);
        self.split_us[1].store(gpu_done.elapsed().as_micros() as u64, Relaxed);
        Ok(())
    }

    /// A mixture of experts' output for `rows` normed inputs `n`: each row's chosen
    /// experts, weighted, plus the shared expert. Rows that chose the same expert go
    /// through it together, so its weights are read once for all of them.
    fn experts(&self, m: &Experts, n: &[f32], rows: usize) -> Vec<f32> {
        let c = &self.cfg;
        let moe = c.moe.as_ref().expect("experts in a model with them");
        let e = c.n_embd;
        let logits = m.router.matmul(n, rows);
        let mut by_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> = Default::default();
        for (r, row) in logits.chunks_exact(moe.experts).enumerate() {
            for (ex, w) in route(row, moe.used, moe.scale) {
                by_expert.entry(ex).or_default().push((r, w));
            }
        }
        let mut out = vec![0.0; rows * e];
        for (ex, users) in by_expert {
            let xs: Vec<f32> = users.iter().flat_map(|&(r, _)| n[r * e..(r + 1) * e].iter().copied()).collect();
            let k = users.len();
            let ff = m.ff;
            let h: Vec<f32> = match &m.gate {
                Some(gate) => {
                    let g = gate.matmul_span(&xs, k, ex * ff..(ex + 1) * ff);
                    let u = m.up.matmul_span(&xs, k, ex * ff..(ex + 1) * ff);
                    g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect()
                }
                None => {
                    let gu = m.up.matmul_span(&xs, k, ex * 2 * ff..(ex + 1) * 2 * ff);
                    gu.chunks_exact(2 * ff).flat_map(|r| r[..ff].iter().zip(&r[ff..]).map(|(g, u)| g / (1.0 + (-g).exp()) * u)).collect()
                }
            };
            let d = m.down.matmul_span(&h, k, ex * e..(ex + 1) * e);
            for (&(r, w), dr) in users.iter().zip(d.chunks_exact(e)) {
                out[r * e..(r + 1) * e].iter_mut().zip(dr).for_each(|(o, v)| *o += w * v);
            }
        }
        if let Some([w, gate, up, down]) = &m.shared {
            let s = swiglu(gate, up, down, n, rows);
            let scale = w.matmul(n, rows);
            for ((o, s), g) in out.chunks_exact_mut(e).zip(s.chunks_exact(e)).zip(&scale) {
                let g = 1.0 / (1.0 + (-g).exp());
                o.iter_mut().zip(s).for_each(|(o, s)| *o += g * s);
            }
        }
        out
    }

    /// An attention layer's output for `rows` normed inputs `n`, its keys and values added
    /// to the cache.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, a: &Attn, n: &[f32], rows: usize, pos: usize, shape: &Shape, kc: &mut Vec<f32>, vc: &mut Vec<f32>) -> Vec<f32> {
        let c = &self.cfg;
        let (hd, kvd, qd) = (c.head_dim, c.kv_dim(), c.n_head * c.head_dim);
        let full = a.wq.matmul(n, rows);
        // Qwen3.5: each head's query is followed by its output gate.
        let (mut q, gate) = if a.gated {
            let mut q = Vec::with_capacity(rows * qd);
            let mut g = Vec::with_capacity(rows * qd);
            for head in full.chunks_exact(2 * hd) {
                q.extend_from_slice(&head[..hd]);
                g.extend_from_slice(&head[hd..]);
            }
            (q, Some(g))
        } else {
            (full, None)
        };
        let mut k = a.wk.matmul(n, rows);
        let mut v = a.wv.matmul(n, rows);
        add_bias(&mut q, a.bq.as_deref());
        add_bias(&mut k, a.bk.as_deref());
        add_bias(&mut v, a.bv.as_deref());
        if let Some(w) = &a.q_norm {
            q = rms_norm(&q, w, c.eps);
        }
        if let Some(w) = &a.k_norm {
            k = rms_norm(&k, w, c.eps);
        }
        for (i, &p) in shape.positions.iter().enumerate() {
            rope(&mut q[i * qd..(i + 1) * qd], c, p, &self.inv_freq);
            rope(&mut k[i * kvd..(i + 1) * kvd], c, p, &self.inv_freq);
        }
        kc.extend_from_slice(&k);
        vc.extend_from_slice(&v);
        let mut att = attention(c, &q, kc, vc, rows, pos, shape.visible.as_deref().map(|v| (shape.base, v)));
        if let Some(g) = gate {
            att.iter_mut().zip(&g).for_each(|(a, g)| *a /= 1.0 + (-g).exp());
        }
        a.wo.matmul(&att, rows)
    }

    /// A gated DeltaNet layer's output for `rows` normed inputs `n`, one token after
    /// another, carrying `state` (see [`Kv`]) from the last token fed.
    fn delta_net(&self, d: &DeltaNet, n: &[f32], rows: usize, state: &mut [f32]) -> Vec<f32> {
        let c = &self.cfg;
        let s = c.delta.as_ref().expect("a DeltaNet layer in a model with them");
        let (cd, vd, hd, taps) = (s.conv_dim(), s.value_dim(), s.head, s.conv);
        let qkv = d.qkv.matmul(n, rows);
        let qkv_w = d.qkv.rows;
        let z = d.z.as_ref().map(|z| z.matmul(n, rows));
        let ba = d.beta.matmul(n, rows);
        let ba_w = d.beta.rows;
        let alpha = d.alpha.as_ref().map(|a| a.matmul(n, rows));
        let (window, mats) = state.split_at_mut((taps - 1) * cd);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut out = vec![0.0; rows * vd];
        let mut x = vec![0.0; cd];
        let mut delta = vec![0.0; hd];
        for t in 0..rows {
            let input = &qkv[t * qkv_w..t * qkv_w + cd];
            // Causal convolution over the last `taps` inputs, then SiLU.
            for (ch, xc) in x.iter_mut().enumerate() {
                let w = &d.conv[ch * taps..(ch + 1) * taps];
                let mut sum = 0.0;
                for (j, wj) in w.iter().enumerate().take(taps - 1) {
                    sum += window[j * cd + ch] * wj;
                }
                sum += input[ch] * w[taps - 1];
                *xc = sum / (1.0 + (-sum).exp());
            }
            window.copy_within(cd.., 0);
            window[(taps - 2) * cd..].copy_from_slice(input);
            let (q, rest) = x.split_at_mut(s.key_dim());
            let (k, v) = rest.split_at_mut(s.key_dim());
            for head in q.chunks_exact_mut(hd).chain(k.chunks_exact_mut(hd)) {
                let inv = 1.0 / dot(head, head).sqrt().max(c.eps);
                head.iter_mut().for_each(|e| *e *= inv);
            }
            let (betas, alphas) = match &alpha {
                Some(a) => (&ba[t * ba_w..(t + 1) * ba_w], &a[t * s.v_heads..(t + 1) * s.v_heads]),
                None => ba[t * ba_w..(t + 1) * ba_w].split_at(s.v_heads),
            };
            let o = &mut out[t * vd..(t + 1) * vd];
            for h in 0..s.v_heads {
                // Value heads take key heads in turn (the file stores them tiled).
                let kh = h % s.k_heads;
                let (qh, kh, vh) = (&q[kh * hd..(kh + 1) * hd], &k[kh * hd..(kh + 1) * hd], &v[h * hd..(h + 1) * hd]);
                let beta = 1.0 / (1.0 + (-betas[h]).exp());
                let pre = alphas[h] + d.dt[h];
                let softplus = if pre > 20.0 { pre } else { pre.exp().ln_1p() };
                let decay = (softplus * d.a[h]).exp();
                let m = &mut mats[h * hd * hd..(h + 1) * hd * hd];
                m.iter_mut().for_each(|e| *e *= decay);
                for (j, row) in m.chunks_exact(hd).enumerate() {
                    delta[j] = (vh[j] - dot(row, kh)) * beta;
                }
                for (row, dj) in m.chunks_exact_mut(hd).zip(&delta) {
                    row.iter_mut().zip(kh).for_each(|(e, k)| *e += dj * k);
                }
                let oh = &mut o[h * hd..(h + 1) * hd];
                for (j, row) in m.chunks_exact(hd).enumerate() {
                    oh[j] = dot(row, qh) * scale;
                }
            }
        }
        // Per head: RMSNorm, then the gate SiLU(z).
        let mut normed = rms_norm(&out, &d.norm, c.eps);
        let (z, zw, zoff) = match &z {
            Some(z) => (z.as_slice(), vd, 0),
            None => (qkv.as_slice(), qkv_w, cd),
        };
        for (t, row) in normed.chunks_exact_mut(vd).enumerate() {
            for (e, g) in row.iter_mut().zip(&z[t * zw + zoff..t * zw + zoff + vd]) {
                *e *= g / (1.0 + (-g).exp());
            }
        }
        d.out.matmul(&normed, rows)
    }

    /// Next-token scores for each row (rows × vocab).
    pub fn logits(&self, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        self.logits_upto(x, rows, self.cfg.vocab)
    }

    /// [`Part::logits`] for the first `upto` tokens of the vocabulary only: `rows` x
    /// `min(upto, vocab)` scores, for a draft model that guesses only common tokens.
    pub fn logits_upto(&self, x: &[f32], rows: usize, upto: usize) -> Result<Vec<f32>> {
        let (norm, out) = self.head.as_ref().context("this part has no output head")?;
        let upto = upto.min(out.rows);
        if let Some(scores) = self.gpu.as_ref().map(|g| g.logits(&self.cfg, x, rows, upto)).transpose()?.flatten() {
            return Ok(scores);
        }
        let n = rms_norm(x, norm, self.cfg.eps);
        debug_assert_eq!(n.len(), rows * self.cfg.n_embd);
        Ok(out.matmul_rows(&n, rows, upto))
    }
}

/// SiLU(W_gate·n) ⊙ W_up·n through W_down, for `rows` rows.
fn swiglu(gate: &QTensor, up: &QTensor, down: &QTensor, n: &[f32], rows: usize) -> Vec<f32> {
    let g = gate.matmul(n, rows);
    let u = up.matmul(n, rows);
    let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
    down.matmul(&h, rows)
}

/// The `used` experts a row of router scores picks, with their weights: softmax over
/// every expert, the likeliest `used` (the lower index first on a tie), weights divided
/// by their sum (at least 2⁻¹⁴, as llama.cpp clamps it) and times `scale` unless 0 or 1.
pub fn route(scores: &[f32], used: usize, scale: f32) -> Vec<(usize, f32)> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let p: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f32 = p.iter().sum();
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
    order.truncate(used);
    let picked: f32 = order.iter().map(|&i| p[i] / total).sum::<f32>().max(2f32.powi(-14));
    let f = if scale != 0.0 && scale != 1.0 { scale } else { 1.0 };
    order.into_iter().map(|i| (i, p[i] / total / picked * f)).collect()
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

/// A row of a tree that follows the cached text directly.
pub const ROOT: u32 = u32::MAX;
/// Rows a tree pass may hold: which earlier rows each sees is one 64-bit mask.
pub const MAX_TREE: usize = 64;

/// Where each row of a batch sits and what it sees.
struct Shape {
    /// RoPE position of each row.
    positions: Vec<usize>,
    /// For a tree, the tree nodes each row sees (its ancestors and itself), as bits: bit
    /// j is slot `base + j`.
    visible: Option<Vec<u64>>,
    /// The tree's first slot.
    base: usize,
}

impl Shape {
    fn new(rows: usize, pos: usize, tree: Option<&[u32]>) -> Result<Self> {
        let Some(parents) = tree else {
            return Ok(Shape { positions: (pos..pos + rows).collect(), visible: None, base: pos });
        };
        if parents.len() < rows || parents.len() > MAX_TREE || parents.len() - rows > pos {
            bail!("a tree of {} nodes for {rows} rows at {pos} (at most {MAX_TREE})", parents.len());
        }
        let base = pos + rows - parents.len();
        let (mut positions, mut visible) = (Vec::with_capacity(rows), Vec::<u64>::with_capacity(rows));
        for (i, &p) in parents.iter().enumerate() {
            let (depth, seen) = match p {
                ROOT => (0, 0),
                p if (p as usize) < i => (positions[p as usize] - base + 1, visible[p as usize]),
                p => bail!("node {i}'s parent {p} does not come before it"),
            };
            positions.push(base + depth);
            visible.push(seen | 1 << i);
        }
        let skip = parents.len() - rows;
        Ok(Shape { positions: positions.split_off(skip), visible: Some(visible.split_off(skip)), base })
    }
}

/// Causal grouped-query attention for `rows` new tokens at `pos..`, against the cache.
/// With `visible` (a tree from slot `base`), row i sees the cache before `base` and only
/// the tree slots its bits name; without, every earlier row (a line).
fn attention(c: &Config, q: &[f32], kc: &[f32], vc: &[f32], rows: usize, pos: usize, visible: Option<(usize, &[u64])>) -> Vec<f32> {
    let (hd, kvd) = (c.head_dim, c.kv_dim());
    let group = c.n_head / c.n_head_kv.max(1);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; rows * c.n_head * hd];
    let mut scores = Vec::new();
    let mut slots: Vec<usize> = Vec::new();
    for i in 0..rows {
        slots.clear();
        match visible {
            None => slots.extend(0..pos + i + 1),
            Some((base, v)) => slots.extend((0..base).chain((0..pos + rows - base).filter(|&j| v[i] >> j & 1 == 1).map(|j| base + j))),
        }
        for h in 0..c.n_head {
            let qh = &q[(i * c.n_head + h) * hd..(i * c.n_head + h + 1) * hd];
            let kvh = h / group.max(1);
            scores.clear();
            scores.extend(slots.iter().map(|&t| dot(qh, &kc[t * kvd + kvh * hd..t * kvd + (kvh + 1) * hd]) * scale));
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            let o = &mut out[(i * c.n_head + h) * hd..(i * c.n_head + h + 1) * hd];
            for (&t, p) in slots.iter().zip(&scores) {
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
    fn a_tree_pass_scores_every_path_as_if_fed_alone() {
        // On the CPU, and on the GPU when there is one: each must match itself exactly,
        // except that the GPU's one-row products round differently from its batched ones
        // (`gpu::MM_FROM`), so a lone row there is only within rounding of a batch.
        let same = |a: &[f32], b: &[f32], exact: bool, what: &str| {
            if exact {
                assert_eq!(a, b, "{what}");
            } else {
                let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                assert!(err <= 1e-4 * (1.0 + b.iter().fold(0f32, |m, v| m.max(v.abs()))), "{what}: max error {err}");
            }
        };
        let gpu = crate::gpu::get().is_some();
        for (neox, on_gpu) in [(false, false), (true, false), (false, true), (true, true)] {
            if on_gpu && !gpu {
                continue;
            }
            let file = tiny_model(3, neox);
            let part = load_all(&file, 0..3, true, true);
            let part = if on_gpu { part } else { part.without_gpu() };
            assert_eq!(part.gpu_layers().0 > 0, on_gpu);
            let v = part.cfg.vocab;
            let prefix = [5u32, 9, 13];
            // Two branches from the text, the first branching again:
            //   40 ─┬─ 41 ── 43
            //       └─ 42
            //   44
            let tokens = [40u32, 41, 42, 43, 44];
            let tree = [ROOT, 0, 0, 1, ROOT];
            let mut kv = part.new_kv();
            run(&[&part], std::slice::from_mut(&mut kv), &prefix, 0);
            let mut x = part.embed(&tokens).unwrap();
            part.forward_tree(&mut x, tokens.len(), prefix.len(), Some(&tree), &mut kv).unwrap();
            let scores = part.logits(&x, tokens.len()).unwrap();
            let paths: [&[u32]; 5] = [&[40], &[40, 41], &[40, 42], &[40, 41, 43], &[44]];
            for (row, path) in paths.iter().enumerate() {
                let mut alone = part.new_kv();
                let line: Vec<u32> = prefix.iter().chain(path.iter()).copied().collect();
                let want = run(&[&part], std::slice::from_mut(&mut alone), &line, 0);
                assert_eq!(&scores[row * v..(row + 1) * v], &want[(line.len() - 1) * v..], "row {row}, path {path:?}");
            }
            // The same tree a level at a time (40 and 44, then 41 and 42, then 43) scores
            // every node the same.
            let mut grown = part.new_kv();
            run(&[&part], std::slice::from_mut(&mut grown), &prefix, 0);
            let order = [40u32, 44, 41, 42, 43];
            let parents = [ROOT, ROOT, 0, 0, 2];
            let mut rows_seen = Vec::new();
            for (from, to) in [(0, 2), (2, 4), (4, 5)] {
                let mut x = part.embed(&order[from..to]).unwrap();
                part.forward_tree(&mut x, to - from, prefix.len() + from, Some(&parents[..to]), &mut grown).unwrap();
                rows_seen.extend(part.logits(&x, to - from).unwrap());
            }
            for (node, row) in [(0, 0), (1, 4), (2, 1), (3, 2), (4, 3)] {
                same(&rows_seen[node * v..(node + 1) * v], &scores[row * v..(row + 1) * v], !on_gpu || node < 4, &format!("node {node}"));
            }
            // Accept 40 → 41 → 43 (rows 0, 1, 3): the cache then holds exactly that text.
            kv.keep(prefix.len(), &[0, 1, 3]).unwrap();
            assert_eq!(kv.len, 6);
            let next = run(&[&part], std::slice::from_mut(&mut kv), &[50], 6);
            let mut fresh = part.new_kv();
            let want = run(&[&part], std::slice::from_mut(&mut fresh), &[5, 9, 13, 40, 41, 43, 50], 0);
            same(&next, &want[6 * v..], !on_gpu, "after keep");
        }
        // Parents must come first, and a path must ascend.
        assert!(Shape::new(2, 0, Some(&[1, ROOT])).is_err());
        let file = tiny_model(1, false);
        let part = load_all(&file, 0..1, true, true).without_gpu();
        let mut kv = part.new_kv();
        run(&[&part], std::slice::from_mut(&mut kv), &[1, 2, 3], 0);
        assert!(kv.keep(0, &[2, 1]).is_err());
        assert!(kv.keep(0, &[0, 5]).is_err());
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
            // On the GPU itself a batch scores each row as a one-token pass would, within
            // rounding (one row and a batch use different products), and a split model
            // matches the whole one bit for bit.
            let mut fresh = [gpu.new_kv()];
            let all = run(&[&gpu], &mut fresh, &[5, 9, 13, 2, 7, 1, 40], 0);
            let mut steps = [gpu.new_kv()];
            run(&[&gpu], &mut steps, &prompt, 0);
            let one = run(&[&gpu], &mut steps, &[40], 6);
            let err = all[6 * gpu.cfg.vocab..].iter().zip(&one).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            assert!(err < 1e-4, "a batch row against a one-token pass: {err}");
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
    fn qwen35_splits_steps_and_layouts_agree() {
        let file = crate::fixtures::tiny_qwen35(4, false);
        let whole = load_all(&file, 0..4, true, true).without_gpu();
        assert!(whole.cfg.recurrent() && whole.cfg.rope_dim == 4 && whole.cfg.rope_neox);
        let prompt = [5u32, 9, 13, 2, 7];
        // The prompt in one batch, then two tokens one at a time…
        let mut kv = [whole.new_kv()];
        run(&[&whole], &mut kv, &prompt, 0);
        run(&[&whole], &mut kv, &[40], 5);
        let stepped = run(&[&whole], &mut kv, &[41], 6);
        // …score what the whole text in one batch does: the state carries the history.
        let mut fresh = [whole.new_kv()];
        let all = run(&[&whole], &mut fresh, &[5, 9, 13, 2, 7, 40, 41], 0);
        let v = whole.cfg.vocab;
        let err = all[6 * v..].iter().zip(&stepped).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "batch against steps: {err}");
        assert!(stepped.iter().all(|x| x.is_finite()) && stepped.iter().any(|&x| x != stepped[0]));
        // Split over three parts: the same scores, bit for bit.
        let (a, b, c) = (load_all(&file, 0..1, true, false).without_gpu(), load_all(&file, 1..3, false, false).without_gpu(), load_all(&file, 3..4, false, true).without_gpu());
        let mut kvs = [a.new_kv(), b.new_kv(), c.new_kv()];
        assert_eq!(run(&[&a, &b, &c], &mut kvs, &[5, 9, 13, 2, 7, 40, 41], 0), all);
        // Starting over at 0 forgets the old text; going back elsewhere is refused.
        assert_eq!(run(&[&whole], &mut kv, &[5, 9, 13, 2, 7, 40, 41], 0), all);
        let mut x = whole.embed(&[1]).unwrap();
        assert!(whole.forward(&mut x, 1, 3, &mut kv[0]).is_err());
        assert!(kv[0].keep(5, &[1]).is_err());
        // One matrix for q, k, v and z (and for β and α) computes what two do.
        let fused = load_all(&crate::fixtures::tiny_qwen35(4, true), 0..4, true, true).without_gpu();
        let mut kf = [fused.new_kv()];
        assert_eq!(run(&[&fused], &mut kf, &[5, 9, 13, 2, 7, 40, 41], 0), all);
    }

    #[test]
    fn qwen35_on_the_gpu_computes_what_the_cpu_does() {
        if crate::gpu::get().is_none() {
            eprintln!("no GPU here: skipped");
            return;
        }
        for fused in [false, true] {
            let file = crate::fixtures::tiny_qwen35(4, fused);
            let gpu = load_all(&file, 0..4, true, true);
            assert_eq!(gpu.gpu_layers(), (4, true), "all of a small model goes on the GPU");
            let cpu = load_all(&file, 0..4, true, true).without_gpu();
            let close = |a: &[f32], b: &[f32], what: &str| {
                let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let top = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!(err <= 0.02 * (1.0 + top), "{what}: max error {err} against scores up to {top}");
            };
            // A prompt in one batch, then single tokens: the GPU's state carries over.
            let (mut kg, mut kc) = ([gpu.new_kv()], [cpu.new_kv()]);
            close(&run(&[&gpu], &mut kg, &[5, 9, 13, 2, 7], 0), &run(&[&cpu], &mut kc, &[5, 9, 13, 2, 7], 0), "prompt");
            for (i, t) in [40u32, 41, 42].into_iter().enumerate() {
                close(&run(&[&gpu], &mut kg, &[t], 5 + i), &run(&[&cpu], &mut kc, &[t], 5 + i), "step");
            }
            // Starting over at 0 resets the GPU's state too.
            close(&run(&[&gpu], &mut kg, &[5, 9], 0), &run(&[&cpu], &mut kc, &[5, 9], 0), "again");
            // Split between parts, the GPU agrees with itself.
            let mut whole = [gpu.new_kv()];
            let all = run(&[&gpu], &mut whole, &[5, 9, 13, 2, 7, 40], 0);
            let (a, b) = (load_all(&file, 0..1, true, false), load_all(&file, 1..4, false, true));
            let mut ks = [a.new_kv(), b.new_kv()];
            assert_eq!(run(&[&a, &b], &mut ks, &[5, 9, 13, 2, 7, 40], 0), all);
        }
    }

    #[test]
    fn routing_picks_the_likeliest_experts_and_renormalises() {
        let picks = route(&[0.0, 2.0, 1.0, 2.0, -1.0], 2, 0.0);
        // Experts 1 and 3 tie at the top: the lower index first, half the weight each.
        assert_eq!(picks.iter().map(|p| p.0).collect::<Vec<_>>(), [1, 3]);
        assert!(picks.iter().all(|p| (p.1 - 0.5).abs() < 1e-6));
        let picks = route(&[3.0, 1.0, 0.0], 2, 2.5);
        let (a, b) = ((3f32).exp(), (1f32).exp());
        assert!((picks[0].1 - 2.5 * a / (a + b)).abs() < 1e-5 && (picks[1].1 - 2.5 * b / (a + b)).abs() < 1e-5);
    }

    #[test]
    fn experts_route_each_row_and_agree_across_layouts() {
        let file = crate::fixtures::tiny_qwen35_moe(4, false);
        let whole = load_all(&file, 0..4, true, true).without_gpu();
        assert_eq!(whole.cfg.moe, Some(Moe { experts: 8, used: 2, scale: 0.0 }));
        // Three rows through the experts at once (grouped by expert) against each row alone.
        let Ffn::Moe(m) = &whole.blocks[1].ffn else { panic!("layer 1 has experts") };
        let e = whole.cfg.n_embd;
        let n: Vec<f32> = (0..3 * e).map(|i| ((i * 7 % 23) as f32 - 11.0) / 9.0).collect();
        let together = whole.experts(m, &n, 3);
        for r in 0..3 {
            let alone = whole.experts(m, &n[r * e..(r + 1) * e], 1);
            let err = alone.iter().zip(&together[r * e..(r + 1) * e]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err < 1e-5, "row {r}: {err}");
        }
        // Batch against steps, a split against the whole, fused against separate weights.
        let mut kv = [whole.new_kv()];
        run(&[&whole], &mut kv, &[5, 9, 13, 2, 7], 0);
        let stepped = run(&[&whole], &mut kv, &[40], 5);
        let mut fresh = [whole.new_kv()];
        let all = run(&[&whole], &mut fresh, &[5, 9, 13, 2, 7, 40], 0);
        let v = whole.cfg.vocab;
        let err = all[5 * v..].iter().zip(&stepped).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "batch against steps: {err}");
        let (a, b) = (load_all(&file, 0..2, true, false).without_gpu(), load_all(&file, 2..4, false, true).without_gpu());
        let mut ks = [a.new_kv(), b.new_kv()];
        assert_eq!(run(&[&a, &b], &mut ks, &[5, 9, 13, 2, 7, 40], 0), all);
        let fused = load_all(&crate::fixtures::tiny_qwen35_moe(4, true), 0..4, true, true).without_gpu();
        let mut kf = [fused.new_kv()];
        assert_eq!(run(&[&fused], &mut kf, &[5, 9, 13, 2, 7, 40], 0), all);
    }

    #[test]
    fn experts_on_the_gpu_compute_what_the_cpu_does() {
        if crate::gpu::get().is_none() {
            eprintln!("no GPU here: skipped");
            return;
        }
        for fused in [false, true] {
            let file = crate::fixtures::tiny_qwen35_moe(4, fused);
            let gpu = load_all(&file, 0..4, true, true);
            assert_eq!(gpu.gpu_layers(), (4, true));
            let cpu = load_all(&file, 0..4, true, true).without_gpu();
            let close = |a: &[f32], b: &[f32], what: &str| {
                let err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
                let top = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!(err <= 0.02 * (1.0 + top), "{what}: max error {err} against scores up to {top}");
            };
            let (mut kg, mut kc) = ([gpu.new_kv()], [cpu.new_kv()]);
            close(&run(&[&gpu], &mut kg, &[5, 9, 13, 2, 7], 0), &run(&[&cpu], &mut kc, &[5, 9, 13, 2, 7], 0), "prompt");
            for (i, t) in [40u32, 41].into_iter().enumerate() {
                close(&run(&[&gpu], &mut kg, &[t], 5 + i), &run(&[&cpu], &mut kc, &[t], 5 + i), "step");
            }
        }
    }

    #[test]
    fn a_delta_net_step_follows_the_rule() {
        // One token, one head, through `delta_net`'s update against the rule written
        // plainly: S ← e^g S; S ← S + β k (v − Sᵀk)ᵀ; out = Sᵀq / √d, S[i][j] indexed as is.
        let file = crate::fixtures::tiny_qwen35(2, false);
        let part = load_all(&file, 0..1, true, false).without_gpu();
        let Mix::Delta(dn) = &part.blocks[0].mix else { panic!("layer 0 is DeltaNet") };
        let s = part.cfg.delta.clone().unwrap();
        let hd = s.head;
        let mut state = part.fresh_state(&part.blocks[0]);
        let mats = (s.conv - 1) * s.conv_dim();
        // A state already full of something, so decay and the update both show.
        for (i, e) in state[mats..].iter_mut().enumerate() {
            *e = ((i * 7 % 13) as f32 - 6.0) / 10.0;
        }
        let before = state.clone();
        let n: Vec<f32> = (0..part.cfg.n_embd).map(|i| ((i * 5 % 11) as f32 - 5.0) / 7.0).collect();
        part.delta_net(dn, &n, 1, &mut state);
        // Recompute head 3's k, v, β and g from the same projections.
        let qkv = dn.qkv.matmul(&n, 1);
        let conv: Vec<f32> = (0..s.conv_dim())
            .map(|ch| {
                let v = qkv[ch] * dn.conv[ch * s.conv + s.conv - 1];
                v / (1.0 + (-v).exp())
            })
            .collect();
        let l2 = |x: &[f32]| {
            let n = x.iter().map(|v| v * v).sum::<f32>().sqrt().max(part.cfg.eps);
            x.iter().map(|v| v / n).collect::<Vec<f32>>()
        };
        let h = 3;
        let kh = h % s.k_heads;
        let k = l2(&conv[s.key_dim() + kh * hd..s.key_dim() + (kh + 1) * hd]);
        let v = &conv[2 * s.key_dim() + h * hd..2 * s.key_dim() + (h + 1) * hd];
        let beta = 1.0 / (1.0 + (-dn.beta.matmul(&n, 1)[h]).exp());
        let pre = dn.alpha.as_ref().unwrap().matmul(&n, 1)[h] + dn.dt[h];
        let decay = (pre.exp().ln_1p() * dn.a[h]).exp();
        // S[i][j] = stored[j][i].
        let old = &before[mats + h * hd * hd..mats + (h + 1) * hd * hd];
        let mut sm = vec![vec![0f32; hd]; hd];
        for i in 0..hd {
            for j in 0..hd {
                sm[i][j] = old[j * hd + i] * decay;
            }
        }
        let stk: Vec<f32> = (0..hd).map(|j| (0..hd).map(|i| sm[i][j] * k[i]).sum()).collect();
        for i in 0..hd {
            for j in 0..hd {
                sm[i][j] += beta * k[i] * (v[j] - stk[j]);
            }
        }
        let new = &state[mats + h * hd * hd..mats + (h + 1) * hd * hd];
        for i in 0..hd {
            for j in 0..hd {
                assert!((new[j * hd + i] - sm[i][j]).abs() < 1e-5, "S[{i}][{j}]");
            }
        }
        // The convolution window now ends with this token's input.
        let w = &state[(s.conv - 2) * s.conv_dim()..mats];
        assert_eq!(w, &qkv[..s.conv_dim()]);
    }

    #[test]
    fn rope_rotates_pairs_by_position() {
        let c = Config { arch: "llama".into(), n_embd: 4, n_layer: 1, n_head: 1, n_head_kv: 1, head_dim: 4, rope_dim: 4, rope_base: 10000.0, rope_neox: false, eps: 1e-5, vocab: 1, context: 8, delta: None, moe: None };
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
