//! The coordinator's side of a language model spread over devices: request in, text out.
//!
//! Three things hide the network's latency:
//! - **Many requests share a pipeline.** Batches are answered in the order sent (every
//!   stage runs them in turn), so requests from several conversations go out back to back
//!   and every stage works on one while the others wait on the network.
//! - **Prompts go in chunks** of [`PREFILL_CHUNK`] tokens, sent without waiting, so the
//!   first stage computes chunk 2 while chunk 1 crosses to the second stage.
//! - **Speculative decoding** (greedy requests, with a draft model on this device): the
//!   draft guesses [`DRAFT_TOKENS`] tokens, one pass through the pipeline checks them all
//!   at once (each row returns the target's own choice there), and the longest agreeing
//!   prefix is kept plus the target's next token. Every pass yields at least one token
//!   and up to `DRAFT_TOKENS + 1`, and the text is exactly what the target alone writes.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bc_infer::gguf::Header;
use bc_infer::{Config, Local, Sampling, Tokenizer};
use iroh::{Endpoint, EndpointAddr};
use tokio::sync::{oneshot, Mutex};

use crate::codec::Encoding;
use crate::msg::{Activation, Meta};
use crate::pipeline::{self, Input, Target};
use crate::plan::{Model, PipelinePlan};

pub const PREFILL_CHUNK: usize = 256;
pub const DRAFT_TOKENS: usize = 4;

type Reply = oneshot::Sender<Result<(Meta, Activation)>>;

/// One pipeline, shared by any number of requests.
pub struct Client {
    input: Mutex<(Input, u64)>,
    pending: Arc<StdMutex<VecDeque<Reply>>>,
    alive: Arc<AtomicBool>,
    in_flight: AtomicUsize,
    /// The plan it was opened with.
    pub plan: PipelinePlan,
    /// Average compute milliseconds per stage for one-token batches, as measured.
    pub stage_ms: Arc<StdMutex<Vec<f32>>>,
}

impl Client {
    pub async fn open(endpoint: &Endpoint, model_url: &str, plan: PipelinePlan, addrs: &[EndpointAddr], encoding: Encoding) -> Result<Arc<Self>> {
        let targets: Vec<Target> = plan.stages.iter().map(|s| Target { addr: addrs[s.device].clone(), layers: s.layers.clone() }).collect();
        let (input, mut output) = pipeline::open(endpoint, model_url, &targets, encoding).await?;
        let pending: Arc<StdMutex<VecDeque<Reply>>> = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let stage_ms = Arc::new(StdMutex::new(vec![0.0; plan.stages.len()]));
        let (p, a, ms) = (pending.clone(), alive.clone(), stage_ms.clone());
        tokio::spawn(async move {
            let failure = loop {
                match output.recv().await {
                    Ok(Some((_, meta, act))) => {
                        // Single tokens only: what decoding speed is made of.
                        if meta.rows == 1 {
                            let mut avg = ms.lock().unwrap();
                            for (a, m) in avg.iter_mut().zip(&meta.stage_ms) {
                                *a = if *a == 0.0 { *m } else { 0.8 * *a + 0.2 * m };
                            }
                        }
                        if let Some(reply) = p.lock().unwrap().pop_front() {
                            let _ = reply.send(Ok((meta, act)));
                        }
                    }
                    Ok(None) => break "the pipeline closed".to_string(),
                    Err(e) => break format!("{e:#}"),
                }
            };
            a.store(false, Ordering::Relaxed);
            for reply in p.lock().unwrap().drain(..) {
                let _ = reply.send(Err(anyhow!("{failure}")));
            }
        });
        Ok(Arc::new(Client { input: Mutex::new((input, 0)), pending, alive, in_flight: AtomicUsize::new(0), plan, stage_ms }))
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    pub fn load(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Sends a batch; the reply arrives on the receiver. Batches sent one after another
    /// are answered in that order.
    pub async fn submit(&self, mut meta: Meta, act: Activation) -> Result<oneshot::Receiver<Result<(Meta, Activation)>>> {
        meta.rows = match &act {
            Activation::Tokens(t) => t.len() as u32,
            Activation::Hidden(t) => t.rows as u32,
        };
        if !self.alive() {
            bail!("the pipeline is down");
        }
        let (tx, rx) = oneshot::channel();
        let mut input = self.input.lock().await;
        self.pending.lock().unwrap().push_back(tx);
        input.1 += 1;
        let seq = input.1;
        input.0.send(seq, meta, &act).await?;
        Ok(rx)
    }

    pub async fn call(&self, meta: Meta, act: Activation) -> Result<Vec<u32>> {
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        let result = async { self.submit(meta, act).await?.await.context("the pipeline went away")? }.await;
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        match result?.1 {
            Activation::Tokens(t) => Ok(t),
            Activation::Hidden(_) => bail!("the last stage sent hidden states"),
        }
    }
}

/// The model's size and shape for the planner, from its header.
pub fn model_spec(name: &str, h: &Header) -> Result<Model> {
    let cfg = Config::from_header(h)?;
    let (mut layer_bytes, mut extra) = (0u64, 0u64);
    for t in &h.tensors {
        let len = t.range.end - t.range.start;
        if t.name.starts_with("blk.") {
            layer_bytes += len;
        } else {
            extra += len;
        }
    }
    Ok(Model {
        name: name.into(),
        layers: cfg.n_layer as u32,
        layer_bytes: layer_bytes / cfg.n_layer.max(1) as u64,
        extra_bytes: extra,
        hidden: cfg.n_embd as u32,
        url: None,
        draft: None,
    })
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub generated: usize,
    /// Trips through the pipeline after the prompt.
    pub passes: usize,
    pub first_token_ms: f64,
    pub total_ms: f64,
}

pub struct Llm {
    pub tokenizer: Tokenizer,
    pub cfg: Config,
    replicas: StdMutex<Vec<Arc<Client>>>,
    draft: Option<Mutex<Local>>,
}

impl Llm {
    pub fn new(header: &Header, replicas: Vec<Arc<Client>>, draft: Option<Local>) -> Result<Self> {
        let tokenizer = Tokenizer::from_header(header)?;
        if let Some(d) = &draft {
            if d.tokenizer.vocab() != tokenizer.vocab() {
                bail!("the draft model's vocabulary ({}) differs from the model's ({})", d.tokenizer.vocab(), tokenizer.vocab());
            }
        }
        Ok(Llm { tokenizer, cfg: Config::from_header(header)?, replicas: StdMutex::new(replicas), draft: draft.map(Mutex::new) })
    }

    pub fn replicas(&self) -> Vec<Arc<Client>> {
        self.replicas.lock().unwrap().clone()
    }

    /// New requests go to these; requests already running finish where they are.
    pub fn set_replicas(&self, replicas: Vec<Arc<Client>>) {
        *self.replicas.lock().unwrap() = replicas;
    }

    /// The replica with the least work per unit of speed.
    fn pick(&self) -> Result<Arc<Client>> {
        self.replicas()
            .into_iter()
            .filter(|c| c.alive())
            .min_by(|a, b| {
                let ka = (a.load() + 1) as f64 / a.plan.tokens_per_s.max(1e-9);
                let kb = (b.load() + 1) as f64 / b.plan.tokens_per_s.max(1e-9);
                ka.total_cmp(&kb)
            })
            .context("no pipeline is running")
    }

    /// Generates up to `max_tokens` after `prompt`, calling `on_text` with each new piece.
    pub async fn generate(&self, prompt: &[u32], max_tokens: usize, sampling: Sampling, mut on_text: impl FnMut(&str)) -> Result<(Vec<u32>, Stats)> {
        if prompt.is_empty() {
            bail!("the prompt is empty");
        }
        if prompt.len() + max_tokens > self.cfg.context {
            bail!("{} prompt tokens plus {max_tokens} to write exceed the model's context of {}", prompt.len(), self.cfg.context);
        }
        let started = Instant::now();
        let client = self.pick()?;
        let stream: u64 = rand::random();
        let mut stats = Stats { prompt_tokens: prompt.len(), ..Default::default() };
        let meta = |pos: usize, all_rows: bool| Meta { stream, pos: pos as u32, all_rows, sampling, ..Default::default() };

        // Prompt: every chunk sent before any answer is read.
        client.in_flight.fetch_add(1, Ordering::Relaxed);
        let first = async {
            let mut waits = Vec::new();
            for (i, chunk) in prompt.chunks(PREFILL_CHUNK).enumerate() {
                waits.push(client.submit(meta(i * PREFILL_CHUNK, false), Activation::Tokens(chunk.to_vec())).await?);
            }
            let mut last = None;
            for w in waits {
                last = Some(w.await.context("the pipeline went away")??);
            }
            match last.map(|l| l.1) {
                Some(Activation::Tokens(t)) if !t.is_empty() => Ok(t[0]),
                _ => bail!("no token came back"),
            }
        }
        .await;
        client.in_flight.fetch_sub(1, Ordering::Relaxed);
        let first = first?;
        stats.first_token_ms = started.elapsed().as_secs_f64() * 1000.0;

        let mut seq: Vec<u32> = prompt.to_vec();
        let mut target_len = prompt.len(); // tokens in the stages' caches
        let mut out: Vec<u32> = Vec::new();
        let mut shown = String::new();
        let mut push = |id: u32, out: &mut Vec<u32>, seq: &mut Vec<u32>| -> bool {
            if self.tokenizer.is_stop(id) || out.len() >= max_tokens {
                return false;
            }
            out.push(id);
            seq.push(id);
            let text = self.tokenizer.decode(out);
            if !text.ends_with('\u{FFFD}') && text.len() > shown.len() && text.starts_with(&shown) {
                on_text(&text[shown.len()..]);
                shown = text;
            }
            true
        };
        if !push(first, &mut out, &mut seq) {
            stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok((out, stats));
        }

        let speculate = sampling.temperature <= 0.0 && self.draft.is_some();
        let mut draft = match &self.draft {
            Some(d) if speculate => Some(d.lock().await),
            _ => None,
        };
        let mut draft_len = 0;
        if let Some(d) = draft.as_mut() {
            d.scores(prompt, 0)?;
            draft_len = prompt.len();
        }
        while out.len() < max_tokens {
            stats.passes += 1;
            let Some(d) = draft.as_mut() else {
                // One token a pass.
                let ids = client.call(meta(target_len, false), Activation::Tokens(seq[target_len..].to_vec())).await?;
                target_len = seq.len();
                if !push(ids[0], &mut out, &mut seq) {
                    break;
                }
                continue;
            };
            // Draft: catch up on accepted tokens, then guess ahead.
            let mut guesses = Vec::with_capacity(DRAFT_TOKENS);
            let mut feed = seq[draft_len..].to_vec();
            let mut pos = draft_len;
            for _ in 0..DRAFT_TOKENS.min(max_tokens - out.len()) {
                let scores = d.scores(&feed, pos)?;
                let next = bc_infer::sample::argmax(&scores[(feed.len() - 1) * self.cfg.vocab..]);
                pos += feed.len();
                guesses.push(next);
                feed = vec![next];
            }
            // Target: the real tokens it has not seen plus the guesses, every row scored.
            let base = seq.len() - target_len;
            let mut rows = seq[target_len..].to_vec();
            rows.extend(&guesses);
            let checked = client.call(meta(target_len, true), Activation::Tokens(rows)).await?;
            let mut accepted = 0;
            while accepted < guesses.len() && guesses[accepted] == checked[base - 1 + accepted] {
                accepted += 1;
            }
            let old_len = seq.len();
            let mut more = true;
            for &g in &guesses[..accepted] {
                if !push(g, &mut out, &mut seq) {
                    more = false;
                    break;
                }
            }
            if more {
                more = push(checked[base - 1 + accepted], &mut out, &mut seq);
            }
            target_len = old_len + accepted;
            draft_len = old_len + accepted.min(guesses.len().saturating_sub(1));
            if !more {
                break;
            }
        }
        stats.generated = out.len();
        stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok((out, stats))
    }
}
