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
//!   prefix is kept plus the target's next token. The text is exactly what the target
//!   alone writes.
//! - **Pipelined speculation** (one conversation on a pipeline of several stages): instead
//!   of one batch going round while every other stage waits, a new batch of guesses
//!   follows each time the first stage is free, continuing where the last left off. Each
//!   stage then works on some batch all the time, so one conversation goes as fast as
//!   the slowest stage, not as slow as all of them in a row, for as long as the guesses
//!   hold. More devices now speed one conversation up.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use bc_infer::gguf::Header;
use bc_infer::{Config, Kv, Local, Part, Sampling, Tokenizer, MAX_TREE, ROOT};
use iroh::{Endpoint, EndpointAddr};
use tokio::sync::{oneshot, Mutex};

/// Runs the draft model off the async threads: its scores after the last of `tokens`,
/// over the [`DRAFT_VOCAB`] tokens it guesses from.
async fn draft_run(d: (Arc<Part>, Kv), tokens: Vec<u32>, pos: usize) -> Result<((Arc<Part>, Kv), Vec<f32>)> {
    tokio::task::spawn_blocking(move || {
        let (part, mut kv) = d;
        let mut x = part.embed(&tokens)?;
        part.forward(&mut x, tokens.len(), pos, &mut kv)?;
        let e = part.cfg.n_embd;
        let scores = part.logits_upto(&x[x.len() - e..], 1, DRAFT_VOCAB)?;
        Ok(((part, kv), scores))
    })
    .await?
}

use crate::msg::{Activation, Meta, StageSetup};

use crate::tree::{add_chain, grow, lookup, top_k, DRAFT_VOCAB, LOOKUP_TOKENS, TREE_NODES, TREE_ROOT_K};

/// [`grow`] off the async threads, under the likeliest first tokens from `root`.
async fn grow_tree(d: (Arc<Part>, Kv), roots: Vec<(u32, f32)>, pos: usize, budget: usize) -> Result<((Arc<Part>, Kv), Vec<(u32, u32)>)> {
    tokio::task::spawn_blocking(move || {
        let (part, mut kv) = d;
        let nodes = grow(&part, &mut kv, pos, roots, budget)?;
        Ok(((part, kv), nodes))
    })
    .await?
}
use crate::pipeline::{self, Input, Target};
use crate::plan::{Model, PipelinePlan};

pub const PREFILL_CHUNK: usize = 256;
pub const DRAFT_TOKENS: usize = 4;

type Reply = oneshot::Sender<Result<(Meta, Activation)>>;
type Answer = oneshot::Receiver<Result<(Meta, Activation)>>;

/// One pipeline, shared by any number of requests.
pub struct Client {
    /// The first remote stage's end, if any stage is remote, and the last batch number.
    input: Mutex<(Option<Input>, u64)>,
    /// This computer's own stage, when the plan gives it the model's first layers: the
    /// coordinator computes them before anything crosses the network.
    local: Option<Arc<StdMutex<Box<dyn crate::stage::Stage>>>>,
    pending: Arc<StdMutex<VecDeque<Reply>>>,
    alive: Arc<AtomicBool>,
    in_flight: AtomicUsize,
    /// The plan it was opened with.
    pub plan: PipelinePlan,
    /// Average compute milliseconds per stage for one-token batches, as measured.
    pub stage_ms: Arc<StdMutex<Vec<f32>>>,
    /// Milliseconds a token, measured, with and without speculation on this pipeline.
    pub speed: StdMutex<Speed>,
}

/// How a request decodes: a token a pass, a chain of guesses pipelined through the
/// stages, a tree of guesses checked in one pass, or a tree of guesses read from the text
/// alone (no draft model: free, and fastest for edits and code, which repeat their input).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Plain = 0,
    Chain = 1,
    Tree = 2,
    Lookup = 3,
}

/// Which way of decoding is fastest on a pipeline, as measured. Guessing pays when the
/// stages check many rows for about the price of one (CPU stages, and memory-bound GPUs)
/// and the draft agrees often; it costs when checking is dear or guesses miss. So each
/// pipeline keeps the milliseconds a token of every way, decodes the fastest, and tries
/// another every [`EXPLORE`] requests in case things changed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Speed {
    /// Milliseconds a token, by [`Mode`]; 0 until measured.
    pub ms: [f64; 4],
    requests: u64,
}

const EXPLORE: u64 = 8;
/// Untried ways go in this order, so a new pipeline's first answer is never the costly
/// experiment.
const MODES: [Mode; 4] = [Mode::Plain, Mode::Lookup, Mode::Chain, Mode::Tree];

impl Speed {
    /// The way to decode next; a chain and a drafted tree need a draft model.
    fn choose(&mut self, draft: bool) -> Mode {
        self.requests += 1;
        let modes: Vec<Mode> = MODES.into_iter().filter(|&m| draft || !matches!(m, Mode::Chain | Mode::Tree)).collect();
        if let Some(&untried) = modes.iter().find(|m| self.ms[**m as usize] == 0.0) {
            return untried;
        }
        let best = *modes.iter().min_by(|a, b| self.ms[**a as usize].total_cmp(&self.ms[**b as usize])).unwrap();
        if self.requests.is_multiple_of(EXPLORE) {
            let other = modes[(self.requests / EXPLORE) as usize % modes.len()];
            if other != best {
                return other;
            }
        }
        best
    }

    fn record(&mut self, mode: Mode, ms_per_token: f64) {
        let slot = &mut self.ms[mode as usize];
        *slot = if *slot == 0.0 { ms_per_token } else { 0.7 * *slot + 0.3 * ms_per_token };
    }
}

impl Client {
    /// Opens a pipeline for `plan`. A stage on a device whose address is this endpoint's
    /// own runs here, and must be the first.
    pub async fn open(endpoint: &Endpoint, model_url: &str, plan: PipelinePlan, addrs: &[EndpointAddr], on_loading: pipeline::OnLoading<'_>) -> Result<Arc<Self>> {
        let is_local = |s: &crate::plan::StagePlan| addrs[s.device].id == endpoint.id();
        if plan.stages.iter().skip(1).any(is_local) {
            bail!("this computer's stage must come first in the pipeline");
        }
        let mine = plan.stages.first().filter(|s| is_local(s));
        let skip = mine.is_some() as usize;
        let targets: Vec<Target> = plan.stages.iter().skip(skip).map(|s| Target { addr: addrs[s.device].clone(), layers: s.layers.clone() }).collect();
        let pending: Arc<StdMutex<VecDeque<Reply>>> = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let stage_ms = Arc::new(StdMutex::new(vec![0.0; plan.stages.len()]));
        // This computer downloads its layers while the other devices download theirs.
        let local = async {
            match mine {
                Some(s) => Ok(Some(Arc::new(StdMutex::new(local_stage(endpoint, model_url, s.layers.clone(), on_loading).await?)))),
                None => anyhow::Ok(None),
            }
        };
        let remote = async {
            if targets.is_empty() {
                return anyhow::Ok(None);
            }
            Ok(Some(pipeline::open(endpoint, model_url, &targets, &|i, done, total| on_loading(i + skip, done, total)).await?))
        };
        let (local, remote) = tokio::join!(local, remote);
        let (local, remote) = (local?, remote?);
        let input = remote.map(|(input, output)| {
            Self::read_replies(output, pending.clone(), alive.clone(), stage_ms.clone());
            input
        });
        Ok(Arc::new(Client { input: Mutex::new((input, 0)), local, pending, alive, in_flight: AtomicUsize::new(0), plan, stage_ms, speed: StdMutex::default() }))
    }

    /// Hands each result from the last stage to the request waiting for it, in order.
    fn read_replies(mut output: pipeline::Output, p: Arc<StdMutex<VecDeque<Reply>>>, a: Arc<AtomicBool>, ms: Arc<StdMutex<Vec<f32>>>) {
        tokio::spawn(async move {
            let failure = loop {
                match output.recv().await {
                    Ok(Some((_, meta, act))) => {
                        // Single tokens only: what decoding speed is made of.
                        if meta.rows == 1 && !meta.skipped {
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
    }

    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    pub fn load(&self) -> usize {
        self.in_flight.load(Ordering::Relaxed)
    }

    /// Sends a batch; the reply arrives on the receiver. Batches sent one after another
    /// are answered in that order.
    pub async fn submit(&self, mut meta: Meta, act: Activation) -> Result<Answer> {
        meta.rows = match &act {
            Activation::Tokens(t) => t.len() as u32,
            Activation::Hidden(t) => t.rows as u32,
        };
        if !self.alive() {
            bail!("the pipeline is down");
        }
        let (tx, rx) = oneshot::channel();
        // Held while this computer computes its layers too, so batches stay in order.
        let mut input = self.input.lock().await;
        let (meta, act) = match &self.local {
            Some(stage) => {
                let stage = stage.clone();
                tokio::task::spawn_blocking(move || -> Result<(Meta, Activation)> {
                    let started = Instant::now();
                    let out = stage.lock().unwrap().forward(&meta, act)?;
                    meta.stage_ms.push(started.elapsed().as_secs_f32() * 1000.0);
                    Ok((meta, out))
                })
                .await??
            }
            None => (meta, act),
        };
        input.1 += 1;
        let seq = input.1;
        match input.0.as_mut() {
            Some(remote) => {
                self.pending.lock().unwrap().push_back(tx);
                remote.send(seq, meta, &act).await?;
            }
            // The whole model is here: the answer is ready.
            None => {
                if meta.rows == 1 {
                    let mut avg = self.stage_ms.lock().unwrap();
                    for (a, m) in avg.iter_mut().zip(&meta.stage_ms) {
                        *a = if *a == 0.0 { *m } else { 0.8 * *a + 0.2 * m };
                    }
                }
                let _ = tx.send(Ok((meta, act)));
            }
        }
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

/// This computer's own stage: the same engine lending devices run, its weights fetched by
/// byte range into memory like theirs, reporting its download as stage 0.
async fn local_stage(endpoint: &Endpoint, model_url: &str, layers: std::ops::Range<u32>, on_loading: pipeline::OnLoading<'_>) -> Result<Box<dyn crate::stage::Stage>> {
    let setup = StageSetup { session: rand::random(), index: 0, count: 1, model: model_url.into(), layers, next: None };
    let (tx, mut rx) = tokio::sync::watch::channel((0u64, 0u64));
    let loading = crate::stage::Loading {
        coordinator: endpoint.id(),
        progress: Arc::new(move |done, total| {
            let _ = tx.send((done, total));
        }),
    };
    let build = crate::engine::factory()(setup, loading);
    tokio::pin!(build);
    loop {
        tokio::select! {
            built = &mut build => {
                let (done, total) = *rx.borrow();
                on_loading(0, done, total);
                return built.context("loading this computer's layers");
            }
            Ok(()) = rx.changed() => {
                let (done, total) = *rx.borrow_and_update();
                on_loading(0, done, total);
            }
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
    /// Shared by every request: each keeps its own cache, so drafts run side by side.
    draft: Option<Arc<Part>>,
    /// The swarm: whole draft models on other devices, each growing its own branch of
    /// the tree while this computer grows the rest (one-stage pipelines).
    drafters: StdMutex<Vec<Arc<Client>>>,
}

impl Llm {
    pub fn new(header: &Header, replicas: Vec<Arc<Client>>, draft: Option<Local>) -> Result<Self> {
        let tokenizer = Tokenizer::from_header(header)?;
        if let Some(d) = &draft {
            if d.tokenizer.vocab() != tokenizer.vocab() {
                bail!("the draft model's vocabulary ({}) differs from the model's ({})", d.tokenizer.vocab(), tokenizer.vocab());
            }
        }
        Ok(Llm { tokenizer, cfg: Config::from_header(header)?, replicas: StdMutex::new(replicas), draft: draft.map(|d| Arc::new(d.part)), drafters: StdMutex::default() })
    }

    /// Devices that draft for tree speculation alongside this computer.
    pub fn set_drafters(&self, drafters: Vec<Arc<Client>>) {
        *self.drafters.lock().unwrap() = drafters;
    }

    pub fn drafters(&self) -> usize {
        self.drafters.lock().unwrap().iter().filter(|d| d.alive()).count()
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
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

    /// Tree speculation, one pass at a time: the draft grows a tree of guesses after the
    /// confirmed text, the whole tree (after the confirmed tokens the stages have not yet
    /// seen) goes through the pipeline as one batch, and the target's own choice is walked
    /// down the tree for as long as a guess matches it; its choice where none does is the
    /// next token. The next batch tells every stage which rows to keep. The text is
    /// exactly what the target alone writes, since each tree row is scored as if its path
    /// had been fed alone.
    #[allow(clippy::too_many_arguments)]
    async fn tree_decode(
        &self,
        client: &Client,
        draft: Option<&Arc<Part>>,
        prompt: &[u32],
        seq: &mut Vec<u32>,
        target_len: &mut usize,
        out: &mut Vec<u32>,
        max_tokens: usize,
        stats: &mut Stats,
        meta: &impl Fn(usize, bool) -> Meta,
        push: &mut impl FnMut(u32, &mut Vec<u32>, &mut Vec<u32>) -> bool,
    ) -> Result<()> {
        let mut d = match draft {
            Some(draft) => Some(draft_run((draft.clone(), draft.new_kv()), prompt.to_vec(), 0).await?.0),
            None => None,
        };
        let mut draft_len = prompt.len();
        let mut keep: Option<(u32, Vec<u32>)> = None;
        // Swarm drafters grow under the local draft's roots, so they need one.
        let drafters: Vec<Arc<Client>> = if d.is_some() { self.drafters.lock().unwrap().iter().filter(|d| d.alive()).cloned().collect() } else { Vec::new() };
        // Confirmed tokens each drafter has seen.
        let mut drafted = vec![0usize; drafters.len()];
        while out.len() < max_tokens {
            let real = seq.len() - *target_len;
            let budget = TREE_NODES.min(MAX_TREE - real);
            // Guesses from the text itself cost nothing; the draft fills the rest.
            let chain = lookup(seq, LOOKUP_TOKENS.min(budget));
            let mut nodes = Vec::new();
            if let Some(back) = d.take() {
                // The draft catches up on the confirmed text; its last scores seed the tree.
                let feed = seq[draft_len..].to_vec();
                let (back, scores) = draft_run(back, feed, draft_len).await?;
                draft_len = seq.len();
                let budget = budget - chain.len();
                // The likeliest first tokens: the second, third… each go to a drafter of
                // the swarm, which grows the subtree below it; this computer grows the rest.
                let mut roots = top_k(&scores, TREE_ROOT_K + drafters.len());
                let swarm = drafters.len().min(roots.len().saturating_sub(1));
                let share = budget / (swarm + 1);
                let mut asks = Vec::with_capacity(swarm);
                for (i, (root, _)) in roots.drain(1..1 + swarm).enumerate() {
                    let mut tokens = seq[drafted[i]..].to_vec();
                    tokens.push(root);
                    let m = Meta { grow: Some(share.saturating_sub(1) as u32), ..meta(drafted[i], false) };
                    drafted[i] = seq.len();
                    asks.push((root, drafters[i].submit(m, Activation::Tokens(tokens)).await));
                }
                let mine = Instant::now();
                let (back, grown) = grow_tree(back, roots, draft_len, budget - swarm * share).await?;
                d = Some(back);
                nodes = grown;
                // Subtrees that arrive while this computer's part takes as long again join
                // the tree; later ones are left out of this pass.
                let wait = mine.elapsed().max(std::time::Duration::from_millis(20));
                for (root, ask) in asks {
                    let Ok(rx) = ask else { continue };
                    let Ok(Ok(Ok((_, Activation::Tokens(flat))))) = tokio::time::timeout(wait, rx).await else { continue };
                    let at = nodes.len() as u32;
                    nodes.push((root, ROOT));
                    for pair in flat.chunks_exact(2).take(share.saturating_sub(1)) {
                        nodes.push((pair[0], if pair[1] == ROOT { at } else { at + 1 + pair[1] }));
                    }
                }
                nodes.truncate(budget);
            }
            add_chain(&mut nodes, &chain, TREE_NODES.min(MAX_TREE - real));
            let mut tokens: Vec<u32> = seq[*target_len..].to_vec();
            let mut parents: Vec<u32> = (0..real as u32).map(|i| if i == 0 { ROOT } else { i - 1 }).collect();
            for &(t, p) in &nodes {
                tokens.push(t);
                parents.push(if p == ROOT { real as u32 - 1 } else { p + real as u32 });
            }
            let m = Meta { tree: Some(parents.clone()), keep: keep.take(), ..meta(*target_len, true) };
            let checked = client.call(m, Activation::Tokens(tokens.clone())).await?;
            stats.passes += 1;
            // Rows the stages keep: the confirmed ones, then every accepted guess.
            let mut path: Vec<u32> = (0..real as u32).collect();
            let mut at = real - 1;
            let mut more = true;
            loop {
                let pred = checked[at];
                let child = (real..tokens.len()).find(|&r| parents[r] as usize == at && tokens[r] == pred);
                if !push(pred, out, seq) {
                    more = false;
                    break;
                }
                match child {
                    Some(c) => {
                        path.push(c as u32);
                        at = c;
                    }
                    None => break,
                }
            }
            *target_len += path.len();
            keep = Some(((*target_len - path.len()) as u32, path));
            if !more {
                break;
            }
        }
        Ok(())
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

        // Decoding speed, for choosing how to decode next time; short answers say little.
        let record = |mode: Mode, stats: &Stats| {
            if stats.generated >= 8 {
                let ms = (stats.total_ms - stats.first_token_ms) / (stats.generated - 1) as f64;
                client.speed.lock().unwrap().record(mode, ms);
            }
        };
        let greedy = sampling.temperature <= 0.0;
        let mode = if greedy { client.speed.lock().unwrap().choose(self.draft.is_some()) } else { Mode::Plain };
        if mode == Mode::Plain {
            // Sampling, or guessing measured slower here: a token a pass.
            while out.len() < max_tokens {
                stats.passes += 1;
                let ids = client.call(meta(target_len, false), Activation::Tokens(seq[target_len..].to_vec())).await?;
                target_len = seq.len();
                if !push(ids[0], &mut out, &mut seq) {
                    break;
                }
            }
            stats.generated = out.len();
            stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
            if greedy {
                record(Mode::Plain, &stats);
            }
            return Ok((out, stats));
        }

        if matches!(mode, Mode::Tree | Mode::Lookup) {
            client.in_flight.fetch_add(1, Ordering::Relaxed);
            let tree = self.tree_decode(&client, self.draft.as_ref().filter(|_| mode == Mode::Tree), prompt, &mut seq, &mut target_len, &mut out, max_tokens, &mut stats, &meta, &mut push).await;
            client.in_flight.fetch_sub(1, Ordering::Relaxed);
            tree?;
            stats.generated = out.len();
            stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
            record(mode, &stats);
            return Ok((out, stats));
        }

        // A chain: `choose` picks one only with a draft model.
        let draft = self.draft.clone().context("no draft model for a chain of guesses")?;

        // Pipelined speculation. The stream is `seq` (confirmed) then `guesses` (the
        // draft's, unchecked); `sent` positions of it are in the target's pipeline. While
        // this is the pipeline's only conversation, up to one batch per stage plus one is
        // in flight, each continuing the draft's guesses where the last left off, so every
        // stage works instead of waiting for one batch to go round. A wrong guess makes
        // every later batch worthless: they are dropped by `epoch` as they return, and the
        // next batch starts at the corrected position, where the stages cut their caches
        // back. The text is exactly what the target alone writes.
        client.in_flight.fetch_add(1, Ordering::Relaxed);
        let spec = async {
            let (mut d, _) = draft_run((draft.clone(), draft.new_kv()), prompt.to_vec(), 0).await?;
            let mut draft_len = prompt.len();
            let mut guesses: Vec<u32> = Vec::new();
            let mut sent = target_len;
            let mut epoch = 0u64;
            // (epoch, first position, rows, answer) of each batch in the pipeline, oldest first.
            let mut flight: VecDeque<(u64, usize, usize, Answer)> = VecDeque::new();
            let depth = client.plan.stages.len() + 1;
            loop {
                // Returned batches first: drafting more only fills time spent waiting.
                let ready = match flight.front_mut() {
                    Some((_, _, _, rx)) => match rx.try_recv() {
                        Ok(r) => Some(r),
                        Err(oneshot::error::TryRecvError::Empty) => None,
                        Err(oneshot::error::TryRecvError::Closed) => bail!("the pipeline went away"),
                    },
                    None => None,
                };
                let room = if client.load() <= 1 { depth } else { 1 };
                if ready.is_none() && flight.len() < room {
                    let want = DRAFT_TOKENS.min(max_tokens.saturating_sub(out.len() + guesses.len()));
                    for _ in 0..want {
                        let stream_len = seq.len() + guesses.len();
                        let feed: Vec<u32> = seq.iter().chain(&guesses).skip(draft_len).copied().collect();
                        let (back, scores) = draft_run(d, feed, draft_len).await?;
                        d = back;
                        draft_len = stream_len;
                        guesses.push(bc_infer::sample::argmax(&scores));
                    }
                    let rows: Vec<u32> = seq.iter().chain(&guesses).skip(sent).copied().collect();
                    if !rows.is_empty() {
                        let len = rows.len();
                        flight.push_back((epoch, sent, len, client.submit(Meta { epoch, ..meta(sent, true) }, Activation::Tokens(rows)).await?));
                        sent += len;
                        stats.passes += 1;
                        continue;
                    }
                }
                let Some((e, pos, len, rx)) = flight.pop_front() else { break };
                let (_, act) = match ready {
                    Some(r) => r?,
                    None => rx.await.context("the pipeline went away")??,
                };
                if e != epoch {
                    continue;
                }
                let Activation::Tokens(checked) = act else { bail!("the last stage sent hidden states") };
                // Row j scored the token after position pos + j.
                for (j, &pred) in checked.iter().enumerate().take(len) {
                    let next = pos + j + 1;
                    if next < seq.len() {
                        continue;
                    }
                    let right = guesses.first() == Some(&pred);
                    let wrong = !right && !guesses.is_empty();
                    if right {
                        guesses.remove(0);
                    } else {
                        guesses.clear();
                    }
                    if !push(pred, &mut out, &mut seq) {
                        return Ok(());
                    }
                    if wrong {
                        epoch += 1;
                        sent = next;
                        draft_len = draft_len.min(next);
                    }
                    if !right {
                        break;
                    }
                }
            }
            anyhow::Ok(())
        }
        .await;
        client.in_flight.fetch_sub(1, Ordering::Relaxed);
        spec?;
        stats.generated = out.len();
        stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
        record(Mode::Chain, &stats);
        Ok((out, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::Speed;

    #[test]
    fn decoding_takes_the_fastest_way_and_rechecks_the_others() {
        use super::Mode::*;
        let mut s = Speed::default();
        // Each way is tried once first, plain first.
        assert_eq!(s.choose(true), Plain);
        s.record(Plain, 20.0);
        assert_eq!(s.choose(true), Lookup);
        s.record(Lookup, 25.0);
        assert_eq!(s.choose(true), Chain);
        s.record(Chain, 30.0);
        assert_eq!(s.choose(true), Tree);
        s.record(Tree, 10.0);
        // Then the fastest, with an occasional recheck of another.
        let picks: Vec<_> = (0..24).map(|_| s.choose(true)).collect();
        let others = picks.iter().filter(|&&m| m != Tree).count();
        assert!((1..=4).contains(&others), "{picks:?}");
        // Plain got faster: it takes over.
        for _ in 0..4 {
            s.record(Plain, 2.0);
        }
        assert!((0..7).filter(|_| s.choose(true) == Plain).count() >= 6);
        // Without a draft model only plain and lookup are ways.
        let mut s = Speed::default();
        assert_eq!(s.choose(false), Plain);
        s.record(Plain, 20.0);
        assert_eq!(s.choose(false), Lookup);
        s.record(Lookup, 10.0);
        assert!((0..40).all(|_| matches!(s.choose(false), Plain | Lookup)));
    }
}
