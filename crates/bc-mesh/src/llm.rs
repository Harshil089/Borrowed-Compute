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
use bc_infer::{Config, Kv, Local, Part, Sampling, Tokenizer};
use iroh::{Endpoint, EndpointAddr};
use tokio::sync::{oneshot, Mutex};

/// Runs the draft model off the async threads: scores after each token in `tokens`.
async fn draft_run(d: (Arc<Part>, Kv), tokens: Vec<u32>, pos: usize) -> Result<((Arc<Part>, Kv), Vec<f32>)> {
    tokio::task::spawn_blocking(move || {
        let (part, mut kv) = d;
        let mut x = part.embed(&tokens)?;
        part.forward(&mut x, tokens.len(), pos, &mut kv)?;
        let scores = part.logits(&x, tokens.len())?;
        Ok(((part, kv), scores))
    })
    .await?
}

use crate::msg::{Activation, Meta};
use crate::pipeline::{self, Input, Target};
use crate::plan::{Model, PipelinePlan};

pub const PREFILL_CHUNK: usize = 256;
pub const DRAFT_TOKENS: usize = 4;

type Reply = oneshot::Sender<Result<(Meta, Activation)>>;
type Answer = oneshot::Receiver<Result<(Meta, Activation)>>;

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
    /// Milliseconds a token, measured, with and without speculation on this pipeline.
    pub speed: StdMutex<Speed>,
}

/// Which way of decoding is faster on a pipeline, as measured. Guessing pays when the
/// stages check several tokens for about the price of one (CPU stages, and memory-bound
/// GPUs) and the draft agrees often; it costs when checking is dear or guesses miss. So
/// each pipeline keeps both times and decodes the faster way, trying the other every
/// [`EXPLORE`] requests in case things changed.
#[derive(Debug, Clone, Copy, Default)]
pub struct Speed {
    pub plain_ms: f64,
    pub spec_ms: f64,
    requests: u64,
}

const EXPLORE: u64 = 8;

impl Speed {
    fn speculate(&mut self) -> bool {
        self.requests += 1;
        let faster = self.spec_ms <= self.plain_ms;
        match (self.spec_ms > 0.0, self.plain_ms > 0.0) {
            (false, _) => true,
            (true, false) => false,
            _ if self.requests.is_multiple_of(EXPLORE) => !faster,
            _ => faster,
        }
    }

    fn record(&mut self, speculated: bool, ms_per_token: f64) {
        let slot = if speculated { &mut self.spec_ms } else { &mut self.plain_ms };
        *slot = if *slot == 0.0 { ms_per_token } else { 0.7 * *slot + 0.3 * ms_per_token };
    }
}

impl Client {
    pub async fn open(endpoint: &Endpoint, model_url: &str, plan: PipelinePlan, addrs: &[EndpointAddr], on_loading: pipeline::OnLoading<'_>) -> Result<Arc<Self>> {
        let targets: Vec<Target> = plan.stages.iter().map(|s| Target { addr: addrs[s.device].clone(), layers: s.layers.clone() }).collect();
        let (input, mut output) = pipeline::open(endpoint, model_url, &targets, on_loading).await?;
        let pending: Arc<StdMutex<VecDeque<Reply>>> = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let stage_ms = Arc::new(StdMutex::new(vec![0.0; plan.stages.len()]));
        let (p, a, ms) = (pending.clone(), alive.clone(), stage_ms.clone());
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
        Ok(Arc::new(Client { input: Mutex::new((input, 0)), pending, alive, in_flight: AtomicUsize::new(0), plan, stage_ms, speed: StdMutex::default() }))
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
    /// Shared by every request: each keeps its own cache, so drafts run side by side.
    draft: Option<Arc<Part>>,
}

impl Llm {
    pub fn new(header: &Header, replicas: Vec<Arc<Client>>, draft: Option<Local>) -> Result<Self> {
        let tokenizer = Tokenizer::from_header(header)?;
        if let Some(d) = &draft {
            if d.tokenizer.vocab() != tokenizer.vocab() {
                bail!("the draft model's vocabulary ({}) differs from the model's ({})", d.tokenizer.vocab(), tokenizer.vocab());
            }
        }
        Ok(Llm { tokenizer, cfg: Config::from_header(header)?, replicas: StdMutex::new(replicas), draft: draft.map(|d| Arc::new(d.part)) })
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

        // Decoding speed, for choosing how to decode next time; short answers say little.
        let record = |speculated: bool, stats: &Stats| {
            if stats.generated >= 8 {
                let ms = (stats.total_ms - stats.first_token_ms) / (stats.generated - 1) as f64;
                client.speed.lock().unwrap().record(speculated, ms);
            }
        };
        let speculate = sampling.temperature <= 0.0 && self.draft.is_some() && client.speed.lock().unwrap().speculate();
        let Some(draft) = self.draft.clone().filter(|_| speculate) else {
            // No draft model, sampling, or guessing measured slower here: a token a pass.
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
            if self.draft.is_some() && sampling.temperature <= 0.0 {
                record(false, &stats);
            }
            return Ok((out, stats));
        };

        // Pipelined speculation. The stream is `seq` (confirmed) then `guesses` (the
        // draft's, unchecked); `sent` positions of it are in the target's pipeline. While
        // this is the pipeline's only conversation, up to one batch per stage plus one is
        // in flight, each continuing the draft's guesses where the last left off, so every
        // stage works instead of waiting for one batch to go round. A wrong guess makes
        // every later batch worthless: they are dropped by `epoch` as they return, and the
        // next batch starts at the corrected position, where the stages cut their caches
        // back. The text is exactly what the target alone writes.
        client.in_flight.fetch_add(1, Ordering::Relaxed);
        let vocab = self.cfg.vocab;
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
                        let n = feed.len();
                        let (back, scores) = draft_run(d, feed, draft_len).await?;
                        d = back;
                        draft_len = stream_len;
                        guesses.push(bc_infer::sample::argmax(&scores[(n - 1) * vocab..]));
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
        record(true, &stats);
        Ok((out, stats))
    }
}

#[cfg(test)]
mod tests {
    use super::Speed;

    #[test]
    fn decoding_takes_the_faster_way_and_rechecks_the_other() {
        let mut s = Speed::default();
        assert!(s.speculate(), "first, try guessing");
        s.record(true, 30.0);
        assert!(!s.speculate(), "then measure without");
        s.record(false, 20.0);
        let choices: Vec<bool> = (0..16).map(|_| s.speculate()).collect();
        assert_eq!(choices.iter().filter(|&&c| c).count(), 2, "plain is faster: guess only to recheck, {choices:?}");
        s.record(true, 5.0);
        s.record(true, 5.0);
        s.record(true, 5.0);
        assert!(s.spec_ms < s.plain_ms);
        assert!((0..7).filter(|_| s.speculate()).count() >= 6, "guessing got faster: use it");
    }
}
