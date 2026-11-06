//! Language-model stages on a lending device.
//!
//! A setup names the model by its published address and the layer range this device
//! runs. The device reads the GGUF header, then only its own tensors, by byte range,
//! straight into memory: a stage with 20 of 80 layers holds a quarter of the weights and
//! never writes any of them to disk. Weights stay in memory while a stage uses them and a
//! short while after (so a rebalanced pipeline that keeps most layers here does not fetch
//! them again), then they are dropped. Quitting the app drops everything.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_infer::gguf::Header;
use bc_infer::quant::QTensor;
use bc_infer::{model, sample, Config, Kv, Part, Source};

use crate::msg::{Activation, Meta, StageSetup, Tensor};
use crate::stage::{Loading, Stage, StageFactory};

/// How long unused weights stay in memory after the last stage using them ends.
const GRACE: Duration = Duration::from_secs(120);
/// Conversations a stage keeps caches for; the least recently used goes first.
const MAX_STREAMS: usize = 16;
/// Share of free memory a stage's weights may take.
const MEMORY_SHARE: f64 = 0.9;

struct Cached {
    tensor: Arc<QTensor>,
    last_used: Instant,
}

#[derive(Default)]
struct Weights {
    headers: HashMap<String, Arc<Header>>,
    tensors: HashMap<(String, String), Cached>,
}

fn weights() -> &'static Mutex<Weights> {
    static W: OnceLock<Mutex<Weights>> = OnceLock::new();
    W.get_or_init(|| {
        // Drops weights no stage holds once the grace period has passed.
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async {
                loop {
                    tokio::time::sleep(Duration::from_secs(30)).await;
                    prune();
                }
            });
        }
        Mutex::default()
    })
}

fn prune() {
    let mut w = weights().lock().unwrap();
    w.tensors.retain(|_, c| Arc::strong_count(&c.tensor) > 1 || c.last_used.elapsed() < GRACE);
    let used: std::collections::HashSet<String> = w.tensors.keys().map(|(u, _)| u.clone()).collect();
    w.headers.retain(|u, _| used.contains(u));
}

/// Bytes of weights held in memory now (for the UI and tests).
pub fn held_bytes() -> u64 {
    weights().lock().unwrap().tensors.values().map(|c| c.tensor.data.len() as u64).sum()
}

/// The factory the agent serves stages with.
pub fn factory() -> StageFactory {
    Arc::new(|setup: StageSetup, loading: Loading| {
        Box::pin(async move {
            // Unlisted if loading fails or is dropped midway (the coordinator went away).
            struct Unlist((u64, u32), bool);
            impl Drop for Unlist {
                fn drop(&mut self) {
                    if !self.1 {
                        hosted().lock().unwrap().remove(&self.0);
                    }
                }
            }
            let mut unlist = Unlist((setup.session, setup.layers.start), false);
            let stage = load(setup, loading).await;
            unlist.1 = stage.is_ok();
            stage.map(|s| Box::new(s) as Box<dyn Stage>)
        })
    })
}

/// A part of a model this device runs for another, for the app to show: lending a device
/// to a mesh should never be invisible on it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Hosted {
    pub session: u64,
    /// The model's file name.
    pub model: String,
    pub layers: std::ops::Range<u32>,
    /// Layers of the model in all.
    pub of: u32,
    /// The device that set the stage up and sends it work.
    pub coordinator: String,
    /// Bytes of weights fetched, of `total`; equal once loaded.
    pub loaded: u64,
    pub total: u64,
    pub running: bool,
    /// Layers on this device's GPU, and its name.
    pub gpu_layers: usize,
    pub gpu: Option<String>,
    /// Batches computed so far, and when the last one was.
    pub batches: u64,
    #[serde(skip)]
    pub last: Option<Instant>,
    /// Seconds since the last batch, for the app.
    pub idle_secs: Option<u64>,
}

/// Keyed by session and first layer: one device may run two parts of one pipeline.
fn hosted() -> &'static Mutex<HashMap<(u64, u32), Hosted>> {
    static H: OnceLock<Mutex<HashMap<(u64, u32), Hosted>>> = OnceLock::new();
    H.get_or_init(Mutex::default)
}

/// The stages this device runs now, loading or running, oldest first.
pub fn hosted_now() -> Vec<Hosted> {
    let mut list: Vec<Hosted> = hosted().lock().unwrap().values().cloned().collect();
    for h in &mut list {
        h.idle_secs = h.last.map(|t| t.elapsed().as_secs());
    }
    list.sort_by_key(|h| (h.session, h.layers.start));
    list
}

async fn header(url: &str, source: &Source) -> Result<Arc<Header>> {
    if let Some(h) = weights().lock().unwrap().headers.get(url) {
        return Ok(h.clone());
    }
    let h = Arc::new(source.header().await.with_context(|| format!("reading the model at {url}"))?);
    weights().lock().unwrap().headers.insert(url.into(), h.clone());
    Ok(h)
}

async fn load(setup: StageSetup, loading: Loading) -> Result<LlmStage> {
    let url = setup.model.clone();
    let source = Source::parse_remote(&url)?;
    let h = header(&url, &source).await?;
    let cfg = Config::from_header(&h)?;
    // By layers, not by place in the pipeline: when the coordinator holds the first
    // layers itself, the first remote stage takes hidden states, not tokens.
    let first = setup.layers.start == 0;
    let last = setup.layers.end as usize == cfg.n_layer;
    let wanted = model::needed(&h, &setup.layers, first, last);
    let total: u64 = wanted.iter().map(|t| t.range.end - t.range.start).sum();
    let key = (setup.session, setup.layers.start);
    hosted().lock().unwrap().insert(
        key,
        Hosted {
            session: setup.session,
            model: url.rsplit('/').next().unwrap_or(&url).trim_end_matches(".gguf").to_string(),
            layers: setup.layers.clone(),
            of: cfg.n_layer as u32,
            coordinator: loading.coordinator.to_string(),
            loaded: 0,
            total,
            running: false,
            gpu_layers: 0,
            gpu: None,
            batches: 0,
            last: None,
            idle_secs: None,
        },
    );
    let report = |done: u64| {
        if let Some(h) = hosted().lock().unwrap().get_mut(&key) {
            h.loaded = done;
        }
        (loading.progress)(done, total);
    };
    let mut have = HashMap::new();
    let mut missing = Vec::new();
    {
        let mut w = weights().lock().unwrap();
        for t in &wanted {
            match w.tensors.get_mut(&(url.clone(), t.name.clone())) {
                Some(c) => {
                    c.last_used = Instant::now();
                    have.insert(t.name.clone(), c.tensor.clone());
                }
                None => missing.push(*t),
            }
        }
    }
    let need: u64 = missing.iter().map(|t| t.range.end - t.range.start).sum();
    let free = crate::probe::memory().1;
    // Zero means the system would not say; then the fetch itself is the test.
    if free > 0 && need as f64 > free as f64 * MEMORY_SHARE {
        bail!(
            "layers {:?} need {:.1} GB and this device has {:.1} GB free",
            setup.layers,
            need as f64 / 1e9,
            free as f64 / 1e9
        );
    }
    // Weights already in memory count as loaded.
    let cached = total - need;
    report(cached);
    let fetched = source.tensors(&missing, |done, _| report(cached + done)).await?;
    {
        let mut w = weights().lock().unwrap();
        for (name, t) in fetched {
            w.tensors.insert((url.clone(), name.clone()), Cached { tensor: t.clone(), last_used: Instant::now() });
            have.insert(name, t);
        }
    }
    let layers = setup.layers.clone();
    let part = tokio::task::spawn_blocking(move || Part::new(cfg, layers, first, last, &have)).await??;
    report(total);
    if let Some(h) = hosted().lock().unwrap().get_mut(&key) {
        h.running = true;
        h.gpu_layers = part.gpu_layers().0;
        h.gpu = bc_infer::gpu::get().filter(|_| h.gpu_layers > 0).map(|g| g.name.clone());
    }
    Ok(LlmStage { part, url, streams: HashMap::new(), hosted: Some(key) })
}

pub struct LlmStage {
    part: Part,
    url: String,
    streams: HashMap<u64, (Kv, Instant)>,
    /// Its entry in [`hosted_now`], when another device set it up.
    hosted: Option<(u64, u32)>,
}

impl LlmStage {
    /// A stage from weights already in memory (the coordinator's own, and tests).
    pub fn new(part: Part) -> Self {
        LlmStage { part, url: String::new(), streams: HashMap::new(), hosted: None }
    }
}

impl Drop for LlmStage {
    fn drop(&mut self) {
        if let Some(k) = self.hosted {
            hosted().lock().unwrap().remove(&k);
        }
        // The weights' grace period starts now.
        if let Ok(mut w) = weights().try_lock() {
            for ((u, _), c) in w.tensors.iter_mut() {
                if *u == self.url {
                    c.last_used = Instant::now();
                }
            }
        }
    }
}

impl Stage for LlmStage {
    fn forward(&mut self, meta: &Meta, input: Activation) -> Result<Activation> {
        if let Some(k) = self.hosted {
            if let Some(h) = hosted().lock().unwrap().get_mut(&k) {
                h.batches += 1;
                h.last = Some(Instant::now());
            }
        }
        let n_embd = self.part.cfg.n_embd;
        let (mut x, rows) = match input {
            Activation::Tokens(t) => (self.part.embed(&t)?, t.len()),
            Activation::Hidden(t) => {
                if t.cols != n_embd {
                    bail!("hidden states of width {} for a model of width {n_embd}", t.cols);
                }
                (t.data, t.rows)
            }
        };
        if !self.streams.contains_key(&meta.stream) && self.streams.len() >= MAX_STREAMS {
            if let Some(oldest) = self.streams.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| *k) {
                self.streams.remove(&oldest);
            }
        }
        let part = &self.part;
        let (kv, used) = self.streams.entry(meta.stream).or_insert_with(|| (part.new_kv(), Instant::now()));
        *used = Instant::now();
        // A stream starts over at position 0, or continues from where it is.
        let pos = meta.pos as usize;
        if pos > kv.len {
            bail!("stream {} is at {} tokens here, the batch starts at {pos}; replay it from the start", meta.stream, kv.len);
        }
        part.forward(&mut x, rows, pos, kv)?;
        if !part.is_last() {
            return Ok(Activation::Hidden(Tensor { rows, cols: n_embd, data: x }));
        }
        let from = if meta.all_rows { 0 } else { rows - 1 };
        let scores = part.logits(&x[from * n_embd..], rows - from)?;
        let vocab = part.cfg.vocab;
        let ids = (from..rows)
            .zip(scores.chunks_exact(vocab))
            .map(|(r, s)| sample::sample(s, &meta.sampling, (pos + r) as u64))
            .collect();
        Ok(Activation::Tokens(ids))
    }
}
