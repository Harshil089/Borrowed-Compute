//! Where each part of a model runs: the load balancer's brain.
//!
//! The model is cut into contiguous layer ranges, one per device, and the devices form a
//! ring: the coordinator sends token ids to the first device, each device runs its layers
//! and hands the hidden state straight to the next, and the last one sends the sampled
//! token back. Only pipeline parallelism crosses devices. Tensor parallelism needs two
//! all-reduces per layer, 160 network round trips per token for an 80-layer model, which
//! no internet link can afford.
//!
//! Two goals pull in different directions:
//! - **Latency** (one conversation): a token visits every stage in turn, so the time per
//!   token is the sum of every stage's work plus every hop. Spreading layers over more
//!   devices never makes one conversation faster; it only adds hops. So the planner uses
//!   the fewest, fastest devices whose memory holds the model.
//! - **Throughput** (many requests at once): with a request in flight on every stage, the
//!   slowest stage sets the pace. The planner balances layers so each stage takes the
//!   same time, and splits the devices into several independent pipelines (replicas) when
//!   that serves more tokens per device.
//!
//! Decoding one token reads every weight once, so a layer's time is its bytes over the
//! device's memory bandwidth, which the probe measures. Prompt processing is compute
//! bound instead, and is not modelled here beyond the bytes it sends between stages.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Share of measured copy bandwidth that inference reaches while streaming weights.
const BANDWIDTH_EFFICIENCY: f64 = 0.7;
/// A new plan replaces a running one only when it is this much better, so a short burst
/// of load on one device does not reshuffle a model back and forth.
pub const HYSTERESIS: f64 = 0.15;
/// Devices searched exhaustively; beyond this the fastest are kept.
const MAX_SEARCH: usize = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub layers: u32,
    /// Weights of one transformer layer, as stored (quantization included).
    pub layer_bytes: u64,
    /// Embeddings and output head: half on the first stage, half on the last.
    pub extra_bytes: u64,
    /// Width of the hidden state sent between stages, in values.
    pub hidden: u32,
    /// Where its GGUF file is published.
    #[serde(default)]
    pub url: Option<String>,
    /// A small model of the same family (same vocabulary) to draft tokens with.
    #[serde(default)]
    pub draft: Option<String>,
}

const GB: u64 = 1 << 30;

/// Models known by name: 4-bit quantized (Q4_K_M) GGUF files on Hugging Face. Sizes are
/// for planning without a download; running a model reads its real header.
pub fn catalog() -> Vec<Model> {
    let hf = |path: &str| Some(format!("https://huggingface.co/{path}"));
    let m = |name: &str, layers: u32, total_gb: f64, extra_gb: f64, hidden: u32, url: Option<String>, draft: Option<&str>| Model {
        name: name.into(),
        layers,
        layer_bytes: (((total_gb - extra_gb) * GB as f64) / layers as f64) as u64,
        extra_bytes: (extra_gb * GB as f64) as u64,
        hidden,
        url,
        draft: draft.map(String::from),
    };
    vec![
        m("qwen2.5-0.5b", 24, 0.46, 0.13, 896, hf("Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"), None),
        m("llama-3.2-1b", 16, 0.75, 0.2, 2048, hf("bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"), None),
        m("qwen2.5-1.5b", 28, 1.04, 0.2, 1536, hf("Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"), Some("qwen2.5-0.5b")),
        m("qwen2.5-7b", 28, 4.36, 0.6, 3584, hf("bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"), Some("qwen2.5-0.5b")),
        m("llama-3.1-8b", 32, 4.58, 0.6, 4096, hf("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"), Some("llama-3.2-1b")),
        m("qwen2.5-32b", 64, 18.5, 0.9, 5120, hf("bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"), Some("qwen2.5-0.5b")),
        m("llama-3.3-70b", 80, 39.6, 1.3, 8192, hf("bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf"), Some("llama-3.2-1b")),
        m("qwen2.5-72b", 80, 44.2, 1.4, 8192, hf("bartowski/Qwen2.5-72B-Instruct-GGUF/resolve/main/Qwen2.5-72B-Instruct-Q4_K_M.gguf"), Some("qwen2.5-0.5b")),
        // Llama 3.1 405B (Meta's own files are gated; Hermes 3 is the open fine-tune), in 7 files.
        m("hermes-3-405b", 126, 226.4, 2.6, 16384, hf("bartowski/Hermes-3-Llama-3.1-405B-GGUF/resolve/main/Hermes-3-Llama-3.1-405B-Q4_K_M/Hermes-3-Llama-3.1-405B-Q4_K_M-00001-of-00007.gguf"), Some("llama-3.2-1b")),
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Device {
    pub name: String,
    /// Memory this device can give the model (weights and its share of the KV cache).
    pub mem_bytes: u64,
    /// Measured memory bandwidth, bytes a second.
    pub mem_bw: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Link {
    pub rtt_ms: f64,
    /// Bytes a second.
    pub bw: f64,
}

/// Links between every pair of devices and the coordinator, symmetric. The coordinator
/// is index `devices.len()`.
#[derive(Debug, Clone)]
pub struct Links {
    n: usize,
    m: Vec<Link>,
}

impl Links {
    /// `devices` devices plus the coordinator, every link `default`.
    pub fn new(devices: usize, default: Link) -> Self {
        let n = devices + 1;
        Links { n, m: vec![default; n * n] }
    }

    pub fn coordinator(&self) -> usize {
        self.n - 1
    }

    pub fn set(&mut self, a: usize, b: usize, link: Link) {
        self.m[a * self.n + b] = link;
        self.m[b * self.n + a] = link;
    }

    pub fn get(&self, a: usize, b: usize) -> Link {
        self.m[a * self.n + b]
    }

    /// Only coordinator links measured: a device-to-device link is taken to cost both
    /// legs through the coordinator's side of the internet, an upper bound.
    pub fn from_coordinator(to_devices: &[Link]) -> Self {
        let mut links = Links::new(to_devices.len(), Link { rtt_ms: 0.0, bw: f64::INFINITY });
        let c = links.coordinator();
        for (i, a) in to_devices.iter().enumerate() {
            links.set(i, c, *a);
            for (j, b) in to_devices.iter().enumerate().skip(i + 1) {
                links.set(i, j, Link { rtt_ms: a.rtt_ms + b.rtt_ms, bw: a.bw.min(b.bw) });
            }
        }
        links
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Goal {
    Latency,
    Throughput,
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub goal: Goal,
    /// Wire bytes per hidden value (about 1.06 for [`crate::codec::Encoding::Int8`]).
    pub bytes_per_value: f64,
    /// Tokens accepted per pipeline pass with speculative decoding (a draft model on the
    /// coordinator proposes tokens, one pass checks them all); 1.0 without it.
    pub tokens_per_pass: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options { goal: Goal::Latency, bytes_per_value: 1.0625, tokens_per_pass: 1.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagePlan {
    pub device: usize,
    pub layers: Range<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipelinePlan {
    pub stages: Vec<StagePlan>,
    /// Milliseconds per generated token for one conversation.
    pub token_ms: f64,
    /// The part of `token_ms` spent on the network.
    pub network_ms: f64,
    /// Tokens a second with a request in flight on every stage.
    pub tokens_per_s: f64,
    /// Network time for 1000 prompt tokens to cross the slowest hop.
    pub prefill_net_ms_per_1k: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub pipelines: Vec<PipelinePlan>,
    pub unused: Vec<usize>,
}

impl Plan {
    pub fn tokens_per_s(&self) -> f64 {
        self.pipelines.iter().map(|p| p.tokens_per_s).sum()
    }
}

fn layer_secs(model: &Model, d: &Device) -> f64 {
    model.layer_bytes as f64 / (d.mem_bw * BANDWIDTH_EFFICIENCY).max(1.0)
}

/// Layers each device in `order` can hold, given which ends carry the extras.
fn caps(model: &Model, devices: &[Device], order: &[usize]) -> Vec<u32> {
    let last = order.len() - 1;
    order
        .iter()
        .enumerate()
        .map(|(pos, &d)| {
            let ends = (pos == 0) as u64 + (pos == last) as u64;
            let free = devices[d].mem_bytes.saturating_sub(model.extra_bytes / 2 * ends);
            (free / model.layer_bytes.max(1)).min(model.layers as u64) as u32
        })
        .collect()
}

/// Layers per device, in `order`, or `None` if they do not fit or a device gets none.
fn split(model: &Model, devices: &[Device], order: &[usize], goal: Goal) -> Option<Vec<u32>> {
    let caps = caps(model, devices, order);
    if caps.iter().map(|&c| c as u64).sum::<u64>() < model.layers as u64 {
        return None;
    }
    let t: Vec<f64> = order.iter().map(|&d| layer_secs(model, &devices[d])).collect();
    let mut n = vec![0u32; order.len()];
    match goal {
        // The sum of n·t is all that matters: fill the fastest first.
        Goal::Latency => {
            let mut by_speed: Vec<usize> = (0..order.len()).collect();
            by_speed.sort_by(|&a, &b| t[a].total_cmp(&t[b]));
            let mut left = model.layers;
            for i in by_speed {
                n[i] = caps[i].min(left);
                left -= n[i];
            }
        }
        // Each layer goes where the stage's new total stays smallest: optimal for the
        // slowest stage, which sets the pace.
        Goal::Throughput => {
            for _ in 0..model.layers {
                let i = (0..order.len())
                    .filter(|&i| n[i] < caps[i])
                    .min_by(|&a, &b| ((n[a] + 1) as f64 * t[a]).total_cmp(&((n[b] + 1) as f64 * t[b])))?;
                n[i] += 1;
            }
        }
    }
    n.iter().all(|&k| k > 0).then_some(n)
}

/// Times for a pipeline with these stages. Also re-estimates a running plan when
/// measurements change.
pub fn estimate(model: &Model, devices: &[Device], links: &Links, opts: &Options, stages: &[StagePlan]) -> PipelinePlan {
    let c = links.coordinator();
    let hidden_bytes = model.hidden as f64 * opts.bytes_per_value;
    let stage_ms: Vec<f64> = stages
        .iter()
        .map(|s| (s.layers.end - s.layers.start) as f64 * layer_secs(model, &devices[s.device]) * 1000.0)
        .collect();
    // Token ids are a few bytes: those hops cost half a round trip. Hidden states also
    // pay for their bytes.
    let mut network_ms = links.get(c, stages[0].device).rtt_ms / 2.0 + links.get(stages[stages.len() - 1].device, c).rtt_ms / 2.0;
    let mut slowest_transfer_ms: f64 = 0.0;
    for pair in stages.windows(2) {
        let link = links.get(pair[0].device, pair[1].device);
        let transfer_ms = hidden_bytes / link.bw * 1000.0;
        network_ms += link.rtt_ms / 2.0 + transfer_ms;
        slowest_transfer_ms = slowest_transfer_ms.max(transfer_ms);
    }
    let compute_ms: f64 = stage_ms.iter().sum();
    let pace_ms = stage_ms.iter().cloned().fold(slowest_transfer_ms, f64::max);
    PipelinePlan {
        stages: stages.to_vec(),
        token_ms: (compute_ms + network_ms) / opts.tokens_per_pass.max(1.0),
        network_ms: network_ms / opts.tokens_per_pass.max(1.0),
        tokens_per_s: 1000.0 / pace_ms.max(1e-9),
        prefill_net_ms_per_1k: slowest_transfer_ms * 1000.0,
    }
}

fn stages_for(order: &[usize], counts: &[u32]) -> Vec<StagePlan> {
    let mut start = 0;
    order
        .iter()
        .zip(counts)
        .map(|(&device, &k)| {
            let s = StagePlan { device, layers: start..start + k };
            start += k;
            s
        })
        .collect()
}

/// True if `a` beats `b` for the goal.
fn better(goal: Goal, a: &PipelinePlan, b: &PipelinePlan) -> bool {
    match goal {
        Goal::Latency => a.token_ms < b.token_ms,
        // Tokens a second per device used: a replica on fewer devices leaves the rest
        // free for another. Ties go to the lower latency.
        Goal::Throughput => {
            let (ea, eb) = (a.tokens_per_s / a.stages.len() as f64, b.tokens_per_s / b.stages.len() as f64);
            if (ea - eb).abs() > eb * 1e-9 {
                ea > eb
            } else {
                a.token_ms < b.token_ms
            }
        }
    }
}

fn permutations(items: &mut Vec<usize>, k: usize, visit: &mut impl FnMut(&[usize])) {
    if k == items.len() {
        visit(items);
        return;
    }
    for i in k..items.len() {
        items.swap(k, i);
        permutations(items, k + 1, visit);
        items.swap(k, i);
    }
}

/// The best single pipeline from `available`, trying every subset and order.
fn best_pipeline(model: &Model, devices: &[Device], links: &Links, opts: &Options, available: &[usize]) -> Option<PipelinePlan> {
    let mut pool = available.to_vec();
    pool.sort_by(|&a, &b| devices[b].mem_bw.total_cmp(&devices[a].mem_bw));
    pool.truncate(MAX_SEARCH);
    let mut best: Option<PipelinePlan> = None;
    for mask in 1u32..(1 << pool.len()) {
        let mut subset: Vec<usize> = (0..pool.len()).filter(|i| mask & (1 << i) != 0).map(|i| pool[i]).collect();
        permutations(&mut subset, 0, &mut |order| {
            if let Some(counts) = split(model, devices, order, opts.goal) {
                let plan = estimate(model, devices, links, opts, &stages_for(order, &counts));
                if best.as_ref().is_none_or(|b| better(opts.goal, &plan, b)) {
                    best = Some(plan);
                }
            }
        });
    }
    best
}

/// Places `model` on `devices`. For latency, one pipeline; for throughput, as many
/// replicas as the devices hold.
pub fn plan(model: &Model, devices: &[Device], links: &Links, opts: &Options) -> Plan {
    let mut available: Vec<usize> = (0..devices.len()).collect();
    let mut pipelines = Vec::new();
    while let Some(p) = best_pipeline(model, devices, links, opts, &available) {
        available.retain(|d| !p.stages.iter().any(|s| s.device == *d));
        pipelines.push(p);
        if opts.goal == Goal::Latency {
            break;
        }
    }
    Plan { pipelines, unused: available }
}

/// Plans again with fresh measurements, keeping `current` unless the new plan is at
/// least [`HYSTERESIS`] better or `current` no longer fits.
pub fn replan(model: &Model, devices: &[Device], links: &Links, opts: &Options, current: &Plan) -> Plan {
    let fresh = plan(model, devices, links, opts);
    let fits = current.pipelines.iter().all(|p| {
        let order: Vec<usize> = p.stages.iter().map(|s| s.device).collect();
        let caps = caps(model, devices, &order);
        p.stages.iter().zip(caps).all(|(s, cap)| s.layers.end - s.layers.start <= cap)
    });
    if !fits || current.pipelines.is_empty() {
        return fresh;
    }
    let now = Plan {
        pipelines: current.pipelines.iter().map(|p| estimate(model, devices, links, opts, &p.stages)).collect(),
        unused: current.unused.clone(),
    };
    let gain = match opts.goal {
        Goal::Latency => now.pipelines[0].token_ms / fresh.pipelines.first().map_or(f64::INFINITY, |p| p.token_ms) - 1.0,
        Goal::Throughput => fresh.tokens_per_s() / now.tokens_per_s() - 1.0,
    };
    if gain >= HYSTERESIS {
        fresh
    } else {
        now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(layers: u32) -> Model {
        Model { name: "m".into(), layers, layer_bytes: GB, extra_bytes: 0, hidden: 8192, url: None, draft: None }
    }

    fn dev(name: &str, mem_gb: u64, bw_gbs: f64) -> Device {
        Device { name: name.into(), mem_bytes: mem_gb * GB, mem_bw: bw_gbs * GB as f64 }
    }

    fn net(n: usize, rtt_ms: f64) -> Links {
        Links::new(n, Link { rtt_ms, bw: 10e6 })
    }

    fn layers(p: &PipelinePlan) -> Vec<(usize, u32)> {
        p.stages.iter().map(|s| (s.device, s.layers.end - s.layers.start)).collect()
    }

    #[test]
    fn latency_uses_one_device_when_the_model_fits() {
        let devices = [dev("a", 64, 100.0), dev("b", 64, 400.0)];
        let p = plan(&model(40), &devices, &net(2, 40.0), &Options::default());
        assert_eq!(layers(&p.pipelines[0]), [(1, 40)], "the fast device alone, no hops");
    }

    #[test]
    fn latency_fills_the_fastest_device_and_spills_the_rest() {
        let devices = [dev("slow", 64, 100.0), dev("fast", 30, 400.0)];
        let p = plan(&model(40), &devices, &net(2, 40.0), &Options::default());
        let mut l = layers(&p.pipelines[0]);
        l.sort();
        assert_eq!(l, [(0, 10), (1, 30)]);
    }

    #[test]
    fn nothing_fits_gives_no_pipeline() {
        let p = plan(&model(40), &[dev("a", 10, 100.0), dev("b", 10, 100.0)], &net(2, 10.0), &Options::default());
        assert!(p.pipelines.is_empty());
        assert_eq!(p.unused, [0, 1]);
    }

    #[test]
    fn throughput_balances_stage_times_by_speed() {
        let devices = [dev("a", 64, 100.0), dev("b", 64, 300.0)];
        let opts = Options { goal: Goal::Throughput, ..Default::default() };
        // Neither fits 80 layers alone, so one pipeline over both, 1:3 by bandwidth.
        let p = plan(&model(80), &[dev("a", 40, 100.0), dev("b", 70, 300.0)], &net(2, 10.0), &opts);
        let mut l = layers(&p.pipelines[0]);
        l.sort();
        assert_eq!(l, [(0, 20), (1, 60)]);
        // When each holds the whole model, two replicas beat one shared pipeline.
        let p = plan(&model(40), &devices, &net(2, 10.0), &opts);
        assert_eq!(p.pipelines.len(), 2);
        assert!(p.unused.is_empty());
    }

    #[test]
    fn throughput_prefers_replicas_on_equal_devices() {
        // Four equal devices, each holding half the model: two 2-stage replicas serve as
        // much as one 4-stage pipeline with half the hops.
        let devices: Vec<Device> = (0..4).map(|i| dev(&i.to_string(), 21, 200.0)).collect();
        let opts = Options { goal: Goal::Throughput, ..Default::default() };
        let p = plan(&model(40), &devices, &net(4, 30.0), &opts);
        assert_eq!(p.pipelines.len(), 2);
        assert!(p.pipelines.iter().all(|p| p.stages.len() == 2));
    }

    #[test]
    fn ring_order_avoids_the_long_link() {
        // b sits far from the coordinator but next to a. Best ring: c → a … but the
        // coordinator-to-b leg is unavoidable only if b is at an end; put b in the middle
        // of three.
        let devices = [dev("a", 14, 100.0), dev("b", 14, 100.0), dev("x", 14, 100.0)];
        let mut links = net(3, 20.0);
        let c = links.coordinator();
        links.set(1, c, Link { rtt_ms: 300.0, bw: 10e6 });
        let p = plan(&model(40), &devices, &links, &Options::default());
        let order: Vec<usize> = p.pipelines[0].stages.iter().map(|s| s.device).collect();
        assert_eq!(order.len(), 3);
        assert_eq!(order[1], 1, "the far device is not next to the coordinator: {order:?}");
    }

    #[test]
    fn estimates_add_up() {
        let devices = [dev("a", 64, 100.0 / 0.7), dev("b", 64, 100.0 / 0.7)];
        let links = net(2, 40.0);
        let stages = [StagePlan { device: 0, layers: 0..10 }, StagePlan { device: 1, layers: 10..30 }];
        let p = estimate(&model(30), &devices, &links, &Options::default(), &stages);
        // 1 GB a layer at an effective 100 GB/s: 10 ms a layer.
        let hop_bytes = 8192.0 * 1.0625;
        let transfer = hop_bytes / 10e6 * 1000.0;
        assert!((p.network_ms - (20.0 + 20.0 + 20.0 + transfer)).abs() < 1e-6, "{}", p.network_ms);
        assert!((p.token_ms - (300.0 + p.network_ms)).abs() < 1e-6);
        assert!((p.tokens_per_s - 1000.0 / 200.0).abs() < 1e-6, "paced by the 20-layer stage");
        let fast = estimate(&model(30), &devices, &links, &Options { tokens_per_pass: 3.0, ..Default::default() }, &stages);
        assert!((fast.token_ms - p.token_ms / 3.0).abs() < 1e-6, "speculation spreads one pass over 3 tokens");
    }

    #[test]
    fn replan_ignores_small_gains_and_moves_on_big_ones() {
        let opts = Options { goal: Goal::Throughput, ..Default::default() };
        let m = model(80);
        let links = net(2, 10.0);
        let mut devices = vec![dev("a", 70, 200.0), dev("b", 70, 200.0)];
        let current = plan(&m, &devices, &links, &opts);
        // b gets a little busier: not worth moving layers.
        devices[1].mem_bw = 190.0 * GB as f64;
        let next = replan(&m, &devices, &links, &opts, &current);
        assert_eq!(layers(&next.pipelines[0]), layers(&current.pipelines[0]));
        // b's owner starts a heavy job: shift layers to a.
        devices[1].mem_bw = 60.0 * GB as f64;
        let next = replan(&m, &devices, &links, &opts, &current);
        let shifted: Vec<(usize, u32)> = layers(&next.pipelines[0]);
        let on_b = shifted.iter().find(|(d, _)| *d == 1).map(|l| l.1).unwrap();
        assert!(on_b < 40, "{shifted:?}");
        assert!(next.tokens_per_s() > current.tokens_per_s() * 0.5);
    }

    #[test]
    fn coordinator_only_links_bound_device_pairs() {
        let links = Links::from_coordinator(&[Link { rtt_ms: 10.0, bw: 5e6 }, Link { rtt_ms: 30.0, bw: 2e6 }]);
        assert_eq!(links.get(0, 1), Link { rtt_ms: 40.0, bw: 2e6 });
        assert_eq!(links.get(1, links.coordinator()).rtt_ms, 30.0);
    }

    #[test]
    fn catalog_sizes_are_consistent() {
        for m in catalog() {
            let total = m.layer_bytes * m.layers as u64 + m.extra_bytes;
            assert!(total > GB / 4 && m.hidden >= 896, "{}", m.name);
            if let Some(d) = &m.draft {
                assert!(catalog().iter().any(|c| &c.name == d), "{}'s draft {d} is in the catalog", m.name);
            }
        }
    }
}
