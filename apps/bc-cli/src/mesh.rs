//! Several devices at once, from this one: measure every paired device together, and
//! plan a model across them. Shared by `bc mesh` and the desktop app.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use bc_auth::CloudClient;
use bc_infer::{Local, Source};
use bc_mesh::llm::{Client, Llm};
use bc_mesh::plan::{self, Device, Goal, Link, Links, Model, Options, Plan};
use bc_mesh::probe::LinkReport;
use bc_proto::cloud::PairingStatus;
use iroh::{Endpoint, EndpointAddr, EndpointId};

/// Share of a device's free memory the planner may fill, leaving room for its owner and
/// for the KV cache.
const MEMORY_SHARE: f64 = 0.85;

#[derive(Clone)]
pub struct MeshDevice {
    pub name: String,
    pub addr: EndpointAddr,
    pub report: Result<LinkReport, String>,
    /// This computer, the coordinator: it holds layers too, ahead of the others.
    pub local: bool,
}

/// This computer as a device of its own mesh: its memory and GPU, measured here, and a
/// link to itself that costs nothing.
pub async fn this_computer(endpoint: &Endpoint) -> MeshDevice {
    let device = bc_mesh::probe::report(true).await;
    // Finite, so it survives JSON for the app.
    let report = LinkReport { rtt_ms: 0.0, up_bytes_s: 1e12, via_relay: false, device };
    MeshDevice { name: "This computer".into(), addr: endpoint.addr(), report: Ok(report), local: true }
}

/// Devices this one is paired with (approved), with the cloud's latest address for each.
pub async fn paired_devices(cloud: &CloudClient, me: &str) -> Result<Vec<(String, EndpointAddr)>> {
    let approved: Vec<String> = cloud
        .pairings(me)
        .await?
        .into_iter()
        .filter(|p| p.client_id == me && p.status == PairingStatus::Approved)
        .map(|p| p.mac_id)
        .collect();
    let mut out = Vec::new();
    for d in cloud.devices().await? {
        // An offline device would only cost a wait before it is counted as away.
        if !approved.contains(&d.endpoint_id) || !d.online {
            continue;
        }
        let Ok(id) = d.endpoint_id.parse::<EndpointId>() else { continue };
        // Trust the cloud's address only for the key it claims to belong to.
        let addr = d
            .addr
            .and_then(|a| serde_json::from_value::<EndpointAddr>(a).ok())
            .filter(|a| a.id == id)
            .unwrap_or_else(|| id.into());
        out.push((d.name, addr));
    }
    Ok(out)
}

/// Devices a Mesh may use: those paired on this account, and computers shared with this
/// device as a guest. A shared computer takes part only if its owner lets guests use it
/// in their Mesh; otherwise it turns the probe away and is left out. Signed out, or with
/// the account's cloud out of reach (the Mac that hosts it closed or asleep), only shared
/// computers: the Mesh still runs on this computer alone rather than failing.
pub async fn mesh_targets(cloud: &CloudClient, me: &str, home: &std::path::Path) -> Result<Vec<(String, EndpointAddr)>> {
    let shared = crate::shared::list(home);
    let mut out = match paired_devices(cloud, me).await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("paired devices left out: {e:#}");
            Vec::new()
        }
    };
    for s in shared {
        if let Ok(id) = s.id.parse::<EndpointId>() {
            if !out.iter().any(|(_, a)| a.id == id) {
                out.push((s.name, id.into()));
            }
        }
    }
    Ok(out)
}

/// Probes every device at the same time over one endpoint.
pub async fn probe_all(endpoint: &Endpoint, devices: Vec<(String, EndpointAddr)>) -> Vec<MeshDevice> {
    let tasks: Vec<_> = devices
        .into_iter()
        .map(|(name, addr)| {
            let endpoint = endpoint.clone();
            tokio::spawn(async move {
                let report = bc_mesh::probe::probe(&endpoint, addr.clone()).await.map_err(|e| format!("{e:#}"));
                MeshDevice { name, addr, report, local: false }
            })
        })
        .collect();
    let mut out = Vec::with_capacity(tasks.len());
    for t in tasks {
        if let Ok(d) = t.await {
            out.push(d);
        }
    }
    out
}

/// Plans `model` over the devices that answered. Plan device indexes point into the
/// returned list.
/// Measured links between devices, by index into the list given to [`plan_for`].
pub type Pairs = Vec<((usize, usize), Link)>;

/// Has every pair of `devices` measure the link between them, all at once. Pairs that
/// fail are left out; the planner then estimates them through this device.
pub async fn probe_pairs(endpoint: &Endpoint, devices: &[MeshDevice]) -> Pairs {
    let mut tasks = Vec::new();
    // This computer's links are the coordinator's own, measured already.
    let remote = |i: usize| !devices[i].local;
    for i in (0..devices.len()).filter(|&i| remote(i)) {
        for j in (i + 1..devices.len()).filter(|&j| remote(j)) {
            let (ep, a, b) = (endpoint.clone(), devices[i].addr.clone(), devices[j].addr.clone());
            tasks.push(tokio::spawn(async move { ((i, j), bc_mesh::probe::probe_link(&ep, a, b).await) }));
        }
    }
    let mut out = Vec::new();
    for t in tasks {
        if let Ok(((i, j), Ok((rtt_ms, bw, _)))) = t.await {
            out.push(((i, j), Link { rtt_ms, bw }));
        }
    }
    out
}

pub fn plan_for<'a>(model: &Model, devices: &'a [MeshDevice], goal: Goal, tokens_per_pass: f64) -> (Plan, Vec<&'a MeshDevice>) {
    plan_with(model, devices, &Vec::new(), goal, tokens_per_pass, &[])
}

/// [`plan_for`] with measured device-to-device links. Pair indexes are into `devices`,
/// which must all have answered. Devices named in `include` get layers even when the
/// plan would be faster without them.
pub fn plan_with<'a>(model: &Model, devices: &'a [MeshDevice], pairs: &Pairs, goal: Goal, tokens_per_pass: f64, include: &[String]) -> (Plan, Vec<&'a MeshDevice>) {
    let usable: Vec<&MeshDevice> = devices.iter().filter(|d| d.report.is_ok()).collect();
    let reports: Vec<&LinkReport> = usable.iter().map(|d| d.report.as_ref().unwrap()).collect();
    let specs: Vec<Device> = usable.iter().zip(&reports).map(|(d, r)| Device { local: d.local, include: include.contains(&d.name), ..device_spec(&d.name, r) }).collect();
    let mut links = Links::from_coordinator(&reports.iter().map(|r| Link { rtt_ms: r.rtt_ms, bw: r.up_bytes_s }).collect::<Vec<_>>());
    if usable.len() == devices.len() {
        for &((i, j), link) in pairs {
            links.set(i, j, link);
        }
    }
    let opts = Options { goal, tokens_per_pass, ..Default::default() };
    (plan::plan(model, &specs, &links, &opts), usable)
}

/// What the planner knows of a device from its report: its memory, and its GPU's when it
/// has one (on unified memory the GPU reaches all of it; a discrete card holds what fits
/// in 85% of its free memory, as the engine does).
pub fn device_spec(name: &str, r: &LinkReport) -> Device {
    let mem_bytes = (r.device.mem_available as f64 * MEMORY_SHARE) as u64;
    let (gpu_bytes, gpu_bw) = match &r.device.gpu {
        Some(g) if g.unified => (mem_bytes, g.bw),
        Some(g) => ((g.mem_bytes as f64 * 0.85) as u64, g.bw),
        None => (0, 0.0),
    };
    Device { name: name.into(), mem_bytes, mem_bw: r.device.mem_bw, gpu_bytes, gpu_bw, local: false, include: false }
}

pub fn find_model(name: &str) -> Option<Model> {
    plan::catalog().into_iter().find(|m| m.name.eq_ignore_ascii_case(name))
}

/// A model running across paired devices, from this one.
pub struct Session {
    pub llm: Arc<Llm>,
    pub name: String,
    pub url: String,
    pub spec: Model,
    pub goal: Goal,
    /// Devices the owner wants used even when not needed.
    pub include: Vec<String>,
    endpoint: Endpoint,
    /// Devices drafting for the swarm.
    pub drafter_names: Vec<String>,
    state: tokio::sync::Mutex<SessionState>,
}

/// Swarm drafters at most: past a few, a wider tree no longer pays for its checking.
const MAX_DRAFTERS: usize = 3;

struct SessionState {
    devices: Vec<MeshDevice>,
    pairs: Pairs,
    plan: Plan,
}

/// A catalog name, or a model's address.
pub fn resolve_model(name: &str) -> Result<(String, String, Option<String>)> {
    if let Some(m) = find_model(name) {
        let url = m.url.with_context(|| format!("{name} has no published address"))?;
        return Ok((m.name, url, m.draft));
    }
    if name.starts_with("https://") || name.starts_with("http://") {
        return Ok((name.rsplit('/').next().unwrap_or(name).to_string(), name.to_string(), None));
    }
    bail!("unknown model {name:?}: give a name from `bc mesh plan` or an https address of a .gguf file")
}

/// Context kept for a conversation even when the measured room is smaller: free memory
/// as measured swings with whatever else runs (an agent prompt with its tools and a web
/// page needs about this much), and 8,192 tokens ran on an 8 GB-free M2 with qwen3-0.6b
/// where the next doubling did not. Past it, the engine reports running out of memory
/// plainly.
const MIN_CONTEXT: usize = 8192;

/// Tokens of context the planned stages can hold: each device's memory left after its
/// weights, over the key/value cache a token takes in its layers (f16 on a GPU, f32 on a
/// CPU). The cache grows by a
/// quarter at a time and the old one stays until copied, so a token needs 2.25 times
/// its share at the moment of growing.
fn context_room(spec: &Model, plan: &Plan, devices: &[&MeshDevice], kv_per_token_layer: u64) -> usize {
    let mut room = usize::MAX;
    for p in &plan.pipelines {
        for (i, st) in p.stages.iter().enumerate() {
            let Some(d) = devices.get(st.device) else { continue };
            let Ok(r) = &d.report else { continue };
            let dev = device_spec(&d.name, r);
            let mem = if dev.gpu_bytes > 0 { dev.gpu_bytes } else { dev.mem_bytes };
            let layers = u64::from(st.layers.end - st.layers.start);
            let ends = (i == 0) as u64 + (i + 1 == p.stages.len()) as u64;
            let weights = spec.layer_bytes * layers + spec.extra_bytes / 2 * ends;
            // The GPU keeps keys and values as f16, half the CPU's f32.
            let kv = if r.device.gpu.is_some() { kv_per_token_layer / 2 } else { kv_per_token_layer };
            let per_token = (layers.max(1) * kv * 9).div_ceil(4);
            room = room.min((mem.saturating_sub(weights) / per_token.max(1)) as usize);
        }
    }
    room
}

fn stage_devices(devices: &[MeshDevice]) -> Vec<&MeshDevice> {
    devices.iter().filter(|d| d.report.as_ref().is_ok_and(|r| r.device.engine)).collect()
}

/// Opens every pipeline of `plan`; `on_loading` hears each device's download as it goes
/// (device name, bytes fetched, bytes in all).
async fn open_all(endpoint: &Endpoint, url: &str, plan: &Plan, usable: &[&MeshDevice], on_loading: &(dyn Fn(&str, u64, u64) + Send + Sync)) -> Result<Vec<Arc<Client>>> {
    let addrs: Vec<EndpointAddr> = usable.iter().map(|d| d.addr.clone()).collect();
    // Every pipeline at once, as every stage within one: all devices download together.
    let opens = plan.pipelines.iter().map(|p| {
        let addrs = &addrs;
        async move {
            let stage_loading = |i: usize, done: u64, total: u64| on_loading(&usable[p.stages[i].device].name, done, total);
            Client::open(endpoint, url, p.clone(), addrs, &stage_loading).await
        }
    });
    futures_util::future::try_join_all(opens).await
}

fn gb(bytes: u64) -> String {
    if bytes < 1 << 30 {
        format!("{} MB", bytes >> 20)
    } else {
        format!("{:.1} GB", bytes as f64 / (1u64 << 30) as f64)
    }
}

impl Session {
    /// Probes the paired devices, plans the model over those that can run stages, and
    /// starts the pipelines. `draft` is a model on this computer for speculative decoding;
    /// devices named in `include` get layers even when not needed.
    pub async fn start(endpoint: Endpoint, targets: Vec<(String, EndpointAddr)>, model: &str, goal: Goal, draft: Option<&str>, include: Vec<String>, status: impl Fn(&str) + Send + Sync) -> Result<Self> {
        let (name, url, suggested) = resolve_model(model)?;
        status(&format!("Reading {name}…"));
        let header = Source::parse_remote(&url)?.header().await?;
        let spec = bc_mesh::llm::model_spec(&name, &header)?;
        status(&format!("Measuring {} devices at once…", targets.len()));
        let devices = probe_all(&endpoint, targets).await;
        // A draft published at a web address can also run on other devices, for the swarm.
        let mut draft_url = None;
        let draft = match draft.or(suggested.as_deref()) {
            Some(d) if d != "none" => {
                status(&format!("Downloading the draft model {d} to this computer's memory…"));
                let src = match resolve_model(d) {
                    Ok((_, u, _)) => {
                        draft_url = Some(u.clone());
                        Source::parse_remote(&u)?
                    }
                    Err(_) => Source::parse_local(d)?,
                };
                Some(Local::load_with(&src, |done, total| status(&format!("Downloading the draft model {d} to this computer's memory: {} of {}", gb(done), gb(total)))).await?)
            }
            _ => None,
        };
        let draft_layers = draft.as_ref().map(|d| d.part.cfg.n_layer as u32);
        // Measured after the draft model is in memory, so its room is not counted twice.
        let mut devices = devices;
        devices.push(this_computer(&endpoint).await);
        let tokens_per_pass = if draft.is_some() { 2.5 } else { 1.0 };
        let owned: Vec<MeshDevice> = stage_devices(&devices).iter().map(|d| (*d).clone()).collect();
        status("Measuring the links between your devices…");
        let pairs = probe_pairs(&endpoint, &owned).await;
        let plan = plan_with(&spec, &owned, &pairs, goal, tokens_per_pass, &include).0;
        if plan.pipelines.is_empty() {
            let need = spec.layer_bytes * spec.layers as u64 + spec.extra_bytes;
            let free: u64 = owned.iter().filter_map(|d| d.report.as_ref().ok()).map(|r| (r.device.mem_available as f64 * MEMORY_SHARE) as u64).sum();
            // A device that did not answer is the likelier cause than memory: say so.
            let silent: Vec<&str> = devices.iter().filter(|d| d.report.is_err()).map(|d| d.name.as_str()).collect();
            let away = if silent.is_empty() { String::new() } else { format!(" {} did not answer and cannot be counted.", silent.join(", ")) };
            bail!(
                "{name} needs {} of memory and the devices that answered, this computer included, have {} free together.{away} Close apps on them, add a device, or pick a smaller model.",
                gb(need),
                gb(free)
            );
        }
        let usable = stage_devices(&devices);
        status("Each device is downloading its own layers…");
        // One line for all of them, since they download at the same time.
        let progress = std::sync::Mutex::new(std::collections::BTreeMap::<String, (u64, u64)>::new());
        let clients = open_all(&endpoint, &url, &plan, &usable, &|device, done, total| {
            let line = {
                let mut p = progress.lock().unwrap();
                p.insert(device.to_string(), (done, total));
                p.iter().map(|(d, (done, total))| format!("{d} {} of {}", gb(*done), gb(*total))).collect::<Vec<_>>().join(" · ")
            };
            status(&format!("Downloading layers: {line}"));
        })
        .await?;
        let mut llm = Llm::new(&header, clients, draft)?;
        // The model's own limit can far exceed what the stages hold: the agent keeps under
        // this one, summarizing before the GPU runs out of room.
        let room = context_room(&spec, &plan, &usable, llm.cfg.kv_bytes_per_token_layer());
        llm.cfg.context = llm.cfg.context.min(room.max(MIN_CONTEXT));
        let llm = Arc::new(llm);
        // The swarm: devices the model does not need each run the draft model and grow a
        // branch of every tree of guesses (docs/frontier.md, phase 1).
        let mut drafter_names = Vec::new();
        if let (Some(durl), Some(layers)) = (draft_url, draft_layers) {
            let idle: Vec<usize> = plan.unused.iter().copied().filter(|&i| !usable[i].local).take(MAX_DRAFTERS).collect();
            if !idle.is_empty() {
                status(&format!("Starting {} swarm drafter(s)…", idle.len()));
                let addrs: Vec<EndpointAddr> = usable.iter().map(|d| d.addr.clone()).collect();
                let opens = idle.iter().map(|&i| {
                    let p = bc_mesh::plan::PipelinePlan { stages: vec![bc_mesh::plan::StagePlan { device: i, layers: 0..layers }], token_ms: 0.0, network_ms: 0.0, tokens_per_s: 0.0, prefill_net_ms_per_1k: 0.0 };
                    let (endpoint, durl, addrs) = (&endpoint, &durl, &addrs);
                    async move { (i, Client::open(endpoint, durl, p, addrs, &|_, _, _| {}).await) }
                });
                let mut drafters = Vec::new();
                for (i, opened) in futures_util::future::join_all(opens).await {
                    // A device that cannot draft is simply left out of the swarm.
                    if let Ok(c) = opened {
                        drafters.push(c);
                        drafter_names.push(usable[i].name.clone());
                    }
                }
                llm.set_drafters(drafters);
            }
        }
        Ok(Session { llm, name, url, spec, goal, include, endpoint, drafter_names, state: tokio::sync::Mutex::new(SessionState { devices, pairs, plan }) })
    }

    /// Stages and devices, for display: (device name, layers) per pipeline.
    pub async fn layout(&self) -> Vec<Vec<(String, std::ops::Range<u32>)>> {
        let s = self.state.lock().await;
        let usable = stage_devices(&s.devices);
        s.plan.pipelines.iter().map(|p| p.stages.iter().map(|st| (usable[st.device].name.clone(), st.layers.clone())).collect()).collect()
    }

    pub async fn plan(&self) -> Plan {
        self.state.lock().await.plan.clone()
    }

    /// Moves layers when measured stage times say another split is clearly faster. The
    /// stages' measured milliseconds stand in for each device's speed: a device whose
    /// owner is busy shows up slower and gets fewer layers. Returns whether it moved.
    pub async fn rebalance(&self) -> Result<bool> {
        let mut s = self.state.lock().await;
        let usable: Vec<MeshDevice> = stage_devices(&s.devices).iter().map(|d| (*d).clone()).collect();
        let reports: Vec<&LinkReport> = usable.iter().map(|d| d.report.as_ref().unwrap()).collect();
        let mut specs: Vec<Device> = usable.iter().zip(&reports).map(|(d, r)| Device { local: d.local, include: self.include.contains(&d.name), ..device_spec(&d.name, r) }).collect();
        for c in self.llm.replicas() {
            for (stage, ms) in c.plan.stages.iter().zip(c.stage_ms.lock().unwrap().iter()) {
                let predicted = plan::stage_secs(&self.spec, &specs[stage.device], stage.layers.end - stage.layers.start) * 1000.0;
                if *ms > 0.0 && predicted > 0.0 {
                    // The device is as much faster or slower as measured, GPU and CPU alike.
                    let d = &mut specs[stage.device];
                    let k = predicted / *ms as f64;
                    d.mem_bw *= k;
                    d.gpu_bw *= k;
                }
            }
        }
        let mut links = Links::from_coordinator(&reports.iter().map(|r| Link { rtt_ms: r.rtt_ms, bw: r.up_bytes_s }).collect::<Vec<_>>());
        for &((i, j), link) in &s.pairs {
            links.set(i, j, link);
        }
        let opts = Options { goal: self.goal, ..Default::default() };
        let next = plan::replan(&self.spec, &specs, &links, &opts, &s.plan);
        let moved = next.pipelines.iter().map(|p| &p.stages).ne(s.plan.pipelines.iter().map(|p| &p.stages));
        if moved {
            let refs: Vec<&MeshDevice> = usable.iter().collect();
            let clients = open_all(&self.endpoint, &self.url, &next, &refs, &|_, _, _| {}).await?;
            self.llm.set_replicas(clients);
        }
        s.plan = next;
        Ok(moved)
    }

    /// After a pipeline fails: measure again, leave out devices that do not answer, plan
    /// and start again.
    pub async fn recover(&self, targets: Vec<(String, EndpointAddr)>) -> Result<()> {
        let (mut devices, me) = tokio::join!(probe_all(&self.endpoint, targets), this_computer(&self.endpoint));
        devices.push(me);
        let usable = stage_devices(&devices);
        let owned: Vec<MeshDevice> = usable.iter().map(|d| (*d).clone()).collect();
        let pairs = probe_pairs(&self.endpoint, &owned).await;
        let (plan, _) = plan_with(&self.spec, &owned, &pairs, self.goal, 1.0, &self.include);
        if plan.pipelines.is_empty() {
            bail!("the devices still reachable cannot hold {}", self.name);
        }
        let clients = open_all(&self.endpoint, &self.url, &plan, &usable, &|_, _, _| {}).await?;
        self.llm.set_replicas(clients);
        *self.state.lock().await = SessionState { devices, pairs, plan };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_mesh::msg::{DeviceReport, GpuReport};
    use bc_mesh::plan::{PipelinePlan, StagePlan};

    fn device(name: &str, free_gb: u64) -> MeshDevice {
        let device = DeviceReport {
            name: name.into(),
            os: "macos".into(),
            cores: 8,
            mem_total: 16 << 30,
            mem_available: free_gb << 30,
            mem_bw: 1e11,
            engine: true,
            gpu: Some(GpuReport { name: "Apple M2".into(), mem_bytes: 0, unified: true, bw: 1e11 }),
        };
        MeshDevice { name: name.into(), addr: iroh::SecretKey::generate().public().into(), report: Ok(LinkReport { rtt_ms: 1.0, up_bytes_s: 1e8, via_relay: false, device }), local: false }
    }

    #[test]
    fn the_context_stops_where_the_smallest_stage_runs_out_of_room() {
        let spec = find_model("qwen3-0.6b").unwrap();
        // qwen3-0.6b: 28 layers, 8 key/value heads of 128, f32: 8 KiB a token a layer.
        let kv = 2 * 8 * 128 * 4;
        let stage = |device, layers| StagePlan { device, layers };
        let pipe = |stages| PipelinePlan { stages, token_ms: 0.0, network_ms: 0.0, tokens_per_s: 0.0, prefill_net_ms_per_1k: 0.0 };
        let (big, small) = (device("big", 8), device("small", 1));
        let whole = Plan { pipelines: vec![pipe(vec![stage(0, 0..28)])], unused: vec![] };
        let room = context_room(&spec, &whole, &[&big], kv);
        // 85% of 8 GiB less the weights, over 28 layers × 4 KiB (f16 on the GPU) × 2.25.
        let free = (8u64 << 30) as f64 * MEMORY_SHARE - (spec.layer_bytes * 28 + spec.extra_bytes) as f64;
        assert!((room as f64 - free / (28.0 * (kv / 2) as f64 * 2.25)).abs() < 2.0, "{room}");
        // Split, the device with less memory decides, for its own layers only.
        let split = Plan { pipelines: vec![pipe(vec![stage(0, 0..14), stage(1, 14..28)])], unused: vec![] };
        let small_room = context_room(&spec, &split, &[&big, &small], kv);
        assert!(small_room < room && small_room > 0, "{small_room} vs {room}");
    }
}
