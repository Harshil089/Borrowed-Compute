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

pub struct MeshDevice {
    pub name: String,
    pub addr: EndpointAddr,
    pub report: Result<LinkReport, String>,
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
        if !approved.contains(&d.endpoint_id) {
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

/// Probes every device at the same time over one endpoint.
pub async fn probe_all(endpoint: &Endpoint, devices: Vec<(String, EndpointAddr)>) -> Vec<MeshDevice> {
    let tasks: Vec<_> = devices
        .into_iter()
        .map(|(name, addr)| {
            let endpoint = endpoint.clone();
            tokio::spawn(async move {
                let report = bc_mesh::probe::probe(&endpoint, addr.clone()).await.map_err(|e| format!("{e:#}"));
                MeshDevice { name, addr, report }
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
    for i in 0..devices.len() {
        for j in i + 1..devices.len() {
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
    plan_with(model, devices, &Vec::new(), goal, tokens_per_pass)
}

/// [`plan_for`] with measured device-to-device links. Pair indexes are into `devices`,
/// which must all have answered.
pub fn plan_with<'a>(model: &Model, devices: &'a [MeshDevice], pairs: &Pairs, goal: Goal, tokens_per_pass: f64) -> (Plan, Vec<&'a MeshDevice>) {
    let usable: Vec<&MeshDevice> = devices.iter().filter(|d| d.report.is_ok()).collect();
    let reports: Vec<&LinkReport> = usable.iter().map(|d| d.report.as_ref().unwrap()).collect();
    let specs: Vec<Device> = usable.iter().zip(&reports).map(|(d, r)| device_spec(&d.name, r)).collect();
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
    Device { name: name.into(), mem_bytes, mem_bw: r.device.mem_bw, gpu_bytes, gpu_bw }
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
    endpoint: Endpoint,
    state: tokio::sync::Mutex<SessionState>,
}

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

fn stage_devices(devices: &[MeshDevice]) -> Vec<&MeshDevice> {
    devices.iter().filter(|d| d.report.as_ref().is_ok_and(|r| r.device.engine)).collect()
}

/// Opens every pipeline of `plan`; `on_loading` hears each device's download as it goes
/// (device name, bytes fetched, bytes in all).
async fn open_all(endpoint: &Endpoint, url: &str, plan: &Plan, usable: &[&MeshDevice], on_loading: &(dyn Fn(&str, u64, u64) + Send + Sync)) -> Result<Vec<Arc<Client>>> {
    let addrs: Vec<EndpointAddr> = usable.iter().map(|d| d.addr.clone()).collect();
    let mut clients = Vec::new();
    for p in &plan.pipelines {
        let stage_loading = |i: usize, done: u64, total: u64| on_loading(&usable[p.stages[i].device].name, done, total);
        clients.push(Client::open(endpoint, url, p.clone(), &addrs, &stage_loading).await?);
    }
    Ok(clients)
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
    /// starts the pipelines. `draft` is a model on this computer for speculative decoding.
    pub async fn start(endpoint: Endpoint, targets: Vec<(String, EndpointAddr)>, model: &str, goal: Goal, draft: Option<&str>, status: impl Fn(&str) + Send + Sync) -> Result<Self> {
        let (name, url, suggested) = resolve_model(model)?;
        status(&format!("Reading {name}…"));
        let header = Source::parse_remote(&url)?.header().await?;
        let spec = bc_mesh::llm::model_spec(&name, &header)?;
        status(&format!("Measuring {} devices at once…", targets.len()));
        let devices = probe_all(&endpoint, targets).await;
        let draft = match draft.or(suggested.as_deref()) {
            Some(d) if d != "none" => {
                status(&format!("Downloading the draft model {d} to this computer's memory…"));
                let src = match resolve_model(d) {
                    Ok((_, u, _)) => Source::parse_remote(&u)?,
                    Err(_) => Source::parse_local(d)?,
                };
                Some(Local::load_with(&src, |done, total| status(&format!("Downloading the draft model {d} to this computer's memory: {} of {}", gb(done), gb(total)))).await?)
            }
            _ => None,
        };
        let tokens_per_pass = if draft.is_some() { 2.5 } else { 1.0 };
        let owned: Vec<MeshDevice> = stage_devices(&devices).iter().map(|d| MeshDevice { name: d.name.clone(), addr: d.addr.clone(), report: d.report.clone() }).collect();
        status("Measuring the links between your devices…");
        let pairs = probe_pairs(&endpoint, &owned).await;
        let plan = plan_with(&spec, &owned, &pairs, goal, tokens_per_pass).0;
        if plan.pipelines.is_empty() {
            let need = spec.layer_bytes * spec.layers as u64 + spec.extra_bytes;
            let free: u64 = owned.iter().filter_map(|d| d.report.as_ref().ok()).map(|r| (r.device.mem_available as f64 * MEMORY_SHARE) as u64).sum();
            bail!(
                "{name} needs {} of memory and the devices it can use have {} free together (this computer coordinates and runs the draft model, but holds no layers). Close apps on them, add a device, or pick a smaller model.",
                gb(need),
                gb(free)
            );
        }
        let usable = stage_devices(&devices);
        status("Each device is downloading its own layers…");
        let clients = open_all(&endpoint, &url, &plan, &usable, &|device, done, total| {
            status(&format!("{device} is downloading its layers: {} of {}", gb(done), gb(total)));
        })
        .await?;
        let llm = Arc::new(Llm::new(&header, clients, draft)?);
        Ok(Session { llm, name, url, spec, goal, endpoint, state: tokio::sync::Mutex::new(SessionState { devices, pairs, plan }) })
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
        let usable: Vec<MeshDevice> = stage_devices(&s.devices).iter().map(|d| MeshDevice { name: d.name.clone(), addr: d.addr.clone(), report: d.report.clone() }).collect();
        let reports: Vec<&LinkReport> = usable.iter().map(|d| d.report.as_ref().unwrap()).collect();
        let mut specs: Vec<Device> = usable.iter().zip(&reports).map(|(d, r)| device_spec(&d.name, r)).collect();
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
        let devices = probe_all(&self.endpoint, targets).await;
        let usable = stage_devices(&devices);
        let owned: Vec<MeshDevice> = usable.iter().map(|d| MeshDevice { name: d.name.clone(), addr: d.addr.clone(), report: d.report.clone() }).collect();
        let pairs = probe_pairs(&self.endpoint, &owned).await;
        let (plan, _) = plan_with(&self.spec, &owned, &pairs, self.goal, 1.0);
        if plan.pipelines.is_empty() {
            bail!("the devices still reachable cannot hold {}", self.name);
        }
        let clients = open_all(&self.endpoint, &self.url, &plan, &usable, &|_, _, _| {}).await?;
        self.llm.set_replicas(clients);
        *self.state.lock().await = SessionState { devices, pairs, plan };
        Ok(())
    }
}
