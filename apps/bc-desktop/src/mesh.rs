//! The Mesh page: run a model across this account's paired devices and serve it on this
//! computer. The steps live in `bc_cli::mesh` and `bc_cli::serve`, shared with `bc mesh`.

use std::sync::Arc;
use std::time::Duration;

use bc_cli::mesh::{paired_devices, plan_for, probe_all, MeshDevice, Session};
use bc_mesh::plan::Goal;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::Mutex;

use crate::{err_chain, AppState, CmdResult};

#[derive(Default)]
pub struct MeshState {
    running: Mutex<Option<Running>>,
}

struct Running {
    session: Arc<Session>,
    port: u16,
    stop: tokio::sync::oneshot::Sender<()>,
}

#[derive(Serialize)]
pub struct ModelRow {
    name: String,
    gb: f64,
    layers: u32,
    runnable: bool,
}

#[tauri::command]
pub fn mesh_models() -> Vec<ModelRow> {
    bc_mesh::plan::catalog()
        .into_iter()
        .map(|m| ModelRow {
            gb: (m.layer_bytes * m.layers as u64 + m.extra_bytes) as f64 / (1u64 << 30) as f64,
            layers: m.layers,
            runnable: m.url.is_some(),
            name: m.name,
        })
        .collect()
}

#[derive(Serialize)]
pub struct DeviceRow {
    name: String,
    error: Option<String>,
    relay: bool,
    rtt_ms: f64,
    up_mbps: f64,
    free_gb: f64,
    total_gb: f64,
    mem_gbs: f64,
    os: String,
    engine: bool,
}

#[derive(Serialize)]
pub struct StageRow {
    device: String,
    start: u32,
    end: u32,
    ms: f32,
}

#[derive(Serialize)]
pub struct PipelineRow {
    stages: Vec<StageRow>,
    token_ms: f64,
    network_ms: f64,
    tokens_per_s: f64,
}

#[derive(Serialize)]
pub struct Probe {
    devices: Vec<DeviceRow>,
    pipelines: Vec<PipelineRow>,
    unused: Vec<String>,
}

fn rows(devices: &[MeshDevice]) -> Vec<DeviceRow> {
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    devices
        .iter()
        .map(|d| match &d.report {
            Ok(r) => DeviceRow {
                name: d.name.clone(),
                error: None,
                relay: r.via_relay,
                rtt_ms: r.rtt_ms,
                up_mbps: r.up_bytes_s * 8.0 / 1e6,
                free_gb: gb(r.device.mem_available),
                total_gb: gb(r.device.mem_total),
                mem_gbs: r.device.mem_bw / 1e9,
                os: r.device.os.clone(),
                engine: r.device.engine,
            },
            Err(e) => DeviceRow { name: d.name.clone(), error: Some(e.clone()), relay: false, rtt_ms: 0.0, up_mbps: 0.0, free_gb: 0.0, total_gb: 0.0, mem_gbs: 0.0, os: String::new(), engine: false },
        })
        .collect()
}

async fn targets(state: &AppState) -> CmdResult<Vec<(String, iroh::EndpointAddr)>> {
    let me = crate::device_id(&state.home)?;
    let cloud = state.cloud();
    paired_devices(&cloud, &me).await.map_err(err_chain)
}

async fn endpoint(state: &AppState) -> CmdResult<iroh::Endpoint> {
    let key = bc_net::load_or_create_key(&bc_cli::flows::key_path(&state.home)).map_err(err_chain)?;
    bc_net::bind(key, bc_cli::flows::mac_net(&state.cloud(), false).await).await.map_err(err_chain)
}

/// Measures every paired device at once and, for `model`, plans it over them.
#[tauri::command]
pub async fn mesh_probe(state: State<'_, AppState>, model: Option<String>, throughput: bool) -> CmdResult<Probe> {
    let targets = targets(&state).await?;
    let ep = endpoint(&state).await?;
    let devices = probe_all(&ep, targets).await;
    bc_net::release(ep).await;
    let mut out = Probe { devices: rows(&devices), pipelines: Vec::new(), unused: Vec::new() };
    if let Some(m) = model.as_deref().and_then(bc_cli::mesh::find_model) {
        let usable: Vec<MeshDevice> = devices
            .into_iter()
            .filter(|d| d.report.as_ref().is_ok_and(|r| r.device.engine))
            .collect();
        let goal = if throughput { Goal::Throughput } else { Goal::Latency };
        let (plan, names) = plan_for(&m, &usable, goal, 1.0);
        out.pipelines = plan
            .pipelines
            .iter()
            .map(|p| PipelineRow {
                stages: p.stages.iter().map(|s| StageRow { device: names[s.device].name.clone(), start: s.layers.start, end: s.layers.end, ms: 0.0 }).collect(),
                token_ms: p.token_ms,
                network_ms: p.network_ms,
                tokens_per_s: p.tokens_per_s,
            })
            .collect();
        out.unused = plan.unused.iter().map(|&i| names[i].name.clone()).collect();
    }
    Ok(out)
}

#[derive(Serialize)]
pub struct MeshStatus {
    running: bool,
    model: String,
    url: String,
    pipelines: Vec<PipelineRow>,
}

async fn status_of(running: &Option<Running>) -> MeshStatus {
    let Some(r) = running else {
        return MeshStatus { running: false, model: String::new(), url: String::new(), pipelines: Vec::new() };
    };
    let layout = r.session.layout().await;
    let replicas = r.session.llm.replicas();
    let pipelines = layout
        .into_iter()
        .zip(replicas.iter())
        .map(|(stages, c)| {
            let ms = c.stage_ms.lock().unwrap().clone();
            PipelineRow {
                stages: stages.into_iter().enumerate().map(|(i, (device, l))| StageRow { device, start: l.start, end: l.end, ms: ms.get(i).copied().unwrap_or(0.0) }).collect(),
                token_ms: c.plan.token_ms,
                network_ms: c.plan.network_ms,
                tokens_per_s: c.plan.tokens_per_s,
            }
        })
        .collect();
    MeshStatus { running: true, model: r.session.name.clone(), url: format!("http://127.0.0.1:{}/v1", r.port), pipelines }
}

#[tauri::command]
pub async fn mesh_status(mesh: State<'_, MeshState>) -> CmdResult<MeshStatus> {
    Ok(status_of(&*mesh.running.lock().await).await)
}

/// Starts `model` across the paired devices and serves it on `port`. Sends `mesh-phase`
/// events while it starts.
#[tauri::command]
pub async fn mesh_start(app: AppHandle, state: State<'_, AppState>, mesh: State<'_, MeshState>, model: String, throughput: bool, port: u16) -> CmdResult<MeshStatus> {
    let mut running = mesh.running.lock().await;
    if running.is_some() {
        return Err("a model is already running; stop it first".into());
    }
    let targets = targets(&state).await?;
    let ep = endpoint(&state).await?;
    let goal = if throughput { Goal::Throughput } else { Goal::Latency };
    let phase = app.clone();
    let session = Arc::new(
        Session::start(ep, targets, &model, goal, None, move |s| {
            let _ = phase.emit("mesh-phase", s.to_string());
        })
        .await
        .map_err(err_chain)?,
    );
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.map_err(|e| format!("port {port}: {e}"))?;
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let home = state.home.clone();
    let served = Arc::new(bc_cli::serve::Served {
        session: session.clone(),
        targets: Arc::new(move || {
            let home = home.clone();
            Box::pin(async move {
                let me = bc_net::load_or_create_key(&bc_cli::flows::key_path(&home))?.public().to_string();
                let cloud = bc_auth::CloudClient::from_home(bc_auth::cloud_url_for(&home), &home);
                paired_devices(&cloud, &me).await
            })
        }),
    });
    let balancer = session.clone();
    tokio::spawn(async move {
        let server = axum::serve(listener, bc_cli::serve::router(served)).with_graceful_shutdown(async {
            let _ = stopped.await;
        });
        let rebalance = async {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let _ = balancer.rebalance().await;
            }
        };
        tokio::select! {
            _ = server => {}
            _ = rebalance => {}
        }
    });
    *running = Some(Running { session, port, stop });
    Ok(status_of(&running).await)
}

/// Stops serving. The devices drop their layers once the grace period passes.
#[tauri::command]
pub async fn mesh_stop(mesh: State<'_, MeshState>) -> CmdResult<()> {
    if let Some(r) = mesh.running.lock().await.take() {
        let _ = r.stop.send(());
        r.session.llm.set_replicas(Vec::new());
    }
    Ok(())
}

#[derive(Serialize)]
pub struct ChatStats {
    prompt_tokens: usize,
    generated: usize,
    passes: usize,
    first_token_ms: f64,
    tokens_per_s: f64,
}

/// Answers one message with the running model, sending `mesh-text` events as it writes.
#[tauri::command]
pub async fn mesh_chat(app: AppHandle, mesh: State<'_, MeshState>, prompt: String) -> CmdResult<ChatStats> {
    let session = mesh.running.lock().await.as_ref().map(|r| r.session.clone()).ok_or("no model is running")?;
    let llm = session.llm.clone();
    let ids = llm.tokenizer.encode(&llm.tokenizer.chat_prompt(&[("user".into(), prompt)]));
    let max = 512.min(llm.cfg.context.saturating_sub(ids.len()));
    let (_, s) = llm
        .generate(&ids, max, bc_infer::Sampling::default(), |t| {
            let _ = app.emit("mesh-text", t.to_string());
        })
        .await
        .map_err(err_chain)?;
    let decode_s = (s.total_ms - s.first_token_ms).max(1.0) / 1000.0;
    Ok(ChatStats { prompt_tokens: s.prompt_tokens, generated: s.generated, passes: s.passes, first_token_ms: s.first_token_ms, tokens_per_s: s.generated.saturating_sub(1) as f64 / decode_s })
}
