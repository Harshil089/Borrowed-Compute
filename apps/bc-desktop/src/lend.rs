//! Lending this device: the agent that runs jobs from paired devices, pairing approval,
//! guests, recent jobs, and the account's cloud when it lives on this device.
//!
//! The agent's endpoint is this device's identity on the network, so it runs whenever the
//! device is signed in or holds the account, lending or not. "Lend this device" only
//! decides whether it accepts jobs and invites (on by default on a Mac, where lending is
//! what the app was for, and off on a PC). Only a lending device sends heartbeats, which is
//! how other devices and the cloud tell that it lends.
//!
//! The cloud (accounts, devices, pairing) lives on whichever device the owner chose at
//! first sign-in, in a SQLite file, and is shared with the other devices over iroh, so no
//! server or VPN is needed.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bc_agent::events::{AgentEvent, JobEnd};
use bc_agent::service::{Agent, AgentPaths};
use bc_auth::{cloud_url_for, load_session, CloudClient, DEFAULT_CLOUD_URL};
use bc_proto::cloud::PairingInfo;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_opener::OpenerExt;
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

use crate::logs::{log, Logs};
use crate::{device_id, err, CmdResult};

const MAX_JOBS: usize = 50;

/// Port of the built-in cloud. It listens on localhost only; the agent shares it with other
/// devices over iroh (`bc_net::CLOUD_ALPN`).
pub const CLOUD_PORT: u16 = 8080;

/// Client id of the GitHub OAuth app used for sign-in. OAuth client ids are public by
/// design (device flow uses no client secret). `BC_GITHUB_CLIENT_ID` overrides it.
const DEFAULT_GITHUB_CLIENT_ID: &str = "Ov23liLwrFmQ0cqqpXfy";

/// Client id of the Google OAuth client (type "TVs and Limited Input devices"). Empty
/// until one is created; Google sign-in is then not offered. Its secret, which Google
/// documents as not confidential for this client type, is built in from the
/// `BC_GOOGLE_CLIENT_SECRET` CI secret rather than kept in the repo.
const DEFAULT_GOOGLE_CLIENT_ID: &str = "";

#[derive(Debug, Clone, Default, Serialize)]
pub struct CloudHost {
    /// True when the account lives on this device.
    pub hosting: bool,
    running: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    job_id: Option<String>,
    client: String,
    argv: Vec<String>,
    started_ms: u64,
    #[serde(default)]
    finished_ms: Option<u64>,
    end: Option<JobEnd>,
    rejected: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CloudState {
    ok: Option<bool>,
    error: Option<String>,
    last_sync_ms: Option<u64>,
}

/// What the window shows about lending.
#[derive(Serialize)]
pub struct LendStatus {
    serving: bool,
    lending: bool,
    start_error: Option<String>,
    cloud: CloudState,
    host: CloudHost,
}

pub struct LendState {
    pub paths: AgentPaths,
    pub host: StdMutex<CloudHost>,
    pub agent: OnceCell<Arc<Agent>>,
    starting: AtomicBool,
    start_error: StdMutex<Option<String>>,
    sync: Mutex<Option<JoinHandle<()>>>,
    jobs: StdMutex<VecDeque<JobRecord>>,
    /// Output of each running job, written to its job file when it ends.
    output: StdMutex<HashMap<String, JobOutput>>,
    cloud_state: StdMutex<CloudState>,
    pub logs: Logs,
    /// Saved in `<home>/lending` as `on` or `off`.
    lending: AtomicBool,
    /// Quitting only hides the app while this is on. Saved in `<home>/run-in-background`.
    pub run_in_background: AtomicBool,
    /// Set by the menu-bar Quit and by update installs, which must really exit.
    pub quitting: AtomicBool,
    /// Per-core CPU use between two reads, for the window's live core grid.
    cpu: StdMutex<sysinfo::System>,
}

fn background_flag(home: &Path) -> PathBuf {
    home.join("run-in-background")
}

fn lending_flag(home: &Path) -> PathBuf {
    home.join("lending")
}

/// Marks that the account lives on this device. A Mac from before the merged app hosted
/// whenever its cloud address was the default; its `cloud/cloud.db` stands in for this.
fn hosts_flag(home: &Path) -> PathBuf {
    home.join("hosts-account")
}

impl LendState {
    pub fn new(home: &Path) -> Self {
        let root = home.parent().unwrap_or(home);
        let lending = match std::fs::read_to_string(lending_flag(home)) {
            Ok(text) => text.trim() == "on",
            Err(_) => cfg!(target_os = "macos"),
        };
        let hosting = cloud_url_for(home) == DEFAULT_CLOUD_URL
            && (hosts_flag(home).exists() || (cfg!(target_os = "macos") && crate::cloud_dir(home).join("cloud.db").exists()));
        let logs = Logs::new(root.join("logs"), home.join("job-history.jsonl"));
        let jobs = logs.recent_jobs(MAX_JOBS).into();
        LendState {
            paths: AgentPaths::new(home.to_path_buf()),
            host: StdMutex::new(CloudHost { hosting, running: false, error: None }),
            agent: OnceCell::new(),
            starting: AtomicBool::new(false),
            start_error: StdMutex::new(None),
            sync: Mutex::new(None),
            jobs: StdMutex::new(jobs),
            output: StdMutex::new(HashMap::new()),
            cloud_state: StdMutex::new(CloudState::default()),
            logs,
            lending: AtomicBool::new(lending),
            run_in_background: AtomicBool::new(background_flag(home).exists()),
            quitting: AtomicBool::new(false),
            cpu: StdMutex::new(sysinfo::System::new()),
        }
    }

    /// A client for this device's cloud address and session, read from disk each time.
    fn cloud(&self) -> CloudClient {
        CloudClient::from_home(cloud_url_for(&self.paths.home), &self.paths.home)
    }

    pub fn status(&self) -> LendStatus {
        let signed_in = load_session(&self.paths.home).is_some();
        LendStatus {
            serving: self.agent.get().is_some(),
            lending: self.lending.load(Ordering::Relaxed),
            start_error: self.start_error.lock().unwrap().clone(),
            cloud: if signed_in { self.cloud_state.lock().unwrap().clone() } else { CloudState::default() },
            host: self.host.lock().unwrap().clone(),
        }
    }

    /// A job from another device is running here.
    pub fn job_running(&self) -> bool {
        self.jobs.lock().unwrap().iter().any(|j| j.end.is_none() && j.rejected.is_none())
    }

    pub fn set_lan_name(&self, name: &str) {
        if let Some(agent) = self.agent.get() {
            agent.set_lan_name(name);
        }
    }

    /// Clears what the last sign-in left and starts or stops syncing to match.
    pub async fn signed_in_or_out(&self) {
        *self.cloud_state.lock().unwrap() = CloudState::default();
        self.restart_sync().await;
    }

    /// Restarts heartbeats and paired-list sync with the current session, or stops them when
    /// signed out or not lending.
    pub async fn restart_sync(&self) {
        let mut sync = self.sync.lock().await;
        if let Some(old) = sync.take() {
            old.abort();
        }
        let cloud = self.cloud();
        let lending = self.lending.load(Ordering::Relaxed);
        if let (Some(agent), true, true) = (self.agent.get(), cloud.token().is_some(), lending) {
            *sync = Some(agent.spawn_cloud_sync(cloud));
        }
    }
}

/// Starts the agent once: on launch when this device is signed in or holds the account,
/// else after the first sign-in.
pub fn start(app: &AppHandle) {
    let state = app.state::<LendState>();
    if state.starting.swap(true, Ordering::SeqCst) {
        return;
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let state = handle.state::<LendState>();
        match Agent::start(&state.paths, false, None, Some(tx)).await {
            Ok(mut agent) => {
                if state.host.lock().unwrap().hosting {
                    agent = agent.share_cloud(CLOUD_PORT);
                }
                agent.set_lending(state.lending.load(Ordering::Relaxed));
                let agent = Arc::new(agent);
                let _ = state.agent.set(agent.clone());
                tauri::async_runtime::spawn(async move { agent.accept_loop().await });
                state.restart_sync().await;
            }
            Err(e) => {
                log(format_args!("agent could not start: {e:#}"));
                *state.start_error.lock().unwrap() = Some(format!("{e:#}"));
            }
        }
        let _ = handle.emit("status-changed", ());
    });

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            let state = handle.state::<LendState>();
            record(&state, &event);
            // The cloud no longer accepts this device's session (expired, or issued by a
            // different cloud): sign out and ask the user to sign in.
            if let AgentEvent::CloudSync { ok: false, error: Some(error) } = &event {
                if error.ends_with("(401 Unauthorized)") {
                    bc_auth::clear_session(&state.paths.home);
                    state.signed_in_or_out().await;
                    crate::show_window(&handle);
                    let _ = handle.emit("status-changed", ());
                }
            }
            // Output goes to the job's file, not to the window.
            if !matches!(event, AgentEvent::JobOutput { .. }) {
                let _ = handle.emit("agent-event", &event);
            }
        }
    });

    // Start or stop syncing when a session appears on or leaves the disk, for example
    // after `bc-agent login`, even while the window is closed.
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            let state = handle.state::<LendState>();
            let should = load_session(&state.paths.home).is_some() && state.lending.load(Ordering::Relaxed);
            let syncing = state.sync.lock().await.is_some();
            if should != syncing && state.agent.get().is_some() {
                state.restart_sync().await;
                let _ = handle.emit("status-changed", ());
            }
        }
    });
}

/// Runs the account's cloud from a SQLite file in `data_dir`, once.
pub fn start_cloud(app: &AppHandle, data_dir: PathBuf) {
    {
        let state = app.state::<LendState>();
        let mut host = state.host.lock().unwrap();
        if host.running {
            return;
        }
        host.running = true;
        host.error = None;
    }
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = host_cloud(data_dir).await;
        let state = handle.state::<LendState>();
        let mut host = state.host.lock().unwrap();
        host.running = false;
        let error = match result {
            Ok(()) => "the built-in cloud stopped".to_string(),
            Err(e) => format!("{e:#}"),
        };
        log(format_args!("built-in cloud: {error}"));
        host.error = Some(error);
        drop(host);
        let _ = handle.emit("status-changed", ());
    });
}

async fn host_cloud(data_dir: PathBuf) -> anyhow::Result<()> {
    std::fs::create_dir_all(&data_dir)?;
    let db = bc_cloud::connect(&bc_cloud::sqlite_url(&data_dir.join("cloud.db"))).await?;
    let client_id = std::env::var("BC_GITHUB_CLIENT_ID").unwrap_or_else(|_| DEFAULT_GITHUB_CLIENT_ID.to_string());
    let state = bc_cloud::AppState {
        db,
        github: Arc::new(bc_cloud::GitHub::new(client_id)),
        google: bc_cloud::google_client(DEFAULT_GOOGLE_CLIENT_ID, option_env!("BC_GOOGLE_CLIENT_SECRET")),
        relay_url: None,
        single_owner: true,
    };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", CLOUD_PORT))
        .await
        .map_err(|e| anyhow::anyhow!("port {CLOUD_PORT} is busy ({e}); quit whatever else uses it"))?;
    bc_cloud::serve(listener, state).await?;
    Ok(())
}

/// Keeps the account on this device: from now on its cloud runs here and other devices
/// sign in through it. Resolves once the cloud answers, so sign-in can follow at once. An
/// agent already running without the cloud cannot take it on, so the app then restarts.
#[tauri::command]
pub async fn keep_account_here(app: AppHandle, state: State<'_, LendState>) -> CmdResult<()> {
    let home = state.paths.home.clone();
    std::fs::write(hosts_flag(&home), "yes\n").map_err(err)?;
    let _ = std::fs::remove_file(home.join("cloud-url"));
    bc_auth::clear_session(&home);
    state.host.lock().unwrap().hosting = true;
    if state.agent.get().is_some() {
        crate::restart_app(&app);
        return Ok(());
    }
    start_cloud(&app, crate::cloud_dir(&home));
    start(&app);
    let local = CloudClient::new(DEFAULT_CLOUD_URL, None);
    for _ in 0..50 {
        if local.health().await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    Err(state.host.lock().unwrap().error.clone().unwrap_or_else(|| "the account's cloud did not start".into()))
}

/// Turns lending on or off. Off refuses new jobs and invites and stops heartbeats, so other
/// devices see this one as not lending; runs in progress finish.
#[tauri::command]
pub async fn set_lending(state: State<'_, LendState>, on: bool) -> CmdResult<()> {
    // The one-time setup that needs an administrator (folder attributes for Node, the
    // firewall rule), if the installer's run of it was declined. Lending works without it.
    #[cfg(windows)]
    if on {
        let asked = tokio::task::spawn_blocking(|| {
            bc_agent::sandbox_windows::setup_needed().then(bc_agent::sandbox_windows::request_setup)
        })
        .await;
        if let Ok(Some(Err(e))) = asked {
            log(format_args!("setup for lending was not completed: {e}"));
        }
    }
    std::fs::write(lending_flag(&state.paths.home), if on { "on\n" } else { "off\n" }).map_err(err)?;
    state.lending.store(on, Ordering::Relaxed);
    if let Some(agent) = state.agent.get() {
        agent.set_lending(on);
    }
    state.restart_sync().await;
    log(format_args!("lending turned {}", if on { "on" } else { "off" }));
    Ok(())
}

/// Pairing requests and paired devices for this device as the lender.
#[tauri::command]
pub async fn list_pairings(state: State<'_, LendState>) -> CmdResult<Vec<PairingInfo>> {
    let me = device_id(&state.paths.home)?;
    let list = state.cloud().pairings(&me).await.map_err(|e| format!("{e:#}"))?;
    Ok(list.into_iter().filter(|p| p.mac_id == me).collect())
}

#[tauri::command]
pub async fn approve(state: State<'_, LendState>, id: i64, code: String) -> CmdResult<PairingInfo> {
    let key = bc_net::load_or_create_key(&state.paths.key()).map_err(|e| format!("{e:#}"))?;
    state.cloud().approve(&key, id, &code).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
pub async fn unpair(state: State<'_, LendState>, id: i64) -> CmdResult<()> {
    state.cloud().unpair(id).await.map_err(|e| format!("{e:#}"))
}

/// Services running on this device, from every paired device.
#[tauri::command]
pub async fn list_services(state: State<'_, LendState>) -> CmdResult<Vec<bc_proto::wire::ServiceInfo>> {
    Ok(match state.agent.get() {
        Some(agent) => agent.services().list_with_ports(None).await,
        None => Vec::new(),
    })
}

/// The owner may stop any service here, whoever started it.
#[tauri::command]
pub fn stop_service(state: State<'_, LendState>, job_id: String) -> CmdResult<()> {
    let agent = running_agent(&state)?;
    agent.services().stop(&job_id, None).map_err(|_| format!("{job_id} is no longer running"))
}

fn running_agent(state: &LendState) -> CmdResult<&Arc<Agent>> {
    state.agent.get().ok_or_else(|| "lending has not started; sign in first".to_string())
}

#[tauri::command]
pub fn guests(state: State<'_, LendState>) -> CmdResult<bc_agent::guests::GuestsView> {
    Ok(running_agent(&state)?.guests())
}

#[tauri::command]
pub fn set_guest_mesh(state: State<'_, LendState>, on: bool) -> CmdResult<()> {
    running_agent(&state)?.set_guest_mesh(on);
    log(format_args!("guests in Mesh turned {}", if on { "on" } else { "off" }));
    Ok(())
}

#[tauri::command]
pub fn set_sharing(state: State<'_, LendState>, on: bool) -> CmdResult<()> {
    running_agent(&state)?.set_sharing(on);
    log(format_args!("sharing turned {}", if on { "on" } else { "off" }));
    Ok(())
}

/// The invite text, shown to the owner once to send to the person.
#[tauri::command]
pub fn create_invite(state: State<'_, LendState>, name: String) -> CmdResult<String> {
    running_agent(&state)?.create_invite(&name)
}

#[tauri::command]
pub fn delete_invite(state: State<'_, LendState>, id: String) -> CmdResult<()> {
    running_agent(&state)?.delete_invite(&id);
    Ok(())
}

#[tauri::command]
pub fn approve_guest(state: State<'_, LendState>, id: u64, code: String) -> CmdResult<()> {
    running_agent(&state)?.approve_guest(id, &code)?;
    log(format_args!("approved a guest request"));
    Ok(())
}

#[tauri::command]
pub fn remove_guest(state: State<'_, LendState>, device: String) -> CmdResult<()> {
    running_agent(&state)?.remove_guest(&device);
    log(format_args!("removed guest {}", short(&device)));
    Ok(())
}

/// Each logical CPU's use since the previous read, from 0.0 to 1.0: the whole computer's
/// load, not only the job's. The window reads it once a second while a run or server is up,
/// so the core grid shows measured numbers. The very first read has nothing to compare
/// against and comes back as zeros.
#[tauri::command]
pub fn cpu_load(state: State<'_, LendState>) -> Vec<f32> {
    let mut sys = state.cpu.lock().unwrap_or_else(|e| e.into_inner());
    cpu_sample(&mut sys)
}

fn cpu_sample(sys: &mut sysinfo::System) -> Vec<f32> {
    sys.refresh_cpu_usage();
    sys.cpus().iter().map(|cpu| (cpu.cpu_usage() / 100.0).clamp(0.0, 1.0)).collect()
}

#[tauri::command]
pub fn recent_jobs(state: State<'_, LendState>) -> Vec<JobRecord> {
    state.jobs.lock().unwrap().iter().cloned().collect()
}

#[tauri::command]
pub fn open_logs(app: AppHandle, state: State<'_, LendState>) -> CmdResult<()> {
    app.opener().open_path(state.logs.dir.to_string_lossy(), None::<&str>).map_err(err)
}

#[derive(Serialize)]
pub struct Settings {
    run_in_background: bool,
    version: String,
}

#[tauri::command]
pub fn get_settings(app: AppHandle, state: State<'_, LendState>) -> Settings {
    Settings {
        run_in_background: state.run_in_background.load(Ordering::Relaxed),
        version: app.package_info().version.to_string(),
    }
}

#[tauri::command]
pub fn set_run_in_background(state: State<'_, LendState>, enabled: bool) -> CmdResult<()> {
    let flag = background_flag(&state.paths.home);
    if enabled {
        std::fs::write(&flag, "on\n").map_err(err)?;
    } else if flag.exists() {
        std::fs::remove_file(&flag).map_err(err)?;
    }
    state.run_in_background.store(enabled, Ordering::Relaxed);
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Keeps the job list and cloud status current.
fn record(state: &LendState, event: &AgentEvent) {
    match event {
        AgentEvent::JobStarted { job_id, client, argv } => {
            let mut jobs = state.jobs.lock().unwrap();
            jobs.push_front(JobRecord {
                job_id: Some(job_id.clone()),
                client: client.clone(),
                argv: argv.clone(),
                started_ms: now_ms(),
                finished_ms: None,
                end: None,
                rejected: None,
            });
            jobs.truncate(MAX_JOBS);
        }
        AgentEvent::JobFinished { job_id, end } => {
            let mut jobs = state.jobs.lock().unwrap();
            if let Some(job) = jobs.iter_mut().find(|j| j.job_id.as_deref() == Some(job_id)) {
                job.end = Some(end.clone());
                job.finished_ms = Some(now_ms());
                state.logs.append_history(job);
                let output = state.output.lock().unwrap().remove(job_id).unwrap_or_default();
                state.logs.write_job_file(&job_file_name(job), &job_file_text(job, &output));
            }
        }
        AgentEvent::JobOutput { job_id, data } => {
            state.output.lock().unwrap().entry(job_id.clone()).or_default().push(data);
        }
        AgentEvent::JobRejected { client, reason } => {
            let mut jobs = state.jobs.lock().unwrap();
            jobs.push_front(JobRecord {
                job_id: None,
                client: client.clone(),
                argv: Vec::new(),
                started_ms: now_ms(),
                finished_ms: None,
                end: None,
                rejected: Some(reason.clone()),
            });
            jobs.truncate(MAX_JOBS);
            if let Some(job) = jobs.front() {
                state.logs.append_history(job);
                state.logs.write_job_file(&job_file_name(job), &job_file_text(job, &JobOutput::default()));
            }
        }
        AgentEvent::GuestsChanged => {}
        AgentEvent::CloudSync { ok, error } => {
            *state.cloud_state.lock().unwrap() = CloudState {
                ok: Some(*ok),
                error: error.clone(),
                last_sync_ms: Some(now_ms()),
            };
        }
    }
}

fn short(id: &str) -> &str {
    id.get(..10).unwrap_or(id)
}

fn timestamp(ms: u64) -> String {
    humantime::format_rfc3339_seconds(UNIX_EPOCH + std::time::Duration::from_millis(ms)).to_string()
}

/// `<start time>-<job id>.log`; the time comes first so files sort oldest first. Colons
/// are left out because Finder shows them as slashes and Windows refuses them.
fn job_file_name(job: &JobRecord) -> String {
    let id = match &job.job_id {
        Some(id) => id.clone(),
        None => format!("rejected-{}", short(&job.client)),
    };
    format!("{}-{id}.log", timestamp(job.started_ms).replace(':', "-"))
}

/// Output kept per job; a longer run keeps its last part.
const MAX_JOB_OUTPUT: usize = 1 << 20;

/// What a job printed, stdout and stderr together, in order.
#[derive(Default)]
struct JobOutput {
    data: VecDeque<u8>,
    dropped: usize,
}

impl JobOutput {
    fn push(&mut self, data: &[u8]) {
        self.data.extend(data);
        let over = self.data.len().saturating_sub(MAX_JOB_OUTPUT);
        self.data.drain(..over);
        self.dropped += over;
    }
}

/// The job's own log: what ran, for whom, how it ended, what it used and what it printed.
fn job_file_text(job: &JobRecord, output: &JobOutput) -> String {
    let mut text = format!(
        "Job: {}\nClient device: {}\nStarted: {}\n",
        job.job_id.as_deref().unwrap_or("none (rejected before it ran)"),
        job.client,
        timestamp(job.started_ms),
    );
    if let Some(reason) = &job.rejected {
        text += &format!("Result: rejected: {reason}\n");
        return text;
    }
    text += &format!("Command: {}\n", job.argv.join(" "));
    if let Some(ms) = job.finished_ms {
        text += &format!("Finished: {}\n", timestamp(ms));
    }
    let result = match &job.end {
        Some(JobEnd::Exited { timed_out: true, .. }) => "timed out".to_string(),
        Some(JobEnd::Exited { code: Some(c), .. }) => format!("exited with code {c}"),
        Some(JobEnd::Exited { code: None, .. }) => "stopped by a signal".to_string(),
        Some(JobEnd::Failed { reason }) => format!("could not run: {reason}"),
        Some(JobEnd::Cancelled) => "cancelled by the client".to_string(),
        Some(JobEnd::Disconnected) => "the client disconnected".to_string(),
        None => "still running".to_string(),
    };
    text += &format!("Result: {result}\n");
    if let Some(JobEnd::Exited { usage: Some(u), .. }) = &job.end {
        let ms = |v: u64| if v < 1000 { format!("{v} ms") } else { format!("{:.1} s", v as f64 / 1000.0) };
        text += &format!(
            "Usage: {}\nCPU time: {} user, {} system\nPeak memory: {} bytes\nRun time: {}\n",
            u.summary(),
            ms(u.user_ms),
            ms(u.system_ms),
            u.max_rss_bytes,
            ms(u.wall_ms),
        );
    }
    if !output.data.is_empty() {
        text += "\nOutput:\n";
        if output.dropped > 0 {
            text += &format!("[first {} bytes left out]\n", output.dropped);
        }
        let (a, b) = output.data.as_slices();
        text += &String::from_utf8_lossy(&[a, b].concat());
        if !text.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_proto::wire::JobUsage;

    #[test]
    fn cpu_sample_reads_every_core_as_a_fraction() {
        let mut sys = sysinfo::System::new();
        cpu_sample(&mut sys);
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        let loads = cpu_sample(&mut sys);
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        assert_eq!(loads.len(), cores);
        assert!(loads.iter().all(|l| (0.0..=1.0).contains(l)), "{loads:?}");
    }

    #[test]
    fn job_file_has_command_result_and_usage() {
        let job = JobRecord {
            job_id: Some("18e1c-352c".into()),
            client: "b0551e4a8e580888".into(),
            argv: vec!["python3".into(), "palindrome_checker.py".into()],
            started_ms: 1_792_931_958_660,
            finished_ms: Some(1_792_931_958_746),
            end: Some(JobEnd::Exited {
                code: Some(0),
                timed_out: false,
                usage: Some(JobUsage { wall_ms: 70, user_ms: 23, system_ms: 17, max_rss_bytes: 14_729_216 }),
            }),
            rejected: None,
        };
        assert_eq!(job_file_name(&job), "2026-10-25T12-39-18Z-18e1c-352c.log");
        let mut output = JobOutput::default();
        output.push(b"is a palindrome\n");
        let text = job_file_text(&job, &output);
        assert!(text.contains("Command: python3 palindrome_checker.py\n"), "{text}");
        assert!(text.contains("Result: exited with code 0\n"), "{text}");
        assert!(text.contains("CPU time: 23 ms user, 17 ms system\n"), "{text}");
        assert!(text.contains("Peak memory: 14729216 bytes\n"), "{text}");
        assert!(text.ends_with("\nOutput:\nis a palindrome\n"), "{text}");

        let mut long = JobOutput::default();
        long.push(&vec![b'a'; MAX_JOB_OUTPUT]);
        long.push(b"end");
        assert_eq!(long.data.len(), MAX_JOB_OUTPUT);
        assert!(job_file_text(&job, &long).contains("[first 3 bytes left out]\n"));
    }
}
