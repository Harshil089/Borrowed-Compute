//! Borrowed Compute Agent: menu-bar app on the Mac.
//!
//! Runs the same agent service as `bc-agent serve`, plus sign-in, pairing approval and a
//! list of recent jobs. Closing the window keeps the agent running; Quit stops it.
//!
//! It also hosts the cloud (accounts, devices, pairing) in a SQLite file on this Mac, and
//! shares it with the user's PC through `tailscale serve`, so no server is needed. When a
//! different cloud address is configured for this Mac, it uses that one instead.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod logs;
mod tailscale;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bc_agent::events::{AgentEvent, JobEnd};
use bc_agent::service::{Agent, AgentPaths};
use bc_auth::{
    clear_session, cloud_url_for, default_device_name, is_unauthorized, load_session, save_session,
    CloudClient, DEFAULT_CLOUD_URL,
};
use bc_net::{home_dir, load_or_create_key};
use bc_proto::cloud::{DeviceKind, PairingInfo};
use logs::{log, Logs};
use serde::{Deserialize, Serialize};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_opener::OpenerExt;
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

const MAX_JOBS: usize = 50;

/// Port of the built-in cloud. It listens on localhost only; Tailscale serve exposes it to
/// the user's tailnet.
const CLOUD_PORT: u16 = 8080;

/// Client id of the GitHub OAuth app used for sign-in. OAuth client ids are public by
/// design (device flow uses no client secret). `BC_GITHUB_CLIENT_ID` overrides it.
const DEFAULT_GITHUB_CLIENT_ID: &str = "Ov23liLwrFmQ0cqqpXfy";

#[derive(Debug, Clone, Default, Serialize)]
struct CloudHost {
    /// True when this Mac hosts the cloud (its cloud address is the built-in default).
    hosting: bool,
    running: bool,
    error: Option<String>,
    /// The cloud address this Mac's agent uses.
    url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JobRecord {
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
struct CloudState {
    ok: Option<bool>,
    error: Option<String>,
    last_sync_ms: Option<u64>,
}

#[derive(Serialize)]
struct Status {
    signed_in: bool,
    device_id: Option<String>,
    serving: bool,
    start_error: Option<String>,
    cloud: CloudState,
    host: CloudHost,
}

#[derive(Serialize)]
struct Account {
    login: String,
    device_name: Option<String>,
}

impl AppState {
    /// A client for this Mac's cloud address and session. Read from disk each time, so a
    /// sign-in with `bc-agent login` shows up here too.
    fn cloud(&self) -> CloudClient {
        CloudClient::from_home(cloud_url_for(&self.paths.home), &self.paths.home)
    }
}

struct AppState {
    paths: AgentPaths,
    host: StdMutex<CloudHost>,
    agent: OnceCell<Arc<Agent>>,
    start_error: StdMutex<Option<String>>,
    sync: Mutex<Option<JoinHandle<()>>>,
    jobs: StdMutex<VecDeque<JobRecord>>,
    cloud_state: StdMutex<CloudState>,
    logs: Logs,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn device_id(state: &AppState) -> CmdResult<String> {
    Ok(load_or_create_key(&state.paths.key()).map_err(|e| format!("{e:#}"))?.public().to_string())
}

/// Restarts heartbeats and paired-list sync with the current session, or stops them when
/// signed out.
async fn restart_sync(state: &AppState) {
    let mut sync = state.sync.lock().await;
    if let Some(old) = sync.take() {
        old.abort();
    }
    let cloud = state.cloud();
    if let (Some(agent), true) = (state.agent.get(), cloud.token().is_some()) {
        *sync = Some(agent.spawn_cloud_sync(cloud));
    }
}

#[tauri::command]
fn get_status(state: State<'_, AppState>) -> CmdResult<Status> {
    let signed_in = load_session(&state.paths.home).is_some();
    Ok(Status {
        signed_in,
        device_id: device_id(&state).ok(),
        serving: state.agent.get().is_some(),
        start_error: state.start_error.lock().unwrap().clone(),
        cloud: if signed_in { state.cloud_state.lock().unwrap().clone() } else { CloudState::default() },
        host: state.host.lock().unwrap().clone(),
    })
}

#[tauri::command]
async fn get_account(state: State<'_, AppState>) -> CmdResult<Account> {
    let cloud = state.cloud();
    let me = match cloud.me().await {
        Ok(me) => me,
        // The session belongs to another cloud (or expired): sign out so the window
        // offers sign-in again instead of failing on every request.
        Err(e) if is_unauthorized(&e) => {
            sign_out_locally(&state).await;
            return Err("signed-out".into());
        }
        Err(e) => return Err(format!("{e:#}")),
    };
    // A session that appeared on disk (for example from `bc-agent login`) needs sync.
    if state.sync.lock().await.is_none() {
        restart_sync(&state).await;
    }
    let id = device_id(&state)?;
    let device_name = cloud
        .devices()
        .await
        .ok()
        .and_then(|list| list.into_iter().find(|d| d.endpoint_id == id))
        .map(|d| d.name);
    Ok(Account { login: me.login, device_name })
}

#[tauri::command]
async fn login(app: AppHandle, state: State<'_, AppState>, name: String) -> CmdResult<Account> {
    // A fresh client, so other commands are not blocked while the user is on GitHub.
    let mut client = CloudClient::new(cloud_url_for(&state.paths.home), None);
    let emitter = app.clone();
    let (token, login) = client
        .login(|start| {
            let _ = emitter.emit("login-code", start);
            let _ = emitter.opener().open_url(&start.verification_uri, None::<&str>);
        })
        .await
        .map_err(|e| format!("{e:#}"))?;
    save_session(&state.paths.home, &token).map_err(err)?;
    let key = load_or_create_key(&state.paths.key()).map_err(err)?;
    let name = match name.trim() {
        "" => default_device_name(),
        n => n.to_string(),
    };
    let device = client.register(&key, &name, DeviceKind::Mac).await.map_err(|e| format!("{e:#}"))?;
    drop(client);
    *state.cloud_state.lock().unwrap() = CloudState::default();
    restart_sync(&state).await;
    Ok(Account { login, device_name: Some(device.name) })
}

#[tauri::command]
async fn logout(state: State<'_, AppState>) -> CmdResult<()> {
    let cloud = state.cloud();
    if cloud.token().is_some() {
        let _ = cloud.logout().await;
    }
    sign_out_locally(&state).await;
    Ok(())
}

async fn sign_out_locally(state: &AppState) {
    clear_session(&state.paths.home);
    *state.cloud_state.lock().unwrap() = CloudState::default();
    restart_sync(state).await;
}

#[tauri::command]
async fn tailscale_status() -> tailscale::TailscaleState {
    tailscale::state(CLOUD_PORT).await
}

#[tauri::command]
async fn tailscale_share() -> CmdResult<tailscale::ShareResult> {
    tailscale::share(CLOUD_PORT).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn tailscale_stop() -> CmdResult<()> {
    tailscale::stop().await.map_err(|e| format!("{e:#}"))
}

/// Runs the cloud from a SQLite file in this Mac's Borrowed Compute folder.
async fn host_cloud(data_dir: std::path::PathBuf) -> anyhow::Result<()> {
    std::fs::create_dir_all(&data_dir)?;
    let db = bc_cloud::connect(&bc_cloud::sqlite_url(&data_dir.join("cloud.db"))).await?;
    let client_id = std::env::var("BC_GITHUB_CLIENT_ID")
        .unwrap_or_else(|_| DEFAULT_GITHUB_CLIENT_ID.to_string());
    let state = bc_cloud::AppState { db, github: Arc::new(bc_cloud::GitHub::new(client_id)) };
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", CLOUD_PORT))
        .await
        .map_err(|e| anyhow::anyhow!("port {CLOUD_PORT} is busy ({e}); quit whatever else uses it"))?;
    bc_cloud::serve(listener, state).await?;
    Ok(())
}

/// Pairing requests and paired devices for this Mac.
#[tauri::command]
async fn list_pairings(state: State<'_, AppState>) -> CmdResult<Vec<PairingInfo>> {
    let me = device_id(&state)?;
    let cloud = state.cloud();
    let list = cloud.pairings(&me).await.map_err(|e| format!("{e:#}"))?;
    Ok(list.into_iter().filter(|p| p.mac_id == me).collect())
}

#[tauri::command]
async fn approve(state: State<'_, AppState>, id: i64, code: String) -> CmdResult<PairingInfo> {
    let cloud = state.cloud();
    cloud.approve(id, &code).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn unpair(state: State<'_, AppState>, id: i64) -> CmdResult<()> {
    let cloud = state.cloud();
    cloud.unpair(id).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
fn recent_jobs(state: State<'_, AppState>) -> Vec<JobRecord> {
    state.jobs.lock().unwrap().iter().cloned().collect()
}

#[tauri::command]
fn open_url(app: AppHandle, url: String) -> CmdResult<()> {
    if !url.starts_with("https://") {
        return Err("only https links can be opened".into());
    }
    app.opener().open_url(url, None::<&str>).map_err(err)
}

/// Keeps the job list and cloud status current, and forwards each event to the window.
fn record(state: &AppState, event: &AgentEvent) {
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
            log(format_args!("job {job_id} started for {}: {}", short(client), argv.join(" ")));
        }
        AgentEvent::JobFinished { job_id, end } => {
            let mut jobs = state.jobs.lock().unwrap();
            if let Some(job) = jobs.iter_mut().find(|j| j.job_id.as_deref() == Some(job_id)) {
                job.end = Some(end.clone());
                job.finished_ms = Some(now_ms());
                state.logs.append_job(job);
            }
            log(format_args!("job {job_id} {}", describe_end(end)));
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
                state.logs.append_job(job);
            }
            log(format_args!("rejected a job from {}: {reason}", short(client)));
        }
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

fn describe_end(end: &JobEnd) -> String {
    match end {
        JobEnd::Exited { code, timed_out, usage } => {
            let how = match (timed_out, code) {
                (true, _) => "timed out".to_string(),
                (false, Some(c)) => format!("exited with code {c}"),
                (false, None) => "was stopped by a signal".to_string(),
            };
            match usage {
                Some(u) => format!("{how} · {}", u.summary()),
                None => how,
            }
        }
        JobEnd::Failed { reason } => format!("failed: {reason}"),
        JobEnd::Cancelled => "was cancelled by the client".to_string(),
        JobEnd::Disconnected => "ended: the client disconnected".to_string(),
    }
}

#[tauri::command]
fn open_logs(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    app.opener().open_path(state.logs.dir.to_string_lossy(), None::<&str>).map_err(err)
}

fn show_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn toggle_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) && window.is_focused().unwrap_or(false) {
            let _ = window.hide();
        } else {
            show_window(app);
        }
    }
}

fn main() {
    let root = home_dir().expect("config directory");
    let home = root.join("agent");
    let paths = AgentPaths::new(home.clone());
    let cloud_url = cloud_url_for(&home);
    let hosting = cloud_url == DEFAULT_CLOUD_URL;
    let signed_in = load_session(&home).is_some();
    let logs = Logs::new(root.join("logs"));
    if let Err(e) = logs.capture_stderr() {
        eprintln!("could not open agent.log: {e}");
    }
    log(format_args!("Borrowed Compute Agent {} starting", env!("CARGO_PKG_VERSION")));
    let jobs = logs.recent_jobs(MAX_JOBS).into();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            paths,
            host: StdMutex::new(CloudHost { hosting, running: false, error: None, url: cloud_url }),
            agent: OnceCell::new(),
            start_error: StdMutex::new(None),
            sync: Mutex::new(None),
            jobs: StdMutex::new(jobs),
            cloud_state: StdMutex::new(CloudState::default()),
            logs,
        })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_account,
            login,
            logout,
            list_pairings,
            approve,
            unpair,
            recent_jobs,
            open_logs,
            open_url,
            tailscale_status,
            tailscale_share,
            tailscale_stop
        ])
        .setup(move |app| {
            // A menu-bar app: no Dock icon.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let open = MenuItem::with_id(app, "open", "Open Borrowed Compute Agent", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit (stops accepting jobs)", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
            TrayIconBuilder::with_id("main")
                .icon(tauri::image::Image::from_bytes(include_bytes!("../icons/tray.png"))?)
                .icon_as_template(true)
                .tooltip("Borrowed Compute Agent")
                .menu(&menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        toggle_window(tray.app_handle());
                    }
                })
                .build(app)?;

            if hosting {
                let handle = app.handle().clone();
                let data_dir = root.join("cloud");
                tauri::async_runtime::spawn(async move {
                    handle.state::<AppState>().host.lock().unwrap().running = true;
                    let result = host_cloud(data_dir).await;
                    let state = handle.state::<AppState>();
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

            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<AppState>();
                match Agent::start(&state.paths, false, Some(tx)).await {
                    Ok(agent) => {
                        let agent = Arc::new(agent);
                        let _ = state.agent.set(agent.clone());
                        tauri::async_runtime::spawn(async move { agent.accept_loop().await });
                        restart_sync(&state).await;
                    }
                    Err(e) => {
                        log(format_args!("agent could not start: {e:#}"));
                        *state.start_error.lock().unwrap() = Some(format!("{e:#}"));
                    }
                }
                let _ = handle.emit("status-changed", ());
            });

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    let state = handle.state::<AppState>();
                    record(&state, &event);
                    // The cloud no longer accepts this Mac's session (expired, or it was
                    // issued by a different cloud): sign out and ask the user to sign in.
                    if let AgentEvent::CloudSync { ok: false, error: Some(error) } = &event {
                        if error.ends_with("(401 Unauthorized)") {
                            sign_out_locally(&state).await;
                            show_window(&handle);
                        }
                    }
                    let _ = handle.emit("agent-event", &event);
                }
            });

            // Start cloud sync when a session appears on disk, for example after
            // `bc-agent login`, even while the window is closed.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    let state = handle.state::<AppState>();
                    let signed_in = load_session(&state.paths.home).is_some();
                    let syncing = state.sync.lock().await.is_some();
                    if signed_in != syncing && state.agent.get().is_some() {
                        restart_sync(&state).await;
                        let _ = handle.emit("status-changed", ());
                    }
                }
            });

            if !signed_in {
                show_window(app.handle());
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window only hides it; the agent keeps serving from the menu bar.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Borrowed Compute Agent");
}
