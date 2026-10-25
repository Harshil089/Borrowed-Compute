//! Borrowed Compute Agent: menu-bar app on the Mac.
//!
//! Runs the same agent service as `bc-agent serve`, plus sign-in, pairing approval and a
//! list of recent jobs. Closing the window keeps the agent running; Quit stops it.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{SystemTime, UNIX_EPOCH};

use bc_agent::events::{AgentEvent, JobEnd};
use bc_agent::service::{Agent, AgentPaths};
use bc_auth::{clear_session, cloud_url, default_device_name, load_session, save_session, CloudClient};
use bc_net::{home_dir, load_or_create_key};
use bc_proto::cloud::{DeviceKind, PairingInfo};
use serde::Serialize;
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, State, WindowEvent};
use tauri_plugin_opener::OpenerExt;
use tokio::sync::{Mutex, OnceCell};
use tokio::task::JoinHandle;

const MAX_JOBS: usize = 50;

#[derive(Debug, Clone, Serialize)]
struct JobRecord {
    job_id: Option<String>,
    client: String,
    argv: Vec<String>,
    started_ms: u64,
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
    cloud_url: String,
}

#[derive(Serialize)]
struct Account {
    login: String,
    device_name: Option<String>,
}

struct AppState {
    paths: AgentPaths,
    cloud: Mutex<CloudClient>,
    agent: OnceCell<Arc<Agent>>,
    start_error: StdMutex<Option<String>>,
    sync: Mutex<Option<JoinHandle<()>>>,
    jobs: StdMutex<VecDeque<JobRecord>>,
    cloud_state: StdMutex<CloudState>,
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
    let cloud = state.cloud.lock().await.clone();
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
        cloud_url: cloud_url(),
    })
}

#[tauri::command]
async fn get_account(state: State<'_, AppState>) -> CmdResult<Account> {
    let cloud = state.cloud.lock().await.clone();
    let me = cloud.me().await.map_err(|e| format!("{e:#}"))?;
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
    let mut client = CloudClient::new(cloud_url(), None);
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
    *state.cloud.lock().await = client;
    *state.cloud_state.lock().unwrap() = CloudState::default();
    restart_sync(&state).await;
    Ok(Account { login, device_name: Some(device.name) })
}

#[tauri::command]
async fn logout(state: State<'_, AppState>) -> CmdResult<()> {
    let cloud = state.cloud.lock().await.clone();
    if cloud.token().is_some() {
        let _ = cloud.logout().await;
    }
    clear_session(&state.paths.home);
    *state.cloud.lock().await = CloudClient::new(cloud_url(), None);
    *state.cloud_state.lock().unwrap() = CloudState::default();
    restart_sync(&state).await;
    Ok(())
}

/// Pairing requests and paired devices for this Mac.
#[tauri::command]
async fn list_pairings(state: State<'_, AppState>) -> CmdResult<Vec<PairingInfo>> {
    let me = device_id(&state)?;
    let cloud = state.cloud.lock().await.clone();
    let list = cloud.pairings(&me).await.map_err(|e| format!("{e:#}"))?;
    Ok(list.into_iter().filter(|p| p.mac_id == me).collect())
}

#[tauri::command]
async fn approve(state: State<'_, AppState>, id: i64, code: String) -> CmdResult<PairingInfo> {
    let cloud = state.cloud.lock().await.clone();
    cloud.approve(id, &code).await.map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn unpair(state: State<'_, AppState>, id: i64) -> CmdResult<()> {
    let cloud = state.cloud.lock().await.clone();
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
                end: None,
                rejected: None,
            });
            jobs.truncate(MAX_JOBS);
        }
        AgentEvent::JobFinished { job_id, end } => {
            let mut jobs = state.jobs.lock().unwrap();
            if let Some(job) = jobs.iter_mut().find(|j| j.job_id.as_deref() == Some(job_id)) {
                job.end = Some(end.clone());
            }
        }
        AgentEvent::JobRejected { client, reason } => {
            let mut jobs = state.jobs.lock().unwrap();
            jobs.push_front(JobRecord {
                job_id: None,
                client: client.clone(),
                argv: Vec::new(),
                started_ms: now_ms(),
                end: None,
                rejected: Some(reason.clone()),
            });
            jobs.truncate(MAX_JOBS);
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
    let home = home_dir().expect("config directory").join("agent");
    let paths = AgentPaths::new(home.clone());
    let cloud = CloudClient::from_home(cloud_url(), &home);
    let signed_in = cloud.token().is_some();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            paths,
            cloud: Mutex::new(cloud),
            agent: OnceCell::new(),
            start_error: StdMutex::new(None),
            sync: Mutex::new(None),
            jobs: StdMutex::new(VecDeque::new()),
            cloud_state: StdMutex::new(CloudState::default()),
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
            open_url
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
                    Err(e) => *state.start_error.lock().unwrap() = Some(format!("{e:#}")),
                }
                let _ = handle.emit("status-changed", ());
            });

            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    record(&handle.state::<AppState>(), &event);
                    let _ = handle.emit("agent-event", &event);
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
