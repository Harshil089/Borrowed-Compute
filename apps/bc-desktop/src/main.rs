//! Borrowed Compute: desktop app for sending jobs to a paired Mac.
//!
//! The same flows as the `bc` command, and the same device identity: it uses the CLI's
//! home directory, so signing in once covers both.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use anyhow::Context;
use bc_auth::{
    clear_session, cloud_url_for, default_device_name, is_unauthorized, load_session, save_cloud_url,
    save_session, CloudClient, DEFAULT_CLOUD_URL,
};
use bc_cli::flows::{
    append_timing, client_home, default_mac, key_path, load_spec, resolve_mac, set_default_mac,
    split_command, wait_for_pairing,
};
use bc_cli::pack::pack;
use bc_cli::run::{run_remote, Exit};
use bc_net::{bind, load_or_create_key, NetConfig};
use bc_proto::cloud::{DeviceInfo, DeviceKind, PairingInfo};
use bc_proto::JobSpec;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tokio::sync::oneshot;

impl AppState {
    /// A client for the saved cloud address and session. Read from disk each time, so a
    /// sign-in with the `bc` command shows up here too.
    fn cloud(&self) -> CloudClient {
        CloudClient::from_home(cloud_url_for(&self.home), &self.home)
    }
}

struct AppState {
    home: PathBuf,
    /// Cancels the running job. At most one job runs at a time.
    job: StdMutex<Option<oneshot::Sender<()>>>,
}

type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn err_chain(e: anyhow::Error) -> String {
    format!("{e:#}")
}

fn device_id(home: &Path) -> CmdResult<String> {
    Ok(load_or_create_key(&key_path(home)).map_err(err_chain)?.public().to_string())
}

#[derive(Serialize)]
struct Status {
    signed_in: bool,
    device_id: Option<String>,
    default_mac: Option<String>,
    cloud_url: String,
    /// False while the address is still the built-in default, which only works when this
    /// computer is the Mac itself.
    cloud_url_set: bool,
}

#[derive(Serialize)]
struct Account {
    login: String,
    device_name: Option<String>,
}

#[derive(Serialize)]
struct Devices {
    me: String,
    devices: Vec<DeviceInfo>,
    /// Pairings where this device is the client.
    pairings: Vec<PairingInfo>,
}

#[derive(Serialize)]
struct ProjectInfo {
    has_spec: bool,
    command: Option<String>,
    spec_error: Option<String>,
}

#[derive(Serialize)]
struct TimingsView {
    connect_ms: u128,
    upload_ms: u128,
    start_ms: Option<u128>,
    first_log_ms: Option<u128>,
    total_ms: u128,
    via_relay: bool,
    archive_bytes: u64,
}

#[derive(Serialize)]
struct JobResult {
    status: i32,
    outcome: &'static str,
    message: String,
    timings: TimingsView,
}

#[derive(Clone, Serialize)]
struct OutputChunk {
    stream: &'static str,
    text: String,
}

#[tauri::command]
fn get_status(state: State<'_, AppState>) -> Status {
    Status {
        signed_in: load_session(&state.home).is_some(),
        device_id: device_id(&state.home).ok(),
        default_mac: default_mac(&state.home),
        cloud_url: cloud_url_for(&state.home),
        cloud_url_set: cloud_url_for(&state.home) != DEFAULT_CLOUD_URL,
    }
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
        Err(e) => return Err(err_chain(e)),
    };
    let id = device_id(&state.home)?;
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
    let mut client = CloudClient::new(cloud_url_for(&state.home), None);
    let emitter = app.clone();
    let (token, login) = client
        .login(|start| {
            let _ = emitter.emit("login-code", start);
            let _ = emitter.opener().open_url(&start.verification_uri, None::<&str>);
        })
        .await
        .map_err(err_chain)?;
    save_session(&state.home, &token).map_err(err_chain)?;
    let key = load_or_create_key(&key_path(&state.home)).map_err(err_chain)?;
    let name = match name.trim() {
        "" => default_device_name(),
        n => n.to_string(),
    };
    let device = client.register(&key, &name, DeviceKind::current()).await.map_err(err_chain)?;
    drop(client);
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
    clear_session(&state.home);
}

/// Checks that a Borrowed Compute cloud answers at `url`, without saving it.
#[tauri::command]
async fn test_cloud_url(url: String) -> CmdResult<String> {
    let url = bc_auth::normalize_cloud_url(&url).map_err(err_chain)?;
    CloudClient::new(&url, None).health().await.map_err(err_chain)?;
    Ok(url)
}

/// Saves the cloud address. A new address means a different cloud, so this device signs
/// out and signs in again there.
#[tauri::command]
async fn set_cloud_url(state: State<'_, AppState>, url: String) -> CmdResult<String> {
    let before = cloud_url_for(&state.home);
    let saved = save_cloud_url(&state.home, &url).map_err(err_chain)?;
    if saved != before {
        sign_out_locally(&state).await;
    }
    Ok(saved)
}

#[tauri::command]
async fn list_devices(state: State<'_, AppState>) -> CmdResult<Devices> {
    let me = device_id(&state.home)?;
    let cloud = state.cloud();
    let devices = cloud.devices().await.map_err(err_chain)?;
    let pairings = cloud
        .pairings(&me)
        .await
        .map_err(err_chain)?
        .into_iter()
        .filter(|p| p.client_id == me)
        .collect();
    Ok(Devices { me, devices, pairings })
}

#[tauri::command]
async fn request_pairing(state: State<'_, AppState>, mac: String) -> CmdResult<PairingInfo> {
    let me = device_id(&state.home)?;
    let cloud = state.cloud();
    cloud.request_pairing(&me, &mac).await.map_err(err_chain)
}

/// Resolves true once the Mac approves, false if the code expires.
#[tauri::command]
async fn wait_pairing(state: State<'_, AppState>, id: i64, expires_in_secs: i64) -> CmdResult<bool> {
    let me = device_id(&state.home)?;
    let cloud = state.cloud();
    let expires = Duration::from_secs(expires_in_secs.max(0) as u64);
    wait_for_pairing(&cloud, &me, id, expires, || {}).await.map_err(err_chain)
}

#[tauri::command]
fn choose_mac(state: State<'_, AppState>, mac: String) -> CmdResult<()> {
    set_default_mac(&state.home, &mac).map_err(err_chain)
}

#[tauri::command]
async fn pick_folder(app: AppHandle) -> CmdResult<Option<String>> {
    let picked = tauri::async_runtime::spawn_blocking(move || {
        app.dialog().file().set_title("Choose the project folder").blocking_pick_folder()
    })
    .await
    .map_err(err)?;
    Ok(picked.and_then(|p| p.into_path().ok()).map(|p| p.display().to_string()))
}

/// What the folder's `bc.toml` would run, so the window can prefill the command.
#[tauri::command]
fn project_info(folder: String) -> ProjectInfo {
    let path = Path::new(&folder).join("bc.toml");
    if !path.exists() {
        return ProjectInfo { has_spec: false, command: None, spec_error: None };
    }
    match std::fs::read_to_string(&path).map_err(anyhow::Error::from).and_then(|t| Ok(JobSpec::from_toml(&t)?)) {
        Ok(spec) => ProjectInfo { has_spec: true, command: Some(join_command(&spec.job.argv)), spec_error: None },
        Err(e) => ProjectInfo { has_spec: true, command: None, spec_error: Some(format!("{e:#}")) },
    }
}

/// Inverse of `split_command` for display: quotes words that need it.
fn join_command(argv: &[String]) -> String {
    argv.iter()
        .map(|a| {
            if a.is_empty() || a.chars().any(|c| c.is_whitespace() || c == '"' || c == '\'') {
                if a.contains('"') { format!("'{a}'") } else { format!("\"{a}\"") }
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn phase(app: &AppHandle, text: impl Into<String>) {
    let _ = app.emit("job-phase", text.into());
}

#[tauri::command]
async fn run_job(
    app: AppHandle,
    state: State<'_, AppState>,
    mac: String,
    folder: String,
    command: String,
    relay_only: bool,
) -> CmdResult<JobResult> {
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    {
        let mut job = state.job.lock().unwrap();
        if job.is_some() {
            return Err("a job is already running".into());
        }
        *job = Some(cancel_tx);
    }
    let result = run_job_inner(&app, &state, &mac, &folder, &command, relay_only, cancel_rx).await;
    state.job.lock().unwrap().take();
    result.map_err(err_chain)
}

async fn run_job_inner(
    app: &AppHandle,
    state: &AppState,
    mac: &str,
    folder: &str,
    command: &str,
    relay_only: bool,
    cancel: oneshot::Receiver<()>,
) -> anyhow::Result<JobResult> {
    let root = PathBuf::from(folder);
    if !root.is_dir() {
        anyhow::bail!("{folder} is not a folder");
    }
    let spec = load_spec(&root, None, split_command(command)?)?;

    phase(app, format!("Finding {mac}…"));
    let cloud = state.cloud();
    let resolved = resolve_mac(&cloud, mac).await?;
    if let Some(warning) = &resolved.offline_warning {
        let _ = app.emit("job-output", OutputChunk { stream: "notice", text: format!("{warning}\n") });
    }

    phase(app, "Packing the project…");
    let excludes = spec.files.exclude.clone();
    let pack_root = root.clone();
    let archive = tauri::async_runtime::spawn_blocking(move || pack(&pack_root, &excludes))
        .await
        .context("packing")??;

    phase(app, format!("Connecting and uploading {}…", human_bytes(archive.len() as u64)));
    let key = load_or_create_key(&key_path(&state.home))?;
    let endpoint = bind(key, NetConfig { relay_only, ..Default::default() }).await?;
    let mut stdout = EmitWriter::new(app.clone(), "stdout");
    let mut stderr = EmitWriter::new(app.clone(), "stderr");
    let started_phase = app.clone();
    let outcome = run_remote(
        &endpoint,
        resolved.addr,
        &spec,
        archive,
        &mut stdout,
        &mut stderr,
        async move {
            let _ = cancel.await;
            phase(&started_phase, "Cancelling…");
        },
    )
    .await;
    endpoint.close().await;
    let outcome = outcome?;
    stdout.flush()?;
    stderr.flush()?;

    append_timing(&state.home, &outcome.timings.to_json(&outcome.job_id, &outcome.exit));
    let (kind, message) = match &outcome.exit {
        Exit::Code(0) => ("success", "Finished with exit code 0".to_string()),
        Exit::Code(c) => ("failure", format!("Finished with exit code {c}")),
        Exit::TimedOut => ("failure", "Timed out on the Mac".to_string()),
        Exit::Killed => ("failure", "Stopped on the Mac without an exit code".to_string()),
        Exit::Failed(r) => ("failure", format!("Could not run on the Mac: {r}")),
        Exit::Rejected(r) => ("rejected", format!("The Mac rejected the job: {r}")),
        Exit::Cancelled => ("cancelled", "Cancelled".to_string()),
    };
    let t = &outcome.timings;
    Ok(JobResult {
        status: outcome.exit.status(),
        outcome: kind,
        message,
        timings: TimingsView {
            connect_ms: t.connect.as_millis(),
            upload_ms: t.upload.as_millis(),
            start_ms: t.start.map(|d| d.as_millis()),
            first_log_ms: t.first_log.map(|d| d.as_millis()),
            total_ms: t.total.as_millis(),
            via_relay: t.via_relay,
            archive_bytes: t.archive_bytes,
        },
    })
}

#[tauri::command]
fn cancel_job(state: State<'_, AppState>) {
    if let Some(cancel) = state.job.lock().unwrap().take() {
        let _ = cancel.send(());
    }
}

#[tauri::command]
fn open_url(app: AppHandle, url: String) -> CmdResult<()> {
    if !url.starts_with("https://") {
        return Err("only https links can be opened".into());
    }
    app.opener().open_url(url, None::<&str>).map_err(err)
}

fn human_bytes(n: u64) -> String {
    match n {
        n if n < 1 << 10 => format!("{n} B"),
        n if n < 1 << 20 => format!("{:.1} KB", n as f64 / 1024.0),
        n if n < 1 << 30 => format!("{:.1} MB", n as f64 / (1 << 20) as f64),
        n => format!("{:.2} GB", n as f64 / (1 << 30) as f64),
    }
}

/// Sends job output to the window as text. Holds back an incomplete UTF-8 sequence at
/// the end of a chunk until the rest arrives, so characters are never split.
struct EmitWriter {
    app: AppHandle,
    stream: &'static str,
    pending: Vec<u8>,
}

impl EmitWriter {
    fn new(app: AppHandle, stream: &'static str) -> Self {
        EmitWriter { app, stream, pending: Vec::new() }
    }

    fn emit(&self, text: String) {
        if !text.is_empty() {
            let _ = self.app.emit("job-output", OutputChunk { stream: self.stream, text });
        }
    }
}

impl Write for EmitWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.pending.extend_from_slice(buf);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // An incomplete sequence at the end: wait for more bytes.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            // Invalid bytes: show them as replacement characters now.
            Err(_) => self.pending.len(),
        };
        let ready: Vec<u8> = self.pending.drain(..valid).collect();
        self.emit(String::from_utf8_lossy(&ready).into_owned());
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let rest = std::mem::take(&mut self.pending);
        self.emit(String::from_utf8_lossy(&rest).into_owned());
        Ok(())
    }
}

fn main() {
    let home = client_home().expect("config directory");
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState { home, job: StdMutex::new(None) })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_account,
            login,
            logout,
            list_devices,
            request_pairing,
            wait_pairing,
            choose_mac,
            pick_folder,
            project_info,
            run_job,
            cancel_job,
            open_url,
            test_cloud_url,
            set_cloud_url
        ])
        .run(tauri::generate_context!())
        .expect("error while running Borrowed Compute");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_command_round_trips_through_split() {
        for argv in [
            vec!["cargo", "test"],
            vec!["sh", "-c", "echo hi; exit 3"],
            vec!["python", "scripts\\run.py", ""],
            vec!["echo", "say \"hi\""],
        ] {
            let argv: Vec<String> = argv.into_iter().map(String::from).collect();
            assert_eq!(split_command(&join_command(&argv)).unwrap(), argv);
        }
    }

    #[test]
    fn human_bytes_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(5 << 20), "5.0 MB");
    }
}
