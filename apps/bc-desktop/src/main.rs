//! Borrowed Compute: desktop app for sending jobs to a paired Mac.
//!
//! The same flows as the `bc` command, and the same device identity: it uses the CLI's
//! home directory, so signing in once covers both.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, SystemTime};

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
use bc_cli::route::RouteMemory;
use bc_cli::run::{run_on_mac, Exit};
use bc_net::{load_or_create_key, NetConfig};
use bc_proto::cloud::{DeviceInfo, DeviceKind, PairingInfo, Provider, Providers};
use bc_proto::wire::JobUsage;
use bc_proto::JobSpec;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_updater::UpdaterExt;
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
    /// What the folder looks like, when it has no bc.toml.
    detected: Option<DetectedView>,
}

#[derive(Serialize)]
struct DetectedView {
    kind: String,
    reason: String,
    suggestions: Vec<SuggestionView>,
}

#[derive(Serialize)]
struct SuggestionView {
    label: String,
    /// The command as the command box shows it; `split_command` turns it back into argv.
    command: String,
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
    usage: Option<JobUsage>,
    /// `JobUsage::summary`, so the window shows the same text as the log.
    usage_text: Option<String>,
    log_path: String,
}

/// Each run's output and result go to `<home>/logs/job-<time>.log`; the oldest beyond this
/// many are deleted.
const MAX_JOB_LOGS: usize = 200;

type SharedLog = Arc<StdMutex<std::fs::File>>;

fn logs_dir(home: &Path) -> PathBuf {
    home.join("logs")
}

/// Creates the log for one run, headed with what is being run where.
fn open_job_log(home: &Path, mac: &str, folder: &str, argv: &[String]) -> anyhow::Result<(PathBuf, SharedLog)> {
    let dir = logs_dir(home);
    std::fs::create_dir_all(&dir)?;
    prune_job_logs(&dir);
    let now = humantime::format_rfc3339_seconds(SystemTime::now()).to_string();
    // Colons are not allowed in Windows file names.
    let path = dir.join(format!("job-{}.log", now.replace(':', "-")));
    let mut file = std::fs::File::create(&path)?;
    writeln!(file, "Started {now}\nMac: {mac}\nFolder: {folder}\nCommand: {}\n", join_command(argv))?;
    Ok((path, Arc::new(StdMutex::new(file))))
}

fn prune_job_logs(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut logs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("job-")))
        .collect();
    // Names start with the time, so they sort oldest first.
    logs.sort();
    let excess = logs.len().saturating_sub(MAX_JOB_LOGS - 1);
    for old in &logs[..excess] {
        let _ = std::fs::remove_file(old);
    }
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
async fn login(
    app: AppHandle,
    state: State<'_, AppState>,
    name: String,
    provider: Option<String>,
) -> CmdResult<Account> {
    let provider = match provider.as_deref() {
        Some("google") => Provider::Google,
        _ => Provider::Github,
    };
    // A fresh client, so other commands are not blocked while the user is on GitHub.
    let mut client = CloudClient::new(cloud_url_for(&state.home), None);
    let emitter = app.clone();
    let (token, login) = client
        .login_with(provider, |start| {
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

/// Which sign-in buttons to show. The cloud decides; an unreachable cloud shows GitHub
/// only, and sign-in then reports the real problem.
#[tauri::command]
async fn providers(state: State<'_, AppState>) -> CmdResult<Providers> {
    let client = CloudClient::new(cloud_url_for(&state.home), None);
    Ok(client.providers().await.unwrap_or(Providers { github: true, google: false }))
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
        let detected = bc_cli::detect::detect(Path::new(&folder)).map(|d| DetectedView {
            kind: d.kind,
            reason: d.reason,
            suggestions: d
                .suggestions
                .into_iter()
                .map(|s| SuggestionView { label: s.label, command: join_command(&s.argv) })
                .collect(),
        });
        return ProjectInfo { has_spec: false, command: None, spec_error: None, detected };
    }
    match std::fs::read_to_string(&path).map_err(anyhow::Error::from).and_then(|t| Ok(JobSpec::from_toml(&t)?)) {
        Ok(spec) => ProjectInfo { has_spec: true, command: Some(join_command(&spec.job.argv)), spec_error: None, detected: None },
        Err(e) => ProjectInfo { has_spec: true, command: None, spec_error: Some(format!("{e:#}")), detected: None },
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
    // Meet the Mac on its own relay when it runs one, keeping public relays out of it.
    let relays = cloud.relay().await.and_then(|url| bc_net::relay_map(&url).ok());
    let (log_path, log) = open_job_log(&state.home, mac, folder, &spec.job.argv)?;
    let mut stdout = EmitWriter::new(app.clone(), "stdout", log.clone());
    let mut stderr = EmitWriter::new(app.clone(), "stderr", log.clone());
    let started_phase = app.clone();
    let notice_app = app.clone();
    let notice_log = log.clone();
    let outcome = run_on_mac(
        &key,
        NetConfig { relay_only, relays, ..Default::default() },
        resolved.addr,
        &spec,
        archive,
        &mut stdout,
        &mut stderr,
        async move {
            let _ = cancel.await;
            phase(&started_phase, "Cancelling…");
        },
        &RouteMemory::at(&state.home),
        move |text: &str| {
            let _ = writeln!(notice_log.lock().unwrap(), "\n{text}");
            let _ = notice_app.emit("job-output", OutputChunk { stream: "notice", text: format!("{text}\n") });
            phase(&notice_app, "Switching route…");
        },
    )
    .await;
    stdout.flush()?;
    stderr.flush()?;
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(e) => {
            let _ = writeln!(log.lock().unwrap(), "\nCould not finish the run: {e:#}");
            return Err(e);
        }
    };

    let timing = outcome.timings.to_json(&outcome.job_id, &outcome.exit, outcome.usage.as_ref());
    append_timing(&state.home, &timing);
    let (kind, message) = match &outcome.exit {
        Exit::Code(0) => ("success", "Finished with exit code 0".to_string()),
        Exit::Code(c) => ("failure", format!("Finished with exit code {c}")),
        Exit::TimedOut => ("failure", "Timed out on the Mac".to_string()),
        Exit::Killed => ("failure", "Stopped on the Mac without an exit code".to_string()),
        Exit::Failed(r) => ("failure", format!("Could not run on the Mac: {r}")),
        Exit::Rejected(r) => ("rejected", format!("The Mac rejected the job: {r}")),
        Exit::Cancelled => ("cancelled", "Cancelled".to_string()),
    };
    let usage_text = outcome.usage.as_ref().map(JobUsage::summary);
    let _ = writeln!(
        log.lock().unwrap(),
        "\n{message}\nUsed on the Mac: {}\nJob {}: {timing}",
        usage_text.as_deref().unwrap_or("not reported"),
        outcome.job_id,
    );
    let t = &outcome.timings;
    Ok(JobResult {
        status: outcome.exit.status(),
        outcome: kind,
        message,
        usage: outcome.usage,
        usage_text,
        log_path: log_path.to_string_lossy().into_owned(),
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

#[derive(Serialize)]
struct UpdateInfo {
    current: String,
    /// The newer version, or `None` when this one is the latest.
    available: Option<String>,
}

#[tauri::command]
async fn check_update(app: AppHandle) -> CmdResult<UpdateInfo> {
    let update = app.updater().map_err(err)?.check().await.map_err(err)?;
    Ok(UpdateInfo {
        current: app.package_info().version.to_string(),
        available: update.map(|u| u.version),
    })
}

/// Downloads, verifies and installs the latest version, then restarts into it. On Windows
/// the installer closes the app itself.
#[tauri::command]
async fn install_update(app: AppHandle, state: State<'_, AppState>) -> CmdResult<()> {
    if state.job.lock().unwrap().is_some() {
        return Err("A job is running. Update when it finishes.".into());
    }
    let update = app.updater().map_err(err)?.check().await.map_err(err)?;
    let Some(update) = update else {
        return Err("Already up to date.".into());
    };
    update.download_and_install(|_, _| {}, || {}).await.map_err(err)?;
    app.restart();
}

/// Opens a run's log. Only files in the logs folder can be opened this way.
#[tauri::command]
fn open_log(app: AppHandle, state: State<'_, AppState>, path: String) -> CmdResult<()> {
    let path = PathBuf::from(path);
    if path.parent() != Some(logs_dir(&state.home).as_path()) {
        return Err("not a job log".into());
    }
    app.opener().open_path(path.to_string_lossy(), None::<&str>).map_err(err)
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
    /// The run's log file, shared by stdout and stderr so it keeps their order.
    log: SharedLog,
}

impl EmitWriter {
    fn new(app: AppHandle, stream: &'static str, log: SharedLog) -> Self {
        EmitWriter { app, stream, pending: Vec::new(), log }
    }

    fn emit(&self, text: String) {
        if !text.is_empty() {
            let _ = self.app.emit("job-output", OutputChunk { stream: self.stream, text });
        }
    }
}

impl Write for EmitWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // A full disk must not stop the job's output reaching the window.
        let _ = self.log.lock().unwrap().write_all(buf);
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
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(AppState { home, job: StdMutex::new(None) })
        .invoke_handler(tauri::generate_handler![
            get_status,
            get_account,
            login,
            providers,
            logout,
            list_devices,
            request_pairing,
            wait_pairing,
            choose_mac,
            pick_folder,
            project_info,
            run_job,
            cancel_job,
            open_log,
            check_update,
            install_update,
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
    fn prunes_oldest_job_logs() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..MAX_JOB_LOGS + 5 {
            std::fs::write(dir.path().join(format!("job-{i:04}.log")), "").unwrap();
        }
        std::fs::write(dir.path().join("other.txt"), "").unwrap();
        prune_job_logs(dir.path());
        let mut left: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        // Room for the run about to start.
        assert_eq!(left.len(), MAX_JOB_LOGS);
        assert_eq!(left[0], "job-0006.log");
        assert!(left.contains(&"other.txt".to_string()));
    }

    #[test]
    fn human_bytes_units() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(2048), "2.0 KB");
        assert_eq!(human_bytes(5 << 20), "5.0 MB");
    }
}
