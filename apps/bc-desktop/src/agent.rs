//! The Agent page: conversations with the model running on the Mesh page, which can use
//! tools on a chosen folder and remember past conversations. The loop, tools and memory
//! live in `bc_cli::agent`; this file connects them to the window and gives `run` its
//! devices: this computer (its own job sandbox, on a copy of the folder) or a paired one.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex as StdMutex};

use bc_cli::agent::format::parse;
use bc_cli::agent::http::{HttpBrain, ModelServer};
use bc_cli::agent::memory::{Conversation, Memory, Summary};
use bc_cli::agent::tools::{RunRequest, Runner, Tools};
use bc_cli::agent::{Agent, Brain, Chat, Decision, Settings};
use bc_proto::wire::{AgentMsg, JobRequest};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::{oneshot, Mutex};

use crate::mesh::MeshState;
use crate::{err_chain, AppState, CmdResult};

/// Open conversations whose streams stay on the stages; older ones start over when reopened.
const OPEN_CHATS: usize = 4;
const THIS_COMPUTER: &str = "This computer";

/// Calls waiting for the person: call id to (conversation, answer).
type Pending = Arc<StdMutex<HashMap<String, (String, oneshot::Sender<Decision>)>>>;

#[derive(Default)]
pub struct AgentState {
    chats: Mutex<Vec<(String, Arc<Mutex<Chat>>)>>,
    pending: Pending,
    stops: StdMutex<HashMap<String, Arc<AtomicBool>>>,
    /// The model server's brain, kept while its settings stay the same, so conversations
    /// keep their token ids between messages.
    server: StdMutex<Option<(ModelServer, Arc<HttpBrain>)>>,
}

fn server_file(state: &AppState) -> std::path::PathBuf {
    state.home.join("model-server.json")
}

fn model_server(state: &AppState) -> ModelServer {
    std::fs::read(server_file(state)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// The model the agent uses: the Mesh page's while one runs, else the model server's.
async fn brain(state: &AppState, mesh: &MeshState, agent: &AgentState) -> CmdResult<(Arc<dyn Brain>, String)> {
    if let Some(session) = mesh.session().await {
        return Ok((session.llm.clone() as Arc<dyn Brain>, session.name.clone()));
    }
    let server = model_server(state);
    if !server.is_set() {
        return Err("Start a model on the Mesh page, or set a model server under Memory.".into());
    }
    let mut kept = agent.server.lock().unwrap();
    if kept.as_ref().is_none_or(|(s, _)| *s != server) {
        *kept = Some((server.clone(), Arc::new(HttpBrain::new(server.clone()).map_err(err_chain)?)));
    }
    let b = kept.as_ref().map(|(_, b)| b.clone()).expect("just set");
    Ok((b as Arc<dyn Brain>, server.model))
}

/// The model the Agent page would use now, by name; empty with none.
#[tauri::command]
pub async fn agent_model(state: State<'_, AppState>, mesh: State<'_, MeshState>, agent: State<'_, AgentState>) -> CmdResult<String> {
    Ok(brain(&state, &mesh, &agent).await.map(|(_, name)| name).unwrap_or_default())
}

#[tauri::command]
pub fn model_server_get(state: State<'_, AppState>) -> ModelServer {
    model_server(&state)
}

#[tauri::command]
pub fn model_server_set(state: State<'_, AppState>, server: ModelServer) -> CmdResult<()> {
    let url = server.url.trim();
    if !(url.is_empty() || url.starts_with("http://") || url.starts_with("https://")) {
        return Err("The address starts with http:// or https://.".into());
    }
    std::fs::write(server_file(&state), serde_json::to_vec_pretty(&server).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn memory(state: &AppState) -> Memory {
    Memory::at(&state.home)
}

/// A conversation, open: kept in memory while recent, so its stream stays on the stages.
async fn chat(agent: &AgentState, mem: &Memory, id: &str) -> CmdResult<Arc<Mutex<Chat>>> {
    let mut chats = agent.chats.lock().await;
    if let Some(i) = chats.iter().position(|(c, _)| c == id) {
        let c = chats.remove(i);
        chats.push(c.clone());
        return Ok(c.1);
    }
    let c = Arc::new(Mutex::new(Chat::new(mem.load(id).map_err(err_chain)?)));
    chats.push((id.to_string(), c.clone()));
    if chats.len() > OPEN_CHATS {
        chats.remove(0);
    }
    Ok(c)
}

/// Starts a conversation, its file tools on `folder` if one is given.
#[tauri::command]
pub fn agent_new(state: State<'_, AppState>, folder: Option<String>) -> CmdResult<String> {
    if let Some(f) = &folder {
        if !std::path::Path::new(f).is_dir() {
            return Err(format!("{f} is not a folder"));
        }
    }
    let c = Conversation::new(folder);
    memory(&state).save(&c).map_err(err_chain)?;
    Ok(c.id)
}

/// Sends a message and works on it until the model answers without a tool. Sends
/// `agent-event` events (`{conversation, event}`) as it goes.
#[tauri::command]
pub async fn agent_send(app: AppHandle, state: State<'_, AppState>, mesh: State<'_, MeshState>, agent: State<'_, AgentState>, conversation: String, text: String, settings: Settings) -> CmdResult<()> {
    let (brain, _) = brain(&state, &mesh, &agent).await?;
    let mem = memory(&state);
    let chat = chat(&agent, &mem, &conversation).await?;
    let mut chat = chat.try_lock().map_err(|_| "This conversation is still working; stop it or wait.".to_string())?;
    let devices = run_devices(&state).await;
    let folder = chat.conv.folder.clone().map(std::path::PathBuf::from);
    let tools = Tools::new(folder.as_deref(), mem.clone(), conversation.clone(), Some(runner(&state)), devices).map_err(err_chain)?;
    let stop = Arc::new(AtomicBool::new(false));
    agent.stops.lock().unwrap().insert(conversation.clone(), stop.clone());
    let (events_app, conv) = (app.clone(), conversation.clone());
    let pending = agent.pending.clone();
    let conv_for_approve = conversation.clone();
    let a = Agent {
        brain,
        tools,
        memory: mem,
        events: Arc::new(move |e| {
            let _ = events_app.emit("agent-event", json!({ "conversation": conv, "event": e }));
        }),
        approve: Arc::new(move |call| {
            let (tx, rx) = oneshot::channel();
            pending.lock().unwrap().insert(call.id, (conv_for_approve.clone(), tx));
            // A dropped answer (Stop, or the window closing) counts as no.
            Box::pin(async move { rx.await.unwrap_or(Decision::Deny) })
        }),
    };
    let result = a.turn(&mut chat, &text, settings, stop).await;
    agent.stops.lock().unwrap().remove(&conversation);
    result.map_err(err_chain)
}

#[tauri::command]
pub fn agent_approve(agent: State<'_, AgentState>, call: String, decision: Decision) {
    if let Some((_, tx)) = agent.pending.lock().unwrap().remove(&call) {
        let _ = tx.send(decision);
    }
}

/// Stops a conversation at its next token; calls waiting for approval are refused.
#[tauri::command]
pub fn agent_stop(agent: State<'_, AgentState>, conversation: String) {
    if let Some(s) = agent.stops.lock().unwrap().get(&conversation) {
        s.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    agent.pending.lock().unwrap().retain(|_, (c, _)| *c != conversation);
}

#[tauri::command]
pub fn agent_conversations(state: State<'_, AppState>) -> Vec<Summary> {
    memory(&state).list()
}

/// A conversation for the window: each message, assistant ones read into text, reasoning
/// and calls.
#[tauri::command]
pub fn agent_open(state: State<'_, AppState>, conversation: String) -> CmdResult<Value> {
    let c = memory(&state).load(&conversation).map_err(err_chain)?;
    let messages: Vec<Value> = c
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "assistant" => {
                let r = parse(&m.content);
                let calls: Vec<Value> = r.calls.iter().map(|c| match c {
                    Ok(c) => json!({ "name": c.name, "arguments": c.arguments }),
                    Err(e) => json!({ "name": "", "error": e }),
                }).collect();
                json!({ "role": "assistant", "text": r.text, "thought": r.thought, "calls": calls })
            }
            role => json!({ "role": role, "text": m.content, "name": m.name }),
        })
        .collect();
    Ok(json!({ "id": c.id, "title": c.title, "folder": c.folder, "messages": messages }))
}

#[tauri::command]
pub async fn agent_delete(state: State<'_, AppState>, agent: State<'_, AgentState>, conversation: Option<String>) -> CmdResult<()> {
    let mem = memory(&state);
    let mut chats = agent.chats.lock().await;
    match conversation {
        Some(id) => {
            chats.retain(|(c, _)| *c != id);
            mem.delete(&id).map_err(err_chain)
        }
        None => {
            chats.clear();
            mem.delete_all().map_err(err_chain)
        }
    }
}

#[tauri::command]
pub fn memory_get(state: State<'_, AppState>) -> String {
    memory(&state).notes()
}

#[tauri::command]
pub fn memory_set(state: State<'_, AppState>, text: String) -> CmdResult<()> {
    memory(&state).set_notes(&text).map_err(err_chain)
}

/// Where `run` can go: this computer, then the paired devices.
async fn run_devices(state: &AppState) -> Vec<String> {
    let mut names = vec![THIS_COMPUTER.to_string()];
    if let Ok(me) = crate::device_id(&state.home) {
        if let Ok(targets) = bc_cli::mesh::mesh_targets(&state.cloud(), &me, &state.home).await {
            names.extend(targets.into_iter().map(|(n, _)| n).filter(|n| n != THIS_COMPUTER));
        }
    }
    names
}

/// Runs a command on a packed copy of the folder: as a job on a paired device, or here in
/// this computer's own job sandbox, which deletes the copy afterwards.
fn runner(state: &AppState) -> Runner {
    let home = state.home.clone();
    Arc::new(move |r: RunRequest| {
        let home = home.clone();
        Box::pin(async move {
            let spec = bc_cli::flows::load_spec(&r.folder, None, r.argv.clone())?;
            let (root, excludes) = (r.folder.clone(), spec.files.exclude.clone());
            let archive = tokio::task::spawn_blocking(move || bc_cli::pack::pack(&root, &excludes)).await??;
            match r.device.as_deref().filter(|d| *d != THIS_COMPUTER) {
                Some(device) => {
                    let cloud = bc_auth::CloudClient::from_home(bc_auth::cloud_url_for(&home), &home);
                    let mac = bc_cli::flows::resolve_mac(&cloud, &home, device).await?;
                    let key = bc_net::load_or_create_key(&bc_cli::flows::key_path(&home))?;
                    let (mut out, mut err) = (Vec::new(), Vec::new());
                    let outcome = bc_cli::run::run_on_mac(
                        &key,
                        bc_cli::flows::mac_net(&cloud, false).await,
                        mac.addr,
                        &spec,
                        archive,
                        &mut out,
                        &mut err,
                        std::future::pending::<()>(),
                        Some(&r.folder),
                        bc_cli::detect::Os::of(mac.kind),
                        &bc_cli::route::RouteMemory::at(&home),
                        |_| {},
                    )
                    .await?;
                    out.extend_from_slice(&err);
                    Ok((outcome.exit.status(), String::from_utf8_lossy(&out).into_owned()))
                }
                None => {
                    let req = JobRequest::from_spec(bc_cli::run::new_job_id(), &spec, archive.len);
                    let file = archive.open()?;
                    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
                    let run = tokio::spawn(async move { bc_agent::runner::run_job(&bc_agent::runner::RunnerConfig::default(), &req, Box::new(file), tx).await });
                    let mut out = Vec::new();
                    let mut code = -1;
                    while let Some(msg) = rx.recv().await {
                        match msg {
                            AgentMsg::Output { data, .. } => out.extend_from_slice(&data),
                            AgentMsg::Exited { code: c, timed_out, .. } => code = if timed_out { 124 } else { c.unwrap_or(-1) },
                            AgentMsg::Failed { reason } => anyhow::bail!("could not run it here: {reason}"),
                            _ => {}
                        }
                    }
                    let _ = run.await;
                    Ok((code, String::from_utf8_lossy(&out).into_owned()))
                }
            }
        })
    })
}

/// Shows the agent's reminders as they come due, as notifications and in the window
/// (`agent-reminder`), in case notifications are off for the app.
pub fn watch_reminders(app: AppHandle, home: std::path::PathBuf) {
    use tauri_plugin_notification::NotificationExt;
    tauri::async_runtime::spawn(async move {
        let reminders = Memory::at(&home).reminders();
        loop {
            for r in reminders.take_due(bc_cli::agent::memory::now() as i64) {
                if let Err(e) = app.notification().builder().title("Reminder").body(&r.text).show() {
                    crate::log(format_args!("could not show a reminder: {e}"));
                }
                let _ = app.emit("agent-reminder", &r);
            }
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        }
    });
}
