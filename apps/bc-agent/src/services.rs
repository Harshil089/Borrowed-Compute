//! Services: jobs that run until someone stops them, such as a dev server.
//!
//! A normal job belongs to its connection and stops when the client goes away. A service
//! belongs to the agent instead: it has no timeout, keeps running when the PC disconnects
//! or sleeps, and ends only when its process exits or it is stopped (by the device that
//! started it, or from the Mac app). Its recent output is kept in memory, never on disk,
//! so a client can attach later and see what it missed. While any service runs, the Mac
//! is kept awake with `caffeinate`.
//!
//! Stopping a service drops its job, which kills the process group and deletes the
//! workspace, the same guarantees as a cancelled job.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use bc_proto::wire::{AgentMsg, JobRequest, ServiceInfo};
use iroh::EndpointId;
use tokio::sync::{broadcast, mpsc};
use tokio::task::AbortHandle;

use crate::events::{AgentEvent, EventSender, JobEnd};
use crate::runner::{run_job_with, RunnerConfig};

/// Output kept per service for clients that attach later.
const BACKLOG_BYTES: usize = 256 << 10;

#[derive(Clone, Default)]
pub struct Services {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    running: HashMap<String, Entry>,
    awake: Option<tokio::process::Child>,
}

struct Entry {
    info: ServiceInfo,
    /// The job's process, which leads its process group.
    pid: Option<u32>,
    started: bool,
    backlog: VecDeque<AgentMsg>,
    backlog_bytes: usize,
    live: broadcast::Sender<AgentMsg>,
    task: AbortHandle,
}

/// A client's view of a running service: what it missed, then what comes next.
pub struct Attached {
    pub backlog: Vec<AgentMsg>,
    pub live: broadcast::Receiver<AgentMsg>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StopError {
    NotFound,
    NotOwner,
}

impl Services {
    /// Starts `req` as a service owned by `owner` and attaches to it.
    pub fn start(
        &self,
        runner: RunnerConfig,
        req: JobRequest,
        archive: Vec<u8>,
        owner: EndpointId,
        events: Option<EventSender>,
    ) -> Attached {
        let job_id = req.job_id.clone();
        let info = ServiceInfo {
            job_id: job_id.clone(),
            argv: req.argv.clone(),
            started_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or_default(),
            owner: owner.to_string(),
            ports: Vec::new(),
        };
        let (live, rx) = broadcast::channel(1024);
        // Made outside the task so a service stopped before it first runs is still removed.
        let mut guard = EndGuard { services: self.clone(), job_id, events, end: JobEnd::Cancelled };
        let mut inner = self.inner.lock().unwrap();
        // Spawned under the lock so the task cannot record output before its entry exists.
        let task = tokio::spawn(async move {
            let (tx, mut msgs) = mpsc::channel(256);
            let (pid_tx, pid_rx) = tokio::sync::oneshot::channel();
            let run = run_job_with(&runner, &req, archive, tx, Some(pid_tx));
            let services = guard.services.clone();
            let id = guard.job_id.clone();
            tokio::spawn(async move {
                if let Ok(pid) = pid_rx.await {
                    if let Some(entry) = services.inner.lock().unwrap().running.get_mut(&id) {
                        entry.pid = Some(pid);
                    }
                }
            });
            let record = async {
                while let Some(msg) = msgs.recv().await {
                    match &msg {
                        AgentMsg::Exited { code, timed_out, usage } => {
                            guard.end = JobEnd::Exited { code: *code, timed_out: *timed_out, usage: *usage };
                        }
                        AgentMsg::Failed { reason } => guard.end = JobEnd::Failed { reason: reason.clone() },
                        AgentMsg::Output { data, .. } => {
                            if let Some(events) = &guard.events {
                                let _ = events.send(AgentEvent::JobOutput { job_id: guard.job_id.clone(), data: data.clone() });
                            }
                        }
                        _ => {}
                    }
                    guard.services.record(&guard.job_id, msg);
                }
            };
            tokio::join!(run, record);
        });
        inner.running.insert(
            info.job_id.clone(),
            Entry { info, pid: None, started: false, backlog: VecDeque::new(), backlog_bytes: 0, live, task: task.abort_handle() },
        );
        if inner.awake.is_none() {
            inner.awake = keep_awake();
        }
        Attached { backlog: Vec::new(), live: rx }
    }

    /// Recent output and a live feed of `job_id`, if `client` started it.
    pub fn attach(&self, job_id: &str, client: EndpointId) -> Result<Attached, StopError> {
        let inner = self.inner.lock().unwrap();
        let entry = inner.running.get(job_id).ok_or(StopError::NotFound)?;
        if entry.info.owner != client.to_string() {
            return Err(StopError::NotOwner);
        }
        let mut backlog = Vec::with_capacity(entry.backlog.len() + 1);
        if entry.started {
            backlog.push(AgentMsg::Started);
        }
        backlog.extend(entry.backlog.iter().cloned());
        Ok(Attached { backlog, live: entry.live.subscribe() })
    }

    /// Stops `job_id`. `client` is the device asking; `None` is the Mac's owner, who may
    /// stop any service.
    pub fn stop(&self, job_id: &str, client: Option<EndpointId>) -> Result<(), StopError> {
        let inner = self.inner.lock().unwrap();
        let entry = inner.running.get(job_id).ok_or(StopError::NotFound)?;
        if client.is_some_and(|c| entry.info.owner != c.to_string()) {
            return Err(StopError::NotOwner);
        }
        let task = entry.task.clone();
        drop(inner);
        // The task's guard removes the entry once the job is dropped.
        task.abort();
        Ok(())
    }

    /// Running services, oldest first; only `client`'s when given.
    pub fn list(&self, client: Option<EndpointId>) -> Vec<ServiceInfo> {
        let inner = self.inner.lock().unwrap();
        let mut list: Vec<ServiceInfo> = inner
            .running
            .values()
            .filter(|e| client.is_none_or(|c| e.info.owner == c.to_string()))
            .map(|e| e.info.clone())
            .collect();
        list.sort_by(|a, b| a.started_at.cmp(&b.started_at).then_with(|| a.job_id.cmp(&b.job_id)));
        list
    }

    /// [`Services::list`] with the ports each service listens on.
    pub async fn list_with_ports(&self, client: Option<EndpointId>) -> Vec<ServiceInfo> {
        let mut list = self.list(client);
        let pids: Vec<Option<u32>> = {
            let inner = self.inner.lock().unwrap();
            list.iter().map(|s| inner.running.get(&s.job_id).and_then(|e| e.pid)).collect()
        };
        let ports = tokio::task::spawn_blocking(move || pids.into_iter().map(|p| p.map(listening_ports).unwrap_or_default()).collect::<Vec<_>>())
            .await
            .unwrap_or_default();
        for (service, ports) in list.iter_mut().zip(ports) {
            service.ports = ports;
        }
        list
    }

    /// Ports `job_id` listens on now; empty if it is not running.
    pub async fn ports(&self, job_id: &str) -> Vec<u16> {
        let pid = self.inner.lock().unwrap().running.get(job_id).and_then(|e| e.pid);
        match pid {
            Some(pid) => tokio::task::spawn_blocking(move || listening_ports(pid)).await.unwrap_or_default(),
            None => Vec::new(),
        }
    }

    fn record(&self, job_id: &str, msg: AgentMsg) {
        let mut inner = self.inner.lock().unwrap();
        let Some(entry) = inner.running.get_mut(job_id) else { return };
        match &msg {
            AgentMsg::Started => entry.started = true,
            AgentMsg::Output { data, .. } => {
                entry.backlog_bytes += data.len();
                entry.backlog.push_back(msg.clone());
                while entry.backlog_bytes > BACKLOG_BYTES {
                    match entry.backlog.pop_front() {
                        Some(AgentMsg::Output { data, .. }) => entry.backlog_bytes -= data.len(),
                        _ => break,
                    }
                }
            }
            _ => {}
        }
        // Nobody attached is fine: the backlog has it.
        let _ = entry.live.send(msg);
    }
}

/// Removes a service once its job ends, however it ends, and reports the end.
struct EndGuard {
    services: Services,
    job_id: String,
    events: Option<EventSender>,
    end: JobEnd,
}

impl Drop for EndGuard {
    fn drop(&mut self) {
        let mut inner = self.services.inner.lock().unwrap();
        inner.running.remove(&self.job_id);
        if inner.running.is_empty() {
            // kill_on_drop ends caffeinate, so the Mac may sleep again.
            inner.awake = None;
        }
        drop(inner);
        if let Some(events) = &self.events {
            let _ = events.send(AgentEvent::JobFinished {
                job_id: std::mem::take(&mut self.job_id),
                end: self.end.clone(),
            });
        }
    }
}

/// TCP ports that any process in group `pgid` listens on, from `lsof`. Forwarding is
/// limited to these, so a paired device reaches its own service and nothing else on the
/// Mac's localhost.
fn listening_ports(pgid: u32) -> Vec<u16> {
    let out = std::process::Command::new("/usr/sbin/lsof")
        .args(["-nP", "-a", "-g", &pgid.to_string(), "-iTCP", "-sTCP:LISTEN", "-Fn"])
        .output();
    let Ok(out) = out else { return Vec::new() };
    let mut ports: Vec<u16> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.strip_prefix('n'))
        .filter_map(|addr| addr.rsplit(':').next()?.parse().ok())
        .collect();
    ports.sort_unstable();
    ports.dedup();
    ports
}

/// Keeps the Mac from sleeping while the returned process lives: `-i` blocks idle sleep,
/// `-s` system sleep on power, and `-w` ends it if the agent itself dies.
fn keep_awake() -> Option<tokio::process::Child> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    tokio::process::Command::new("/usr/bin/caffeinate")
        .args(["-is", "-w", &std::process::id().to_string()])
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| eprintln!("could not keep the Mac awake: {e}"))
        .ok()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn request(id: &str, argv: &[&str]) -> JobRequest {
        JobRequest {
            job_id: id.into(),
            argv: argv.iter().map(|s| s.to_string()).collect(),
            workdir: ".".into(),
            env: BTreeMap::new(),
            timeout_secs: Some(1),
            service: true,
            archive_bytes: 0,
        }
    }

    fn empty_archive() -> Vec<u8> {
        let mut b = tar::Builder::new(Vec::new());
        b.finish().unwrap();
        b.into_inner().unwrap()
    }

    fn device(n: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[n; 32]).public()
    }

    async fn wait_for(mut done: impl FnMut() -> bool) {
        for _ in 0..200 {
            if done() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("timed out waiting");
    }

    #[tokio::test]
    async fn service_outlives_timeout_keeps_backlog_and_stops_for_owner_only() {
        let root = tempfile::tempdir().unwrap();
        let runner = RunnerConfig { work_root: root.path().into(), default_timeout: Duration::from_secs(1), denied_ports: Vec::new() };
        let services = Services::default();
        let owner = device(1);
        let req = request("svc", &["sh", "-c", "echo up; while true; do sleep 0.1; done"]);
        drop(services.start(runner, req, empty_archive(), owner, None));

        // Past the 1 s timeout it still runs, and a late attach sees its earlier output.
        tokio::time::sleep(Duration::from_millis(1500)).await;
        assert_eq!(services.list(Some(owner)).len(), 1);
        assert!(services.list(Some(device(2))).is_empty());
        let attached = services.attach("svc", owner).unwrap();
        assert_eq!(attached.backlog.first(), Some(&AgentMsg::Started));
        assert!(attached.backlog.iter().any(|m| matches!(m, AgentMsg::Output { data, .. } if data == b"up\n")));

        assert_eq!(services.attach("svc", device(2)).err(), Some(StopError::NotOwner));
        assert_eq!(services.stop("svc", Some(device(2))), Err(StopError::NotOwner));
        services.stop("svc", Some(owner)).unwrap();
        wait_for(|| services.list(None).is_empty()).await;
        // The workspace goes with it.
        wait_for(|| std::fs::read_dir(root.path()).unwrap().next().is_none()).await;
        assert_eq!(services.stop("svc", None), Err(StopError::NotFound));
    }

    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn finds_the_ports_a_service_listens_on() {
        let root = tempfile::tempdir().unwrap();
        let runner = RunnerConfig { work_root: root.path().into(), default_timeout: Duration::from_secs(30), denied_ports: Vec::new() };
        let services = Services::default();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let script = format!("python3 -m http.server {port} -b 127.0.0.1");
        drop(services.start(runner, request("web", &["sh", "-c", &script]), empty_archive(), device(1), None));
        for _ in 0..100 {
            if services.ports("web").await == vec![port] {
                assert_eq!(services.list_with_ports(None).await[0].ports, vec![port]);
                services.stop("web", None).unwrap();
                return;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        panic!("port {port} never showed up");
    }

    #[tokio::test]
    async fn service_that_exits_is_removed() {
        let root = tempfile::tempdir().unwrap();
        let runner = RunnerConfig { work_root: root.path().into(), default_timeout: Duration::from_secs(30), denied_ports: Vec::new() };
        let services = Services::default();
        let mut attached = services.start(runner, request("quick", &["sh", "-c", "exit 3"]), empty_archive(), device(1), None);
        loop {
            match attached.live.recv().await.unwrap() {
                AgentMsg::Exited { code, .. } => break assert_eq!(code, Some(3)),
                _ => continue,
            }
        }
        wait_for(|| services.list(None).is_empty()).await;
    }
}

impl std::fmt::Debug for Services {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.list(None)).finish()
    }
}
