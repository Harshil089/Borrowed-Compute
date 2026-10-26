//! Runs one job on a Mac and streams its output to local writers.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_net::{bind, recv_msg, send_msg, EndpointAddr, NetConfig, JOB_ALPN};
use bc_proto::wire::{AgentMsg, ClientMsg, JobRequest, JobUsage, OutputStream, ServiceInfo, PROTOCOL_VERSION};
use bc_proto::JobSpec;
use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::{Endpoint, SecretKey};

use crate::autofix;
use crate::route::RouteMemory;

const UPLOAD_CHUNK: usize = 1 << 20;

/// How long the Mac has to answer before a direct attempt counts as stuck.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// An upload chunk that makes no progress for this long means the path is dead, even
/// though QUIC itself would wait longer before timing out.
const UPLOAD_STALL: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, PartialEq)]
pub enum Exit {
    Code(i32),
    TimedOut,
    /// Ended by a signal, or the agent could not report a code.
    Killed,
    Failed(String),
    Rejected(String),
    Cancelled,
    /// A service this client stopped watching; it keeps running on the Mac.
    Detached,
}

impl Exit {
    /// Exit status for the local `bc` process. 124 matches the `timeout` command.
    pub fn status(&self) -> i32 {
        match self {
            Exit::Code(c) => *c,
            Exit::TimedOut => 124,
            Exit::Killed => 137,
            Exit::Cancelled => 130,
            Exit::Detached => 0,
            Exit::Failed(_) | Exit::Rejected(_) => 1,
        }
    }
}

/// Time spent in each phase, measured from the start of the run.
#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub connect: Duration,
    pub upload: Duration,
    pub start: Option<Duration>,
    pub first_log: Option<Duration>,
    pub total: Duration,
    pub via_relay: bool,
    pub archive_bytes: u64,
}

impl Timings {
    /// One JSON object per line, for the local timing log.
    pub fn to_json(&self, job_id: &str, exit: &Exit, usage: Option<&JobUsage>) -> String {
        let ms = |d: Duration| d.as_millis().to_string();
        let opt = |d: Option<Duration>| d.map(ms).unwrap_or_else(|| "null".into());
        let usage = usage
            .map(|u| serde_json::to_string(u).unwrap_or_else(|_| "null".into()))
            .unwrap_or_else(|| "null".into());
        format!(
            "{{\"job\":\"{job_id}\",\"connect_ms\":{},\"upload_ms\":{},\"start_ms\":{},\"first_log_ms\":{},\"total_ms\":{},\"via_relay\":{},\"archive_bytes\":{},\"exit\":{},\"usage\":{usage}}}",
            ms(self.connect),
            ms(self.upload),
            opt(self.start),
            opt(self.first_log),
            ms(self.total),
            self.via_relay,
            self.archive_bytes,
            exit.status(),
        )
    }
}

pub struct RunOutcome {
    pub job_id: String,
    pub exit: Exit,
    pub timings: Timings,
    /// What the job used on the Mac; `None` if it never ran or the agent is too old to say.
    pub usage: Option<JobUsage>,
}

pub fn new_job_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    format!("{nanos:x}-{:x}", std::process::id())
}

/// Sends the job and writes its output as it arrives. Resolves when the job ends or
/// `cancel` fires, in which case the agent is told to stop.
pub async fn run_remote(
    endpoint: &Endpoint,
    mac: EndpointAddr,
    spec: &JobSpec,
    archive: Vec<u8>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
) -> Result<RunOutcome> {
    let mut started = false;
    run_attempt(endpoint, mac, spec, archive, stdout, stderr, cancel, &mut started).await
}

/// Runs the job, trying a direct path first. Some networks (a VPN on the Mac, for one)
/// let a direct connection open and then drop it mid-upload; when that happens before
/// the job starts on the Mac, the run is retried once through the relay, which works
/// wherever HTTPS does. `memory` records the failure so the next runs to that Mac go
/// through the relay first and skip the wait. `notice` explains each switch to the person
/// watching.
///
/// With `project` (the folder being sent), the run also fixes what a fresh workspace
/// lacks: [`autofix::preflight`] installs dependencies or builds first, and after a
/// failed run [`autofix::after_failure`] reads the output and, when it recognizes a
/// missing module, command or build file, runs once more with the fix.
#[allow(clippy::too_many_arguments)]
pub async fn run_on_mac(
    key: &SecretKey,
    net: NetConfig,
    mac: EndpointAddr,
    spec: &JobSpec,
    archive: Vec<u8>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
    project: Option<&std::path::Path>,
    memory: &RouteMemory,
    notice: impl Fn(&str),
) -> Result<RunOutcome> {
    tokio::pin!(cancel);
    let mut spec = spec.clone();
    if let Some(fix) = project.and_then(|p| autofix::preflight(p, &spec.job.argv)) {
        notice(&format!("Auto-fix: {}. Running: {}", fix.reason, fix.argv.join(" ")));
        spec.job.argv = fix.argv;
    }
    // The end of the output, to recognize what a failed run was missing.
    let tail = std::sync::Mutex::new(Vec::new());
    let outcome = {
        let mut out = Tail { inner: &mut *stdout, tail: &tail };
        let mut err = Tail { inner: &mut *stderr, tail: &tail };
        run_routed(key, net.clone(), mac.clone(), &spec, archive.clone(), &mut out, &mut err, cancel.as_mut(), memory, &notice)
            .await?
    };
    let failed = matches!(outcome.exit, Exit::Code(c) if c != 0);
    if let (true, Some(project)) = (failed, project) {
        let output = String::from_utf8_lossy(&tail.lock().unwrap()).into_owned();
        if let Some(fix) = autofix::after_failure(project, &spec.job.argv, &output) {
            notice(&format!("Auto-fix: {}. Running again: {}", fix.reason, fix.argv.join(" ")));
            spec.job.argv = fix.argv;
            return run_routed(key, net, mac, &spec, archive, stdout, stderr, cancel.as_mut(), memory, &notice).await;
        }
    }
    Ok(outcome)
}

/// Keeps the last [`TAIL_BYTES`] written, in memory only, while passing everything on.
const TAIL_BYTES: usize = 64 * 1024;

struct Tail<'a, W: Write> {
    inner: &'a mut W,
    tail: &'a std::sync::Mutex<Vec<u8>>,
}

impl<W: Write> Write for Tail<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        let mut tail = self.tail.lock().unwrap();
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > TAIL_BYTES {
            let excess = tail.len() - TAIL_BYTES;
            tail.drain(..excess);
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Runs the job, choosing direct or relay as described on [`run_on_mac`].
#[allow(clippy::too_many_arguments)]
async fn run_routed(
    key: &SecretKey,
    net: NetConfig,
    mac: EndpointAddr,
    spec: &JobSpec,
    archive: Vec<u8>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
    memory: &RouteMemory,
    notice: impl Fn(&str),
) -> Result<RunOutcome> {
    tokio::pin!(cancel);
    if net.relay_only {
        let endpoint = bind(key.clone(), net).await?;
        let mut started = false;
        let outcome = run_attempt(&endpoint, mac, spec, archive, stdout, stderr, cancel.as_mut(), &mut started).await;
        endpoint.close().await;
        return outcome;
    }

    let relay_first = memory.relay_first(&mac);
    if relay_first {
        notice("Using the relay: the direct path to this Mac failed within the last hour.");
    }
    // Each attempt is "relay only" or not; the second is the fallback for the first.
    let order = if relay_first { [true, false] } else { [false, true] };
    let mut last_error = None;
    for (i, relay_only) in order.into_iter().enumerate() {
        let endpoint = bind(key.clone(), NetConfig { relay_only, ..net.clone() }).await?;
        let mut started = false;
        let attempt = run_attempt(
            &endpoint, mac.clone(), spec, archive.clone(), stdout, stderr, cancel.as_mut(), &mut started,
        )
        .await;
        endpoint.close().await;
        match attempt {
            Ok(outcome) => {
                if !relay_only && !outcome.timings.via_relay {
                    memory.direct_worked(&mac);
                }
                return Ok(outcome);
            }
            // A job that started on the Mac must not run twice.
            Err(e) if started || i == 1 => return Err(e),
            Err(e) if relay_only => notice(&format!("The relay failed ({e:#}). Trying a direct connection…")),
            Err(e) => {
                memory.direct_failed(&mac);
                notice(&format!("Direct connection failed ({e:#}). Retrying through the relay…"));
                last_error = Some(e);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no attempt ran")))
}

/// One attempt over `endpoint`. Sets `started` once the job is running on the Mac, after
/// which a retry would run it twice.
#[allow(clippy::too_many_arguments)]
async fn run_attempt(
    endpoint: &Endpoint,
    mac: EndpointAddr,
    spec: &JobSpec,
    archive: Vec<u8>,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
    started: &mut bool,
) -> Result<RunOutcome> {
    let t0 = Instant::now();
    let job_id = new_job_id();
    let mut timings = Timings {
        archive_bytes: archive.len() as u64,
        ..Default::default()
    };

    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(mac, JOB_ALPN))
        .await
        .map_err(|_| anyhow::anyhow!("the Mac did not answer within {}s", CONNECT_TIMEOUT.as_secs()))?
        .context("connecting to the Mac")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &ClientMsg::Hello { protocol: PROTOCOL_VERSION }).await?;
    match recv_msg::<AgentMsg>(&mut recv).await? {
        AgentMsg::Hello { protocol } if protocol == PROTOCOL_VERSION => {}
        AgentMsg::Rejected { reason } => return Ok(done(job_id, Exit::Rejected(reason), timings, t0, None)),
        other => bail!("unexpected handshake reply {other:?}"),
    }
    timings.connect = t0.elapsed();

    let req = JobRequest::from_spec(job_id.clone(), spec, archive.len() as u64);
    send_msg(&mut send, &ClientMsg::Submit(req)).await?;
    match recv_msg::<AgentMsg>(&mut recv).await? {
        AgentMsg::Accepted { .. } => {}
        AgentMsg::Rejected { reason } => return Ok(done(job_id, Exit::Rejected(reason), timings, t0, None)),
        other => bail!("unexpected reply to Submit {other:?}"),
    }
    for chunk in archive.chunks(UPLOAD_CHUNK) {
        tokio::time::timeout(UPLOAD_STALL, send_msg(&mut send, &ClientMsg::Upload { data: chunk.to_vec() }))
            .await
            .map_err(|_| anyhow::anyhow!("upload stalled for {}s", UPLOAD_STALL.as_secs()))??;
    }
    send_msg(&mut send, &ClientMsg::UploadDone).await?;
    timings.upload = t0.elapsed();

    let (exit, usage) =
        stream_job(&mut send, &mut recv, spec.job.service, &mut timings, t0, stdout, stderr, cancel, started).await?;

    timings.via_relay = bc_net::via_relay(&conn);
    let _ = send.finish();
    conn.close(0u32.into(), b"done");
    Ok(done(job_id, exit, timings, t0, usage))
}

/// Writes a job's output until it ends or `cancel` fires. Cancelling a service only
/// detaches from it (the connection closes, the service keeps running); cancelling a job
/// stops it.
#[allow(clippy::too_many_arguments)]
async fn stream_job(
    send: &mut SendStream,
    recv: &mut RecvStream,
    service: bool,
    timings: &mut Timings,
    t0: Instant,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
    started: &mut bool,
) -> Result<(Exit, Option<JobUsage>)> {
    tokio::pin!(cancel);
    let mut usage = None;
    let exit = loop {
        let msg = tokio::select! {
            msg = recv_msg::<AgentMsg>(recv) => msg.context("connection to the Mac lost")?,
            _ = &mut cancel => {
                if service {
                    break Exit::Detached;
                }
                let _ = send_msg(send, &ClientMsg::Cancel).await;
                break Exit::Cancelled;
            }
        };
        match msg {
            AgentMsg::Started => {
                *started = true;
                timings.start = Some(t0.elapsed());
            }
            AgentMsg::Output { stream, data, .. } => {
                timings.first_log.get_or_insert_with(|| t0.elapsed());
                let out: &mut dyn Write = match stream {
                    OutputStream::Stdout => stdout,
                    OutputStream::Stderr => stderr,
                };
                out.write_all(&data)?;
                out.flush()?;
            }
            AgentMsg::Exited { code, timed_out, usage: used } => {
                usage = used;
                break match (timed_out, code) {
                    (true, _) => Exit::TimedOut,
                    (false, Some(c)) => Exit::Code(c),
                    (false, None) => Exit::Killed,
                };
            }
            AgentMsg::Failed { reason } => break Exit::Failed(reason),
            other => bail!("unexpected message during job {other:?}"),
        }
    };
    Ok((exit, usage))
}

/// A connection to the Mac that has passed the handshake, for the service commands.
pub(crate) struct Session {
    /// Owned when the session bound its own endpoint, closed with it.
    endpoint: Option<Endpoint>,
    pub(crate) conn: Connection,
    pub(crate) send: SendStream,
    pub(crate) recv: RecvStream,
}

/// An endpoint for talking to `mac`: through the relay when `memory` says the direct path
/// to it fails.
pub(crate) async fn endpoint_for(key: &SecretKey, net: NetConfig, mac: &EndpointAddr, memory: &RouteMemory) -> Result<Endpoint> {
    let relay_only = net.relay_only || memory.relay_first(mac);
    bind(key.clone(), NetConfig { relay_only, ..net }).await
}

impl Session {
    async fn open(key: &SecretKey, net: NetConfig, mac: EndpointAddr, memory: &RouteMemory) -> Result<Self> {
        let endpoint = endpoint_for(key, net, &mac, memory).await?;
        let mut session = Session::connect(&endpoint, mac).await?;
        session.endpoint = Some(endpoint);
        Ok(session)
    }

    pub(crate) async fn connect(endpoint: &Endpoint, mac: EndpointAddr) -> Result<Self> {
        let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(mac, JOB_ALPN))
            .await
            .map_err(|_| anyhow::anyhow!("the Mac did not answer within {}s", CONNECT_TIMEOUT.as_secs()))?
            .context("connecting to the Mac")?;
        let (mut send, mut recv) = conn.open_bi().await?;
        send_msg(&mut send, &ClientMsg::Hello { protocol: PROTOCOL_VERSION }).await?;
        match recv_msg::<AgentMsg>(&mut recv).await? {
            AgentMsg::Hello { protocol } if protocol == PROTOCOL_VERSION => {}
            AgentMsg::Rejected { reason } => bail!(reason),
            other => bail!("unexpected handshake reply {other:?}"),
        }
        Ok(Session { endpoint: None, conn, send, recv })
    }

    /// Sends `msg` and returns the one reply. A pairing refusal comes back as an error.
    pub(crate) async fn ask(&mut self, msg: &ClientMsg) -> Result<AgentMsg> {
        send_msg(&mut self.send, msg).await?;
        match recv_msg::<AgentMsg>(&mut self.recv).await? {
            AgentMsg::Rejected { reason } => bail!(reason),
            reply => Ok(reply),
        }
    }

    pub(crate) async fn close(self) {
        let _ = { self.send }.finish();
        self.conn.close(0u32.into(), b"done");
        if let Some(endpoint) = self.endpoint {
            endpoint.close().await;
        }
    }
}

/// The services this device started on `mac`.
pub async fn list_services(key: &SecretKey, net: NetConfig, mac: EndpointAddr, memory: &RouteMemory) -> Result<Vec<ServiceInfo>> {
    let mut session = Session::open(key, net, mac, memory).await?;
    let reply = session.ask(&ClientMsg::ListServices).await;
    session.close().await;
    match reply? {
        AgentMsg::Services(list) => Ok(list),
        other => bail!("unexpected reply to ListServices {other:?}"),
    }
}

/// Stops a service this device started; its process group and workspace go with it.
pub async fn stop_service(key: &SecretKey, net: NetConfig, mac: EndpointAddr, memory: &RouteMemory, job_id: &str) -> Result<()> {
    let mut session = Session::open(key, net, mac, memory).await?;
    let reply = session.ask(&ClientMsg::Stop { job_id: job_id.into() }).await;
    session.close().await;
    match reply? {
        AgentMsg::Accepted { .. } => Ok(()),
        other => bail!("unexpected reply to Stop {other:?}"),
    }
}

/// Watches a running service: its recent output first, then new output until it ends or
/// `cancel` fires, which detaches and leaves it running.
#[allow(clippy::too_many_arguments)]
pub async fn attach_service(
    key: &SecretKey,
    net: NetConfig,
    mac: EndpointAddr,
    memory: &RouteMemory,
    job_id: &str,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    cancel: impl std::future::Future<Output = ()>,
) -> Result<RunOutcome> {
    let t0 = Instant::now();
    let mut timings = Timings::default();
    let mut session = Session::open(key, net, mac, memory).await?;
    timings.connect = t0.elapsed();
    let streamed = match session.ask(&ClientMsg::Attach { job_id: job_id.into() }).await {
        Ok(AgentMsg::Accepted { .. }) => {
            let mut started = false;
            let Session { send, recv, .. } = &mut session;
            stream_job(send, recv, true, &mut timings, t0, stdout, stderr, cancel, &mut started).await
        }
        Ok(other) => Err(anyhow::anyhow!("unexpected reply to Attach {other:?}")),
        Err(e) => Err(e),
    };
    timings.via_relay = bc_net::via_relay(&session.conn);
    session.close().await;
    let (exit, usage) = streamed?;
    Ok(done(job_id.to_string(), exit, timings, t0, usage))
}

fn done(job_id: String, exit: Exit, mut timings: Timings, t0: Instant, usage: Option<JobUsage>) -> RunOutcome {
    timings.total = t0.elapsed();
    RunOutcome { job_id, exit, timings, usage }
}

