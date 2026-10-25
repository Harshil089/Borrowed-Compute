//! Runs one job on a Mac and streams its output to local writers.

use std::io::Write;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_net::{bind, recv_msg, send_msg, EndpointAddr, NetConfig, JOB_ALPN};
use bc_proto::wire::{AgentMsg, ClientMsg, JobRequest, JobUsage, OutputStream, PROTOCOL_VERSION};
use bc_proto::JobSpec;
use iroh::{Endpoint, SecretKey};

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
}

impl Exit {
    /// Exit status for the local `bc` process. 124 matches the `timeout` command.
    pub fn status(&self) -> i32 {
        match self {
            Exit::Code(c) => *c,
            Exit::TimedOut => 124,
            Exit::Killed => 137,
            Exit::Cancelled => 130,
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

    tokio::pin!(cancel);
    let mut usage = None;
    let exit = loop {
        let msg = tokio::select! {
            msg = recv_msg::<AgentMsg>(&mut recv) => msg.context("connection to the Mac lost")?,
            _ = &mut cancel => {
                let _ = send_msg(&mut send, &ClientMsg::Cancel).await;
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

    timings.via_relay = bc_net::via_relay(&conn);
    let _ = send.finish();
    conn.close(0u32.into(), b"done");
    Ok(done(job_id, exit, timings, t0, usage))
}

fn done(job_id: String, exit: Exit, mut timings: Timings, t0: Instant, usage: Option<JobUsage>) -> RunOutcome {
    timings.total = t0.elapsed();
    RunOutcome { job_id, exit, timings, usage }
}

