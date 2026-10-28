//! Handles one connection from a client: handshake, upload, run, stream back.

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg};
use bc_proto::wire::{AgentMsg, ClientMsg, PROTOCOL_VERSION};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use iroh::EndpointId;
use tokio::sync::{broadcast, mpsc};

use crate::events::{AgentEvent, EventSender, JobEnd};
use crate::paired::PairedDevices;
use crate::runner::{run_job, RunnerConfig};
use crate::services::{Attached, Services, StopError};

/// Largest project archive accepted in Phase one, before chunked sync exists.
pub const MAX_ARCHIVE: u64 = 2 << 30;
/// Runs (not services) one device may have going at once.
pub const MAX_JOBS_PER_DEVICE: usize = 4;
/// Services one device may keep running at once; each also keeps the Mac awake.
pub const MAX_SERVICES_PER_DEVICE: usize = 8;
/// Upload bytes held in memory at once across all devices, before they are unpacked.
pub const MAX_UPLOAD_BYTES: u64 = 4 << 30;

/// What paired devices are using now, so one device cannot exhaust the Mac.
#[derive(Debug, Clone, Default)]
pub struct Limits(Arc<Mutex<LimitsInner>>);

#[derive(Debug, Default)]
struct LimitsInner {
    jobs: HashMap<EndpointId, usize>,
    upload_bytes: u64,
}

/// A job's share of [`Limits`], given back when dropped.
pub struct Slot {
    limits: Limits,
    peer: EndpointId,
    counts_as_job: bool,
    upload_bytes: u64,
}

impl Limits {
    fn claim(&self, peer: EndpointId, upload_bytes: u64, service: bool, services: &Services) -> Result<Slot, String> {
        if service && services.list(Some(peer)).len() >= MAX_SERVICES_PER_DEVICE {
            return Err(format!(
                "this device already keeps {MAX_SERVICES_PER_DEVICE} services running on the Mac; stop one first"
            ));
        }
        let mut inner = self.0.lock().unwrap();
        let jobs = inner.jobs.get(&peer).copied().unwrap_or(0);
        if !service && jobs >= MAX_JOBS_PER_DEVICE {
            return Err(format!("this device already has {MAX_JOBS_PER_DEVICE} runs going on the Mac; wait for one to finish"));
        }
        if inner.upload_bytes + upload_bytes > MAX_UPLOAD_BYTES {
            return Err("the Mac is receiving too many large projects at once; try again in a moment".into());
        }
        inner.upload_bytes += upload_bytes;
        if !service {
            *inner.jobs.entry(peer).or_default() += 1;
        }
        Ok(Slot { limits: self.clone(), peer, counts_as_job: !service, upload_bytes })
    }
}

impl Slot {
    /// The upload is unpacked or handed on, so its bytes no longer count.
    fn uploaded(&mut self) {
        let mut inner = self.limits.0.lock().unwrap();
        inner.upload_bytes -= std::mem::take(&mut self.upload_bytes);
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.uploaded();
        if self.counts_as_job {
            let mut inner = self.limits.0.lock().unwrap();
            if let Some(n) = inner.jobs.get_mut(&self.peer) {
                *n -= 1;
                if *n == 0 {
                    inner.jobs.remove(&self.peer);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub runner: RunnerConfig,
    pub paired: PairedDevices,
    /// Job activity for a UI. `None` when nobody is watching.
    pub events: Option<EventSender>,
    pub services: Services,
    /// The built-in cloud's port on localhost, shared over [`bc_net::CLOUD_ALPN`] with any
    /// device (sign-in and pairing come before a device is paired). `None` shares nothing.
    pub cloud_port: Option<u16>,
    pub limits: Limits,
    /// Cloud connections served at once; any device may open them.
    pub cloud_slots: std::sync::Arc<tokio::sync::Semaphore>,
    pub rejections: Rejections,
}

/// How often one unpaired device's attempts are reported. Anyone who knows this Mac's
/// code can connect, so reporting every attempt would let them flood the job list and
/// the log files.
const REJECTION_QUIET: Duration = Duration::from_secs(10 * 60);
/// Devices remembered at once; past this, new unpaired devices are refused unreported.
const MAX_REJECTED_DEVICES: usize = 1000;

/// When each unpaired device was last reported.
#[derive(Debug, Clone, Default)]
pub struct Rejections(Arc<Mutex<HashMap<EndpointId, Instant>>>);

impl Rejections {
    /// True for a device's first attempt, and again once it has been quiet for a while.
    fn should_report(&self, peer: EndpointId) -> bool {
        let mut seen = self.0.lock().unwrap();
        let now = Instant::now();
        if seen.get(&peer).is_some_and(|t| now.duration_since(*t) < REJECTION_QUIET) {
            return false;
        }
        if seen.len() >= MAX_REJECTED_DEVICES {
            seen.retain(|_, t| now.duration_since(*t) < REJECTION_QUIET);
            if seen.len() >= MAX_REJECTED_DEVICES {
                return false;
            }
        }
        seen.insert(peer, now);
        true
    }
}

/// Cloud connections served at once, and requests in flight on each.
pub const MAX_CLOUD_CONNECTIONS: usize = 32;
pub const MAX_CLOUD_STREAMS: u32 = 16;

impl AgentConfig {
    pub fn new(runner: RunnerConfig, paired: PairedDevices) -> Self {
        AgentConfig {
            runner,
            paired,
            events: None,
            services: Services::default(),
            cloud_port: None,
            limits: Limits::default(),
            cloud_slots: std::sync::Arc::new(tokio::sync::Semaphore::new(MAX_CLOUD_CONNECTIONS)),
            rejections: Rejections::default(),
        }
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
    }
}

pub async fn serve_connection(config: &AgentConfig, conn: Connection) -> Result<()> {
    let peer = conn.remote_id();
    let (mut send, mut recv) = conn.accept_bi().await?;

    match recv_msg::<ClientMsg>(&mut recv).await? {
        ClientMsg::Hello { protocol } if protocol == PROTOCOL_VERSION => {}
        ClientMsg::Hello { protocol } => {
            let reason = format!("protocol {protocol} not supported, agent speaks {PROTOCOL_VERSION}");
            send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
            bail!(reason);
        }
        other => bail!("expected Hello, got {other:?}"),
    }
    send_msg(&mut send, &AgentMsg::Hello { protocol: PROTOCOL_VERSION }).await?;

    let first = recv_msg::<ClientMsg>(&mut recv).await?;
    // Checked after the handshake so an unpaired device gets a clear message.
    if !config.paired.contains(&peer) {
        let reason = format!("device {peer} is not paired with this Mac");
        // The device always gets the answer; the Mac's job list and logs hear of it once
        // in a while. Not an error, so the agent log stays quiet too.
        if config.rejections.should_report(peer) {
            config.emit(AgentEvent::JobRejected { client: peer.to_string(), reason: reason.clone() });
        }
        send_msg(&mut send, &AgentMsg::Rejected { reason }).await?;
        send.finish()?;
        conn.closed().await;
        return Ok(());
    }
    let req = match first {
        ClientMsg::Submit(req) => req,
        ClientMsg::Attach { job_id } => {
            return match config.services.attach(&job_id, peer) {
                Ok(attached) => {
                    send_msg(&mut send, &AgentMsg::Accepted { job_id: job_id.clone() }).await?;
                    stream_service(config, &conn, send, recv, &job_id, attached).await
                }
                Err(e) => reply_and_close(&conn, send, AgentMsg::Rejected { reason: service_error(&job_id, e) }).await,
            };
        }
        ClientMsg::Stop { job_id } => {
            let reply = match config.services.stop(&job_id, Some(peer)) {
                Ok(()) => AgentMsg::Accepted { job_id },
                Err(e) => AgentMsg::Rejected { reason: service_error(&job_id, e) },
            };
            return reply_and_close(&conn, send, reply).await;
        }
        ClientMsg::ListServices => {
            let list = config.services.list_with_ports(Some(peer)).await;
            return reply_and_close(&conn, send, AgentMsg::Services(list)).await;
        }
        ClientMsg::Forward { job_id } => {
            return match config.services.attach(&job_id, peer) {
                Ok(attached) => {
                    send_msg(&mut send, &AgentMsg::Accepted { job_id: job_id.clone() }).await?;
                    forward(config, &conn, attached, &job_id).await;
                    Ok(())
                }
                Err(e) => reply_and_close(&conn, send, AgentMsg::Rejected { reason: service_error(&job_id, e) }).await,
            };
        }
        other => bail!("expected Submit, Attach, Stop or ListServices, got {other:?}"),
    };
    if req.archive_bytes > MAX_ARCHIVE {
        let reason = format!("project is {} bytes, limit is {MAX_ARCHIVE}", req.archive_bytes);
        send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
        bail!(reason);
    }
    let mut slot = match config.limits.claim(peer, req.archive_bytes, req.service, &config.services) {
        Ok(slot) => slot,
        Err(reason) => return reply_and_close(&conn, send, AgentMsg::Rejected { reason }).await,
    };
    send_msg(&mut send, &AgentMsg::Accepted { job_id: req.job_id.clone() }).await?;

    let mut archive = Vec::with_capacity(req.archive_bytes as usize);
    loop {
        match recv_msg::<ClientMsg>(&mut recv).await? {
            ClientMsg::Upload { data } => {
                archive.extend_from_slice(&data);
                if archive.len() as u64 > req.archive_bytes {
                    bail!("upload is larger than the declared {} bytes", req.archive_bytes);
                }
            }
            ClientMsg::UploadDone => break,
            ClientMsg::Cancel => return Ok(()),
            other => bail!("expected Upload, got {other:?}"),
        }
    }
    if archive.len() as u64 != req.archive_bytes {
        bail!("upload is {} bytes, declared {}", archive.len(), req.archive_bytes);
    }

    let job_id = req.job_id.clone();
    config.emit(AgentEvent::JobStarted {
        job_id: job_id.clone(),
        client: peer.to_string(),
        argv: req.argv.clone(),
    });
    if req.service {
        // The service, not this connection, owns the job from here, and reports its end.
        let attached = config.services.start(config.runner.clone(), req, archive, peer, config.events.clone());
        drop(slot);
        return stream_service(config, &conn, send, recv, &job_id, attached).await;
    }
    // Reports how the job ended on every return path, including a lost connection.
    let mut end = JobEndGuard { config, job_id, end: JobEnd::Disconnected };

    let (tx, mut rx) = mpsc::channel(256);
    let runner = config.runner.clone();
    // Any early return (Cancel, lost connection) aborts the job; kill_on_drop stops it.
    let job = AbortOnDrop(tokio::spawn(async move {
        run_job(&runner, &req, archive, tx).await
    }));
    // Read from the client in its own task so a partly read frame is never dropped.
    let mut cancel = AbortOnDrop(tokio::spawn(async move {
        recv_msg::<ClientMsg>(&mut recv).await
    }));
    let mut client_done = false;
    loop {
        tokio::select! {
            msg = rx.recv() => match msg {
                Some(msg) => {
                    // The runner reports in only after unpacking, which frees the upload.
                    slot.uploaded();
                    match &msg {
                        AgentMsg::Exited { code, timed_out, usage } => {
                            end.end = JobEnd::Exited { code: *code, timed_out: *timed_out, usage: *usage };
                        }
                        AgentMsg::Failed { reason } => end.end = JobEnd::Failed { reason: reason.clone() },
                        AgentMsg::Output { data, .. } => {
                            config.emit(AgentEvent::JobOutput { job_id: end.job_id.clone(), data: data.clone() })
                        }
                        _ => {}
                    }
                    send_msg(&mut send, &msg).await.context("streaming to client")?
                }
                None => break,
            },
            msg = &mut cancel.0, if !client_done => {
                if matches!(msg, Ok(Ok(ClientMsg::Cancel)) | Ok(Err(_))) {
                    end.end = JobEnd::Cancelled;
                    return Ok(());
                }
                client_done = true;
            }
        }
    }
    drop(job);
    send.finish()?;
    conn.closed().await;
    Ok(())
}

/// Streams a service to an attached client until it ends or the client leaves. Leaving
/// (closing the connection) detaches; `Cancel` stops the service.
async fn stream_service(
    config: &AgentConfig,
    conn: &Connection,
    mut send: SendStream,
    mut recv: RecvStream,
    job_id: &str,
    attached: Attached,
) -> Result<()> {
    let Attached { backlog, mut live } = attached;
    for msg in backlog {
        send_msg(&mut send, &msg).await.context("streaming to client")?;
    }
    let mut client = AbortOnDrop(tokio::spawn(async move { recv_msg::<ClientMsg>(&mut recv).await }));
    loop {
        tokio::select! {
            msg = live.recv() => match msg {
                Ok(msg) => {
                    let end = matches!(msg, AgentMsg::Exited { .. } | AgentMsg::Failed { .. });
                    send_msg(&mut send, &msg).await.context("streaming to client")?;
                    if end {
                        break;
                    }
                }
                // A slow client misses some output rather than holding the service back.
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => break,
            },
            msg = &mut client.0 => {
                if matches!(msg, Ok(Ok(ClientMsg::Cancel))) {
                    let _ = config.services.stop(job_id, None);
                    // Keep streaming: the service's end follows.
                    client = AbortOnDrop(tokio::spawn(std::future::pending()));
                    continue;
                }
                // Gone or detached; the service keeps running.
                return Ok(());
            }
        }
    }
    send.finish()?;
    conn.closed().await;
    Ok(())
}

/// Carries the client's forwarded TCP connections to the service's ports on this Mac's
/// localhost, one stream each, until the service ends or the client closes the connection.
async fn forward(config: &AgentConfig, conn: &Connection, attached: Attached, job_id: &str) {
    let mut live = attached.live;
    let service_ended = async move {
        loop {
            if let Err(broadcast::error::RecvError::Closed) = live.recv().await {
                return;
            }
        }
    };
    tokio::pin!(service_ended);
    loop {
        tokio::select! {
            stream = conn.accept_bi() => match stream {
                Ok((send, recv)) => {
                    let services = config.services.clone();
                    let job_id = job_id.to_string();
                    tokio::spawn(async move {
                        if let Err(e) = forward_stream(&services, &job_id, send, recv).await {
                            eprintln!("forwarding for {job_id} ended: {e:#}");
                        }
                    });
                }
                Err(_) => return,
            },
            _ = &mut service_ended => {
                conn.close(0u32.into(), b"service ended");
                return;
            }
        }
    }
}

async fn forward_stream(services: &Services, job_id: &str, mut send: SendStream, mut recv: RecvStream) -> Result<()> {
    let port = match recv_msg::<ClientMsg>(&mut recv).await? {
        ClientMsg::Connect { port } => port,
        other => bail!("expected Connect, got {other:?}"),
    };
    // Only the service's own ports: anything else on localhost is not the client's.
    if !services.ports(job_id).await.contains(&port) {
        let reason = format!("service {job_id} is not listening on port {port}");
        send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
        send.finish()?;
        bail!(reason);
    }
    let tcp = match tokio::net::TcpStream::connect(("127.0.0.1", port)).await {
        Ok(tcp) => tcp,
        // Some servers listen on IPv6 localhost only.
        Err(_) => tokio::net::TcpStream::connect(("::1", port)).await?,
    };
    let _ = tcp.set_nodelay(true);
    send_msg(&mut send, &AgentMsg::Accepted { job_id: job_id.to_string() }).await?;
    let (mut tcp_read, mut tcp_write) = tcp.into_split();
    let upstream = async {
        tokio::io::copy(&mut recv, &mut tcp_write).await?;
        tokio::io::AsyncWriteExt::shutdown(&mut tcp_write).await
    };
    let downstream = async {
        tokio::io::copy(&mut tcp_read, &mut send).await?;
        send.finish().map_err(std::io::Error::other)
    };
    let (up, down) = tokio::join!(upstream, downstream);
    up.and(down)?;
    Ok(())
}

async fn reply_and_close(conn: &Connection, mut send: SendStream, msg: AgentMsg) -> Result<()> {
    send_msg(&mut send, &msg).await?;
    send.finish()?;
    conn.closed().await;
    Ok(())
}

fn service_error(job_id: &str, e: StopError) -> String {
    match e {
        StopError::NotFound => format!("no service {job_id} is running on this Mac"),
        StopError::NotOwner => format!("service {job_id} was started by another device"),
    }
}

struct JobEndGuard<'a> {
    config: &'a AgentConfig,
    job_id: String,
    end: JobEnd,
}

impl Drop for JobEndGuard<'_> {
    fn drop(&mut self) {
        self.config.emit(AgentEvent::JobFinished {
            job_id: std::mem::take(&mut self.job_id),
            end: self.end.clone(),
        });
    }
}

struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_runs_per_device_and_upload_bytes() {
        let limits = Limits::default();
        let services = Services::default();
        let a = iroh::SecretKey::from_bytes(&[1; 32]).public();
        let b = iroh::SecretKey::from_bytes(&[2; 32]).public();
        let slots: Vec<Slot> = (0..MAX_JOBS_PER_DEVICE).map(|_| limits.claim(a, 10, false, &services).unwrap()).collect();
        assert!(limits.claim(a, 10, false, &services).err().unwrap().contains("runs going"));
        assert!(limits.claim(b, 10, false, &services).is_ok(), "another device is not affected");
        drop(slots);
        let mut big = limits.claim(a, MAX_UPLOAD_BYTES, false, &services).unwrap();
        assert!(limits.claim(b, 1, false, &services).err().unwrap().contains("too many large projects"));
        big.uploaded();
        assert!(limits.claim(b, 1, false, &services).is_ok(), "unpacked bytes no longer count");
        drop(big);
        assert!(limits.0.lock().unwrap().jobs.is_empty());
    }

    #[test]
    fn reports_an_unpaired_device_once_in_a_while() {
        let rejections = Rejections::default();
        let a = iroh::SecretKey::from_bytes(&[1; 32]).public();
        let b = iroh::SecretKey::from_bytes(&[2; 32]).public();
        assert!(rejections.should_report(a));
        assert!(!rejections.should_report(a), "a second attempt right away stays quiet");
        assert!(rejections.should_report(b), "another device is reported");
        rejections.0.lock().unwrap().insert(a, Instant::now() - REJECTION_QUIET);
        assert!(rejections.should_report(a), "reported again after a quiet spell");
    }
}
