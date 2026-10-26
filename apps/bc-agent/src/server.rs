//! Handles one connection from a client: handshake, upload, run, stream back.

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg};
use bc_proto::wire::{AgentMsg, ClientMsg, PROTOCOL_VERSION};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use tokio::sync::{broadcast, mpsc};

use crate::events::{AgentEvent, EventSender, JobEnd};
use crate::paired::PairedDevices;
use crate::runner::{run_job, RunnerConfig};
use crate::services::{Attached, Services, StopError};

/// Largest project archive accepted in Phase one, before chunked sync exists.
pub const MAX_ARCHIVE: u64 = 2 << 30;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub runner: RunnerConfig,
    pub paired: PairedDevices,
    /// Job activity for a UI. `None` when nobody is watching.
    pub events: Option<EventSender>,
    pub services: Services,
}

impl AgentConfig {
    pub fn new(runner: RunnerConfig, paired: PairedDevices) -> Self {
        AgentConfig { runner, paired, events: None, services: Services::default() }
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
        config.emit(AgentEvent::JobRejected { client: peer.to_string(), reason: reason.clone() });
        send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
        send.finish()?;
        conn.closed().await;
        bail!(reason);
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
            return reply_and_close(&conn, send, AgentMsg::Services(config.services.list(Some(peer)))).await;
        }
        other => bail!("expected Submit, Attach, Stop or ListServices, got {other:?}"),
    };
    if req.archive_bytes > MAX_ARCHIVE {
        let reason = format!("project is {} bytes, limit is {MAX_ARCHIVE}", req.archive_bytes);
        send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
        bail!(reason);
    }
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
                    match &msg {
                        AgentMsg::Exited { code, timed_out, usage } => {
                            end.end = JobEnd::Exited { code: *code, timed_out: *timed_out, usage: *usage };
                        }
                        AgentMsg::Failed { reason } => end.end = JobEnd::Failed { reason: reason.clone() },
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
