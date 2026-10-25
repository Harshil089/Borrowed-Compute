//! Handles one connection from a client: handshake, upload, run, stream back.

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg};
use bc_proto::wire::{AgentMsg, ClientMsg, PROTOCOL_VERSION};
use iroh::endpoint::Connection;
use tokio::sync::mpsc;

use crate::paired::PairedDevices;
use crate::runner::{run_job, RunnerConfig};

/// Largest project archive accepted in Phase one, before chunked sync exists.
pub const MAX_ARCHIVE: u64 = 2 << 30;

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub runner: RunnerConfig,
    pub paired: PairedDevices,
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

    let req = match recv_msg::<ClientMsg>(&mut recv).await? {
        ClientMsg::Submit(req) => req,
        other => bail!("expected Submit, got {other:?}"),
    };
    // Checked after the handshake so an unpaired device gets a clear message.
    if !config.paired.contains(&peer) {
        let reason = format!("device {peer} is not paired with this Mac");
        send_msg(&mut send, &AgentMsg::Rejected { reason: reason.clone() }).await?;
        send.finish()?;
        conn.closed().await;
        bail!(reason);
    }
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
                Some(msg) => send_msg(&mut send, &msg).await.context("streaming to client")?,
                None => break,
            },
            msg = &mut cancel.0, if !client_done => {
                if matches!(msg, Ok(Ok(ClientMsg::Cancel)) | Ok(Err(_))) {
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

struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}
