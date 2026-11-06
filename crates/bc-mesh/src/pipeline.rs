//! The coordinator's side of a pipeline: set the stages up, then stream batches in at
//! one end and read results at the other.
//!
//! The coordinator talks to the first stage (input) and the last (output) only. Hidden
//! states move stage to stage on their own connections, so each crosses one link once.
//! Send several batches before reading results: that is what keeps every stage busy.

use std::ops::Range;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg, MESH_ALPN};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::{Endpoint, EndpointAddr};

use crate::msg::{Activation, Batch, MeshMsg, Meta, Payload, StageSetup};
use crate::ticket::Ticket;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

pub struct Target {
    pub addr: EndpointAddr,
    pub layers: Range<u32>,
}

/// The first stage's end: send batches, then `finish`.
pub struct Input {
    send: SendStream,
    _controls: Vec<SendStream>,
    _conns: Vec<Connection>,
}

/// The last stage's end: results in the order sent.
pub struct Output {
    recv: RecvStream,
    _conns: Vec<Connection>,
}

/// Sets every stage up, last first, so each one is waiting when the one before it
/// connects.
pub async fn open(endpoint: &Endpoint, model: &str, targets: &[Target]) -> Result<(Input, Output)> {
    if targets.is_empty() {
        bail!("a pipeline needs at least one stage");
    }
    let session: u64 = rand::random();
    let count = targets.len();
    let mut opened = Vec::with_capacity(count);
    for i in (0..count).rev() {
        let target = &targets[i];
        let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(target.addr.clone(), MESH_ALPN))
            .await
            .with_context(|| format!("stage {i} did not answer"))?
            .with_context(|| format!("connecting to stage {i}"))?;
        let (mut send, mut recv) = conn.open_bi().await?;
        let next = targets.get(i + 1).map(|n| {
            (n.addr.clone(), Ticket::sign(endpoint.secret_key(), session, target.addr.id, n.addr.id))
        });
        let setup = StageSetup {
            session,
            index: i as u16,
            count: count as u16,
            model: model.into(),
            layers: target.layers.clone(),
            next,
        };
        send_msg(&mut send, &MeshMsg::Setup(setup)).await?;
        match recv_msg::<MeshMsg>(&mut recv).await? {
            MeshMsg::Ready => {}
            MeshMsg::Error { reason } => bail!("stage {i}: {reason}"),
            other => bail!("stage {i}: unexpected reply {other:?}"),
        }
        opened.push((conn, send, recv));
    }
    opened.reverse();
    let conns: Vec<Connection> = opened.iter().map(|(c, _, _)| c.clone()).collect();
    let mut first = None;
    let mut last = None;
    let mut controls = Vec::new();
    for (i, (_, send, recv)) in opened.into_iter().enumerate() {
        if i == 0 {
            first = Some(send);
        } else {
            controls.push(send);
        }
        if i == count - 1 {
            last = Some(recv);
        }
    }
    Ok((
        Input { send: first.unwrap(), _controls: controls, _conns: conns.clone() },
        Output { recv: last.unwrap(), _conns: conns },
    ))
}

impl Input {
    pub async fn send(&mut self, seq: u64, meta: Meta, activation: &Activation) -> Result<()> {
        let payload = Payload::encode(activation);
        send_msg(&mut self.send, &MeshMsg::Batch(Batch { seq, payload, meta })).await
    }

    /// No more batches. Results already in flight still arrive.
    pub async fn finish(mut self) -> Result<()> {
        send_msg(&mut self.send, &MeshMsg::End).await?;
        self.send.finish()?;
        let _ = tokio::time::timeout(Duration::from_secs(30), self.send.stopped()).await;
        Ok(())
    }
}

impl Output {
    /// The next result, or `None` once the pipeline has finished.
    pub async fn recv(&mut self) -> Result<Option<(u64, Meta, Activation)>> {
        match recv_msg::<MeshMsg>(&mut self.recv).await? {
            MeshMsg::Batch(b) => Ok(Some((b.seq, b.meta, b.payload.decode()?))),
            MeshMsg::End => Ok(None),
            MeshMsg::Error { reason } => bail!("the pipeline failed: {reason}"),
            other => bail!("unexpected frame {other:?}"),
        }
    }
}
