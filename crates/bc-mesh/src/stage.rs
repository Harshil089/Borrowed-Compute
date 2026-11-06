//! The device side of the mesh protocol: answering probes and running one stage of a
//! pipeline.
//!
//! A stage reads batches from the previous device (or the coordinator, for the first),
//! runs its layers and writes the result to the next device (or back to the coordinator,
//! for the last). Reading, computing and writing run as three tasks joined by short
//! queues, so a stage receives batch n+1 and sends batch n-1 while it computes batch n:
//! with several batches in flight every device in the pipeline works at once.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, recv_msg_max, send_msg, MESH_ALPN};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::{Endpoint, EndpointId};
use tokio::sync::{mpsc, oneshot};

use crate::msg::{Activation, Batch, MeshMsg, Meta, Payload, StageSetup};
use crate::probe;
use crate::ticket::Ticket;

/// Largest first frame, before the sender is known to be trusted.
const FIRST_FRAME: usize = 1 << 20;
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
/// How long a stage waits for the previous one to start feeding it.
const FEED_TIMEOUT: Duration = Duration::from_secs(60);
/// Batches queued between a stage's reader, compute and writer.
const QUEUE: usize = 4;

/// One device's part of a model. `forward` runs on a blocking thread.
pub trait Stage: Send {
    fn forward(&mut self, meta: &Meta, input: Activation) -> Result<Activation>;
}

/// Builds the stage a setup asks for (loading weights can take a while), or says why
/// this device cannot.
pub type StageFactory = Arc<dyn Fn(StageSetup) -> BoxFuture<Result<Box<dyn Stage>>> + Send + Sync>;
pub type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;
/// Whether a device may probe this one or set a stage up on it: the agent's paired list.
pub type Trust = Arc<dyn Fn(&EndpointId) -> bool + Send + Sync>;

struct Waiting {
    coordinator: EndpointId,
    feed: oneshot::Sender<(Connection, RecvStream)>,
}

/// Serves mesh connections for one endpoint.
#[derive(Clone)]
pub struct Host {
    endpoint: Endpoint,
    trust: Trust,
    factory: Option<StageFactory>,
    waiting: Arc<Mutex<HashMap<u64, Waiting>>>,
}

impl Host {
    /// Without a `factory` the device answers probes but turns pipelines away.
    pub fn new(endpoint: Endpoint, trust: Trust, factory: Option<StageFactory>) -> Self {
        Host { endpoint, trust, factory, waiting: Arc::default() }
    }

    pub async fn serve(&self, conn: Connection) -> Result<()> {
        let peer = conn.remote_id();
        // Anyone with this device's code can connect: never wait on them for long.
        let (mut send, mut recv, first) = tokio::time::timeout(FIRST_FRAME_TIMEOUT, async {
            let (send, mut recv) = conn.accept_bi().await?;
            let first = recv_msg_max::<MeshMsg>(&mut recv, FIRST_FRAME).await?;
            anyhow::Ok((send, recv, first))
        })
        .await
        .context("no first frame in time")??;
        // A feed comes from a device this one may never have paired with; its ticket,
        // signed by the trusted coordinator, is what lets it in.
        if let MeshMsg::Feed { ticket } = first {
            return self.feed(conn, send, peer, recv, ticket).await;
        }
        // A link probe from another lender, vouched for by a coordinator this device trusts.
        if let MeshMsg::PeerHello { coordinator, ticket } = first {
            let allowed = if (self.trust)(&coordinator) {
                ticket.check(ticket.session, &coordinator, &peer, &self.endpoint.id())
            } else {
                Err(anyhow::anyhow!("device {coordinator} is not paired with this device"))
            };
            if let Err(e) = allowed {
                return refuse(&conn, send, format!("{e:#}")).await;
            }
            send_msg(&mut send, &MeshMsg::Ready).await?;
            let next = recv_msg::<MeshMsg>(&mut recv).await?;
            return probe::answer(next, &mut send, &mut recv, self.factory.is_some()).await;
        }
        if !(self.trust)(&peer) {
            return refuse(&conn, send, format!("device {peer} is not paired with this device")).await;
        }
        match first {
            MeshMsg::Setup(setup) => self.run_stage(conn, send, recv, peer, setup).await,
            MeshMsg::ProbePeer { addr, ticket } => {
                let reply = match probe::measure_peer(&self.endpoint, peer, addr, ticket).await {
                    Ok(m) => m,
                    Err(e) => MeshMsg::Error { reason: format!("{e:#}") },
                };
                send_msg(&mut send, &reply).await?;
                send.finish()?;
                let _ = tokio::time::timeout(Duration::from_secs(10), conn.closed()).await;
                Ok(())
            }
            other => probe::answer(other, &mut send, &mut recv, self.factory.is_some()).await,
        }
    }

    async fn feed(&self, conn: Connection, mut send: SendStream, peer: EndpointId, recv: RecvStream, ticket: Ticket) -> Result<()> {
        let claimed = {
            let mut waiting = self.waiting.lock().unwrap();
            match waiting.get(&ticket.session) {
                None => Err(anyhow::anyhow!("no stage is waiting for session {}", ticket.session)),
                Some(w) => ticket
                    .check(ticket.session, &w.coordinator, &peer, &self.endpoint.id())
                    .map(|()| waiting.remove(&ticket.session).unwrap()),
            }
        };
        let w = match claimed {
            Ok(w) => w,
            Err(e) => return refuse(&conn, send, format!("{e:#}")).await,
        };
        send_msg(&mut send, &MeshMsg::Ready).await?;
        send.finish()?;
        let _ = w.feed.send((conn, recv));
        Ok(())
    }

    async fn run_stage(&self, conn: Connection, mut control: SendStream, setup_recv: RecvStream, coordinator: EndpointId, setup: StageSetup) -> Result<()> {
        let built = match &self.factory {
            None => Err(anyhow::anyhow!("this device has no inference engine")),
            Some(factory) => factory(setup.clone()).await,
        };
        let stage = match built {
            Ok(s) => s,
            Err(e) => return refuse(&conn, control, format!("{e:#}")).await,
        };
        // Registered before Ready, so the previous stage, set up after this one, can feed it.
        let feed = (setup.index > 0).then(|| {
            let (tx, rx) = oneshot::channel();
            self.waiting.lock().unwrap().insert(setup.session, Waiting { coordinator, feed: tx });
            rx
        });
        let session = setup.session;
        let result = async {
            let next = match &setup.next {
                Some((addr, ticket)) => Some(open_feed(&self.endpoint, addr.clone(), ticket.clone()).await?),
                None => None,
            };
            send_msg(&mut control, &MeshMsg::Ready).await?;
            let (input, feed_conn) = match feed {
                Some(rx) => {
                    let (c, r) = tokio::time::timeout(FEED_TIMEOUT, rx)
                        .await
                        .context("the previous stage never connected")?
                        .context("the previous stage never connected")?;
                    (r, Some(c))
                }
                None => (setup_recv, None),
            };
            // The last stage answers the coordinator on the stream it was set up on.
            let (out, next_conn) = match next {
                Some((c, s)) => (s, Some(c)),
                None => (control, None),
            };
            pump(stage, input, out).await?;
            drop((feed_conn, next_conn, conn));
            Ok(())
        }
        .await;
        self.waiting.lock().unwrap().remove(&session);
        result
    }
}

async fn refuse(conn: &Connection, mut send: SendStream, reason: String) -> Result<()> {
    send_msg(&mut send, &MeshMsg::Error { reason }).await?;
    send.finish()?;
    let _ = tokio::time::timeout(Duration::from_secs(10), conn.closed()).await;
    Ok(())
}

/// Connects to the next stage and presents the ticket.
async fn open_feed(endpoint: &Endpoint, addr: iroh::EndpointAddr, ticket: Ticket) -> Result<(Connection, SendStream)> {
    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(addr, MESH_ALPN))
        .await
        .context("the next stage did not answer")?
        .context("connecting to the next stage")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &MeshMsg::Feed { ticket }).await?;
    match recv_msg::<MeshMsg>(&mut recv).await? {
        MeshMsg::Ready => Ok((conn, send)),
        MeshMsg::Error { reason } => bail!("the next stage refused: {reason}"),
        other => bail!("unexpected reply from the next stage {other:?}"),
    }
}

/// Reads, computes and writes until `End`, which it passes on.
async fn pump(mut stage: Box<dyn Stage>, mut input: RecvStream, mut out: SendStream) -> Result<()> {
    let (in_tx, mut in_rx) = mpsc::channel::<Batch>(QUEUE);
    let (out_tx, mut out_rx) = mpsc::channel::<MeshMsg>(QUEUE);
    let reader = tokio::spawn(async move {
        loop {
            match recv_msg::<MeshMsg>(&mut input).await? {
                MeshMsg::Batch(b) => {
                    if in_tx.send(b).await.is_err() {
                        return Ok(());
                    }
                }
                MeshMsg::End => return Ok(()),
                MeshMsg::Error { reason } => bail!("upstream: {reason}"),
                other => bail!("expected a batch, got {other:?}"),
            }
        }
    });
    let writer = tokio::spawn(async move {
        while let Some(msg) = out_rx.recv().await {
            send_msg(&mut out, &msg).await?;
        }
        out.finish()?;
        // Wait until the receiver has everything before the connection may close.
        let _ = tokio::time::timeout(Duration::from_secs(30), out.stopped()).await;
        anyhow::Ok(())
    });
    let mut failure = None;
    while let Some(batch) = in_rx.recv().await {
        let seq = batch.seq;
        let mut meta = batch.meta;
        let (back, result, meta) = tokio::task::spawn_blocking(move || {
            let started = std::time::Instant::now();
            let r = batch.payload.decode().and_then(|a| stage.forward(&meta, a));
            meta.stage_ms.push(started.elapsed().as_secs_f32() * 1000.0);
            (stage, r, meta)
        })
        .await?;
        stage = back;
        match result {
            Ok(a) => {
                let msg = MeshMsg::Batch(Batch { seq, payload: Payload::encode(&a), meta });
                if out_tx.send(msg).await.is_err() {
                    break;
                }
            }
            Err(e) => {
                failure = Some(format!("batch {seq}: {e:#}"));
                break;
            }
        }
    }
    drop(in_rx);
    let read = reader.await?;
    let end = match (&failure, &read) {
        (Some(reason), _) => MeshMsg::Error { reason: reason.clone() },
        (None, Err(e)) => MeshMsg::Error { reason: format!("{e:#}") },
        (None, Ok(())) => MeshMsg::End,
    };
    let _ = out_tx.send(end).await;
    drop(out_tx);
    writer.await??;
    match failure {
        Some(reason) => bail!(reason),
        None => read,
    }
}
