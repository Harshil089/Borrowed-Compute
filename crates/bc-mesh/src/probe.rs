//! Measuring a link and the device at its other end, which the planner needs: round trip
//! time and upload rate for the hops, memory and memory bandwidth for the layers.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg, MESH_ALPN};
use iroh::{Endpoint, EndpointAddr};
use serde::{Deserialize, Serialize};

use crate::msg::{DeviceReport, GpuReport, MeshMsg};

const PINGS: usize = 5;
const BULK_BYTES: usize = 2 << 20;
const BULK_CHUNK: usize = 64 << 10;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkReport {
    pub rtt_ms: f64,
    /// Upload rate to the device, bytes a second.
    pub up_bytes_s: f64,
    pub via_relay: bool,
    pub device: DeviceReport,
}

/// Probes one device. Several run at once from the same endpoint, one per device.
pub async fn probe(endpoint: &Endpoint, addr: impl Into<EndpointAddr>) -> Result<LinkReport> {
    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(addr, MESH_ALPN))
        .await
        .context("the device did not answer")?
        .context("connecting")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &MeshMsg::Report).await?;
    let device = match recv_msg::<MeshMsg>(&mut recv).await? {
        MeshMsg::Device(d) => d,
        MeshMsg::Error { reason } => bail!(reason),
        other => bail!("unexpected reply {other:?}"),
    };

    let (rtt_ms, up_bytes_s) = measure(&mut send, &mut recv, BULK_BYTES).await?;
    let via_relay = bc_net::via_relay(&conn);
    send.finish()?;
    conn.close(0u32.into(), b"done");
    Ok(LinkReport { rtt_ms, up_bytes_s, via_relay, device })
}

/// Median round trip of a few pings, then the upload rate for `bulk` bytes, on a stream
/// whose other end runs [`answer`].
pub(crate) async fn measure(send: &mut iroh::endpoint::SendStream, recv: &mut iroh::endpoint::RecvStream, bulk: usize) -> Result<(f64, f64)> {
    let mut rtts = Vec::with_capacity(PINGS);
    for nonce in 0..PINGS as u64 {
        let t = Instant::now();
        send_msg(send, &MeshMsg::Ping { nonce }).await?;
        match recv_msg::<MeshMsg>(recv).await? {
            MeshMsg::Pong { nonce: n } if n == nonce => rtts.push(t.elapsed().as_secs_f64() * 1000.0),
            other => bail!("unexpected reply {other:?}"),
        }
    }
    rtts.sort_by(f64::total_cmp);
    let chunk = vec![0x5a; BULK_CHUNK];
    for _ in 0..bulk / BULK_CHUNK {
        send_msg(send, &MeshMsg::Bulk { data: chunk.clone() }).await?;
    }
    send_msg(send, &MeshMsg::BulkDone).await?;
    let up = match recv_msg::<MeshMsg>(recv).await? {
        MeshMsg::BulkAck { bytes, micros } => bytes as f64 / (micros.max(1) as f64 / 1e6),
        other => bail!("unexpected reply {other:?}"),
    };
    Ok((rtts[PINGS / 2], up))
}

/// Bytes sent to measure a link between two devices: less than a coordinator's probe,
/// since every pair is measured.
const PEER_BULK: usize = 512 << 10;

/// Asks device `a` to measure its link to device `b`, with a ticket that lets `b` accept
/// `a`. Both must be paired with this device; they need not be with each other.
pub async fn probe_link(endpoint: &Endpoint, a: EndpointAddr, b: EndpointAddr) -> Result<(f64, f64, bool)> {
    let ticket = crate::ticket::Ticket::sign(endpoint.secret_key(), rand::random(), a.id, b.id);
    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(a, MESH_ALPN))
        .await
        .context("the device did not answer")?
        .context("connecting")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &MeshMsg::ProbePeer { addr: b, ticket }).await?;
    let reply = tokio::time::timeout(Duration::from_secs(60), recv_msg::<MeshMsg>(&mut recv)).await.context("no answer in time")??;
    conn.close(0u32.into(), b"done");
    match reply {
        MeshMsg::PeerLink { rtt_ms, up_bytes_s, via_relay } => Ok((rtt_ms, up_bytes_s, via_relay)),
        MeshMsg::Error { reason } => bail!(reason),
        other => bail!("unexpected reply {other:?}"),
    }
}

/// On device `a`: connects to `b` with the coordinator's ticket and measures the link.
pub(crate) async fn measure_peer(endpoint: &Endpoint, coordinator: iroh::EndpointId, b: EndpointAddr, ticket: crate::ticket::Ticket) -> Result<MeshMsg> {
    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(b, MESH_ALPN))
        .await
        .context("the other device did not answer")?
        .context("connecting to the other device")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &MeshMsg::PeerHello { coordinator, ticket }).await?;
    match recv_msg::<MeshMsg>(&mut recv).await? {
        MeshMsg::Ready => {}
        MeshMsg::Error { reason } => bail!("the other device refused: {reason}"),
        other => bail!("unexpected reply {other:?}"),
    }
    let (rtt_ms, up_bytes_s) = measure(&mut send, &mut recv, PEER_BULK).await?;
    let via_relay = bc_net::via_relay(&conn);
    send.finish()?;
    conn.close(0u32.into(), b"done");
    Ok(MeshMsg::PeerLink { rtt_ms, up_bytes_s, via_relay })
}

/// This device, as a probe reports it.
pub async fn report(engine: bool) -> DeviceReport {
    let mem_bw = tokio::task::spawn_blocking(memory_bandwidth).await.unwrap_or(0.0);
    let (mem_total, mem_available) = memory();
    DeviceReport {
        name: sysinfo::System::host_name().unwrap_or_default(),
        os: std::env::consts::OS.into(),
        cores: std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1),
        mem_total,
        mem_available,
        mem_bw,
        engine,
        gpu: if engine { tokio::task::spawn_blocking(gpu).await.unwrap_or(None) } else { None },
    }
}

/// The engine's GPU and its measured speed, once per process (the measurement reads a
/// 38 MB matrix eight times).
fn gpu() -> Option<GpuReport> {
    static GPU: OnceLock<Option<(String, bool, f64)>> = OnceLock::new();
    let (name, unified, bw) = GPU.get_or_init(|| bc_infer::gpu::get().and_then(|g| Some((g.name.clone(), g.unified(), g.bandwidth().ok()?)))).clone()?;
    let g = bc_infer::gpu::get()?;
    Some(GpuReport { name, mem_bytes: if unified { 0 } else { g.memory() }, unified, bw })
}

/// Total and available memory. macOS can report no available memory at all under
/// pressure, though inactive and cached pages would be handed over; then what is not in
/// use counts as available.
pub fn memory() -> (u64, u64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    (total, sys.available_memory().max(total.saturating_sub(sys.used_memory())))
}

/// Copy bandwidth (read plus write), measured once per process: the best of a few
/// copies, so a busy moment does not stick.
fn memory_bandwidth() -> f64 {
    static BW: OnceLock<f64> = OnceLock::new();
    *BW.get_or_init(|| {
        const SIZE: usize = 64 << 20;
        let src = vec![1u8; SIZE];
        let mut dst = vec![0u8; SIZE];
        let mut best = f64::MAX;
        for _ in 0..4 {
            let t = Instant::now();
            dst.copy_from_slice(&src);
            std::hint::black_box(&dst);
            best = best.min(t.elapsed().as_secs_f64());
        }
        2.0 * SIZE as f64 / best.max(1e-9)
    })
}

/// Answers probe frames on one stream until it ends.
pub(crate) async fn answer(
    first: MeshMsg,
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
    engine: bool,
) -> Result<()> {
    let mut msg = first;
    let mut bulk: Option<(Instant, u64)> = None;
    loop {
        match msg {
            MeshMsg::Report => send_msg(send, &MeshMsg::Device(report(engine).await)).await?,
            MeshMsg::Ping { nonce } => send_msg(send, &MeshMsg::Pong { nonce }).await?,
            MeshMsg::Bulk { data } => {
                let (_, bytes) = bulk.get_or_insert((Instant::now(), 0));
                *bytes += data.len() as u64;
            }
            MeshMsg::BulkDone => {
                let (start, bytes) = bulk.take().unwrap_or((Instant::now(), 0));
                send_msg(send, &MeshMsg::BulkAck { bytes, micros: start.elapsed().as_micros() as u64 }).await?;
            }
            other => bail!("unexpected probe frame {other:?}"),
        }
        msg = match recv_msg::<MeshMsg>(recv).await {
            Ok(m) => m,
            Err(_) => return Ok(()), // the prober finished
        };
    }
}
