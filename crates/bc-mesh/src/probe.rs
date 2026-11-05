//! Measuring a link and the device at its other end, which the planner needs: round trip
//! time and upload rate for the hops, memory and memory bandwidth for the layers.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use bc_net::{recv_msg, send_msg, MESH_ALPN};
use iroh::{Endpoint, EndpointAddr};
use serde::{Deserialize, Serialize};

use crate::msg::{DeviceReport, MeshMsg};

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

    let mut rtts = Vec::with_capacity(PINGS);
    for nonce in 0..PINGS as u64 {
        let t = Instant::now();
        send_msg(&mut send, &MeshMsg::Ping { nonce }).await?;
        match recv_msg::<MeshMsg>(&mut recv).await? {
            MeshMsg::Pong { nonce: n } if n == nonce => rtts.push(t.elapsed().as_secs_f64() * 1000.0),
            other => bail!("unexpected reply {other:?}"),
        }
    }
    rtts.sort_by(f64::total_cmp);

    let chunk = vec![0x5a; BULK_CHUNK];
    for _ in 0..BULK_BYTES / BULK_CHUNK {
        send_msg(&mut send, &MeshMsg::Bulk { data: chunk.clone() }).await?;
    }
    send_msg(&mut send, &MeshMsg::BulkDone).await?;
    let up_bytes_s = match recv_msg::<MeshMsg>(&mut recv).await? {
        MeshMsg::BulkAck { bytes, micros } => bytes as f64 / (micros.max(1) as f64 / 1e6),
        other => bail!("unexpected reply {other:?}"),
    };
    let via_relay = bc_net::via_relay(&conn);
    send.finish()?;
    conn.close(0u32.into(), b"done");
    Ok(LinkReport { rtt_ms: rtts[PINGS / 2], up_bytes_s, via_relay, device })
}

/// This device, as a probe reports it.
pub async fn report(engine: bool) -> DeviceReport {
    let mem_bw = tokio::task::spawn_blocking(memory_bandwidth).await.unwrap_or(0.0);
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    DeviceReport {
        name: sysinfo::System::host_name().unwrap_or_default(),
        os: std::env::consts::OS.into(),
        cores: std::thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(1),
        mem_total: sys.total_memory(),
        mem_available: sys.available_memory(),
        mem_bw,
        engine,
    }
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
