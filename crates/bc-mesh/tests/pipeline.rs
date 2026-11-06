//! Several devices as one: a coordinator probes three devices at once and runs a model
//! split across them, through a local relay with direct paths off (the worst case).

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use bc_mesh::msg::{Activation, Meta, StageSetup, Tensor};
use bc_mesh::pipeline::{self, Target};
use bc_mesh::probe::probe;
use bc_mesh::stage::{Host, Stage, StageFactory};
use bc_mesh::ticket::Ticket;
use bc_net::{bind, recv_msg, send_msg, NetConfig, MESH_ALPN};
use iroh::{Endpoint, EndpointId, SecretKey};

const WIDTH: usize = 64;
const LAYERS: u32 = 6;

/// A stand-in model: layer `l` is `tanh(W_l x + b_l)` with fixed pseudo-random weights.
fn layer(l: u32, x: &[f32]) -> Vec<f32> {
    (0..WIDTH)
        .map(|i| {
            let mut acc = ((l as usize * 31 + i * 7) % 13) as f32 / 13.0 - 0.5;
            for (j, v) in x.iter().enumerate() {
                let w = ((((l as usize + 1) * 2654435761) ^ ((i * WIDTH + j) * 40503)) % 1000) as f32 / 1000.0 - 0.5;
                acc += w * v * 0.25;
            }
            acc.tanh()
        })
        .collect()
}

fn run_layers(layers: std::ops::Range<u32>, t: &Tensor) -> Tensor {
    let mut data = Vec::with_capacity(t.data.len());
    for row in t.data.chunks(t.cols) {
        let mut x = row.to_vec();
        for l in layers.clone() {
            x = layer(l, &x);
        }
        data.extend(x);
    }
    Tensor { rows: t.rows, cols: WIDTH, data }
}

struct ToyStage {
    layers: std::ops::Range<u32>,
    delay: Duration,
}

impl Stage for ToyStage {
    fn forward(&mut self, _meta: &Meta, input: Activation) -> Result<Activation> {
        std::thread::sleep(self.delay);
        match input {
            Activation::Hidden(t) => Ok(Activation::Hidden(run_layers(self.layers.clone(), &t))),
            Activation::Tokens(_) => anyhow::bail!("toy stages take hidden states"),
        }
    }
}

fn factory(delay: Duration) -> StageFactory {
    Arc::new(move |s: StageSetup| Box::pin(async move { Ok(Box::new(ToyStage { layers: s.layers.clone(), delay }) as Box<dyn Stage>) }))
}

struct Mesh {
    coordinator: Endpoint,
    devices: Vec<Endpoint>,
    _relay: iroh_relay::server::Server,
}

async fn mesh(devices: usize, delay: Duration, engine: bool) -> Mesh {
    let (relay_map, _url, relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let config = NetConfig { relays: Some(relay_map), relay_only: true, insecure_relay_tls: true };
    let coordinator = bind(SecretKey::generate(), config.clone()).await.unwrap();
    let coordinator_id = coordinator.id();
    let mut eps = Vec::new();
    for _ in 0..devices {
        let ep = bind(SecretKey::generate(), config.clone()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), ep.online()).await.unwrap();
        // Each device trusts only the coordinator, as if paired with it and not with each other.
        let trust = Arc::new(move |id: &EndpointId| *id == coordinator_id);
        let host = Host::new(ep.clone(), trust, engine.then(|| factory(delay)));
        let accept = ep.clone();
        tokio::spawn(async move {
            while let Some(incoming) = accept.accept().await {
                let host = host.clone();
                tokio::spawn(async move {
                    if let Ok(conn) = incoming.await {
                        let _ = host.serve(conn).await;
                    }
                });
            }
        });
        eps.push(ep);
    }
    Mesh { coordinator, devices: eps, _relay: relay }
}

fn input(seq: u64) -> Tensor {
    let rows = 3;
    let data = (0..rows * WIDTH).map(|i| ((i as u64 * 17 + seq * 5) % 23) as f32 / 23.0 - 0.5).collect();
    Tensor { rows, cols: WIDTH, data }
}

fn targets(m: &Mesh, splits: &[std::ops::Range<u32>]) -> Vec<Target> {
    m.devices.iter().zip(splits).map(|(d, l)| Target { addr: d.addr(), layers: l.clone() }).collect()
}

async fn run(m: &Mesh, splits: &[std::ops::Range<u32>], batches: u64) -> Vec<(u64, Tensor)> {
    let (mut tx, mut rx) = pipeline::open(&m.coordinator, "toy", &targets(m, splits)).await.unwrap();
    let sender = tokio::spawn(async move {
        for seq in 0..batches {
            tx.send(seq, Meta::default(), &Activation::Hidden(input(seq))).await.unwrap();
        }
        tx.finish().await.unwrap();
    });
    let mut out = Vec::new();
    while let Some((seq, _, a)) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.unwrap().unwrap() {
        let Activation::Hidden(t) = a else { panic!("expected hidden states") };
        out.push((seq, t));
    }
    sender.await.unwrap();
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn a_model_split_over_three_devices_matches_running_it_on_one() {
    let m = mesh(3, Duration::ZERO, true).await;
    let splits = [0..2, 2..5, 5..LAYERS];
    let exact = run(&m, &splits, 4).await;
    assert_eq!(exact.len(), 4);
    for (seq, t) in &exact {
        assert_eq!(t, &run_layers(0..LAYERS, &input(*seq)), "batch {seq} in order and bit for bit");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn stages_work_at_the_same_time() {
    let delay = Duration::from_millis(150);
    let m = mesh(3, delay, true).await;
    let start = Instant::now();
    let out = run(&m, &[0..2, 2..4, 4..LAYERS], 8).await;
    let took = start.elapsed();
    assert_eq!(out.len(), 8);
    // One after another: 8 batches × 3 stages × 150 ms = 3.6 s. Pipelined: about
    // (8 + 2) × 150 ms = 1.5 s plus the network.
    assert!(took < Duration::from_millis(2900), "took {took:?}");
}

#[tokio::test(flavor = "multi_thread")]
async fn probes_several_devices_at_once() {
    let m = mesh(3, Duration::ZERO, false).await;
    let reports = futures_join(&m).await;
    for r in reports {
        let r = r.unwrap();
        assert!(r.via_relay);
        assert!(r.rtt_ms >= 0.0 && r.up_bytes_s > 0.0);
        assert!(r.device.mem_total > 0 && r.device.mem_bw > 0.0 && r.device.cores > 0);
        assert!(!r.device.engine);
    }
}

async fn futures_join(m: &Mesh) -> Vec<Result<bc_mesh::probe::LinkReport>> {
    let tasks: Vec<_> = m
        .devices
        .iter()
        .map(|d| {
            let (ep, addr) = (m.coordinator.clone(), d.addr());
            tokio::spawn(async move { probe(&ep, addr).await })
        })
        .collect();
    let mut out = Vec::new();
    for t in tasks {
        out.push(t.await.unwrap());
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn devices_without_an_engine_refuse_stages() {
    let m = mesh(1, Duration::ZERO, false).await;
    let whole = vec![0..LAYERS; 1];
    let err = match pipeline::open(&m.coordinator, "toy", &targets(&m, &whole)).await {
        Ok(_) => panic!("a device without an engine accepted a stage"),
        Err(e) => e,
    };
    assert!(format!("{err:#}").contains("no inference engine"), "{err:#}");
}

#[tokio::test(flavor = "multi_thread")]
async fn strangers_cannot_probe_set_up_or_feed() {
    let m = mesh(1, Duration::ZERO, true).await;
    let stranger = bind(SecretKey::generate(), m_config(&m)).await.unwrap();
    // Not paired: probes and setups are refused.
    let err = probe(&stranger, m.devices[0].addr()).await.expect_err("refused");
    assert!(format!("{err:#}").contains("not paired"), "{err:#}");
    // A feed with a ticket the coordinator never signed is refused too.
    let conn = stranger.connect(m.devices[0].addr(), MESH_ALPN).await.unwrap();
    let (mut send, mut recv) = conn.open_bi().await.unwrap();
    let forged = Ticket::sign(stranger.secret_key(), 1, stranger.id(), m.devices[0].id());
    send_msg(&mut send, &bc_mesh::msg::MeshMsg::Feed { ticket: forged }).await.unwrap();
    match recv_msg::<bc_mesh::msg::MeshMsg>(&mut recv).await.unwrap() {
        bc_mesh::msg::MeshMsg::Error { reason } => assert!(reason.contains("no stage is waiting"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

fn m_config(m: &Mesh) -> NetConfig {
    // The stranger reaches the devices through the same relay.
    let relays = m.devices[0].addr().relay_urls().next().cloned().map(iroh::RelayMap::from);
    NetConfig { relays, relay_only: true, insecure_relay_tls: true }
}

#[tokio::test(flavor = "multi_thread")]
async fn devices_measure_the_link_between_them_on_the_coordinators_word() {
    let m = mesh(2, Duration::ZERO, false).await;
    // The two devices never paired with each other; the coordinator's ticket lets them talk.
    let (rtt, up, relay) = bc_mesh::probe::probe_link(&m.coordinator, m.devices[0].addr(), m.devices[1].addr()).await.unwrap();
    assert!(rtt >= 0.0 && up > 0.0 && relay);
    // A coordinator the devices do not trust gets nowhere.
    let stranger = bind(SecretKey::generate(), m_config(&m)).await.unwrap();
    let err = bc_mesh::probe::probe_link(&stranger, m.devices[0].addr(), m.devices[1].addr()).await.expect_err("refused");
    assert!(format!("{err:#}").contains("not paired"), "{err:#}");
}
