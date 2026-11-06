//! A language model split over three devices, end to end: the devices fetch only their
//! own layers from a web server by byte range, the coordinator streams requests through
//! them over a relay, and the text matches running the whole model on one computer.

use std::sync::Arc;
use std::time::Duration;

use bc_infer::{Local, Sampling, Source};
use bc_mesh::llm::{Client, Llm};
use bc_mesh::plan::{PipelinePlan, StagePlan};
use bc_mesh::stage::Host;
use bc_net::{bind, NetConfig};
use iroh::{Endpoint, EndpointId, SecretKey};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Serves `file` over plain HTTP on localhost, honouring `Range`. Counts bytes served.
async fn serve(file: Vec<u8>) -> (String, Arc<std::sync::atomic::AtomicU64>) {
    serve_slow(file, Duration::ZERO).await
}

/// [`serve`], answering each request only after `delay`, like a slow download.
async fn serve_slow(file: Vec<u8>, delay: Duration) -> (String, Arc<std::sync::atomic::AtomicU64>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/model.gguf", listener.local_addr().unwrap());
    let file = Arc::new(file);
    let served = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let counter = served.clone();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let (file, counter) = (file.clone(), counter.clone());
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = s.read(&mut buf).await.unwrap_or(0);
                tokio::time::sleep(delay).await;
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let range = req.lines().find_map(|l| l.to_lowercase().strip_prefix("range: bytes=").map(String::from));
                let (start, end) = match range.and_then(|r| r.split_once('-').map(|(a, b)| (a.trim().parse::<usize>().ok(), b.trim().parse::<usize>().ok()))) {
                    Some((Some(a), Some(b))) => (a, (b + 1).min(file.len())),
                    _ => (0, file.len()),
                };
                if start >= file.len() {
                    let _ = s.write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\ncontent-length: 0\r\n\r\n").await;
                    return;
                }
                let body = &file[start..end];
                counter.fetch_add(body.len() as u64, std::sync::atomic::Ordering::Relaxed);
                let head = format!("HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\ncontent-range: bytes {start}-{}/{}\r\nconnection: close\r\n\r\n", body.len(), end - 1, file.len());
                let _ = s.write_all(head.as_bytes()).await;
                let _ = s.write_all(body).await;
            });
        }
    });
    (url, served)
}

struct Mesh {
    coordinator: Endpoint,
    devices: Vec<Endpoint>,
    _relay: iroh_relay::server::Server,
}

async fn mesh(n: usize) -> Mesh {
    mesh_slow(n, Duration::ZERO).await
}

/// Stages that take `delay` longer per batch, like a slower device or link.
struct Slow(Box<dyn bc_mesh::stage::Stage>, Duration);

impl bc_mesh::stage::Stage for Slow {
    fn forward(&mut self, meta: &bc_mesh::msg::Meta, input: bc_mesh::msg::Activation) -> anyhow::Result<bc_mesh::msg::Activation> {
        std::thread::sleep(self.1);
        self.0.forward(meta, input)
    }
}

async fn mesh_slow(n: usize, delay: Duration) -> Mesh {
    let (relay_map, _url, relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let config = NetConfig { relays: Some(relay_map), relay_only: true, insecure_relay_tls: true };
    let coordinator = bind(SecretKey::generate(), config.clone()).await.unwrap();
    let cid = coordinator.id();
    let mut devices = Vec::new();
    for _ in 0..n {
        let ep = bind(SecretKey::generate(), config.clone()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), ep.online()).await.unwrap();
        let engine = bc_mesh::engine::factory();
        let factory: bc_mesh::stage::StageFactory = Arc::new(move |setup, loading| {
            let engine = engine.clone();
            Box::pin(async move { Ok(Box::new(Slow(engine(setup, loading).await?, delay)) as Box<dyn bc_mesh::stage::Stage>) })
        });
        let host = Host::new(ep.clone(), Arc::new(move |id: &EndpointId| *id == cid), Some(factory));
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
        devices.push(ep);
    }
    Mesh { coordinator, devices, _relay: relay }
}

fn plan(splits: &[(usize, std::ops::Range<u32>)]) -> PipelinePlan {
    PipelinePlan {
        stages: splits.iter().map(|(d, l)| StagePlan { device: *d, layers: l.clone() }).collect(),
        token_ms: 0.0,
        network_ms: 0.0,
        tokens_per_s: 1.0,
        prefill_net_ms_per_1k: 0.0,
    }
}

/// What the whole model writes on one computer, greedily.
async fn local_text(file: &[u8], prompt: &[u32], n: usize) -> Vec<u32> {
    let dir = tempfile_dir();
    let path = dir.join("whole.gguf");
    std::fs::write(&path, file).unwrap();
    let mut m = Local::load(&Source::File(path)).await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let mut out = Vec::new();
    let mut feed = prompt.to_vec();
    let mut pos = 0;
    for _ in 0..n {
        let s = m.scores(&feed, pos).unwrap();
        pos += feed.len();
        let id = bc_infer::sample::argmax(&s[(feed.len() - 1) * m.part.cfg.vocab..]);
        if m.tokenizer.is_stop(id) {
            break;
        }
        out.push(id);
        feed = vec![id];
    }
    out
}

/// A fresh folder for model copies; each test deletes it once the model is in memory.
fn tempfile_dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("bc-mesh-llm-{}-{}", std::process::id(), rand::random::<u32>()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[tokio::test(flavor = "multi_thread")]
async fn three_devices_write_what_one_computer_writes() {
    let file = bc_infer::fixtures::tiny_model(6, true);
    let (url, served) = serve(file.clone()).await;
    let m = mesh(3).await;
    let addrs: Vec<_> = m.devices.iter().map(|d| d.addr()).collect();
    let header = Source::parse_remote(&url).unwrap().header().await.unwrap();
    let before = served.load(std::sync::atomic::Ordering::Relaxed);
    // Each stage reports its download; the last report says it has everything.
    let progress = std::sync::Mutex::new(std::collections::HashMap::new());
    let on_loading = |stage: usize, done: u64, total: u64| {
        progress.lock().unwrap().insert(stage, (done, total));
    };
    let client = Client::open(&m.coordinator, &url, plan(&[(0, 0..2), (1, 2..5), (2, 5..6)]), &addrs, &on_loading).await.unwrap();
    let progress = progress.into_inner().unwrap();
    assert_eq!(progress.len(), 3, "{progress:?}");
    assert!(progress.values().all(|(done, total)| done == total && *total > 0), "{progress:?}");
    // Each device fetched only its own tensors (plus the header): the three together
    // read about one file's worth, not three.
    let fetched = served.load(std::sync::atomic::Ordering::Relaxed) - before;
    assert!(fetched < file.len() as u64 + 3 * (2 << 20), "devices fetched {fetched} bytes of a {}-byte file", file.len());

    let llm = Llm::new(&header, vec![client], None).unwrap();
    let prompt: Vec<u32> = vec![1, 40, 41, 42, 43, 44];
    let want = local_text(&file, &prompt, 20).await;
    let mut text = String::new();
    let (got, stats) = llm.generate(&prompt, 20, Sampling::default(), |t| text.push_str(t)).await.unwrap();
    assert_eq!(got, want, "split over three devices, greedy, f32 on the wire");
    assert_eq!(stats.passes + 1, stats.generated.max(1).max(stats.passes + 1).min(stats.generated + 1));
    assert_eq!(text, llm.tokenizer.decode(&got));
    // Each lender lists the part it runs, so its app can show it.
    let mine: Vec<_> = bc_mesh::engine::hosted_now().into_iter().filter(|h| h.coordinator == m.coordinator.id().to_string() && h.of == 6).collect();
    assert!(mine.len() >= 3 && mine.iter().all(|h| h.running && h.batches > 0 && h.loaded == h.total), "{mine:?}");
    assert!(mine.iter().all(|h| h.gpu_ms + h.cpu_ms > 0.0), "each part says where its time goes: {mine:?}");

    // A second conversation and a long prompt in chunks, at the same time.
    let long: Vec<u32> = (0..300).map(|i| 3 + (i * 7 % 90) as u32).collect();
    let want_long = local_text(&file, &long, 8).await;
    let (a, b) = tokio::join!(llm.generate(&prompt, 20, Sampling::default(), |_| {}), llm.generate(&long, 8, Sampling::default(), |_| {}));
    assert_eq!(a.unwrap().0, want);
    assert_eq!(b.unwrap().0, want_long, "a 300-token prompt sent in chunks");
}

#[tokio::test(flavor = "multi_thread")]
async fn speculative_decoding_writes_the_same_text_in_fewer_passes() {
    let file = bc_infer::fixtures::tiny_model(4, false);
    let (url, _) = serve(file.clone()).await;
    let m = mesh(2).await;
    let addrs: Vec<_> = m.devices.iter().map(|d| d.addr()).collect();
    let header = Source::parse_remote(&url).unwrap().header().await.unwrap();
    let prompt: Vec<u32> = vec![1, 50, 51, 52];
    let want = local_text(&file, &prompt, 24).await;

    // The model itself as its own draft: every guess is right.
    let dir = tempfile_dir();
    std::fs::write(dir.join("d.gguf"), &file).unwrap();
    let same = Local::load(&Source::File(dir.join("d.gguf"))).await.unwrap();
    let client = Client::open(&m.coordinator, &url, plan(&[(0, 0..2), (1, 2..4)]), &addrs, &|_, _, _| {}).await.unwrap();
    let llm = Llm::new(&header, vec![client], Some(same)).unwrap();
    let (got, stats) = llm.generate(&prompt, 24, Sampling::default(), |_| {}).await.unwrap();
    assert_eq!(got, want);
    if got.len() > 5 {
        assert!(stats.passes * 3 <= got.len() + 3, "{} passes for {} tokens", stats.passes, got.len());
    }

    // A different, smaller draft: guesses are sometimes wrong, the text never is.
    std::fs::write(dir.join("d2.gguf"), bc_infer::fixtures::tiny_model(2, false)).unwrap();
    let other = Local::load(&Source::File(dir.join("d2.gguf"))).await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let client = Client::open(&m.coordinator, &url, plan(&[(1, 0..1), (0, 1..4)]), &addrs, &|_, _, _| {}).await.unwrap();
    let llm = Llm::new(&header, vec![client], Some(other)).unwrap();
    let (got, _) = llm.generate(&prompt, 24, Sampling::default(), |_| {}).await.unwrap();
    assert_eq!(got, want);
}

#[tokio::test(flavor = "multi_thread")]
async fn one_conversation_keeps_every_stage_busy() {
    // Three stages of 60 ms each and a perfect draft: one batch at a time would take
    // 180 ms a pass; with a batch per stage in flight, one comes back every 60 ms.
    let delay = Duration::from_millis(60);
    let file = bc_infer::fixtures::tiny_model(3, false);
    let (url, _) = serve(file.clone()).await;
    let m = mesh_slow(3, delay).await;
    let addrs: Vec<_> = m.devices.iter().map(|d| d.addr()).collect();
    let header = Source::parse_remote(&url).unwrap().header().await.unwrap();
    let dir = tempfile_dir();
    std::fs::write(dir.join("d.gguf"), &file).unwrap();
    let same = Local::load(&Source::File(dir.join("d.gguf"))).await.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let client = Client::open(&m.coordinator, &url, plan(&[(0, 0..1), (1, 1..2), (2, 2..3)]), &addrs, &|_, _, _| {}).await.unwrap();
    let llm = Llm::new(&header, vec![client], Some(same)).unwrap();
    let prompt: Vec<u32> = vec![1, 60, 61, 62];
    let want = local_text(&file, &prompt, 40).await;
    let (got, stats) = llm.generate(&prompt, 40, Sampling::default(), |_| {}).await.unwrap();
    assert_eq!(got, want);
    if got.len() < 20 {
        return; // the tiny model stopped early; nothing to time
    }
    let decode = Duration::from_secs_f64((stats.total_ms - stats.first_token_ms) / 1000.0);
    let one_at_a_time = delay * 3 * stats.passes as u32;
    println!("{} tokens in {} batches: {decode:?}, against {one_at_a_time:?} one batch at a time", got.len(), stats.passes);
    assert!(decode < one_at_a_time * 7 / 10, "{} batches took {decode:?}; one at a time would take {one_at_a_time:?}", stats.passes);
}

#[tokio::test(flavor = "multi_thread")]
async fn this_computer_holds_the_first_layers_itself() {
    let file = bc_infer::fixtures::tiny_model(6, true);
    let (url, _) = serve(file.clone()).await;
    let m = mesh(1).await;
    // Device 0 is this computer (the coordinator), device 1 a lending device.
    let addrs = vec![m.coordinator.addr(), m.devices[0].addr()];
    let header = Source::parse_remote(&url).unwrap().header().await.unwrap();
    let prompt: Vec<u32> = vec![1, 40, 41, 42, 43, 44];
    let want = local_text(&file, &prompt, 16).await;

    let progress = std::sync::Mutex::new(std::collections::HashMap::new());
    let on_loading = |stage: usize, done: u64, total: u64| {
        progress.lock().unwrap().insert(stage, (done, total));
    };
    let client = Client::open(&m.coordinator, &url, plan(&[(0, 0..3), (1, 3..6)]), &addrs, &on_loading).await.unwrap();
    assert_eq!(progress.into_inner().unwrap().len(), 2, "both this computer and the device report their download");
    let llm = Llm::new(&header, vec![client.clone()], None).unwrap();
    let (got, _) = llm.generate(&prompt, 16, Sampling::default(), |_| {}).await.unwrap();
    assert_eq!(got, want, "split between this computer and a device");
    assert!(client.stage_ms.lock().unwrap().iter().all(|&ms| ms > 0.0), "both stages measured");

    // The whole model on this computer: no network at all.
    let alone = Client::open(&m.coordinator, &url, plan(&[(0, 0..6)]), &addrs, &|_, _, _| {}).await.unwrap();
    let llm = Llm::new(&header, vec![alone], None).unwrap();
    assert_eq!(llm.generate(&prompt, 16, Sampling::default(), |_| {}).await.unwrap().0, want);

    // Only the first stage may be this computer's.
    let err = Client::open(&m.coordinator, &url, plan(&[(1, 0..3), (0, 3..6)]), &addrs, &|_, _, _| {}).await.err().expect("refused");
    assert!(format!("{err:#}").contains("must come first"), "{err:#}");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_cancelled_start_frees_the_models_memory() {
    let file = bc_infer::fixtures::tiny_model(6, false);
    let (url, _) = serve_slow(file.clone(), Duration::from_millis(1500)).await;
    let m = mesh(1).await;
    let addrs: Vec<_> = m.devices.iter().map(|d| d.addr()).collect();
    // A first run with half the layers leaves them cached for a while, for a quick restart.
    let first = Client::open(&m.coordinator, &url, plan(&[(0, 0..3)]), &addrs, &|_, _, _| {}).await.unwrap();
    drop(first);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(bc_mesh::engine::held_bytes_of(&url) > 0, "kept for the grace period");
    // A second start needs the other half from the slow server and is cancelled midway:
    // every weight of that model leaves memory, the cached half included.
    let start = tokio::spawn({
        let (ep, url, addrs) = (m.coordinator.clone(), url.clone(), addrs.clone());
        async move { Client::open(&ep, &url, plan(&[(0, 0..6)]), &addrs, &|_, _, _| {}).await.map(|_| ()) }
    });
    tokio::time::sleep(Duration::from_millis(700)).await;
    start.abort();
    let freed = async {
        while bc_mesh::engine::held_bytes_of(&url) > 0 {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(5), freed).await.expect("the cancelled model left memory");
}

/// A real model over three devices: `BC_TEST_MODEL=<model.gguf> [BC_TEST_DRAFT=<draft.gguf>]
/// cargo test -p bc-mesh --release --test llm real_model -- --ignored --nocapture`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn real_model_over_three_devices() {
    let Ok(path) = std::env::var("BC_TEST_MODEL") else { return };
    let file = std::fs::read(&path).unwrap();
    let (url, _) = serve(file.clone()).await;
    let m = mesh(3).await;
    let addrs: Vec<_> = m.devices.iter().map(|d| d.addr()).collect();
    let header = Source::parse_remote(&url).unwrap().header().await.unwrap();
    let n = bc_infer::Config::from_header(&header).unwrap().n_layer as u32;
    let (a, b) = (n / 3, 2 * n / 3);
    let t = std::time::Instant::now();
    let client = Client::open(&m.coordinator, &url, plan(&[(0, 0..a), (1, a..b), (2, b..n)]), &addrs, &|_, _, _| {}).await.unwrap();
    println!("three stages loaded their layers in {:.1}s", t.elapsed().as_secs_f64());
    let draft = match std::env::var("BC_TEST_DRAFT") {
        Ok(d) => Some(Local::load(&Source::File(d.into())).await.unwrap()),
        Err(_) => None,
    };
    let llm = Llm::new(&header, vec![client], draft).unwrap();
    let prompt = llm.tokenizer.encode(&llm.tokenizer.chat_prompt(&[("user".into(), "Explain in three sentences why the sky is blue.".into())]));
    let mut text = String::new();
    let (ids, stats) = llm.generate(&prompt, 96, Sampling::default(), |t| text.push_str(t)).await.unwrap();
    println!("---\n{text}\n---");
    let decode_s = (stats.total_ms - stats.first_token_ms) / 1000.0;
    println!(
        "{} prompt tokens, first token {:.0} ms; {} tokens in {} passes, {:.1} tokens/s; stage ms {:?}",
        stats.prompt_tokens, stats.first_token_ms, stats.generated, stats.passes,
        (stats.generated.saturating_sub(1)) as f64 / decode_s.max(1e-3),
        llm.replicas()[0].stage_ms.lock().unwrap()
    );
    let want = local_text(&file, &prompt, ids.len()).await;
    assert_eq!(ids, want, "the split model writes exactly what the whole one does");
    println!("matches one computer for all {} tokens", ids.len());
}
