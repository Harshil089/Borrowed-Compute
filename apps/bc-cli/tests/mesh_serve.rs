//! `bc mesh serve` end to end: two devices run a model's layers, fetched from a web
//! server by byte range, and an OpenAI-style client chats with it on localhost.

use std::sync::Arc;
use std::time::Duration;

use bc_cli::mesh::Session;
use bc_cli::serve::{router, Served};
use bc_mesh::plan::Goal;
use bc_mesh::stage::Host;
use bc_net::{bind, NetConfig};
use iroh::{EndpointId, SecretKey};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn serve_file(file: Vec<u8>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/tiny.gguf", listener.local_addr().unwrap());
    let file = Arc::new(file);
    tokio::spawn(async move {
        loop {
            let (mut s, _) = listener.accept().await.unwrap();
            let file = file.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = s.read(&mut buf).await.unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_lowercase();
                let (a, b) = req
                    .lines()
                    .find_map(|l| l.strip_prefix("range: bytes=").and_then(|r| r.split_once('-')).map(|(a, b)| (a.trim().parse::<usize>().unwrap(), b.trim().parse::<usize>().unwrap() + 1)))
                    .unwrap_or((0, file.len()));
                let b = b.min(file.len());
                let head = format!("HTTP/1.1 206 Partial Content\r\ncontent-length: {}\r\nconnection: close\r\n\r\n", b - a);
                let _ = s.write_all(head.as_bytes()).await;
                let _ = s.write_all(&file[a..b]).await;
            });
        }
    });
    url
}

#[tokio::test(flavor = "multi_thread")]
async fn chats_through_an_openai_endpoint_backed_by_two_devices() {
    let url = serve_file(bc_infer::fixtures::tiny_model(4, true)).await;
    let (relay_map, _u, _relay) = iroh::test_utils::run_relay_server().await.unwrap();
    let config = NetConfig { relays: Some(relay_map), relay_only: true, insecure_relay_tls: true };
    let coordinator = bind(SecretKey::generate(), config.clone()).await.unwrap();
    let cid = coordinator.id();
    let mut targets = Vec::new();
    for i in 0..2 {
        let ep = bind(SecretKey::generate(), config.clone()).await.unwrap();
        tokio::time::timeout(Duration::from_secs(10), ep.online()).await.unwrap();
        let host = Host::new(ep.clone(), Arc::new(move |id: &EndpointId| *id == cid), Some(bc_mesh::engine::factory()));
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
        targets.push((format!("device-{i}"), ep.addr()));
    }
    let session = Arc::new(Session::start(coordinator, targets.clone(), &url, Goal::Throughput, Some("none"), |_| {}).await.unwrap());
    assert!(!session.layout().await.is_empty());
    let t = targets.clone();
    let served = Arc::new(Served { session: session.clone(), targets: Arc::new(move || { let t = t.clone(); Box::pin(async move { Ok(t) }) }) });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router(served)).await.unwrap() });

    let http = reqwest::Client::new();
    let models: serde_json::Value = http.get(format!("{base}/models")).send().await.unwrap().json().await.unwrap();
    assert!(models["data"][0]["id"].as_str().unwrap().ends_with(".gguf"));

    let body = serde_json::json!({ "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 12 });
    let reply: serde_json::Value = http.post(format!("{base}/chat/completions")).json(&body).send().await.unwrap().json().await.unwrap();
    let content = reply["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert!(reply["usage"]["prompt_tokens"].as_u64().unwrap() > 5, "{reply}");

    // Streaming gives the same text, in pieces.
    let mut stream_body = body.clone();
    stream_body["stream"] = true.into();
    let text = http.post(format!("{base}/chat/completions")).json(&stream_body).send().await.unwrap().text().await.unwrap();
    let mut streamed = String::new();
    for line in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            break;
        }
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(c);
        }
    }
    assert!(text.trim_end().ends_with("data: [DONE]"));
    assert_eq!(streamed, content, "greedy: the same answer streamed or not");
    // Nothing changed on the devices, so a rebalance keeps the plan.
    assert!(!session.rebalance().await.unwrap());
}
