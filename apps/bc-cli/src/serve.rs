//! An OpenAI-compatible endpoint on this computer for a model running across devices, so
//! any chat app or SDK pointed at `http://127.0.0.1:<port>/v1` uses the mesh.
//!
//! `POST /v1/chat/completions` (streamed or not) and `GET /v1/models`. Only localhost
//! listens: whoever can reach the endpoint uses the owner's devices.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bc_infer::Sampling;
use futures_util::stream::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;

use crate::mesh::Session;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Asks again for the paired devices, when a pipeline breaks.
pub type Targets = Arc<dyn Fn() -> futures_util::future::BoxFuture<'static, Result<Vec<(String, iroh::EndpointAddr)>>> + Send + Sync>;

pub struct Served {
    pub session: Arc<Session>,
    pub targets: Targets,
}

pub fn router(served: Arc<Served>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(served)
}

async fn models(State(s): State<Arc<Served>>) -> Json<serde_json::Value> {
    Json(json!({ "object": "list", "data": [{ "id": s.session.name, "object": "model", "owned_by": "borrowed-compute" }] }))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default()
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(json!({ "error": { "message": message, "type": "server_error" } }))).into_response()
}

async fn chat(State(s): State<Arc<Served>>, Json(req): Json<ChatRequest>) -> Response {
    let llm = s.session.llm.clone();
    let messages: Vec<(String, String)> = req.messages.into_iter().map(|m| (m.role, m.content)).collect();
    let prompt = llm.tokenizer.encode(&llm.tokenizer.chat_prompt(&messages));
    let max = req.max_tokens.unwrap_or(512).min(llm.cfg.context.saturating_sub(prompt.len()));
    let sampling = Sampling { temperature: req.temperature.unwrap_or(0.0), top_p: req.top_p.unwrap_or(1.0), seed: req.seed.unwrap_or_else(rand::random) };
    let id = format!("chatcmpl-{:016x}", rand::random::<u64>());
    let model = s.session.name.clone();

    if !req.stream {
        let mut result = llm.generate(&prompt, max, sampling, |_| {}).await;
        if result.is_err() {
            // A device dropped out: start the pipelines again without it, then retry once.
            if let Ok(targets) = (s.targets)().await {
                if s.session.recover(targets).await.is_ok() {
                    result = llm.generate(&prompt, max, sampling, |_| {}).await;
                }
            }
        }
        return match result {
            Ok((ids, stats)) => Json(json!({
                "id": id, "object": "chat.completion", "created": now(), "model": model,
                "choices": [{ "index": 0, "message": { "role": "assistant", "content": llm.tokenizer.decode(&ids) },
                              "finish_reason": if ids.len() >= max { "length" } else { "stop" } }],
                "usage": { "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.generated, "total_tokens": stats.prompt_tokens + stats.generated },
            }))
            .into_response(),
            Err(e) => error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
        };
    }

    let (tx, rx) = mpsc::unbounded_channel::<Event>();
    let chunk = move |delta: serde_json::Value, finish: Option<&str>| {
        Event::default().data(json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }).to_string())
    };
    tokio::spawn(async move {
        let _ = tx.send(chunk(json!({ "role": "assistant" }), None));
        let t = tx.clone();
        let c = chunk.clone();
        let result = llm.generate(&prompt, max, sampling, |text| {
            let _ = t.send(c(json!({ "content": text }), None));
        }).await;
        match result {
            Ok((ids, _)) => {
                let _ = tx.send(chunk(json!({}), Some(if ids.len() >= max { "length" } else { "stop" })));
            }
            Err(e) => {
                let _ = tx.send(Event::default().data(json!({ "error": { "message": format!("{e:#}") } }).to_string()));
            }
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    });
    Sse::new(events(rx)).into_response()
}

fn events(mut rx: mpsc::UnboundedReceiver<Event>) -> impl Stream<Item = Result<Event, Infallible>> {
    futures_util::stream::poll_fn(move |cx| rx.poll_recv(cx).map(|e| e.map(Ok)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_openai_requests() {
        let r: ChatRequest = serde_json::from_str(r#"{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true,"temperature":0.2}"#).unwrap();
        assert!(r.stream);
        assert_eq!(r.messages[0].content, "hi");
        assert_eq!(r.temperature, Some(0.2));
        assert_eq!(r.max_tokens, None);
    }
}
