//! An OpenAI-compatible endpoint on this computer for a model running across devices, so
//! any chat app or SDK pointed at `http://127.0.0.1:<port>/v1` uses the mesh.
//!
//! `POST /v1/chat/completions` (streamed or not, with tools) and `GET /v1/models`. Only
//! localhost listens: whoever can reach the endpoint uses the owner's devices.
//!
//! Chat clients send the whole conversation every time. The endpoint keeps a few
//! conversations' streams on the stages ([`Cached`]) and gives each request the one that
//! shares the longest start with it, so only the new messages are computed.

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

use crate::agent::format::{parse, system_prompt, ToolSpec, Turns};
use crate::mesh::Session;
use bc_mesh::llm::Cached;

/// Conversations whose streams stay on the stages (well under the stages' own limit of 16).
const KEPT_STREAMS: usize = 4;

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    /// vLLM's way to turn thinking off: `{"enable_thinking": false}`.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
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
    /// Text, or a list of parts of which the text ones count; none for a turn that only calls tools.
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct Tool {
    pub function: ToolSpecIn,
}

#[derive(Debug, Deserialize)]
pub struct ToolSpecIn {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ToolCall {
    pub function: CallIn,
}

#[derive(Debug, Deserialize)]
pub struct CallIn {
    pub name: String,
    /// A JSON object, as a string.
    #[serde(default)]
    pub arguments: String,
}

impl Message {
    fn text(&self) -> String {
        match &self.content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str())).collect::<Vec<_>>().join(""),
            _ => String::new(),
        }
    }
}

/// Asks again for the paired devices, when a pipeline breaks.
pub type Targets = Arc<dyn Fn() -> futures_util::future::BoxFuture<'static, Result<Vec<(String, iroh::EndpointAddr)>>> + Send + Sync>;

pub struct Served {
    pub session: Arc<Session>,
    pub targets: Targets,
    streams: std::sync::Mutex<Vec<Cached>>,
}

impl Served {
    pub fn new(session: Arc<Session>, targets: Targets) -> Self {
        Served { session, targets, streams: Default::default() }
    }

    /// The kept stream that shares the longest start with `prompt`, or a new one.
    fn stream_for(&self, prompt: &[u32]) -> Cached {
        let mut kept = self.streams.lock().unwrap();
        let common = |c: &Cached| c.held().iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let best = (0..kept.len()).max_by_key(|&i| common(&kept[i]));
        match best {
            Some(i) if common(&kept[i]) > 0 || kept.len() >= KEPT_STREAMS => kept.remove(i),
            _ => Cached::default(),
        }
    }

    fn keep(&self, c: Cached) {
        let mut kept = self.streams.lock().unwrap();
        kept.push(c);
        if kept.len() > KEPT_STREAMS {
            kept.remove(0);
        }
    }
}

/// A chat request as the model reads it: the system prompt (with the tools), then each
/// message in the model's own format, ending where the assistant speaks.
pub fn render(tokenizer: &bc_infer::Tokenizer, req: &ChatRequest) -> Vec<u32> {
    let think = req.chat_template_kwargs.as_ref().and_then(|k| k.get("enable_thinking")).and_then(|v| v.as_bool()).unwrap_or(true);
    let turns = Turns { tokenizer, think };
    let tools: Vec<ToolSpec> = req.tools.iter().map(|t| ToolSpec { name: t.function.name.clone(), description: t.function.description.clone(), parameters: t.function.parameters.clone() }).collect();
    let base = req.messages.iter().filter(|m| m.role == "system").map(Message::text).collect::<Vec<_>>().join("\n\n");
    let base = if base.is_empty() { "You are a helpful assistant.".to_string() } else { base };
    let mut ids = tokenizer.encode(&turns.open(&system_prompt(&base, &tools)));
    let mut results: Vec<(String, String)> = Vec::new();
    for m in req.messages.iter().filter(|m| m.role != "system") {
        if m.role == "tool" {
            results.push((String::new(), m.text()));
            continue;
        }
        if !results.is_empty() {
            ids.extend(tokenizer.encode_segment(&turns.results(&std::mem::take(&mut results))));
        }
        if m.role == "assistant" {
            let mut text = m.text();
            for c in &m.tool_calls {
                let args: serde_json::Value = serde_json::from_str(&c.function.arguments).unwrap_or_else(|_| json!({}));
                text.push_str(&format!("\n<tool_call>\n{}\n</tool_call>", json!({ "name": c.function.name, "arguments": args })));
            }
            ids.extend(tokenizer.encode_segment(&format!("{}{}", text.trim_start(), turns.close())));
        } else {
            ids.extend(tokenizer.encode_segment(&turns.user(&m.text())));
        }
    }
    if !results.is_empty() {
        ids.extend(tokenizer.encode_segment(&turns.results(&results)));
    }
    ids
}

/// The assistant's message for an answer: text, reasoning if it thought, and tool calls.
fn reply_message(raw: &str, with_tools: bool) -> (serde_json::Value, bool) {
    let r = parse(raw);
    let calls: Vec<serde_json::Value> = if with_tools {
        r.calls
            .iter()
            .filter_map(|c| c.as_ref().ok())
            .map(|c| json!({ "id": format!("call_{:016x}", rand::random::<u64>()), "type": "function", "function": { "name": c.name, "arguments": c.arguments.to_string() } }))
            .collect()
    } else {
        Vec::new()
    };
    let text = if with_tools { r.text } else { raw.split_once("</think>").map_or(raw, |(_, a)| a).trim_start().to_string() };
    let mut m = json!({ "role": "assistant", "content": if text.is_empty() && !calls.is_empty() { serde_json::Value::Null } else { json!(text) } });
    if !r.thought.is_empty() {
        m["reasoning_content"] = json!(r.thought);
    }
    let called = !calls.is_empty();
    if called {
        m["tool_calls"] = json!(calls);
    }
    (m, called)
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
    let prompt = render(&llm.tokenizer, &req);
    let with_tools = !req.tools.is_empty();
    let max = req.max_tokens.unwrap_or(512).min(llm.cfg.context.saturating_sub(prompt.len()));
    let sampling = Sampling { temperature: req.temperature.unwrap_or(0.0), top_p: req.top_p.unwrap_or(1.0), seed: req.seed.unwrap_or_else(rand::random) };
    let id = format!("chatcmpl-{:016x}", rand::random::<u64>());
    let model = s.session.name.clone();

    if !req.stream || with_tools {
        let mut cache = s.stream_for(&prompt);
        let mut result = llm.generate_cached(&mut cache, &prompt, max, sampling, |_| {}).await;
        if result.is_err() {
            // A device dropped out: start the pipelines again without it, then retry once.
            if let Ok(targets) = (s.targets)().await {
                if s.session.recover(targets).await.is_ok() {
                    result = llm.generate_cached(&mut cache, &prompt, max, sampling, |_| {}).await;
                }
            }
        }
        s.keep(cache);
        let finish = |ids: &[u32], called: bool| if called { "tool_calls" } else if ids.len() >= max { "length" } else { "stop" };
        if req.stream {
            // With tools the answer is read whole first, then sent as one chunk.
            return match result {
                Ok((ids, _)) => {
                    let (m, called) = reply_message(&llm.tokenizer.decode(&ids), with_tools);
                    let mut delta = m;
                    if let Some(calls) = delta.get_mut("tool_calls").and_then(|c| c.as_array_mut()) {
                        for (i, c) in calls.iter_mut().enumerate() {
                            c["index"] = json!(i);
                        }
                    }
                    let chunk = |delta: serde_json::Value, finish: Option<&str>| json!({ "id": id, "object": "chat.completion.chunk", "created": now(), "model": model, "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] }).to_string();
                    let events = vec![Event::default().data(chunk(delta, None)), Event::default().data(chunk(json!({}), Some(finish(&ids, called)))), Event::default().data("[DONE]")];
                    Sse::new(futures_util::stream::iter(events.into_iter().map(Ok::<_, Infallible>))).into_response()
                }
                Err(e) => error(StatusCode::BAD_GATEWAY, format!("{e:#}")),
            };
        }
        return match result {
            Ok((ids, stats)) => {
                let (message, called) = reply_message(&llm.tokenizer.decode(&ids), with_tools);
                Json(json!({
                    "id": id, "object": "chat.completion", "created": now(), "model": model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish(&ids, called) }],
                    "usage": { "prompt_tokens": stats.prompt_tokens, "completion_tokens": stats.generated, "total_tokens": stats.prompt_tokens + stats.generated,
                               "prompt_tokens_details": { "cached_tokens": stats.cached_tokens } },
                }))
                .into_response()
            }
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
        let mut cache = s.stream_for(&prompt);
        let result = llm.generate_cached(&mut cache, &prompt, max, sampling, |text| {
            let _ = t.send(c(json!({ "content": text }), None));
        }).await;
        s.keep(cache);
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
        assert_eq!(r.messages[0].text(), "hi");
        assert_eq!(r.temperature, Some(0.2));
        assert_eq!(r.max_tokens, None);
    }

    #[test]
    fn tool_calls_and_results_render_and_read_back() {
        let r: ChatRequest = serde_json::from_str(r#"{"messages":[
            {"role":"user","content":[{"type":"text","text":"weather?"}]},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Pune\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"31 C"}],
            "tools":[{"type":"function","function":{"name":"weather","description":"Weather now","parameters":{"type":"object"}}}]}"#).unwrap();
        assert_eq!(r.messages[0].text(), "weather?");
        assert_eq!(r.tools[0].function.name, "weather");
        let (m, called) = reply_message("<think>\nhm\n</think>\n\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Pune\"}}\n</tool_call>", true);
        assert!(called);
        assert_eq!(m["content"], serde_json::Value::Null);
        assert_eq!(m["reasoning_content"], "hm");
        assert_eq!(m["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Pune\"}");
        let (m, called) = reply_message("<think>\n\n</think>\n\nIt is 31 C.", false);
        assert!(!called && m["content"] == "It is 31 C." && m.get("reasoning_content").is_none());
    }
}
