//! A model on an OpenAI-compatible server (Ollama, LM Studio, llama.cpp's server, vLLM),
//! for when no model runs on the Mesh page. The agent lays out its prompt in the model's
//! chat format with a byte-level vocabulary ([`Tokenizer::byte_level`]) and sends it as
//! text to `/v1/completions`, which every one of those servers takes; the server keeps
//! its own prompt cache, so the agent's append-only prompts reuse it the same way.

use std::time::Instant;

use anyhow::{bail, Context, Result};
use bc_infer::tokenizer::Template;
use bc_infer::Tokenizer;
use bc_mesh::llm::{Cached, Stats};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Where the model is, as the person set it on the Agent page.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelServer {
    /// Such as `http://127.0.0.1:11434/v1` (Ollama) or `http://127.0.0.1:1234/v1`.
    pub url: String,
    /// The server's name for the model, such as `qwen3:8b`.
    pub model: String,
    /// The model's context in its own tokens; 0 means 8,192.
    #[serde(default)]
    pub context: usize,
}

impl ModelServer {
    pub fn is_set(&self) -> bool {
        !self.url.trim().is_empty() && !self.model.trim().is_empty()
    }

    fn completions(&self) -> String {
        let base = self.url.trim().trim_end_matches('/');
        let base = if base.ends_with("/v1") { base.to_string() } else { format!("{base}/v1") };
        format!("{base}/completions")
    }
}

/// A byte token is about this share of a real one, so a context counted in real tokens
/// holds this many times as many byte tokens; kept low so the count errs on the safe side.
const BYTES_PER_TOKEN: usize = 3;

pub struct HttpBrain {
    server: ModelServer,
    tk: Tokenizer,
    client: reqwest::Client,
}

impl HttpBrain {
    pub fn new(server: ModelServer) -> Result<Self> {
        if !server.is_set() {
            bail!("set the model server's address and model name first");
        }
        let name = server.model.to_lowercase();
        // Llama 3 models use their own format; Hermes builds on Llama but speaks ChatML,
        // as Qwen and most others do.
        let template = if name.contains("llama") && !name.contains("hermes") { Template::Llama3 } else { Template::ChatMl };
        let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(10)).build()?;
        Ok(HttpBrain { server, tk: Tokenizer::byte_level(template), client })
    }

    fn stops(&self) -> Vec<&'static str> {
        match self.tk.template {
            Template::Llama3 => vec!["<|eot_id|>"],
            _ => vec!["<|im_end|>", "<|endoftext|>"],
        }
    }

    /// Sends `prompt`, streams the text to `on_text`, and gives it with the usage.
    async fn send(&self, prompt: &str, max_tokens: usize, stream: bool, on_text: &mut (dyn FnMut(&str) + Send)) -> Result<(String, Value)> {
        let body = json!({
            "model": self.server.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": stream,
            "stop": self.stops(),
            // llama.cpp's server keeps the prompt's cache only when asked.
            "cache_prompt": true,
        });
        let url = self.server.completions();
        let mut resp = self.client.post(&url).json(&body).send().await.with_context(|| format!("cannot reach the model server at {url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            bail!("the model server answered {status}: {}", super::format::clip(&text, 300));
        }
        if !stream {
            let v: Value = resp.json().await?;
            let text = v["choices"][0]["text"].as_str().unwrap_or_default().to_string();
            return Ok((text, v["usage"].clone()));
        }
        let (mut text, mut usage, mut pending) = (String::new(), Value::Null, String::new());
        while let Some(chunk) = resp.chunk().await? {
            pending.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(end) = pending.find('\n') {
                let line = pending[..end].trim().to_string();
                pending.drain(..=end);
                let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
                if data == "[DONE]" {
                    return Ok((text, usage));
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                if let Some(t) = v["choices"][0]["text"].as_str() {
                    on_text(t);
                    text.push_str(t);
                }
                if !v["usage"].is_null() {
                    usage = v["usage"].clone();
                }
            }
        }
        Ok((text, usage))
    }
}

impl super::Brain for HttpBrain {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tk
    }

    fn context(&self) -> usize {
        (if self.server.context == 0 { 8192 } else { self.server.context }) * BYTES_PER_TOKEN
    }

    fn complete<'a>(&'a self, _: &'a mut Cached, ids: &'a [u32], max: usize, on_text: &'a mut (dyn FnMut(&str) + Send)) -> BoxFuture<'a, Result<(Vec<u32>, Stats)>> {
        Box::pin(async move {
            let started = Instant::now();
            let prompt = self.tk.decode(ids);
            let (text, usage) = self.send(&prompt, (max / BYTES_PER_TOKEN).max(16), true, on_text).await?;
            let count = |k: &str| usage[k].as_u64().map(|n| n as usize);
            let out = self.tk.encode_segment(&text);
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let stats = Stats {
                prompt_tokens: count("prompt_tokens").unwrap_or(ids.len() / BYTES_PER_TOKEN),
                cached_tokens: usage["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap_or(0) as usize,
                generated: count("completion_tokens").unwrap_or(out.len() / BYTES_PER_TOKEN),
                total_ms: ms,
                ..Default::default()
            };
            Ok((out, stats))
        })
    }

    /// A server's tokens are not this vocabulary's, so it cannot be held to a range: the
    /// first digit it writes is read instead, and anything else counts as no choice.
    fn choose<'a>(&'a self, _: &'a mut Cached, ids: &'a [u32], range: [u32; 2]) -> BoxFuture<'a, Result<u32>> {
        Box::pin(async move {
            let (text, _) = self.send(&self.tk.decode(ids), 3, false, &mut |_| {}).await?;
            let digit = text.trim().chars().next().filter(char::is_ascii_digit).context("no number in the answer")?;
            let id = self.tk.token_id(&digit.to_string()).context("no such token")?;
            if !(range[0]..range[1]).contains(&id) {
                bail!("{digit} is not one of the choices");
            }
            Ok(id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Brain;

    /// A stand-in OpenAI-compatible server: one request, answered as `reply` says.
    async fn server(reply: &'static str, stream: bool) -> (String, tokio::task::JoinHandle<String>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 65536];
            loop {
                let n = sock.read(&mut buf).await.unwrap();
                req.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&req).to_string();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let len: usize = head.lines().find_map(|l| l.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap_or(0);
                    if body.len() >= len {
                        break;
                    }
                }
            }
            let body = if stream {
                format!("data: {{\"choices\":[{{\"text\":\"{reply}\"}}]}}\n\ndata: {{\"choices\":[{{\"text\":\"!\"}}],\"usage\":{{\"prompt_tokens\":12,\"completion_tokens\":3,\"prompt_tokens_details\":{{\"cached_tokens\":10}}}}}}\n\ndata: [DONE]\n\n")
            } else {
                format!("{{\"choices\":[{{\"text\":\"{reply}\"}}]}}")
            };
            let ctype = if stream { "text/event-stream" } else { "application/json" };
            sock.write_all(format!("HTTP/1.1 200 OK\r\ncontent-type: {ctype}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            String::from_utf8_lossy(&req).to_string()
        });
        (url, task)
    }

    #[tokio::test]
    async fn a_server_model_answers_in_the_agents_format() {
        let (url, req) = server("Hello", true).await;
        let brain = HttpBrain::new(ModelServer { url, model: "qwen3:8b".into(), context: 0 }).unwrap();
        let tk = brain.tokenizer();
        let ids = tk.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
        let mut shown = String::new();
        let (out, stats) = brain.complete(&mut Cached::default(), &ids, 600, &mut |t| shown.push_str(t)).await.unwrap();
        assert_eq!((tk.decode(&out).as_str(), shown.as_str()), ("Hello!", "Hello!"));
        assert_eq!((stats.prompt_tokens, stats.cached_tokens, stats.generated), (12, 10, 3));
        let req = req.await.unwrap();
        assert!(req.starts_with("POST /v1/completions"), "{req}");
        assert!(req.contains(r#""prompt":"<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n""#), "{req}");
        assert!(req.contains(r#""stop":["<|im_end|>","<|endoftext|>"]"#) && req.contains(r#""max_tokens":200"#), "{req}");
        assert_eq!(brain.context(), 8192 * BYTES_PER_TOKEN);
    }

    #[tokio::test]
    async fn a_server_model_picks_by_the_digit_it_writes() {
        let (url, _) = server("3", false).await;
        let brain = HttpBrain::new(ModelServer { url: format!("{url}/v1/"), model: "llama3.1:8b".into(), context: 0 }).unwrap();
        assert_eq!(brain.tokenizer().template, Template::Llama3);
        let one = brain.tokenizer().token_id("1").unwrap();
        assert_eq!(brain.choose(&mut Cached::default(), &[], [one, one + 4]).await.unwrap(), one + 2);
        let (url, _) = server("sure", false).await;
        let brain = HttpBrain::new(ModelServer { url, model: "m".into(), context: 0 }).unwrap();
        assert!(brain.choose(&mut Cached::default(), &[], [one, one + 4]).await.is_err());
        assert!(HttpBrain::new(ModelServer::default()).is_err());
    }
}
