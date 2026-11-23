//! A Model Context Protocol client: the person's own MCP servers, set in
//! `<home>/mcp.json` in the shape other apps use (`{"mcpServers": {"name": {...}}}`),
//! give the agent their tools, named `<server>__<tool>`. A server is a program started
//! here and spoken to over its stdin and stdout, or an address spoken to over Streamable
//! HTTP. Either way it is the person's choice to run it; its tools ask before each call
//! unless the server is marked read-only, and what they return is untrusted, like any
//! tool output.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{oneshot, Mutex};

use super::format::ToolSpec;

const PROTOCOL: &str = "2025-06-18";
const START_TIMEOUT: Duration = Duration::from_secs(30);
const CALL_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    /// A program to start: `npx`, `uvx`, a path…
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Or the address of a server already running (Streamable HTTP).
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Its tools only read, so they run without asking.
    #[serde(default, rename = "readOnly", alias = "read_only")]
    pub read_only: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default, rename = "mcpServers", alias = "servers")]
    pub servers: BTreeMap<String, ServerConfig>,
}

impl McpConfig {
    pub fn parse(text: &str) -> Result<Self> {
        if text.trim().is_empty() {
            return Ok(McpConfig::default());
        }
        let c: McpConfig = serde_json::from_str(text).context("mcp.json is not valid")?;
        for (name, s) in &c.servers {
            if name.is_empty() || name.contains("__") || !name.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_') {
                bail!("the server name {name:?} may use letters, digits, - and single _ only");
            }
            if s.command.is_none() == s.url.is_none() {
                bail!("the server {name} needs either a command or a url");
            }
        }
        Ok(c)
    }
}

/// One running server and its tools.
pub struct McpServer {
    pub name: String,
    pub read_only: bool,
    pub tools: Vec<ToolSpec>,
    link: Link,
    next: AtomicU64,
}

enum Link {
    Stdio {
        stdin: Mutex<tokio::process::ChildStdin>,
        waiting: Arc<std::sync::Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
        _child: tokio::process::Child,
    },
    Http {
        url: String,
        headers: BTreeMap<String, String>,
        client: reqwest::Client,
        session: std::sync::Mutex<Option<String>>,
    },
}

impl McpServer {
    /// Starts or reaches the server, introduces this client and lists its tools. `path`
    /// is the PATH a started program gets: an app opened from the Finder has a short one.
    pub async fn start(name: &str, config: &ServerConfig, path: Option<&std::ffi::OsStr>) -> Result<Arc<McpServer>> {
        let link = match (&config.command, &config.url) {
            (Some(command), _) => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(&config.args).envs(&config.env).stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).kill_on_drop(true);
                if let (Some(p), false) = (path, config.env.contains_key("PATH")) {
                    cmd.env("PATH", p);
                }
                let mut child = cmd.spawn().with_context(|| format!("cannot start {command}"))?;
                let stdin = child.stdin.take().context("no stdin")?;
                let stdout = child.stdout.take().context("no stdout")?;
                let waiting: Arc<std::sync::Mutex<HashMap<u64, oneshot::Sender<Value>>>> = Arc::default();
                let w = waiting.clone();
                tokio::spawn(async move {
                    let mut lines = BufReader::new(stdout).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
                        // Answers to this client's requests; the server's own requests and
                        // notifications are not used.
                        if let (Some(id), false) = (msg["id"].as_u64(), msg.get("method").is_some()) {
                            if let Some(tx) = w.lock().unwrap().remove(&id) {
                                let _ = tx.send(msg);
                            }
                        }
                    }
                });
                Link::Stdio { stdin: Mutex::new(stdin), waiting, _child: child }
            }
            (None, Some(url)) => Link::Http { url: url.clone(), headers: config.headers.clone(), client: reqwest::Client::new(), session: Default::default() },
            (None, None) => bail!("the server {name} needs either a command or a url"),
        };
        let mut server = McpServer { name: name.to_string(), read_only: config.read_only, tools: Vec::new(), link, next: AtomicU64::new(1) };
        let init = json!({ "protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": { "name": "borrowed-compute", "version": env!("CARGO_PKG_VERSION") } });
        tokio::time::timeout(START_TIMEOUT, server.request("initialize", init)).await.with_context(|| format!("{name} did not answer within {}s", START_TIMEOUT.as_secs()))??;
        server.notify("notifications/initialized").await?;
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor.as_ref().map_or(json!({}), |c| json!({ "cursor": c }));
            let page = tokio::time::timeout(START_TIMEOUT, server.request("tools/list", params)).await.context("listing tools timed out")??;
            for t in page["tools"].as_array().into_iter().flatten() {
                let Some(tool) = t["name"].as_str() else { continue };
                server.tools.push(ToolSpec {
                    name: format!("{name}__{tool}"),
                    description: super::format::clip(t["description"].as_str().unwrap_or_default(), 600),
                    parameters: t.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object" })),
                });
            }
            cursor = page["nextCursor"].as_str().map(String::from);
            if cursor.is_none() {
                break;
            }
        }
        Ok(Arc::new(server))
    }

    /// A server that is not there, with these tools, for tests of what lists them.
    #[cfg(test)]
    pub fn fake(name: &str, read_only: bool, tools: Vec<ToolSpec>) -> Arc<McpServer> {
        let link = Link::Http { url: String::new(), headers: BTreeMap::new(), client: reqwest::Client::new(), session: Default::default() };
        Arc::new(McpServer { name: name.into(), read_only, tools, link, next: AtomicU64::new(1) })
    }

    /// Calls one of its tools by its own name (without the server's prefix).
    pub async fn call(&self, tool: &str, arguments: &Value) -> Result<String> {
        let result = tokio::time::timeout(CALL_TIMEOUT, self.request("tools/call", json!({ "name": tool, "arguments": arguments })))
            .await
            .with_context(|| format!("{}__{tool} did not finish within {}s", self.name, CALL_TIMEOUT.as_secs()))??;
        let text: Vec<String> = result["content"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| match c["type"].as_str() {
                Some("text") => c["text"].as_str().unwrap_or_default().to_string(),
                Some("resource") => c["resource"]["text"].as_str().map(String::from).unwrap_or_else(|| format!("[resource {}]", c["resource"]["uri"].as_str().unwrap_or_default())),
                Some(other) => format!("[{other} content left out]"),
                None => String::new(),
            })
            .collect();
        let text = text.join("\n");
        if result["isError"].as_bool() == Some(true) {
            bail!("{text}");
        }
        Ok(if text.is_empty() { result["structuredContent"].to_string() } else { text })
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let msg = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let answer = match &self.link {
            Link::Stdio { stdin, waiting, .. } => {
                let (tx, rx) = oneshot::channel();
                waiting.lock().unwrap().insert(id, tx);
                let mut line = msg.to_string();
                line.push('\n');
                stdin.lock().await.write_all(line.as_bytes()).await.context("the server's input closed")?;
                rx.await.context("the server stopped")?
            }
            Link::Http { .. } => self.post(&msg).await?.with_context(|| format!("no answer to {method}"))?,
        };
        if let Some(e) = answer.get("error") {
            bail!("{}", e["message"].as_str().unwrap_or("the server refused"));
        }
        Ok(answer["result"].clone())
    }

    async fn notify(&self, method: &str) -> Result<()> {
        let msg = json!({ "jsonrpc": "2.0", "method": method });
        match &self.link {
            Link::Stdio { stdin, .. } => {
                let mut line = msg.to_string();
                line.push('\n');
                stdin.lock().await.write_all(line.as_bytes()).await?;
            }
            Link::Http { .. } => {
                self.post(&msg).await?;
            }
        }
        Ok(())
    }

    /// One message over Streamable HTTP; the answer comes as JSON or in an event stream.
    async fn post(&self, msg: &Value) -> Result<Option<Value>> {
        let Link::Http { url, headers, client, session } = &self.link else { unreachable!() };
        let mut req = client.post(url).header("accept", "application/json, text/event-stream").header("mcp-protocol-version", PROTOCOL).json(msg).timeout(CALL_TIMEOUT);
        for (k, v) in headers {
            req = req.header(k, v);
        }
        if let Some(s) = session.lock().unwrap().clone() {
            req = req.header("mcp-session-id", s);
        }
        let resp = req.send().await.with_context(|| format!("cannot reach {url}"))?;
        if let Some(s) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            *session.lock().unwrap() = Some(s.to_string());
        }
        if !resp.status().is_success() {
            bail!("{url} answered {}", resp.status());
        }
        let sse = resp.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|t| t.starts_with("text/event-stream"));
        let body = resp.text().await?;
        if body.trim().is_empty() {
            return Ok(None);
        }
        let want = msg["id"].clone();
        if !sse {
            return Ok(Some(serde_json::from_str(&body)?));
        }
        Ok(body.lines().filter_map(|l| l.strip_prefix("data:")).filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok()).find(|m| m["id"] == want))
    }
}

/// Starts every server in `config`, each on its own; one that fails is named with why.
pub async fn start_all(config: &McpConfig, path: Option<&std::ffi::OsStr>) -> (Vec<Arc<McpServer>>, Vec<String>) {
    let started = futures_util::future::join_all(config.servers.iter().map(|(name, s)| async move { (name.clone(), McpServer::start(name, s, path).await) })).await;
    let mut servers = Vec::new();
    let mut failed = Vec::new();
    for (name, r) in started {
        match r {
            Ok(s) => servers.push(s),
            Err(e) => failed.push(format!("{name}: {e:#}")),
        }
    }
    (servers, failed)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A stand-in MCP server in Python: initialize, one page of tools, and an echo tool.
    const SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m:
        continue
    if m["method"] == "initialize":
        r = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "t", "version": "1"}}
    elif m["method"] == "tools/list":
        r = {"tools": [{"name": "echo", "description": "Says it back.", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}
    elif m["method"] == "tools/call":
        t = m["params"]["arguments"].get("text", "")
        r = {"content": [{"type": "text", "text": "echo: " + t}], "isError": t == "fail"}
    else:
        r = {}
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#;

    #[tokio::test]
    async fn a_stdio_server_lists_and_runs_its_tools() {
        if std::process::Command::new("python3").arg("--version").output().is_err() {
            return;
        }
        let config = McpConfig::parse(&json!({ "mcpServers": { "t": { "command": "python3", "args": ["-c", SERVER], "readOnly": true } } }).to_string()).unwrap();
        let (servers, failed) = start_all(&config, None).await;
        assert!(failed.is_empty(), "{failed:?}");
        let s = &servers[0];
        assert!(s.read_only);
        assert_eq!(s.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["t__echo"]);
        assert_eq!(s.call("echo", &json!({ "text": "hi" })).await.unwrap(), "echo: hi");
        assert_eq!(format!("{:#}", s.call("echo", &json!({ "text": "fail" })).await.unwrap_err()), "echo: fail");
    }

    #[test]
    fn the_config_is_checked() {
        assert!(McpConfig::parse("").unwrap().servers.is_empty());
        assert!(McpConfig::parse(r#"{"mcpServers": {"a__b": {"command": "x"}}}"#).is_err());
        assert!(McpConfig::parse(r#"{"mcpServers": {"a": {}}}"#).is_err());
        assert!(McpConfig::parse(r#"{"mcpServers": {"a": {"url": "https://example.com/mcp"}}}"#).is_ok());
    }
}
