//! Sharing the Mac's built-in cloud with the user's other devices through Tailscale.
//!
//! `tailscale serve` gives the cloud a private HTTPS address inside the user's tailnet,
//! such as `https://studio.tail1234.ts.net`, reachable only from their own devices.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;

/// Where the Tailscale CLI lives. A GUI app does not inherit the shell's PATH, so check
/// the usual install locations directly.
const CANDIDATES: &[&str] = &[
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/opt/homebrew/bin/tailscale",
    "/usr/local/bin/tailscale",
];

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum TailscaleState {
    NotInstalled,
    /// Installed but not connected: logged out, stopped, or still starting.
    NotRunning { detail: String },
    /// Connected, but the tailnet has not enabled HTTPS certificates, which serve needs.
    HttpsDisabled,
    /// Connected and ready; the cloud is not shared yet.
    Ready { host: String },
    Sharing { url: String },
    Error { message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum ShareResult {
    Shared { url: String },
    /// Tailscale wants the tailnet admin to allow serve; open `url` to do it.
    NeedsApproval { url: String },
}

fn cli() -> Option<PathBuf> {
    CANDIDATES.iter().map(PathBuf::from).find(|p| p.exists())
}

async fn run(cli: &Path, args: &[&str]) -> anyhow::Result<String> {
    let out = tokio::time::timeout(
        Duration::from_secs(10),
        Command::new(cli).args(args).stdin(Stdio::null()).output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("tailscale {} timed out", args.join(" ")))??;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        anyhow::bail!("tailscale {} failed: {}", args.join(" "), err.trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Status {
    backend_state: String,
    #[serde(rename = "Self")]
    me: Option<Peer>,
    #[serde(default)]
    cert_domains: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct Peer {
    #[serde(rename = "DNSName")]
    dns_name: String,
}

/// Reads `tailscale status --json` and `tailscale serve status --json`.
pub async fn state(local_port: u16) -> TailscaleState {
    let Some(cli) = cli() else { return TailscaleState::NotInstalled };
    let status = match run(&cli, &["status", "--json"]).await {
        Ok(json) => json,
        // A stopped Tailscale makes `status` fail; that is "not running", not an error.
        Err(e) => return TailscaleState::NotRunning { detail: format!("{e:#}") },
    };
    let serve = run(&cli, &["serve", "status", "--json"]).await.unwrap_or_default();
    interpret(&status, &serve, local_port)
}

fn interpret(status_json: &str, serve_json: &str, local_port: u16) -> TailscaleState {
    let status: Status = match serde_json::from_str(status_json) {
        Ok(s) => s,
        Err(e) => return TailscaleState::Error { message: format!("unexpected tailscale status: {e}") },
    };
    if status.backend_state != "Running" {
        return TailscaleState::NotRunning { detail: status.backend_state };
    }
    let host = status
        .me
        .map(|p| p.dns_name.trim_end_matches('.').to_string())
        .filter(|h| !h.is_empty());
    let Some(host) = host else {
        return TailscaleState::Error { message: "Tailscale did not report this Mac's name".into() };
    };
    if status.cert_domains.unwrap_or_default().is_empty() {
        return TailscaleState::HttpsDisabled;
    }
    if serves_port(serve_json, local_port) {
        TailscaleState::Sharing { url: format!("https://{host}") }
    } else {
        TailscaleState::Ready { host }
    }
}

/// True if serve proxies some HTTPS address to the local cloud port.
fn serves_port(serve_json: &str, port: u16) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(serve_json) else { return false };
    let targets = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    value
        .get("Web")
        .and_then(|w| w.as_object())
        .into_iter()
        .flat_map(|web| web.values())
        .filter_map(|site| site.get("Handlers").and_then(|h| h.as_object()))
        .flat_map(|handlers| handlers.values())
        .filter_map(|h| h.get("Proxy").and_then(|p| p.as_str()))
        .any(|proxy| targets.iter().any(|t| proxy.ends_with(t.as_str())))
}

/// Shares `http://127.0.0.1:<port>` at the Mac's tailnet HTTPS address.
pub async fn share(local_port: u16) -> anyhow::Result<ShareResult> {
    let cli = cli().ok_or_else(|| anyhow::anyhow!("Tailscale is not installed"))?;
    let target = format!("http://127.0.0.1:{local_port}");
    let mut child = Command::new(&cli)
        .args(["serve", "--bg", &target])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    // When the tailnet has not allowed serve, the CLI prints an approval link and waits.
    // Watch both streams for it rather than waiting for the process.
    let mut out = BufReader::new(child.stdout.take().expect("piped")).lines();
    let mut err = BufReader::new(child.stderr.take().expect("piped")).lines();
    let mut seen = String::new();
    let deadline = tokio::time::sleep(Duration::from_secs(20));
    tokio::pin!(deadline);
    let (mut out_done, mut err_done) = (false, false);
    loop {
        let line = tokio::select! {
            l = out.next_line(), if !out_done => l?.or_else(|| { out_done = true; None }),
            l = err.next_line(), if !err_done => l?.or_else(|| { err_done = true; None }),
            status = child.wait(), if out_done && err_done => {
                if status?.success() {
                    break;
                }
                anyhow::bail!("tailscale serve failed: {}", seen.trim());
            }
            _ = &mut deadline => anyhow::bail!("tailscale serve did not finish: {}", seen.trim()),
        };
        if let Some(line) = line {
            if let Some(url) = approval_link(&line) {
                return Ok(ShareResult::NeedsApproval { url });
            }
            seen.push_str(&line);
            seen.push('\n');
        }
    }
    match state(local_port).await {
        TailscaleState::Sharing { url } => Ok(ShareResult::Shared { url }),
        other => anyhow::bail!("tailscale serve ran, but sharing is not active ({other:?})"),
    }
}

/// The tailnet HTTPS port the relay is shared on, next to the cloud on 443. Tailscale
/// serve allows HTTPS on 443, 8443 and 10000.
pub const RELAY_HTTPS_PORT: u16 = 8443;

/// The relay's address for a cloud shared at `cloud_url` (`https://<mac>.ts.net`).
pub fn relay_url(cloud_url: &str) -> String {
    format!("{}:{RELAY_HTTPS_PORT}/", cloud_url.trim_end_matches('/'))
}

/// Shares the relay listening on `http://127.0.0.1:<relay_port>` at the Mac's tailnet
/// address on port 8443. Does nothing when it is already shared.
pub async fn share_relay(relay_port: u16) -> anyhow::Result<()> {
    let cli = cli().ok_or_else(|| anyhow::anyhow!("Tailscale is not installed"))?;
    let serve = run(&cli, &["serve", "status", "--json"]).await.unwrap_or_default();
    if serves_port(&serve, relay_port) {
        return Ok(());
    }
    let https = format!("--https={RELAY_HTTPS_PORT}");
    let target = format!("http://127.0.0.1:{relay_port}");
    run(&cli, &["serve", "--bg", &https, &target]).await.map(|_| ())
}

/// Stops sharing the cloud and the relay. Other serve settings are left alone.
pub async fn stop() -> anyhow::Result<()> {
    let cli = cli().ok_or_else(|| anyhow::anyhow!("Tailscale is not installed"))?;
    run(&cli, &["serve", "--https=443", "off"]).await?;
    let relay = format!("--https={RELAY_HTTPS_PORT}");
    let _ = run(&cli, &["serve", &relay, "off"]).await;
    Ok(())
}

fn approval_link(line: &str) -> Option<String> {
    line.split_whitespace()
        .find(|w| w.starts_with("https://login.tailscale.com/"))
        .map(|w| w.trim_end_matches(['.', ',']).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATUS: &str = r#"{
        "Version": "1.90.0",
        "BackendState": "Running",
        "Self": { "DNSName": "studio.tail1234.ts.net.", "HostName": "studio", "Online": true },
        "CertDomains": ["studio.tail1234.ts.net"],
        "CurrentTailnet": { "Name": "you@example.com", "MagicDNSSuffix": "tail1234.ts.net", "MagicDNSEnabled": true }
    }"#;
    const SERVING: &str = r#"{
        "TCP": { "443": { "HTTPS": true } },
        "Web": { "studio.tail1234.ts.net:443": { "Handlers": { "/": { "Proxy": "http://127.0.0.1:8080" } } } }
    }"#;

    #[test]
    fn reads_sharing_state() {
        assert_eq!(
            interpret(STATUS, SERVING, 8080),
            TailscaleState::Sharing { url: "https://studio.tail1234.ts.net".into() }
        );
        assert_eq!(
            interpret(STATUS, "{}", 8080),
            TailscaleState::Ready { host: "studio.tail1234.ts.net".into() }
        );
        // Serving something else does not count.
        assert_eq!(
            interpret(STATUS, &SERVING.replace("8080", "3000"), 8080),
            TailscaleState::Ready { host: "studio.tail1234.ts.net".into() }
        );
    }

    #[test]
    fn reads_not_ready_states() {
        let stopped = STATUS.replace("\"Running\"", "\"NeedsLogin\"");
        assert!(matches!(interpret(&stopped, "", 8080), TailscaleState::NotRunning { .. }));
        let no_https = STATUS.replace("\"CertDomains\": [\"studio.tail1234.ts.net\"],", "");
        assert_eq!(interpret(&no_https, "", 8080), TailscaleState::HttpsDisabled);
        assert!(matches!(interpret("not json", "", 8080), TailscaleState::Error { .. }));
    }

    #[test]
    fn finds_approval_link() {
        let line = "Serve is not enabled on your tailnet. To enable, visit: https://login.tailscale.com/f/serve?node=abc123.";
        assert_eq!(approval_link(line).as_deref(), Some("https://login.tailscale.com/f/serve?node=abc123"));
        assert_eq!(approval_link("Available within your tailnet: https://studio.ts.net"), None);
    }
}
