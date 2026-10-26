//! Client for the cloud API, shared by `bc` and `bc-agent`.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use bc_proto::cloud::{
    register_message, sha256_hex, ApiErrorBody, ApprovePairing, CreatePairing, DeviceFlowPoll,
    DeviceFlowStart, DeviceFlowStatus, DeviceInfo, DeviceKind, Heartbeat, Me, PairedClients,
    PairingInfo, RegisterDevice, Provider, Providers};
use iroh_base::SecretKey;
use serde::de::DeserializeOwned;
use serde::Serialize;

/// Where the cloud runs when nothing else is configured: the Mac app's built-in cloud.
pub const DEFAULT_CLOUD_URL: &str = "http://127.0.0.1:8080";

/// Cloud base URL for a device: `BC_CLOUD_URL`, else the address saved in the device's home
/// folder (set from the app's settings), else [`DEFAULT_CLOUD_URL`].
pub fn cloud_url_for(home: &Path) -> String {
    std::env::var("BC_CLOUD_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .or_else(|| {
            std::fs::read_to_string(home.join("cloud-url"))
                .ok()
                .map(|u| u.trim().to_string())
                .filter(|u| !u.is_empty())
        })
        .unwrap_or_else(|| DEFAULT_CLOUD_URL.to_string())
}

/// Checks and normalizes a cloud address typed by a person.
pub fn normalize_cloud_url(input: &str) -> Result<String> {
    let url = input.trim().trim_end_matches('/');
    let url = if url.contains("://") { url.to_string() } else { format!("https://{url}") };
    let parsed = reqwest::Url::parse(&url).with_context(|| format!("{input:?} is not a web address"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        bail!("{input:?} must be an http or https address");
    }
    if parsed.path() != "/" || parsed.query().is_some() {
        bail!("give only the address, without a path: for example https://my-mac.example.ts.net");
    }
    Ok(url)
}

/// Saves the cloud address for this device. An empty address returns to the default.
pub fn save_cloud_url(home: &Path, url: &str) -> Result<String> {
    std::fs::create_dir_all(home)?;
    let path = home.join("cloud-url");
    if url.trim().is_empty() {
        let _ = std::fs::remove_file(path);
        return Ok(DEFAULT_CLOUD_URL.to_string());
    }
    let url = normalize_cloud_url(url)?;
    std::fs::write(&path, format!("{url}\n")).with_context(|| format!("writing {}", path.display()))?;
    Ok(url)
}

#[derive(Clone)]
pub struct CloudClient {
    base: String,
    token: Option<String>,
    http: reqwest::Client,
}

impl CloudClient {
    pub fn new(base: impl Into<String>, token: Option<String>) -> Self {
        CloudClient {
            base: base.into().trim_end_matches('/').to_string(),
            token,
            http: reqwest::Client::builder()
                .user_agent("borrowed-compute")
                .timeout(Duration::from_secs(30))
                .build()
                .expect("http client"),
        }
    }

    /// Client for `base` using the session saved in `home`, if any.
    pub fn from_home(base: impl Into<String>, home: &Path) -> Self {
        Self::new(base, load_session(home))
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// Confirms a Borrowed Compute cloud answers at this address.
    pub async fn health(&self) -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Health {
            service: String,
        }
        let health: Health = self.get("/v1/health").await?;
        if health.service != "borrowed-compute-cloud" {
            bail!("{} answered, but it is not a Borrowed Compute cloud", self.base);
        }
        Ok(())
    }

    async fn call<T: DeserializeOwned>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&(impl Serialize + ?Sized)>,
    ) -> Result<T> {
        let mut req = self.http.request(method, format!("{}{path}", self.base));
        if let Some(token) = &self.token {
            req = req.bearer_auth(token);
        }
        if let Some(body) = body {
            req = req.json(body);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("cannot reach the cloud at {}", self.base))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            let msg = serde_json::from_str::<ApiErrorBody>(&text)
                .map(|e| e.error)
                .unwrap_or(text);
            bail!("{msg} ({status})");
        }
        let text = if text.is_empty() { "null" } else { &text };
        serde_json::from_str(text).with_context(|| format!("unexpected response from {path}"))
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        self.call(reqwest::Method::GET, path, None::<&()>).await
    }

    async fn post<T: DeserializeOwned>(&self, path: &str, body: &impl Serialize) -> Result<T> {
        self.call(reqwest::Method::POST, path, Some(body)).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.call::<()>(reqwest::Method::DELETE, path, None::<&()>).await
    }

    /// Signs in with GitHub device flow. `show` receives the code and URL to show the
    /// user. Returns the session token and the account name.
    pub async fn login(&mut self, show: impl FnOnce(&DeviceFlowStart)) -> Result<(String, String)> {
        self.login_with(Provider::Github, show).await
    }

    /// Signs in with `provider`'s device flow. Identities from different providers that
    /// share a verified email address sign in to the same account.
    pub async fn login_with(
        &mut self,
        provider: Provider,
        show: impl FnOnce(&DeviceFlowStart),
    ) -> Result<(String, String)> {
        let path = match provider {
            Provider::Github => "/v1/auth/device",
            Provider::Google => "/v1/auth/google/device",
        };
        let start: DeviceFlowStart = self.post(path, &()).await?;
        show(&start);
        let mut interval = Duration::from_secs(start.interval_secs.max(1));
        loop {
            tokio::time::sleep(interval).await;
            let poll = DeviceFlowPoll { flow_id: start.flow_id.clone() };
            match self.post::<DeviceFlowStatus>("/v1/auth/device/poll", &poll).await? {
                DeviceFlowStatus::Pending => {}
                DeviceFlowStatus::SlowDown { interval_secs } => {
                    interval = Duration::from_secs(interval_secs.max(interval.as_secs() + 5));
                }
                DeviceFlowStatus::Expired => bail!("the sign-in code expired; run login again"),
                DeviceFlowStatus::Denied => bail!("sign-in was denied on {}", provider.display()),
                DeviceFlowStatus::Complete { session_token, login } => {
                    self.token = Some(session_token.clone());
                    return Ok((session_token, login));
                }
            }
        }
    }

    pub async fn logout(&self) -> Result<()> {
        self.call::<()>(reqwest::Method::POST, "/v1/auth/logout", None::<&()>).await
    }

    /// Which sign-in providers the cloud offers. Clouds older than Google sign-in answer
    /// 404, which means GitHub only.
    pub async fn providers(&self) -> Result<Providers> {
        match self.get::<Providers>("/v1/auth/providers").await {
            Ok(p) => Ok(p),
            Err(e) if format!("{e:#}").contains("404") => Ok(Providers { github: true, google: false }),
            Err(e) => Err(e),
        }
    }

    pub async fn me(&self) -> Result<Me> {
        self.get("/v1/me").await
    }

    /// Registers this device's key, proving possession with a signature bound to the
    /// current session.
    pub async fn register(&self, key: &SecretKey, name: &str, kind: DeviceKind) -> Result<DeviceInfo> {
        let token = self.token.as_deref().context("sign in first")?;
        let endpoint_id = key.public().to_string();
        let signed_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let message = register_message(&endpoint_id, signed_at, &sha256_hex(token.as_bytes()));
        let body = RegisterDevice {
            endpoint_id,
            name: name.to_string(),
            kind,
            signed_at,
            signature: hex::encode(key.sign(&message).to_bytes()),
        };
        self.post("/v1/devices", &body).await
    }

    pub async fn devices(&self) -> Result<Vec<DeviceInfo>> {
        self.get("/v1/devices").await
    }

    /// Finds a device on this account by name or endpoint id.
    pub async fn find_device(&self, name_or_id: &str) -> Result<DeviceInfo> {
        self.devices()
            .await?
            .into_iter()
            .find(|d| d.name == name_or_id || d.endpoint_id == name_or_id)
            .with_context(|| format!("no device named {name_or_id:?} on this account"))
    }

    pub async fn remove_device(&self, endpoint_id: &str) -> Result<()> {
        self.delete(&format!("/v1/devices/{endpoint_id}")).await
    }

    pub async fn heartbeat(&self, endpoint_id: &str, addr: Option<serde_json::Value>) -> Result<()> {
        self.post::<()>(&format!("/v1/devices/{endpoint_id}/heartbeat"), &Heartbeat { addr })
            .await
    }

    pub async fn request_pairing(&self, client: &str, mac: &str) -> Result<PairingInfo> {
        let body = CreatePairing { client: client.to_string(), mac: mac.to_string() };
        self.post("/v1/pairings", &body).await
    }

    pub async fn pairings(&self, device: &str) -> Result<Vec<PairingInfo>> {
        self.get(&format!("/v1/pairings?device={device}")).await
    }

    pub async fn approve(&self, pairing_id: i64, code: &str) -> Result<PairingInfo> {
        self.post(
            &format!("/v1/pairings/{pairing_id}/approve"),
            &ApprovePairing { code: code.to_string() },
        )
        .await
    }

    pub async fn unpair(&self, pairing_id: i64) -> Result<()> {
        self.delete(&format!("/v1/pairings/{pairing_id}")).await
    }

    pub async fn paired_clients(&self, mac: &str) -> Result<Vec<String>> {
        Ok(self
            .get::<PairedClients>(&format!("/v1/devices/{mac}/paired"))
            .await?
            .clients)
    }
}

fn session_path(home: &Path) -> PathBuf {
    home.join("session")
}

/// Phase one keeps the session token in a file readable only by its owner. The OS
/// keychain replaces this later.
pub fn save_session(home: &Path, token: &str) -> Result<()> {
    std::fs::create_dir_all(home)?;
    let path = session_path(home);
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options.open(&path).with_context(|| format!("writing {}", path.display()))?;
    std::io::Write::write_all(&mut file, token.as_bytes())?;
    Ok(())
}

pub fn load_session(home: &Path) -> Option<String> {
    std::fs::read_to_string(session_path(home))
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

pub fn clear_session(home: &Path) {
    let _ = std::fs::remove_file(session_path(home));
}

/// A readable default device name, such as the computer's host name.
pub fn default_device_name() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
        })
        .map(|n| n.trim().trim_end_matches(".local").to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "my-device".into())
}

/// True if the cloud rejected the session, meaning the device must sign in again (for
/// example after switching to a different cloud).
pub fn is_unauthorized(err: &anyhow::Error) -> bool {
    err.to_string().ends_with("(401 Unauthorized)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cloud_addresses() {
        assert_eq!(normalize_cloud_url("my-mac.tail1234.ts.net").unwrap(), "https://my-mac.tail1234.ts.net");
        assert_eq!(normalize_cloud_url(" https://my-mac.tail1234.ts.net/ ").unwrap(), "https://my-mac.tail1234.ts.net");
        assert_eq!(normalize_cloud_url("http://127.0.0.1:8080").unwrap(), "http://127.0.0.1:8080");
        assert!(normalize_cloud_url("ftp://x").is_err());
        assert!(normalize_cloud_url("https://x/v1/health").is_err());
        assert!(normalize_cloud_url("not a url at all").is_err());
    }

    #[test]
    fn saved_address_is_used_and_cleared() {
        let dir = std::env::temp_dir().join(format!("bc-auth-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // BC_CLOUD_URL would win; this test relies on it being unset in the test run.
        if std::env::var("BC_CLOUD_URL").is_ok() {
            return;
        }
        assert_eq!(cloud_url_for(&dir), DEFAULT_CLOUD_URL);
        save_cloud_url(&dir, "my-mac.tail1234.ts.net").unwrap();
        assert_eq!(cloud_url_for(&dir), "https://my-mac.tail1234.ts.net");
        save_cloud_url(&dir, "").unwrap();
        assert_eq!(cloud_url_for(&dir), DEFAULT_CLOUD_URL);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
