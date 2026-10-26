//! GitHub OAuth device flow, run by the cloud so the GitHub token never reaches devices.
//!
//! Base URLs are configurable so tests can point at a fake GitHub.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct GitHub {
    pub client_id: String,
    /// `https://github.com`
    pub web_base: String,
    /// `https://api.github.com`
    pub api_base: String,
    http: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct DeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    pub interval: u64,
}

#[derive(Debug, PartialEq)]
pub enum Poll {
    Pending,
    SlowDown(u64),
    Expired,
    Denied,
    Token(String),
}

#[derive(Debug, Deserialize)]
pub struct User {
    pub id: i64,
    pub login: String,
}

#[derive(Debug, Deserialize)]
struct Email {
    email: String,
    verified: bool,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
    interval: Option<u64>,
}

impl GitHub {
    pub fn new(client_id: String) -> Self {
        Self::with_bases(client_id, "https://github.com".into(), "https://api.github.com".into())
    }

    pub fn with_bases(client_id: String, web_base: String, api_base: String) -> Self {
        GitHub {
            client_id,
            web_base,
            api_base,
            http: reqwest::Client::builder()
                .user_agent("borrowed-compute-cloud")
                .build()
                .expect("http client"),
        }
    }

    pub async fn start(&self) -> Result<DeviceCode> {
        let resp = self
            .http
            .post(format!("{}/login/device/code", self.web_base))
            .header("Accept", "application/json")
            // user:email lets the cloud read verified addresses, which link a GitHub
            // sign-in to the same person's Google sign-in.
            .form(&[("client_id", self.client_id.as_str()), ("scope", "user:email")])
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            bail!("GitHub device code request failed: {status} {body}");
        }
        serde_json::from_str(&body).with_context(|| format!("GitHub device code response: {body}"))
    }

    pub async fn poll(&self, device_code: &str) -> Result<Poll> {
        let resp: TokenResponse = self
            .http
            .post(format!("{}/login/oauth/access_token", self.web_base))
            .header("Accept", "application/json")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("device_code", device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ])
            .send()
            .await?
            .json()
            .await?;
        if let Some(token) = resp.access_token {
            return Ok(Poll::Token(token));
        }
        Ok(match resp.error.as_deref() {
            Some("authorization_pending") => Poll::Pending,
            Some("slow_down") => Poll::SlowDown(resp.interval.unwrap_or(10)),
            Some("expired_token") => Poll::Expired,
            Some("access_denied") => Poll::Denied,
            other => bail!("GitHub token poll failed: {other:?}"),
        })
    }

    pub async fn user(&self, token: &str) -> Result<User> {
        let resp = self
            .http
            .get(format!("{}/user", self.api_base))
            .bearer_auth(token)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!("GitHub user request failed: {}", resp.status());
        }
        Ok(resp.json().await?)
    }

    /// The user's verified email addresses. Empty when the token lacks the `user:email`
    /// scope, as for sign-ins approved before it was asked for.
    pub async fn verified_emails(&self, token: &str) -> Result<Vec<String>> {
        let resp = self
            .http
            .get(format!("{}/user/emails", self.api_base))
            .bearer_auth(token)
            .header("Accept", "application/vnd.github+json")
            .send()
            .await?;
        if !resp.status().is_success() {
            return Ok(Vec::new());
        }
        let emails: Vec<Email> = resp.json().await?;
        Ok(emails.into_iter().filter(|e| e.verified).map(|e| e.email.to_lowercase()).collect())
    }
}
