//! Google sign-in through the OAuth device flow for limited-input devices, run by the
//! cloud so Google tokens never reach devices.
//!
//! Google issues installed-app clients a "secret" that it documents as not confidential;
//! it ships inside the apps like the client id. Base URLs are configurable so tests can
//! point at a fake Google.

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::github::{DeviceCode, Poll};

#[derive(Debug, Clone)]
pub struct Google {
    pub client_id: String,
    pub client_secret: String,
    /// `https://oauth2.googleapis.com`
    pub oauth_base: String,
    /// `https://openidconnect.googleapis.com`
    pub userinfo_base: String,
    http: reqwest::Client,
}

/// The Google account behind a token: its stable id and, when Google has verified it,
/// its email address.
#[derive(Debug, Deserialize)]
pub struct UserInfo {
    pub sub: String,
    pub email: Option<String>,
    #[serde(default)]
    pub email_verified: bool,
}

#[derive(Deserialize)]
struct GoogleDeviceCode {
    device_code: String,
    user_code: String,
    verification_url: String,
    expires_in: u64,
    interval: Option<u64>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    error: Option<String>,
}

impl Google {
    pub fn new(client_id: String, client_secret: String) -> Self {
        Self::with_bases(
            client_id,
            client_secret,
            "https://oauth2.googleapis.com".into(),
            "https://openidconnect.googleapis.com".into(),
        )
    }

    pub fn with_bases(client_id: String, client_secret: String, oauth_base: String, userinfo_base: String) -> Self {
        Google {
            client_id,
            client_secret,
            oauth_base,
            userinfo_base,
            http: reqwest::Client::builder()
                .user_agent("borrowed-compute-cloud")
                .build()
                .expect("http client"),
        }
    }

    pub async fn start(&self) -> Result<DeviceCode> {
        let resp = self
            .http
            .post(format!("{}/device/code", self.oauth_base))
            .form(&[("client_id", self.client_id.as_str()), ("scope", "openid email")])
            .send()
            .await?;
        let status = resp.status();
        let body = resp.text().await?;
        if !status.is_success() {
            bail!("Google device code request failed: {status} {body}");
        }
        let code: GoogleDeviceCode =
            serde_json::from_str(&body).with_context(|| format!("Google device code response: {body}"))?;
        Ok(DeviceCode {
            device_code: code.device_code,
            user_code: code.user_code,
            verification_uri: code.verification_url,
            expires_in: code.expires_in,
            interval: code.interval.unwrap_or(5),
        })
    }

    pub async fn poll(&self, device_code: &str) -> Result<Poll> {
        let resp: TokenResponse = self
            .http
            .post(format!("{}/token", self.oauth_base))
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
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
            // Google gives no new interval; RFC 8628 says to add five seconds.
            Some("slow_down") => Poll::SlowDown(10),
            Some("expired_token") => Poll::Expired,
            Some("access_denied") => Poll::Denied,
            other => bail!("Google token poll failed: {other:?}"),
        })
    }

    pub async fn user(&self, token: &str) -> Result<UserInfo> {
        let resp = self
            .http
            .get(format!("{}/v1/userinfo", self.userinfo_base))
            .bearer_auth(token)
            .send()
            .await?;
        if !resp.status().is_success() {
            bail!("Google userinfo request failed: {}", resp.status());
        }
        Ok(resp.json().await?)
    }
}
