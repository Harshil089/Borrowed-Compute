//! Request and response bodies for the `bc-cloud` HTTP API (JSON).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Prefix of session tokens, so a leaked token is easy to recognize.
pub const SESSION_PREFIX: &str = "bcs_";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceFlowStart {
    pub flow_id: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval_secs: u64,
    pub expires_in_secs: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceFlowPoll {
    pub flow_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum DeviceFlowStatus {
    Pending,
    SlowDown { interval_secs: u64 },
    Expired,
    Denied,
    Complete { session_token: String, login: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Me {
    pub login: String,
}

/// A way to sign in. Identities from different providers that report the same verified
/// email address belong to one account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Github,
    Google,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Provider::Github => "github",
            Provider::Google => "google",
        }
    }

    pub fn display(self) -> &'static str {
        match self {
            Provider::Github => "GitHub",
            Provider::Google => "Google",
        }
    }
}

/// The relay devices on this cloud should use, when the Mac runs one. `None` means
/// iroh's public relays.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelayInfo {
    pub url: Option<String>,
}

/// Which sign-in providers this cloud offers. GitHub is always available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Providers {
    pub github: bool,
    pub google: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Mac,
    Windows,
    Other,
}

impl DeviceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Mac => "mac",
            DeviceKind::Windows => "windows",
            DeviceKind::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mac" => Some(DeviceKind::Mac),
            "windows" => Some(DeviceKind::Windows),
            "other" => Some(DeviceKind::Other),
            _ => None,
        }
    }

    /// The kind of the machine this code runs on.
    pub fn current() -> Self {
        if cfg!(target_os = "macos") {
            DeviceKind::Mac
        } else if cfg!(windows) {
            DeviceKind::Windows
        } else {
            DeviceKind::Other
        }
    }
}

/// Registers a device key with the signed-in account. The signature over
/// [`register_message`] proves the caller holds the key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterDevice {
    pub endpoint_id: String,
    pub name: String,
    pub kind: DeviceKind,
    /// Unix seconds. The cloud accepts a few minutes of clock skew.
    pub signed_at: u64,
    /// Hex-encoded ed25519 signature.
    pub signature: String,
}

/// Bytes a device signs to register. Binding the session token's hash means a signature
/// cannot be replayed to register the key under another account.
pub fn register_message(endpoint_id: &str, signed_at: u64, session_token_sha256_hex: &str) -> Vec<u8> {
    format!("borrowed-compute/register/v1\n{endpoint_id}\n{signed_at}\n{session_token_sha256_hex}")
        .into_bytes()
}

pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub endpoint_id: String,
    pub name: String,
    pub kind: DeviceKind,
    pub online: bool,
    pub last_seen_secs_ago: Option<i64>,
    /// The device's latest iroh `EndpointAddr`, as JSON. Lets clients dial without
    /// public address lookup.
    pub addr: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub addr: Option<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreatePairing {
    /// Endpoint id of the device asking to send jobs.
    pub client: String,
    /// Mac name or endpoint id.
    pub mac: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PairingStatus {
    Pending,
    Approved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairingInfo {
    pub id: i64,
    pub client_id: String,
    pub client_name: String,
    pub mac_id: String,
    pub mac_name: String,
    pub status: PairingStatus,
    /// Shown on both screens while pending.
    pub code: Option<String>,
    pub expires_in_secs: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovePairing {
    pub code: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedClients {
    pub clients: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_flow_status_is_tagged() {
        let json = serde_json::to_string(&DeviceFlowStatus::SlowDown { interval_secs: 10 }).unwrap();
        assert_eq!(json, r#"{"status":"slow_down","interval_secs":10}"#);
    }

    #[test]
    fn register_message_binds_all_fields() {
        let a = register_message("id", 1, "aa");
        assert_ne!(a, register_message("id", 2, "aa"));
        assert_ne!(a, register_message("id", 1, "bb"));
        assert_ne!(a, register_message("other", 1, "aa"));
    }

    #[test]
    fn hashes_hex() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
