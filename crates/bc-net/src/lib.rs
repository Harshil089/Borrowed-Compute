//! Encrypted device-to-device connections.
//!
//! Each device is an iroh endpoint whose public key is its identity. Connections try a
//! direct path and fall back to a relay, which forwards traffic it cannot read.
//! Messages are `bc-proto` frames sent over QUIC streams.

use anyhow::{Context, Result};
use bc_proto::wire;
use iroh::endpoint::{presets, Connection, RecvStream, SendStream};
use iroh::{Endpoint, RelayMap, RelayMode, SecretKey};
use serde::{de::DeserializeOwned, Serialize};

pub use iroh::{EndpointAddr, EndpointId};

/// ALPN for the job protocol. The version suffix tracks [`wire::PROTOCOL_VERSION`].
pub const JOB_ALPN: &[u8] = b"borrowed-compute/job/1";

#[derive(Debug, Clone, Default)]
pub struct NetConfig {
    /// Our own relays. `None` uses iroh's public relays, which is for development only.
    pub relays: Option<RelayMap>,
    /// Disables direct paths so every byte goes through the relay. Used to test the
    /// worst case, as in a strict corporate network.
    pub relay_only: bool,
    /// Accepts self-signed relay certificates. Only exists with the `test-utils` feature,
    /// so release builds always verify relay certificates.
    #[cfg(feature = "test-utils")]
    pub insecure_relay_tls: bool,
}

/// Binds an endpoint with this device's key, ready to dial and accept job connections.
pub async fn bind(secret: SecretKey, config: NetConfig) -> Result<Endpoint> {
    let mut builder = Endpoint::builder(presets::N0)
        .secret_key(secret)
        .alpns(vec![JOB_ALPN.to_vec()]);
    if let Some(relays) = config.relays {
        builder = builder.relay_mode(RelayMode::Custom(relays));
    }
    if config.relay_only {
        builder = builder.clear_ip_transports();
    }
    #[cfg(feature = "test-utils")]
    if config.insecure_relay_tls {
        builder = builder.ca_tls_config(iroh_relay::tls::CaTlsConfig::insecure_skip_verify());
    }
    builder.bind().await.context("binding endpoint")
}

/// Loads this device's secret key, or creates one on first run.
///
/// Phase one stores the key in a file readable only by its owner. The OS keychain
/// (Windows Credential Manager, macOS Keychain) replaces this later.
pub fn load_or_create_key(path: &std::path::Path) -> Result<SecretKey> {
    if let Ok(bytes) = std::fs::read(path) {
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("{} is not a 32-byte key", path.display()))?;
        return Ok(SecretKey::from_bytes(&bytes));
    }
    let key = SecretKey::generate();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let mut file = options
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    std::io::Write::write_all(&mut file, &key.to_bytes())?;
    Ok(key)
}

/// Directory for this device's key and settings: `BC_HOME`, or the OS config directory.
pub fn home_dir() -> Result<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("BC_HOME") {
        return Ok(dir.into());
    }
    dirs::config_dir()
        .map(|d| d.join("BorrowedCompute"))
        .context("no config directory; set BC_HOME")
}

/// True if the connection currently uses a relay rather than a direct path.
pub fn via_relay(conn: &Connection) -> bool {
    let paths = conn.paths();
    let mut any = false;
    for path in paths.iter() {
        if path.is_ip() {
            return false;
        }
        any |= path.is_relay();
    }
    any
}

pub async fn send_msg<T: Serialize>(send: &mut SendStream, msg: &T) -> Result<()> {
    let frame = wire::encode(msg)?;
    send.write_all(&frame).await.context("writing frame")?;
    Ok(())
}

pub async fn recv_msg<T: DeserializeOwned>(recv: &mut RecvStream) -> Result<T> {
    let mut prefix = [0u8; 4];
    recv.read_exact(&mut prefix)
        .await
        .context("reading frame length")?;
    let mut body = vec![0u8; wire::frame_len(prefix)?];
    recv.read_exact(&mut body)
        .await
        .context("reading frame body")?;
    Ok(wire::decode_body(&body)?)
}
