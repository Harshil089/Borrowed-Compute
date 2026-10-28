//! Macs shared with this device by someone on another account ([`bc_proto::guest`]).
//!
//! Joining takes only the invite: no sign-in, no cloud. The Mac is reached by its
//! endpoint ID through iroh's discovery, and this device's list of shared Macs lives in
//! `shared-macs.json` in the client home, so runs can find them by name without a cloud.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use bc_net::{bind, recv_msg, send_msg, EndpointAddr, EndpointId, NetConfig, GUEST_ALPN};
use iroh::{Endpoint, SecretKey};
use bc_proto::guest::{GuestMsg, GuestReply, Invite};
use serde::{Deserialize, Serialize};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedMac {
    /// The Mac's computer name, made unique among this device's shared Macs.
    pub name: String,
    pub id: String,
    /// Unix seconds.
    pub added: u64,
}

fn path(home: &Path) -> PathBuf {
    home.join("shared-macs.json")
}

pub fn list(home: &Path) -> Vec<SharedMac> {
    std::fs::read(path(home)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save(home: &Path, macs: &[SharedMac]) -> Result<()> {
    std::fs::create_dir_all(home)?;
    std::fs::write(path(home), serde_json::to_vec_pretty(macs)?).context("saving shared Macs")
}

/// The shared Mac called `target`, by name (any case) or endpoint ID.
pub fn find(home: &Path, target: &str) -> Option<SharedMac> {
    list(home).into_iter().find(|m| m.id == target || m.name.eq_ignore_ascii_case(target))
}

/// Forgets a shared Mac on this device. The Mac's owner still lists this device until
/// they remove it there.
pub fn remove(home: &Path, target: &str) -> Result<bool> {
    let mut macs = list(home);
    let before = macs.len();
    macs.retain(|m| m.id != target && !m.name.eq_ignore_ascii_case(target));
    save(home, &macs)?;
    Ok(macs.len() != before)
}

/// Adds or renames the entry for `id`, keeping names unique so `bc use <name>` is never
/// ambiguous.
fn remember(home: &Path, id: &str, mac_name: &str) -> Result<SharedMac> {
    let mut macs = list(home);
    macs.retain(|m| m.id != id);
    let base = match mac_name.trim() {
        "" => format!("shared-{}", &id[..6]),
        n => n.to_string(),
    };
    let mut name = base.clone();
    let mut n = 2;
    while macs.iter().any(|m| m.name.eq_ignore_ascii_case(&name)) {
        name = format!("{base}-{n}");
        n += 1;
    }
    let added = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mac = SharedMac { name, id: id.to_string(), added };
    macs.push(mac.clone());
    save(home, &macs)?;
    Ok(mac)
}

/// Presents `invite` to its Mac and waits while the owner approves it there. `on_code` gets
/// the code to read to the owner. Returns the Mac as saved in this device's list.
pub async fn join(
    home: &Path,
    key: SecretKey,
    net: NetConfig,
    invite: &str,
    device_name: &str,
    on_code: impl FnOnce(&str),
) -> Result<SharedMac> {
    let invite = Invite::parse(invite).context("that is not an invite; it starts with bc-invite:")?;
    let id: EndpointId = invite.mac.parse().context("the invite names no valid Mac")?;
    let endpoint = bind(key, net).await?;
    let mac_name = request_access(&endpoint, id.into(), &invite.token, device_name, on_code).await;
    endpoint.close().await;
    remember(home, &id.to_string(), &mac_name?)
}

/// The exchange behind [`join`] on an existing endpoint. Returns the Mac's name.
pub async fn request_access(
    endpoint: &Endpoint,
    mac: EndpointAddr,
    token: &str,
    device_name: &str,
    on_code: impl FnOnce(&str),
) -> Result<String> {
    let conn = tokio::time::timeout(CONNECT_TIMEOUT, endpoint.connect(mac, GUEST_ALPN))
        .await
        .map_err(|_| anyhow::anyhow!("the Mac did not answer within {}s; is it on and is its app open?", CONNECT_TIMEOUT.as_secs()))?
        .context("connecting to the Mac")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &GuestMsg::Request { token: token.to_string(), device_name: device_name.to_string() }).await?;
    let mut on_code = Some(on_code);
    let mac_name = loop {
        match recv_msg::<GuestReply>(&mut recv).await.context("the Mac closed the request")? {
            GuestReply::Pending { code } => {
                if let Some(f) = on_code.take() {
                    f(&code);
                }
            }
            GuestReply::Approved { mac_name } => break mac_name,
            GuestReply::Rejected { reason } => bail!("{reason}"),
        }
    };
    conn.close(0u32.into(), b"done");
    Ok(mac_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_names_unique_and_finds_by_name_or_id() {
        let dir = tempfile::tempdir().unwrap();
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        assert_eq!(remember(dir.path(), &a, "Studio").unwrap().name, "Studio");
        assert_eq!(remember(dir.path(), &b, "studio").unwrap().name, "studio-2");
        assert_eq!(remember(dir.path(), &a, "Studio").unwrap().name, "Studio", "joining again keeps one entry");
        assert_eq!(list(dir.path()).len(), 2);
        assert_eq!(find(dir.path(), "STUDIO-2").unwrap().id, b);
        assert_eq!(find(dir.path(), &a).unwrap().name, "Studio");
        assert!(remove(dir.path(), "studio").unwrap());
        assert!(find(dir.path(), &a).is_none());
        assert_eq!(remember(dir.path(), &a, " ").unwrap().name, "shared-aaaaaa");
    }
}
