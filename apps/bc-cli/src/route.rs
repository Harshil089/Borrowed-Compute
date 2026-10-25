//! Remembers, per Mac, that its direct path recently failed, so later runs go through the
//! relay first instead of waiting on a path that will not connect (a VPN on the Mac, for
//! example, can block direct paths entirely).
//!
//! A memory is forgotten after [`RELAY_FIRST_FOR`], when the Mac's published IP addresses
//! change (it moved networks), or when a direct run succeeds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bc_net::EndpointAddr;
use serde::{Deserialize, Serialize};

/// How long a failed direct path sends runs to the relay first.
pub const RELAY_FIRST_FOR: Duration = Duration::from_secs(60 * 60);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Entry {
    failed_at: u64,
    /// The Mac's IP addresses when it failed, without ports: ports change on every agent
    /// restart, while a changed address means the Mac is on a different network.
    addrs: String,
}

/// Route memory stored as JSON in the client's home folder. A failure to read or write it
/// never fails a run; it only costs the wait on the direct path.
#[derive(Debug, Clone)]
pub struct RouteMemory {
    path: Option<PathBuf>,
}

impl RouteMemory {
    pub fn at(home: &Path) -> Self {
        RouteMemory { path: Some(home.join("route-memory.json")) }
    }

    /// Remembers nothing; every run tries the direct path first.
    pub fn none() -> Self {
        RouteMemory { path: None }
    }

    pub fn relay_first(&self, mac: &EndpointAddr) -> bool {
        self.relay_first_at(mac, now())
    }

    pub fn direct_failed(&self, mac: &EndpointAddr) {
        self.direct_failed_at(mac, now());
    }

    pub fn direct_worked(&self, mac: &EndpointAddr) {
        let mut entries = self.load();
        if entries.remove(&mac.id.to_string()).is_some() {
            self.save(&entries);
        }
    }

    fn relay_first_at(&self, mac: &EndpointAddr, now: u64) -> bool {
        self.load().get(&mac.id.to_string()).is_some_and(|e| {
            now.saturating_sub(e.failed_at) < RELAY_FIRST_FOR.as_secs() && e.addrs == fingerprint(mac)
        })
    }

    fn direct_failed_at(&self, mac: &EndpointAddr, now: u64) {
        let mut entries = self.load();
        entries.insert(mac.id.to_string(), Entry { failed_at: now, addrs: fingerprint(mac) });
        self.save(&entries);
    }

    fn load(&self) -> BTreeMap<String, Entry> {
        self.path
            .as_ref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    fn save(&self, entries: &BTreeMap<String, Entry>) {
        let Some(path) = &self.path else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(json) = serde_json::to_vec_pretty(entries) {
            let _ = std::fs::write(path, json);
        }
    }
}

fn fingerprint(mac: &EndpointAddr) -> String {
    let mut ips: Vec<String> = mac.ip_addrs().map(|a| a.ip().to_string()).collect();
    ips.sort();
    ips.dedup();
    ips.join(",")
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::{SecretKey, TransportAddr};
    use std::collections::BTreeSet;
    use std::net::SocketAddr;

    fn mac(key: &SecretKey, ips: &[&str]) -> EndpointAddr {
        let addrs: BTreeSet<TransportAddr> =
            ips.iter().map(|ip| TransportAddr::Ip(ip.parse::<SocketAddr>().unwrap())).collect();
        EndpointAddr { id: key.public(), addrs }
    }

    #[test]
    fn remembers_until_expiry_address_change_or_success() {
        let dir = tempfile::tempdir().unwrap();
        let memory = RouteMemory::at(dir.path());
        let key = SecretKey::generate();
        let here = mac(&key, &["10.0.0.5:4000", "159.26.101.3:4000"]);
        let t = 1_000_000;

        assert!(!memory.relay_first_at(&here, t), "nothing remembered yet");
        memory.direct_failed_at(&here, t);
        assert!(memory.relay_first_at(&here, t + 60));

        // An agent restart changes ports, not the network: still relay first.
        let restarted = mac(&key, &["10.0.0.5:5111", "159.26.101.3:5111"]);
        assert!(memory.relay_first_at(&restarted, t + 60));

        // A new address means a new network: try direct again.
        let moved = mac(&key, &["192.168.1.20:4000"]);
        assert!(!memory.relay_first_at(&moved, t + 60));

        assert!(!memory.relay_first_at(&here, t + RELAY_FIRST_FOR.as_secs()), "expired");

        memory.direct_worked(&here);
        assert!(!memory.relay_first_at(&here, t + 60), "a direct success forgets");
    }

    #[test]
    fn other_macs_and_no_memory_are_unaffected() {
        let dir = tempfile::tempdir().unwrap();
        let memory = RouteMemory::at(dir.path());
        let a = mac(&SecretKey::generate(), &["10.0.0.5:4000"]);
        let b = mac(&SecretKey::generate(), &["10.0.0.5:4000"]);
        memory.direct_failed_at(&a, 10);
        assert!(!memory.relay_first_at(&b, 20));

        let none = RouteMemory::none();
        none.direct_failed_at(&a, 10);
        assert!(!none.relay_first_at(&a, 20));
    }
}
