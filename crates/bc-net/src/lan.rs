//! Finding Borrowed Compute Macs on the local network.
//!
//! The Mac advertises its endpoint over mDNS under its own service name, with its name in
//! the endpoint's user data. A PC listening on the same network learns the Mac's code (its
//! endpoint id) without anyone typing it. Advertising tells nearby devices only what the
//! code already is: anything a Mac code unlocks still needs sign-in and the owner's approval.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result};
use iroh::endpoint_info::UserData;
use iroh::{Endpoint, EndpointId, SecretKey};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::watch;

/// Our own service name, so the PC hears only Borrowed Compute Macs and not every iroh app.
const SERVICE: &str = "borrowedcompute";
/// Prefix on the user data, so data from anything else under the same service is ignored.
const PREFIX: &str = "bc-mac:";

/// A Mac seen on this network.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NearbyMac {
    /// Its Mac code.
    pub id: String,
    pub name: String,
}

/// Starts advertising `endpoint` on the local network under `name`.
pub fn advertise(endpoint: &Endpoint, name: &str) -> Result<()> {
    // The name goes first, so the first announcement already carries it.
    set_name(endpoint, name);
    let mdns = MdnsAddressLookup::builder()
        .service_name(SERVICE)
        .build(endpoint.id())
        .context("starting local network discovery")?;
    endpoint.address_lookup().context("endpoint has no address lookup")?.add(mdns);
    Ok(())
}

/// Changes the name nearby devices see, as when the owner names the Mac at sign-in.
pub fn set_name(endpoint: &Endpoint, name: &str) {
    endpoint.set_user_data_for_address_lookup(user_data(name));
}

/// User data is at most 245 bytes, so long names are cut on a character boundary.
fn user_data(name: &str) -> Option<UserData> {
    let mut text = format!("{PREFIX}{}", name.trim());
    while text.len() > 240 {
        text.pop();
    }
    text.parse().ok()
}

fn name_of(data: &str) -> Option<String> {
    data.strip_prefix(PREFIX).map(|n| n.trim().to_string()).filter(|n| !n.is_empty())
}

/// Listens for Macs on the local network and keeps the receiver's list current. The
/// listener only hears; it advertises nothing, so it needs no key of this device's own.
/// The task stops when every receiver is dropped.
pub fn watch_nearby() -> Result<watch::Receiver<Vec<NearbyMac>>> {
    let listener_id = SecretKey::generate().public();
    let mdns = MdnsAddressLookup::builder()
        .service_name(SERVICE)
        .advertise(false)
        .discovery_cadence(Duration::from_secs(2))
        .build(listener_id)
        .context("starting local network discovery")?;
    let (tx, rx) = watch::channel(Vec::new());
    tokio::spawn(async move {
        let mut seen: BTreeMap<EndpointId, String> = BTreeMap::new();
        let mut events = mdns.subscribe().await;
        while let Some(event) = events.next().await {
            let changed = match event {
                DiscoveryEvent::Discovered { endpoint_info, .. } => {
                    match endpoint_info.data.user_data().and_then(|d| name_of(d.as_ref())) {
                        Some(name) => seen.insert(endpoint_info.endpoint_id, name.clone()).as_ref() != Some(&name),
                        None => seen.remove(&endpoint_info.endpoint_id).is_some(),
                    }
                }
                DiscoveryEvent::Expired { endpoint_id } => seen.remove(&endpoint_id).is_some(),
                _ => false,
            };
            if changed {
                let list = seen.iter().map(|(id, name)| NearbyMac { id: id.to_string(), name: name.clone() }).collect();
                if tx.send(list).is_err() {
                    break;
                }
            }
        }
    });
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_through_user_data() {
        let data = user_data("  studio ").unwrap();
        assert_eq!(name_of(data.as_ref()).as_deref(), Some("studio"));
        assert_eq!(name_of("something-else"), None);
        assert_eq!(name_of("bc-mac:  "), None);
        let long = user_data(&"é".repeat(300)).unwrap();
        assert!(long.as_ref().len() <= 245);
    }
}
