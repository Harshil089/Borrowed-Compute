//! Finding the device that holds the account on the local network, Mac or PC.
//!
//! That device advertises its endpoint over mDNS under our own service name, with its kind
//! and name in the endpoint's user data. Another device listening on the same network
//! learns its code (its endpoint id) without anyone typing it. Advertising tells nearby
//! devices only what the code already is: anything the code unlocks still needs sign-in and
//! the owner's approval.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result};
use iroh::endpoint_info::UserData;
use iroh::{Endpoint, EndpointId, SecretKey};
use iroh_mdns_address_lookup::{DiscoveryEvent, MdnsAddressLookup};
use n0_future::StreamExt;
use tokio::sync::watch;

/// Our own service name, so we hear only Borrowed Compute devices and not every iroh app.
const SERVICE: &str = "borrowedcompute";
/// Prefix on the user data, followed by `<kind>:<name>`, so data from anything else under
/// the same service is ignored.
const PREFIX: &str = "bc-host:";
/// What Macs from before any OS could hold the account advertise: `bc-mac:<name>`.
const OLD_PREFIX: &str = "bc-mac:";

/// A device seen on this network.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NearbyDevice {
    /// Its device code.
    pub id: String,
    pub name: String,
    /// `mac`, `windows` or `other`, as `bc_proto::cloud::DeviceKind` names them.
    pub kind: String,
}

/// This machine's kind, as advertised.
fn this_kind() -> &'static str {
    if cfg!(target_os = "macos") {
        "mac"
    } else if cfg!(windows) {
        "windows"
    } else {
        "other"
    }
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

/// Changes the name nearby devices see, as when the owner names the device at sign-in.
pub fn set_name(endpoint: &Endpoint, name: &str) {
    endpoint.set_user_data_for_address_lookup(user_data(name));
}

/// User data is at most 245 bytes, so long names are cut on a character boundary.
fn user_data(name: &str) -> Option<UserData> {
    let mut text = format!("{PREFIX}{}:{}", this_kind(), name.trim());
    while text.len() > 240 {
        text.pop();
    }
    text.parse().ok()
}

/// The kind and name in a device's user data.
fn parse(data: &str) -> Option<(String, String)> {
    let (kind, name) = match data.strip_prefix(PREFIX) {
        Some(rest) => rest.split_once(':')?,
        None => ("mac", data.strip_prefix(OLD_PREFIX)?),
    };
    let name = name.trim();
    (!name.is_empty() && !kind.is_empty()).then(|| (kind.to_string(), name.to_string()))
}

/// Listens for devices on the local network and keeps the receiver's list current. The
/// listener only hears; it advertises nothing, so it needs no key of this device's own.
/// The task stops when every receiver is dropped.
pub fn watch_nearby() -> Result<watch::Receiver<Vec<NearbyDevice>>> {
    let listener_id = SecretKey::generate().public();
    let mdns = MdnsAddressLookup::builder()
        .service_name(SERVICE)
        .advertise(false)
        .discovery_cadence(Duration::from_secs(2))
        .build(listener_id)
        .context("starting local network discovery")?;
    let (tx, rx) = watch::channel(Vec::new());
    tokio::spawn(async move {
        let mut seen: BTreeMap<EndpointId, (String, String)> = BTreeMap::new();
        let mut events = mdns.subscribe().await;
        while let Some(event) = events.next().await {
            let changed = match event {
                DiscoveryEvent::Discovered { endpoint_info, .. } => {
                    match endpoint_info.data.user_data().and_then(|d| parse(d.as_ref())) {
                        Some(device) => seen.insert(endpoint_info.endpoint_id, device.clone()).as_ref() != Some(&device),
                        None => seen.remove(&endpoint_info.endpoint_id).is_some(),
                    }
                }
                DiscoveryEvent::Expired { endpoint_id } => seen.remove(&endpoint_id).is_some(),
                _ => false,
            };
            if changed {
                let list = seen
                    .iter()
                    .map(|(id, (kind, name))| NearbyDevice { id: id.to_string(), name: name.clone(), kind: kind.clone() })
                    .collect();
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
        assert_eq!(parse(data.as_ref()), Some((this_kind().into(), "studio".into())));
        assert_eq!(parse("bc-host:windows:desk: pc"), Some(("windows".into(), "desk: pc".into())));
        // Macs from before advertise without a kind.
        assert_eq!(parse("bc-mac: studio"), Some(("mac".into(), "studio".into())));
        assert_eq!(parse("something-else"), None);
        assert_eq!(parse("bc-mac:  "), None);
        let long = user_data(&"é".repeat(300)).unwrap();
        assert!(long.as_ref().len() <= 245);
    }
}
