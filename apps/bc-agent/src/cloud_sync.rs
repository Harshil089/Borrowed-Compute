//! Keeps the cloud informed that this Mac is online, and keeps the local paired list in
//! step with the cloud's approved pairings.

use std::collections::BTreeSet;
use std::time::Duration;

use bc_auth::CloudClient;
use bc_net::EndpointId;
use iroh::Endpoint;

use crate::paired::PairedDevices;

/// Revoking a pairing in the cloud takes effect within this interval.
pub const SYNC_INTERVAL: Duration = Duration::from_secs(15);

pub async fn run(cloud: CloudClient, endpoint: Endpoint, paired: PairedDevices) {
    let me = endpoint.id().to_string();
    let mut failing = false;
    loop {
        let result = sync_once(&cloud, &endpoint, &me, &paired).await;
        match (&result, failing) {
            (Err(e), false) => eprintln!("cloud sync failing, keeping last paired list: {e:#}"),
            (Ok(()), true) => eprintln!("cloud sync recovered"),
            _ => {}
        }
        failing = result.is_err();
        tokio::time::sleep(SYNC_INTERVAL).await;
    }
}

async fn sync_once(
    cloud: &CloudClient,
    endpoint: &Endpoint,
    me: &str,
    paired: &PairedDevices,
) -> anyhow::Result<()> {
    let addr = serde_json::to_value(endpoint.addr()).ok();
    cloud.heartbeat(me, addr).await?;
    let clients: BTreeSet<EndpointId> = cloud
        .paired_clients(me)
        .await?
        .iter()
        .filter_map(|id| id.parse().ok())
        .collect();
    paired.replace(&clients)
}
