//! Several devices at once, from this one: measure every paired device together, and
//! plan a model across them. Shared by `bc mesh` and the desktop app.

use anyhow::Result;
use bc_auth::CloudClient;
use bc_mesh::plan::{self, Device, Goal, Link, Links, Model, Options, Plan};
use bc_mesh::probe::LinkReport;
use bc_proto::cloud::PairingStatus;
use iroh::{Endpoint, EndpointAddr, EndpointId};

/// Share of a device's free memory the planner may fill, leaving room for its owner and
/// for the KV cache.
const MEMORY_SHARE: f64 = 0.85;

pub struct MeshDevice {
    pub name: String,
    pub addr: EndpointAddr,
    pub report: Result<LinkReport, String>,
}

/// Devices this one is paired with (approved), with the cloud's latest address for each.
pub async fn paired_devices(cloud: &CloudClient, me: &str) -> Result<Vec<(String, EndpointAddr)>> {
    let approved: Vec<String> = cloud
        .pairings(me)
        .await?
        .into_iter()
        .filter(|p| p.client_id == me && p.status == PairingStatus::Approved)
        .map(|p| p.mac_id)
        .collect();
    let mut out = Vec::new();
    for d in cloud.devices().await? {
        if !approved.contains(&d.endpoint_id) {
            continue;
        }
        let Ok(id) = d.endpoint_id.parse::<EndpointId>() else { continue };
        // Trust the cloud's address only for the key it claims to belong to.
        let addr = d
            .addr
            .and_then(|a| serde_json::from_value::<EndpointAddr>(a).ok())
            .filter(|a| a.id == id)
            .unwrap_or_else(|| id.into());
        out.push((d.name, addr));
    }
    Ok(out)
}

/// Probes every device at the same time over one endpoint.
pub async fn probe_all(endpoint: &Endpoint, devices: Vec<(String, EndpointAddr)>) -> Vec<MeshDevice> {
    let tasks: Vec<_> = devices
        .into_iter()
        .map(|(name, addr)| {
            let endpoint = endpoint.clone();
            tokio::spawn(async move {
                let report = bc_mesh::probe::probe(&endpoint, addr.clone()).await.map_err(|e| format!("{e:#}"));
                MeshDevice { name, addr, report }
            })
        })
        .collect();
    let mut out = Vec::with_capacity(tasks.len());
    for t in tasks {
        if let Ok(d) = t.await {
            out.push(d);
        }
    }
    out
}

/// Plans `model` over the devices that answered. Plan device indexes point into the
/// returned list.
pub fn plan_for<'a>(model: &Model, devices: &'a [MeshDevice], goal: Goal, tokens_per_pass: f64) -> (Plan, Vec<&'a MeshDevice>) {
    let usable: Vec<&MeshDevice> = devices.iter().filter(|d| d.report.is_ok()).collect();
    let reports: Vec<&LinkReport> = usable.iter().map(|d| d.report.as_ref().unwrap()).collect();
    let specs: Vec<Device> = usable
        .iter()
        .zip(&reports)
        .map(|(d, r)| Device {
            name: d.name.clone(),
            mem_bytes: (r.device.mem_available as f64 * MEMORY_SHARE) as u64,
            mem_bw: r.device.mem_bw,
        })
        .collect();
    let links = Links::from_coordinator(&reports.iter().map(|r| Link { rtt_ms: r.rtt_ms, bw: r.up_bytes_s }).collect::<Vec<_>>());
    let opts = Options { goal, tokens_per_pass, ..Default::default() };
    (plan::plan(model, &specs, &links, &opts), usable)
}

pub fn find_model(name: &str) -> Option<Model> {
    plan::catalog().into_iter().find(|m| m.name.eq_ignore_ascii_case(name))
}
