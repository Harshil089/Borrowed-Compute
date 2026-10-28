//! The running agent: one endpoint accepting jobs, plus optional cloud sync.
//! Shared by `bc-agent serve` and the menu-bar app.

use std::path::PathBuf;

use anyhow::Result;
use bc_auth::CloudClient;
use bc_net::{bind, load_or_create_key, NetConfig};
use iroh::Endpoint;
use tokio::task::JoinHandle;

use crate::events::EventSender;
use crate::guests::{Guests, GuestsView};
use crate::paired::PairedDevices;
use crate::runner::RunnerConfig;
use crate::server::{serve_connection, serve_guest, AgentConfig};

/// Files the agent keeps under its home directory.
pub struct AgentPaths {
    pub home: PathBuf,
}

impl AgentPaths {
    pub fn new(home: PathBuf) -> Self {
        AgentPaths { home }
    }
    pub fn key(&self) -> PathBuf {
        self.home.join("device.key")
    }
    pub fn paired(&self) -> PairedDevices {
        PairedDevices::new(self.home.join("paired"))
    }
    pub fn guests(&self) -> Guests {
        Guests::load(self.home.join("guests.json"))
    }
}

pub struct Agent {
    pub endpoint: Endpoint,
    config: AgentConfig,
}

impl Agent {
    /// Binds this Mac's endpoint and waits until it can be reached.
    /// `relays` replaces iroh's public relays, as with the relay the Mac app runs itself.
    pub async fn start(
        paths: &AgentPaths,
        relay_only: bool,
        relays: Option<bc_net::RelayMap>,
        events: Option<EventSender>,
    ) -> Result<Self> {
        let key = load_or_create_key(&paths.key())?;
        let endpoint = bind(key, NetConfig { relay_only, relays, ..Default::default() }).await?;
        endpoint.online().await;
        let mut config = AgentConfig::new(RunnerConfig::default(), paths.paired());
        config.events = events;
        config.guests = paths.guests();
        Ok(Agent { endpoint, config })
    }

    /// Shares the built-in cloud on `127.0.0.1:port` with other devices over iroh.
    pub fn share_cloud(mut self, port: u16) -> Self {
        self.config.cloud_port = Some(port);
        // Jobs, a guest's above all, have no business with the cloud's database API.
        self.config.runner.denied_ports.push(port);
        self
    }

    /// This Mac's code: the endpoint ID other devices enter to reach its cloud.
    pub fn code(&self) -> String {
        self.endpoint.id().to_string()
    }

    /// Running services, for the Mac app to show and stop.
    pub fn services(&self) -> &crate::services::Services {
        &self.config.services
    }

    pub fn guests(&self) -> GuestsView {
        self.config.guests.view()
    }

    pub fn create_invite(&self, name: &str) -> Result<String, String> {
        self.config.guests.create_invite(name, &self.code())
    }

    pub fn delete_invite(&self, id: &str) {
        self.config.guests.delete_invite(id)
    }

    pub fn approve_guest(&self, id: u64, code: &str) -> Result<(), String> {
        self.config.guests.approve(id, code)
    }

    pub fn deny_guest(&self, id: u64) {
        self.config.guests.deny(id)
    }

    /// Turning sharing off also stops every guest's services, so nothing of theirs keeps
    /// running. Their runs in progress finish or time out as usual.
    pub fn set_sharing(&self, on: bool) {
        self.config.guests.set_sharing(on);
        if !on {
            for guest in self.config.guests.view().guests {
                self.stop_services_of(&guest.device);
            }
        }
    }

    /// Forgets a guest and stops their services.
    pub fn remove_guest(&self, device: &str) -> bool {
        let removed = self.config.guests.remove_guest(device);
        self.stop_services_of(device);
        removed
    }

    fn stop_services_of(&self, device: &str) {
        for service in self.config.services.list(None) {
            if service.owner == device {
                let _ = self.config.services.stop(&service.job_id, None);
            }
        }
    }

    /// Accepts job and cloud connections until the endpoint closes.
    pub async fn accept_loop(&self) {
        while let Some(incoming) = self.endpoint.accept().await {
            let config = self.config.clone();
            tokio::spawn(async move {
                let result = async {
                    let conn = incoming.await?;
                    if conn.alpn() == bc_net::CLOUD_ALPN {
                        // A busy cloud turns new connections away rather than queueing them.
                        let (Some(port), Ok(_slot)) = (config.cloud_port, config.cloud_slots.clone().try_acquire_owned())
                        else {
                            conn.close(1u32.into(), b"busy");
                            return Ok(());
                        };
                        conn.set_max_concurrent_bi_streams(crate::server::MAX_CLOUD_STREAMS.into());
                        bc_net::serve_cloud(conn, port).await;
                        return Ok(());
                    }
                    if conn.alpn() == bc_net::GUEST_ALPN {
                        return serve_guest(&config, conn).await;
                    }
                    serve_connection(&config, conn).await
                }
                .await;
                if let Err(e) = result {
                    eprintln!("job connection ended: {e:#}");
                }
            });
        }
    }

    /// Starts heartbeats and paired-list sync. Abort the handle on sign-out.
    pub fn spawn_cloud_sync(&self, cloud: CloudClient) -> JoinHandle<()> {
        tokio::spawn(crate::cloud_sync::run(
            cloud,
            self.endpoint.clone(),
            self.config.paired.clone(),
            self.config.events.clone(),
        ))
    }
}
