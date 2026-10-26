//! Port forwarding: a service's ports on the Mac become the same ports on this PC's
//! localhost, so a dev server running on the Mac opens in the PC's browser at
//! `http://localhost:5173`.
//!
//! [`forward_services`] asks the Mac every two seconds which ports this device's services
//! listen on, and forwards each new one. Each local TCP connection is carried as its own
//! stream on one connection per service. The Mac only accepts ports the service itself
//! listens on, so nothing else on the Mac's localhost is reachable.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{bail, Result};
use bc_net::{recv_msg, send_msg, EndpointAddr, NetConfig};
use bc_proto::wire::{AgentMsg, ClientMsg, ServiceInfo};
use iroh::endpoint::Connection;
use iroh::{Endpoint, SecretKey};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;

use crate::route::RouteMemory;
use crate::run::{endpoint_for, Session};

const POLL: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forwarded {
    pub job_id: String,
    /// The service's port on the Mac.
    pub remote: u16,
    /// Where it is on this PC: the same port when it was free.
    pub local: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForwardEvent {
    Opened(Forwarded),
    /// The service ended or stopped being reachable; its ports are closed here.
    Closed { job_id: String },
}

struct Active {
    session: Session,
    /// Remote port to local port.
    ports: HashMap<u16, u16>,
    listeners: JoinSet<()>,
}

/// Forwards the ports of every service this device runs on `mac` until `cancel` fires.
pub async fn forward_services(
    key: &SecretKey,
    net: NetConfig,
    mac: EndpointAddr,
    memory: &RouteMemory,
    on_event: impl Fn(ForwardEvent),
    cancel: impl std::future::Future<Output = ()>,
) -> Result<()> {
    let endpoint = endpoint_for(key, net, &mac, memory).await?;
    tokio::pin!(cancel);
    let mut active: HashMap<String, Active> = HashMap::new();
    loop {
        // A failed poll (the Mac asleep, a network change) keeps what is open and tries again.
        if let Ok(services) = list(&endpoint, mac.clone()).await {
            update(&endpoint, &mac, &mut active, services, &on_event).await;
        }
        tokio::select! {
            _ = &mut cancel => break,
            _ = tokio::time::sleep(POLL) => {}
        }
    }
    for (_, a) in active.drain() {
        a.session.close().await;
    }
    endpoint.close().await;
    Ok(())
}

async fn list(endpoint: &Endpoint, mac: EndpointAddr) -> Result<Vec<ServiceInfo>> {
    let mut session = Session::connect(endpoint, mac).await?;
    let reply = session.ask(&ClientMsg::ListServices).await;
    session.close().await;
    match reply? {
        AgentMsg::Services(list) => Ok(list),
        other => bail!("unexpected reply to ListServices {other:?}"),
    }
}

async fn update(
    endpoint: &Endpoint,
    mac: &EndpointAddr,
    active: &mut HashMap<String, Active>,
    services: Vec<ServiceInfo>,
    on_event: &impl Fn(ForwardEvent),
) {
    let gone: Vec<String> = active
        .iter()
        .filter(|(id, a)| a.session.conn.close_reason().is_some() || !services.iter().any(|s| &s.job_id == *id))
        .map(|(id, _)| id.clone())
        .collect();
    for id in gone {
        if let Some(a) = active.remove(&id) {
            a.session.close().await;
        }
        on_event(ForwardEvent::Closed { job_id: id });
    }

    for service in services.into_iter().filter(|s| !s.ports.is_empty()) {
        if !active.contains_key(&service.job_id) {
            let Ok(mut session) = Session::connect(endpoint, mac.clone()).await else { continue };
            match session.ask(&ClientMsg::Forward { job_id: service.job_id.clone() }).await {
                Ok(AgentMsg::Accepted { .. }) => {}
                _ => {
                    session.close().await;
                    continue;
                }
            }
            active.insert(service.job_id.clone(), Active { session, ports: HashMap::new(), listeners: JoinSet::new() });
        }
        let a = active.get_mut(&service.job_id).expect("inserted above");
        for remote in service.ports {
            if a.ports.contains_key(&remote) {
                continue;
            }
            let Ok(listener) = listen(remote).await else { continue };
            let local = listener.local_addr().map(|a| a.port()).unwrap_or(remote);
            a.ports.insert(remote, local);
            a.listeners.spawn(accept_loop(listener, a.session.conn.clone(), remote));
            on_event(ForwardEvent::Opened(Forwarded { job_id: service.job_id.clone(), remote, local }));
        }
    }
}

/// The same port on this PC's localhost when free, else any free one.
async fn listen(port: u16) -> std::io::Result<TcpListener> {
    match TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => Ok(listener),
        Err(_) => TcpListener::bind(("127.0.0.1", 0)).await,
    }
}

async fn accept_loop(listener: TcpListener, conn: Connection, remote: u16) {
    while let Ok((tcp, _)) = listener.accept().await {
        let conn = conn.clone();
        tokio::spawn(async move {
            let _ = pipe(conn, tcp, remote).await;
        });
    }
}

async fn pipe(conn: Connection, tcp: TcpStream, remote: u16) -> Result<()> {
    let _ = tcp.set_nodelay(true);
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &ClientMsg::Connect { port: remote }).await?;
    match recv_msg::<AgentMsg>(&mut recv).await? {
        AgentMsg::Accepted { .. } => {}
        AgentMsg::Rejected { reason } => bail!(reason),
        other => bail!("unexpected reply to Connect {other:?}"),
    }
    let (mut tcp_read, mut tcp_write) = tcp.into_split();
    let upstream = async {
        tokio::io::copy(&mut tcp_read, &mut send).await?;
        send.finish().map_err(std::io::Error::other)
    };
    let downstream = async {
        tokio::io::copy(&mut recv, &mut tcp_write).await?;
        tokio::io::AsyncWriteExt::shutdown(&mut tcp_write).await
    };
    let (up, down) = tokio::join!(upstream, downstream);
    up.and(down)?;
    Ok(())
}
