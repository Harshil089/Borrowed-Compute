//! The Mac's built-in cloud, carried over iroh so other devices reach it without a VPN.
//!
//! The Mac accepts [`CLOUD_ALPN`] connections and pipes each stream to its cloud's HTTP
//! port on localhost. Another device names the Mac by its endpoint ID (the "Mac code") and
//! gets a local HTTP address from [`cloud_proxy`] that leads there.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use crate::NetConfig;

/// ALPN for HTTP requests to the Mac's cloud, one QUIC stream per TCP connection.
pub const CLOUD_ALPN: &[u8] = b"borrowed-compute/cloud/1";

/// Pipes each stream of `conn` to the cloud at `127.0.0.1:port` until the connection ends.
pub async fn serve_cloud(conn: Connection, port: u16) {
    while let Ok((send, recv)) = conn.accept_bi().await {
        tokio::spawn(async move {
            if let Ok(tcp) = TcpStream::connect(("127.0.0.1", port)).await {
                let _ = pipe(tcp, send, recv).await;
            }
        });
    }
}

/// A local HTTP address that leads to the cloud on the Mac with this ID. The first call
/// for a Mac starts the proxy; it then lives as long as the process.
pub async fn cloud_proxy(mac: EndpointId) -> Result<SocketAddr> {
    static PROXIES: OnceLock<Mutex<HashMap<EndpointId, SocketAddr>>> = OnceLock::new();
    let mut proxies = PROXIES.get_or_init(Default::default).lock().await;
    if let Some(addr) = proxies.get(&mac) {
        return Ok(*addr);
    }
    let addr = start_proxy(mac.into()).await?;
    proxies.insert(mac, addr);
    Ok(addr)
}

/// Starts a proxy to the cloud on the Mac at `mac`, found through discovery when it
/// carries no addresses. [`cloud_proxy`] keeps one per Mac; tests call this directly.
pub async fn start_proxy(mac: EndpointAddr) -> Result<SocketAddr> {
    // Its own key: the cloud identifies devices by their sign-in, not by this connection.
    let endpoint = crate::bind_with(SecretKey::generate(), NetConfig::default(), vec![]).await?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.context("starting the cloud proxy")?;
    let addr = listener.local_addr()?;
    tokio::spawn(run_proxy(endpoint, listener, mac));
    Ok(addr)
}

async fn run_proxy(endpoint: Endpoint, listener: TcpListener, mac: EndpointAddr) {
    let conn: Mutex<Option<Connection>> = Mutex::new(None);
    let conn = std::sync::Arc::new(conn);
    while let Ok((tcp, _)) = listener.accept().await {
        let endpoint = endpoint.clone();
        let conn = conn.clone();
        let mac = mac.clone();
        tokio::spawn(async move {
            let streams = {
                let mut conn = conn.lock().await;
                // One connection for all requests; dial again once it is gone.
                if conn.as_ref().is_none_or(|c| c.close_reason().is_some()) {
                    match endpoint.connect(mac, CLOUD_ALPN).await {
                        Ok(c) => *conn = Some(c),
                        Err(e) => {
                            eprintln!("cannot reach the Mac's cloud: {e:#}");
                            return;
                        }
                    }
                }
                let streams = conn.as_ref().unwrap().open_bi().await;
                if streams.is_err() {
                    *conn = None;
                }
                streams
            };
            if let Ok((send, recv)) = streams {
                let _ = pipe(tcp, send, recv).await;
            }
        });
    }
}

async fn pipe(tcp: TcpStream, mut send: SendStream, mut recv: RecvStream) -> Result<()> {
    let (mut tcp_read, mut tcp_write) = tcp.into_split();
    let up = async {
        tokio::io::copy(&mut tcp_read, &mut send).await?;
        send.finish().map_err(std::io::Error::other)
    };
    let down = async {
        tokio::io::copy(&mut recv, &mut tcp_write).await?;
        tcp_write.shutdown().await
    };
    let (up, down) = tokio::join!(up, down);
    up.and(down)?;
    Ok(())
}
