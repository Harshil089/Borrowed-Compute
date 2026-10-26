//! Two endpoints exchange job messages through a local relay with direct paths disabled.

use std::time::Duration;

use bc_net::{bind, recv_msg, send_msg, via_relay, NetConfig, JOB_ALPN};
use bc_proto::wire::{AgentMsg, ClientMsg, OutputStream, PROTOCOL_VERSION};
use iroh::SecretKey;

#[tokio::test(flavor = "multi_thread")]
async fn job_messages_flow_through_relay() -> anyhow::Result<()> {
    let (relay_map, _relay_url, _relay) = iroh::test_utils::run_relay_server().await?;
    let config = NetConfig {
        relays: Some(relay_map),
        relay_only: true,
        insecure_relay_tls: true,
    };

    let mac = bind(SecretKey::generate(), config.clone()).await?;
    let windows = bind(SecretKey::generate(), config).await?;
    tokio::time::timeout(Duration::from_secs(10), mac.online()).await?;
    let mac_addr = mac.addr();
    let windows_id = windows.id();

    let agent = tokio::spawn(async move {
        let conn = mac.accept().await.expect("incoming").await?;
        assert_eq!(conn.remote_id(), windows_id, "peer identity is the dialer's key");
        let (mut send, mut recv) = conn.accept_bi().await?;
        let hello: ClientMsg = recv_msg(&mut recv).await?;
        assert_eq!(hello, ClientMsg::Hello { protocol: PROTOCOL_VERSION });
        send_msg(&mut send, &AgentMsg::Hello { protocol: PROTOCOL_VERSION }).await?;
        send_msg(
            &mut send,
            &AgentMsg::Output {
                stream: OutputStream::Stdout,
                seq: 0,
                data: b"hello from the mac\n".to_vec(),
            },
        )
        .await?;
        send_msg(&mut send, &AgentMsg::Exited { code: Some(0), timed_out: false, usage: None }).await?;
        send.finish()?;
        conn.closed().await;
        mac.close().await;
        anyhow::Ok(())
    });

    let conn = tokio::time::timeout(Duration::from_secs(15), windows.connect(mac_addr, JOB_ALPN))
        .await??;
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &ClientMsg::Hello { protocol: PROTOCOL_VERSION }).await?;

    let hello: AgentMsg = recv_msg(&mut recv).await?;
    assert_eq!(hello, AgentMsg::Hello { protocol: PROTOCOL_VERSION });
    let out: AgentMsg = recv_msg(&mut recv).await?;
    assert!(matches!(out, AgentMsg::Output { seq: 0, .. }));
    let exit: AgentMsg = recv_msg(&mut recv).await?;
    assert_eq!(exit, AgentMsg::Exited { code: Some(0), timed_out: false, usage: None });

    assert!(via_relay(&conn), "direct paths are disabled, so traffic must use the relay");

    conn.close(0u32.into(), b"done");
    windows.close().await;
    agent.await??;
    Ok(())
}

/// The Mac app's relay: plain HTTP on localhost (Tailscale adds the HTTPS in real use),
/// reached by both devices through `relay_map`, with direct paths disabled.
#[tokio::test(flavor = "multi_thread")]
async fn job_messages_flow_through_a_plain_http_relay() -> anyhow::Result<()> {
    use iroh_relay::server::{RelayConfig, Server, ServerConfig};
    let mut relay = RelayConfig::new((std::net::Ipv4Addr::LOCALHOST, 0));
    relay.key_cache_capacity = Some(1024);
    let mut config = ServerConfig::default();
    config.relay = Some(relay);
    let server = Server::spawn(config).await?;
    let url = format!("http://{}", server.http_addr().expect("http relay"));
    let net = NetConfig { relays: Some(bc_net::relay_map(&url)?), relay_only: true, insecure_relay_tls: false };

    let mac = bind(SecretKey::generate(), net.clone()).await?;
    let pc = bind(SecretKey::generate(), net).await?;
    tokio::time::timeout(Duration::from_secs(10), mac.online()).await?;
    let mac_addr = mac.addr();
    assert!(mac_addr.relay_urls().any(|u| u.as_str().starts_with(&url)), "the Mac's home relay is ours");

    let agent = tokio::spawn(async move {
        let conn = mac.accept().await.expect("incoming").await?;
        let (mut send, mut recv) = conn.accept_bi().await?;
        let hello: ClientMsg = recv_msg(&mut recv).await?;
        assert_eq!(hello, ClientMsg::Hello { protocol: PROTOCOL_VERSION });
        send_msg(&mut send, &AgentMsg::Exited { code: Some(0), timed_out: false, usage: None }).await?;
        send.finish()?;
        conn.closed().await;
        mac.close().await;
        anyhow::Ok(())
    });

    let conn = tokio::time::timeout(Duration::from_secs(20), pc.connect(mac_addr, JOB_ALPN)).await??;
    assert!(via_relay(&conn), "went through our relay");
    let (mut send, mut recv) = conn.open_bi().await?;
    send_msg(&mut send, &ClientMsg::Hello { protocol: PROTOCOL_VERSION }).await?;
    let exit: AgentMsg = recv_msg(&mut recv).await?;
    assert_eq!(exit, AgentMsg::Exited { code: Some(0), timed_out: false, usage: None });
    conn.close(0u32.into(), b"done");
    agent.await??;
    pc.close().await;
    Ok(())
}

/// Same exchange through a real deployed relay, such as the Mac app's behind Tailscale:
/// set `BC_TEST_RELAY_URL=https://<mac>.ts.net:8443/`. Skipped otherwise.
#[tokio::test(flavor = "multi_thread")]
async fn job_messages_flow_through_a_deployed_relay() -> anyhow::Result<()> {
    let Ok(url) = std::env::var("BC_TEST_RELAY_URL") else {
        eprintln!("BC_TEST_RELAY_URL not set; skipping");
        return Ok(());
    };
    let net = NetConfig { relays: Some(bc_net::relay_map(&url)?), relay_only: true, insecure_relay_tls: false };
    let mac = bind(SecretKey::generate(), net.clone()).await?;
    let pc = bind(SecretKey::generate(), net).await?;
    tokio::time::timeout(Duration::from_secs(15), mac.online()).await?;
    let mac_addr = mac.addr();
    let agent = tokio::spawn(async move {
        let conn = mac.accept().await.expect("incoming").await?;
        let (mut send, mut recv) = conn.accept_bi().await?;
        let data: ClientMsg = recv_msg(&mut recv).await?;
        let ClientMsg::Upload { data } = data else { anyhow::bail!("expected upload") };
        send_msg(&mut send, &AgentMsg::Output { stream: OutputStream::Stdout, seq: 0, data }).await?;
        send.finish()?;
        conn.closed().await;
        mac.close().await;
        anyhow::Ok(())
    });
    let conn = tokio::time::timeout(Duration::from_secs(20), pc.connect(mac_addr, JOB_ALPN)).await??;
    let (mut send, mut recv) = conn.open_bi().await?;
    // A megabyte, like one upload chunk, to exercise more than a handshake.
    let payload = vec![7u8; 1 << 20];
    send_msg(&mut send, &ClientMsg::Upload { data: payload.clone() }).await?;
    let AgentMsg::Output { data, .. } = recv_msg::<AgentMsg>(&mut recv).await? else { anyhow::bail!("expected output") };
    assert_eq!(data, payload);
    assert!(via_relay(&conn));
    conn.close(0u32.into(), b"done");
    agent.await??;
    pc.close().await;
    Ok(())
}
