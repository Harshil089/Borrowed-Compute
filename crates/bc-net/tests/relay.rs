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
        send_msg(&mut send, &AgentMsg::Exited { code: Some(0), timed_out: false }).await?;
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
    assert_eq!(exit, AgentMsg::Exited { code: Some(0), timed_out: false });

    assert!(via_relay(&conn), "direct paths are disabled, so traffic must use the relay");

    conn.close(0u32.into(), b"done");
    windows.close().await;
    agent.await??;
    Ok(())
}
