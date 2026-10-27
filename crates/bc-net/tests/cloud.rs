//! The Mac's cloud reached over iroh: an HTTP request to the local proxy arrives at the
//! cloud's port on the Mac, and the answer comes back.

use std::time::Duration;

use bc_net::{bind, cloud, NetConfig, CLOUD_ALPN};
use iroh::SecretKey;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test(flavor = "multi_thread")]
async fn http_reaches_the_cloud_on_the_mac() -> anyhow::Result<()> {
    // Stands in for the cloud: answers one request per connection.
    let cloud_port = {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        let port = listener.local_addr()?.port();
        tokio::spawn(async move {
            while let Ok((mut tcp, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let n = tcp.read(&mut buf).await.unwrap();
                    assert!(buf[..n].starts_with(b"GET /v1/health"));
                    tcp.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok").await.unwrap();
                });
            }
        });
        port
    };

    let mac = bind(SecretKey::generate(), NetConfig::default()).await?;
    let mac_accept = mac.clone();
    tokio::spawn(async move {
        while let Some(incoming) = mac_accept.accept().await {
            let conn = incoming.await.unwrap();
            assert_eq!(conn.alpn(), CLOUD_ALPN);
            tokio::spawn(bc_net::serve_cloud(conn, cloud_port));
        }
    });

    let proxy = cloud::start_proxy(mac.addr()).await?;
    for _ in 0..2 {
        let mut tcp = tokio::net::TcpStream::connect(proxy).await?;
        tcp.write_all(b"GET /v1/health HTTP/1.1\r\nhost: x\r\n\r\n").await?;
        let mut reply = String::new();
        tokio::time::timeout(Duration::from_secs(20), tcp.read_to_string(&mut reply)).await??;
        assert!(reply.starts_with("HTTP/1.1 200 OK") && reply.ends_with("ok"), "{reply}");
    }
    Ok(())
}

/// Same, but the proxy knows only the Mac's code and finds it through iroh's public
/// discovery, as a PC does. Needs the internet: `cargo test -p bc-net --test cloud -- --ignored`.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn finds_the_mac_by_its_code() -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let cloud_port = listener.local_addr()?.port();
    tokio::spawn(async move {
        while let Ok((mut tcp, _)) = listener.accept().await {
            let mut buf = [0u8; 1024];
            let _ = tcp.read(&mut buf).await;
            let _ = tcp.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok").await;
        }
    });
    let mac = bind(SecretKey::generate(), NetConfig::default()).await?;
    mac.online().await;
    let mac_accept = mac.clone();
    tokio::spawn(async move {
        while let Some(incoming) = mac_accept.accept().await {
            tokio::spawn(bc_net::serve_cloud(incoming.await.unwrap(), cloud_port));
        }
    });
    let proxy = bc_net::cloud_proxy(mac.id()).await?;
    let mut tcp = tokio::net::TcpStream::connect(proxy).await?;
    tcp.write_all(b"GET /v1/health HTTP/1.1\r\nhost: x\r\n\r\n").await?;
    let mut reply = String::new();
    tokio::time::timeout(Duration::from_secs(30), tcp.read_to_string(&mut reply)).await??;
    assert!(reply.ends_with("ok"), "{reply}");
    Ok(())
}
