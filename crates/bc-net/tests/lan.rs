//! A Mac advertising on the local network is heard by a listener on the same machine.
//! Needs multicast on the host, so it is ignored by default: run with `-- --ignored`.

use std::time::Duration;

use bc_net::{bind, lan, NetConfig};
use iroh::SecretKey;

#[tokio::test]
#[ignore]
async fn a_listener_hears_an_advertising_mac() -> anyhow::Result<()> {
    let mac = bind(SecretKey::generate(), NetConfig::default()).await?;
    lan::advertise(&mac, "studio")?;
    mac.online().await;
    let mut nearby = lan::watch_nearby()?;
    let found = tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            if nearby.borrow().iter().any(|m| m.id == mac.id().to_string() && m.name == "studio") {
                return;
            }
            nearby.changed().await.unwrap();
        }
    })
    .await;
    assert!(found.is_ok(), "the Mac was not heard within 20 seconds");
    Ok(())
}
