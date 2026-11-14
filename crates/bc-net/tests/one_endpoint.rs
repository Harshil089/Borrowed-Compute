//! One endpoint per key in a process, however the callers race.

use bc_net::{bind, release, share_endpoint, NetConfig};
use iroh::SecretKey;

#[tokio::test]
async fn callers_with_one_key_share_one_endpoint() {
    let key = SecretKey::generate();
    // Two at once, as the app's port forwarding and its agent at launch.
    let (a, b) = tokio::join!(bind(key.clone(), NetConfig::default()), bind(key.clone(), NetConfig::default()));
    let (a, b) = (a.unwrap(), b.unwrap());
    assert_eq!(a.bound_sockets(), b.bound_sockets(), "the same endpoint, not two with one key");
    let other = bind(SecretKey::generate(), NetConfig::default()).await.unwrap();
    assert_ne!(other.bound_sockets(), a.bound_sockets(), "another key gets its own");
    // It closes with its last holder.
    release(a).await;
    assert!(!b.is_closed());
    release(b.clone()).await;
    assert!(b.is_closed());
    // The agent's is never closed by a client that releases it.
    let agent = bind(key.clone(), NetConfig::default()).await.unwrap();
    share_endpoint(&agent);
    let client = bind(key, NetConfig::default()).await.unwrap();
    assert_eq!(client.bound_sockets(), agent.bound_sockets());
    release(client).await;
    assert!(!agent.is_closed());
    release(other).await;
}
