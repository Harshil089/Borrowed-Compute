//! Cloud API tests: real databases, fake GitHub, real HTTP through `bc-auth`.
//!
//! Every test runs against a fresh SQLite file. Set `BC_TEST_DATABASE_URL` to a Postgres
//! admin connection string, for example `postgres://bc@127.0.0.1:54329/postgres`, to also
//! run it against a fresh Postgres database.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use bc_auth::CloudClient;
use bc_cloud::{app, connect, AppState, GitHub};
use bc_proto::cloud::{DeviceKind, PairingStatus};
use iroh_base::SecretKey;
use serde_json::{json, Value};
use sqlx::postgres::PgPoolOptions;
use sqlx::Executor;

#[derive(Default)]
struct FakeGitHub {
    /// Users who will sign in next, in order.
    next: Mutex<VecDeque<(i64, String)>>,
    /// Device codes polled once already.
    polled: Mutex<Vec<String>>,
}

fn fake_github(state: Arc<FakeGitHub>) -> Router {
    async fn device_code(State(s): State<Arc<FakeGitHub>>) -> Json<Value> {
        let (id, login) = s.next.lock().unwrap().pop_front().expect("queue a user first");
        Json(json!({
            "device_code": format!("dc-{id}-{login}"),
            "user_code": "WDJB-MJHT",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 1
        }))
    }
    async fn token(
        State(s): State<Arc<FakeGitHub>>,
        axum::Form(form): axum::Form<Vec<(String, String)>>,
    ) -> Json<Value> {
        let code = form.iter().find(|(k, _)| k == "device_code").unwrap().1.clone();
        let mut polled = s.polled.lock().unwrap();
        if !polled.contains(&code) {
            polled.push(code);
            return Json(json!({ "error": "authorization_pending" }));
        }
        Json(json!({ "access_token": code.replacen("dc-", "gh-", 1), "token_type": "bearer" }))
    }
    async fn user(headers: axum::http::HeaderMap) -> Json<Value> {
        let auth = headers["authorization"].to_str().unwrap();
        let token = auth.strip_prefix("Bearer gh-").unwrap();
        let (id, login) = token.split_once('-').unwrap();
        Json(json!({ "id": id.parse::<i64>().unwrap(), "login": login }))
    }
    Router::new()
        .route("/login/device/code", post(device_code))
        .route("/login/oauth/access_token", post(token))
        .route("/user", get(user))
        .with_state(state)
}

async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

struct Env {
    cloud: String,
    github: Arc<FakeGitHub>,
}

/// Database URLs to run each test against: a fresh SQLite file always, plus a fresh
/// Postgres database when `BC_TEST_DATABASE_URL` points at a Postgres admin connection.
async fn databases() -> Vec<(String, Option<tempfile::TempDir>)> {
    let dir = tempfile::tempdir().unwrap();
    let mut urls = vec![(bc_cloud::sqlite_url(&dir.path().join("cloud test.db")), Some(dir))];
    if let Ok(admin_url) = std::env::var("BC_TEST_DATABASE_URL") {
        let admin = PgPoolOptions::new().max_connections(1).connect(&admin_url).await.unwrap();
        let name = format!("bc_test_{}", bc_cloud::auth::random_hex(6));
        admin.execute(format!("create database {name}").as_str()).await.unwrap();
        let (base, _) = admin_url.rsplit_once('/').unwrap();
        urls.push((format!("{base}/{name}"), None));
    } else {
        eprintln!("BC_TEST_DATABASE_URL not set; testing SQLite only");
    }
    urls
}

async fn setup(database_url: &str) -> Env {
    let db = connect(database_url).await.unwrap();
    let github = Arc::new(FakeGitHub::default());
    let gh_url = serve(fake_github(github.clone())).await;
    let state = AppState {
        db,
        github: Arc::new(GitHub::with_bases("test-client".into(), gh_url.clone(), gh_url)),
    };
    let cloud = serve(app(state)).await;
    Env { cloud, github }
}

impl Env {
    async fn login(&self, github_id: i64, login: &str) -> CloudClient {
        self.github.next.lock().unwrap().push_back((github_id, login.into()));
        let mut client = CloudClient::new(&self.cloud, None);
        let mut shown = None;
        let (_, who) = client.login(|s| shown = Some(s.user_code.clone())).await.unwrap();
        assert_eq!(who, login);
        assert_eq!(shown.as_deref(), Some("WDJB-MJHT"));
        client
    }
}

#[tokio::test]
async fn sign_in_register_pair_and_revoke() {
    for (url, _dir) in databases().await {
        sign_in_register_pair_and_revoke_on(setup(&url).await).await;
    }
}

async fn sign_in_register_pair_and_revoke_on(env: Env) {
    let alice = env.login(1, "alice").await;
    assert_eq!(alice.me().await.unwrap().login, "alice");

    let mac_key = SecretKey::generate();
    let pc_key = SecretKey::generate();
    let mac_id = mac_key.public().to_string();
    let pc_id = pc_key.public().to_string();
    alice.register(&mac_key, "studio", DeviceKind::Mac).await.unwrap();
    alice.register(&pc_key, "laptop", DeviceKind::Windows).await.unwrap();

    // Presence and address come from heartbeats.
    let before = alice.find_device("studio").await.unwrap();
    assert!(!before.online);
    let addr = json!({ "id": mac_id, "addrs": [] });
    alice.heartbeat(&mac_id, Some(addr.clone())).await.unwrap();
    let after = alice.find_device("studio").await.unwrap();
    assert!(after.online);
    assert_eq!(after.addr, Some(addr));
    assert_eq!(alice.devices().await.unwrap().len(), 2);

    // The laptop asks; the Mac sees the same code and approves with it.
    let request = alice.request_pairing(&pc_id, "studio").await.unwrap();
    assert_eq!(request.status, PairingStatus::Pending);
    let code = request.code.clone().unwrap();
    assert_eq!(code.len(), 6);
    let on_mac = alice.pairings(&mac_id).await.unwrap();
    assert_eq!(on_mac.len(), 1);
    assert_eq!(on_mac[0].code.as_deref(), Some(code.as_str()));
    assert_eq!(on_mac[0].client_name, "laptop");

    let wrong = if code == "000000" { "111111" } else { "000000" };
    assert!(alice.approve(request.id, wrong).await.is_err());
    assert!(alice.paired_clients(&mac_id).await.unwrap().is_empty());

    let approved = alice.approve(request.id, &code).await.unwrap();
    assert_eq!(approved.status, PairingStatus::Approved);
    assert_eq!(approved.code, None);
    assert_eq!(alice.paired_clients(&mac_id).await.unwrap(), vec![pc_id.clone()]);

    // Asking again keeps the approval.
    let again = alice.request_pairing(&pc_id, &mac_id).await.unwrap();
    assert_eq!(again.status, PairingStatus::Approved);

    alice.unpair(request.id).await.unwrap();
    assert!(alice.paired_clients(&mac_id).await.unwrap().is_empty());
}

#[tokio::test]
async fn rejects_bad_proofs_and_other_accounts() {
    for (url, _dir) in databases().await {
        rejects_bad_proofs_and_other_accounts_on(setup(&url).await).await;
    }
}

async fn rejects_bad_proofs_and_other_accounts_on(env: Env) {
    let alice = env.login(1, "alice").await;
    let mallory = env.login(2, "mallory").await;

    let mac_key = SecretKey::generate();
    let mac_id = mac_key.public().to_string();
    alice.register(&mac_key, "studio", DeviceKind::Mac).await.unwrap();

    // Someone else's key cannot be claimed, even with a valid signature from that key.
    let err = mallory.register(&mac_key, "stolen", DeviceKind::Mac).await.unwrap_err();
    assert!(err.to_string().contains("another account"), "{err}");

    // Registering requires a signature from the key being registered.
    let pc_key = SecretKey::generate();
    let other = SecretKey::generate();
    let body = json!({
        "endpoint_id": pc_key.public().to_string(),
        "name": "laptop",
        "kind": "windows",
        "signed_at": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "signature": hex::encode(other.sign(b"anything").to_bytes()),
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/v1/devices", env.cloud))
        .bearer_auth(alice.token().unwrap())
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // Device names are unique per account.
    let err = alice.register(&pc_key, "studio", DeviceKind::Windows).await.unwrap_err();
    assert!(err.to_string().contains("already named"), "{err}");
    alice.register(&pc_key, "laptop", DeviceKind::Windows).await.unwrap();
    let pc_id = pc_key.public().to_string();

    // Other accounts see nothing and cannot act on Alice's devices.
    assert!(mallory.devices().await.unwrap().is_empty());
    assert!(mallory.paired_clients(&mac_id).await.unwrap().is_empty());
    let request = alice.request_pairing(&pc_id, "studio").await.unwrap();
    let code = request.code.clone().unwrap();
    assert!(mallory.approve(request.id, &code).await.is_err());
    assert!(mallory.pairings(&mac_id).await.unwrap().is_empty());
    assert!(mallory.heartbeat(&mac_id, None).await.is_err());
    assert!(mallory.unpair(request.id).await.is_err());

    // Only Macs can be paired to.
    let err = alice.request_pairing(&mac_id, "laptop").await.unwrap_err();
    assert!(err.to_string().contains("not a Mac"), "{err}");

    // Missing and revoked sessions are refused.
    assert!(CloudClient::new(&env.cloud, None).devices().await.is_err());
    alice.logout().await.unwrap();
    let err = alice.devices().await.unwrap_err();
    assert!(err.to_string().contains("401"), "{err}");
}
