//! Cloud API tests: real databases, fake GitHub and Google, real HTTP through `bc-auth`.
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
use bc_cloud::{app, connect, AppState, GitHub, Google};
use bc_proto::cloud::{DeviceKind, PairingStatus, Provider};
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
    /// Verified email addresses, by GitHub user id.
    emails: Mutex<std::collections::HashMap<i64, Vec<String>>>,
}

#[derive(Default)]
struct FakeGoogle {
    /// (sub, email, email_verified) of whoever signs in next.
    next: Mutex<VecDeque<(String, String, bool)>>,
    polled: Mutex<Vec<String>>,
    users: Mutex<std::collections::HashMap<String, (String, bool)>>,
}

fn fake_google(state: Arc<FakeGoogle>) -> Router {
    async fn device_code(State(s): State<Arc<FakeGoogle>>) -> Json<Value> {
        let (sub, email, verified) = s.next.lock().unwrap().pop_front().expect("queue a Google user first");
        s.users.lock().unwrap().insert(sub.clone(), (email, verified));
        Json(json!({
            "device_code": format!("gdc-{sub}"),
            "user_code": "GQVQ-JKEC",
            "verification_url": "https://www.google.com/device",
            "expires_in": 1800,
            "interval": 1
        }))
    }
    async fn token(
        State(s): State<Arc<FakeGoogle>>,
        axum::Form(form): axum::Form<Vec<(String, String)>>,
    ) -> Json<Value> {
        let get = |k: &str| form.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).unwrap_or_default();
        assert_eq!(get("client_secret"), "test-secret");
        let code = get("device_code");
        let mut polled = s.polled.lock().unwrap();
        if !polled.contains(&code) {
            polled.push(code);
            return Json(json!({ "error": "authorization_pending" }));
        }
        Json(json!({ "access_token": code.replacen("gdc-", "gat-", 1), "id_token": "unused" }))
    }
    async fn userinfo(State(s): State<Arc<FakeGoogle>>, headers: axum::http::HeaderMap) -> Json<Value> {
        let sub = headers["authorization"].to_str().unwrap().strip_prefix("Bearer gat-").unwrap().to_string();
        let (email, verified) = s.users.lock().unwrap()[&sub].clone();
        Json(json!({ "sub": sub, "email": email, "email_verified": verified }))
    }
    Router::new()
        .route("/device/code", post(device_code))
        .route("/token", post(token))
        .route("/v1/userinfo", get(userinfo))
        .with_state(state)
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
    async fn emails(State(s): State<Arc<FakeGitHub>>, headers: axum::http::HeaderMap) -> Json<Value> {
        let auth = headers["authorization"].to_str().unwrap();
        let id: i64 = auth.strip_prefix("Bearer gh-").unwrap().split_once('-').unwrap().0.parse().unwrap();
        let list = s.emails.lock().unwrap().get(&id).cloned().unwrap_or_default();
        // GitHub also lists unverified addresses; the cloud must ignore those.
        let mut out: Vec<Value> = list.iter().map(|e| json!({ "email": e, "verified": true, "primary": true })).collect();
        out.push(json!({ "email": "unverified@example.com", "verified": false, "primary": false }));
        Json(Value::Array(out))
    }
    Router::new()
        .route("/login/device/code", post(device_code))
        .route("/login/oauth/access_token", post(token))
        .route("/user", get(user))
        .route("/user/emails", get(emails))
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
    google: Arc<FakeGoogle>,
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
    let google = Arc::new(FakeGoogle::default());
    let g_url = serve(fake_google(google.clone())).await;
    let state = AppState {
        db,
        github: Arc::new(GitHub::with_bases("test-client".into(), gh_url.clone(), gh_url)),
        google: Some(Arc::new(Google::with_bases("g-client".into(), "test-secret".into(), g_url.clone(), g_url))),
        relay_url: Some("https://studio.tail1234.ts.net:8443/".into()),
    };
    let cloud = serve(app(state)).await;
    Env { cloud, github, google }
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

    /// Signs in with Google as `sub`; returns the client and the account name shown.
    async fn google_login(&self, sub: &str, email: &str, verified: bool) -> (CloudClient, String) {
        self.google.next.lock().unwrap().push_back((sub.into(), email.into(), verified));
        let mut client = CloudClient::new(&self.cloud, None);
        let (_, who) = client.login_with(Provider::Google, |_| {}).await.unwrap();
        (client, who)
    }
}

#[tokio::test]
async fn google_and_github_link_by_verified_email() {
    for (url, _dir) in databases().await {
        google_and_github_link_by_verified_email_on(setup(&url).await).await;
    }
}

async fn google_and_github_link_by_verified_email_on(env: Env) {
    let providers = CloudClient::new(&env.cloud, None).providers().await.unwrap();
    assert!(providers.github && providers.google);
    assert_eq!(
        CloudClient::new(&env.cloud, None).relay().await.as_deref(),
        Some("https://studio.tail1234.ts.net:8443/"),
        "the cloud advertises the Mac's relay"
    );

    // Alice signs in with GitHub on her Mac and registers it.
    env.github.emails.lock().unwrap().insert(1, vec!["alice@example.com".into()]);
    let alice_gh = env.login(1, "alice").await;
    let mac = SecretKey::generate();
    alice_gh.register(&mac, "studio", DeviceKind::Mac).await.unwrap();

    // Her PC signs in with Google using the same verified address: same account.
    let (alice_g, who) = env.google_login("g-alice", "Alice@Example.com", true).await;
    assert_eq!(who, "alice", "linked to the existing account and its name");
    let names: Vec<String> = alice_g.devices().await.unwrap().into_iter().map(|d| d.name).collect();
    assert_eq!(names, ["studio"], "the Google session sees the GitHub session's Mac");

    // Signing in with Google again reuses the identity.
    let (_, again) = env.google_login("g-alice", "alice@example.com", true).await;
    assert_eq!(again, "alice");

    // An unverified Google address never links, even when it matches.
    let (eve, eve_name) = env.google_login("g-eve", "alice@example.com", false).await;
    assert_eq!(eve_name, "google-g-eve");
    assert!(eve.devices().await.unwrap().is_empty(), "no access to Alice's devices");

    // GitHub's unverified addresses are ignored too.
    let (bob_g, bob_name) = env.google_login("g-bob", "unverified@example.com", true).await;
    assert_eq!(bob_name, "unverified@example.com", "a new account named by its email");
    assert!(bob_g.devices().await.unwrap().is_empty());

    // A GitHub user with no verified email gets an account of their own.
    let carol = env.login(3, "carol").await;
    assert!(carol.devices().await.unwrap().is_empty());
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

    // The laptop asks and is shown the code; the Mac sees the request but must type it.
    let request = alice.request_pairing(&pc_id, "studio").await.unwrap();
    assert_eq!(request.status, PairingStatus::Pending);
    let code = request.code.clone().unwrap();
    assert_eq!(code.len(), 9);
    let on_mac = alice.pairings(&mac_id).await.unwrap();
    assert_eq!(on_mac.len(), 1);
    assert_eq!(on_mac[0].code, None);
    assert_eq!(on_mac[0].client_name, "laptop");

    assert!(alice.approve(request.id, "WRNG-CODE").await.is_err());
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

/// Upgrades a copy of an existing database through `connect`, the way an app update does,
/// and checks no row was lost. Point `BC_TEST_UPGRADE_DB` at a copy of a real `cloud.db`;
/// skipped otherwise.
#[tokio::test]
async fn upgrades_an_existing_database() {
    let Ok(path) = std::env::var("BC_TEST_UPGRADE_DB") else {
        eprintln!("BC_TEST_UPGRADE_DB not set; skipping");
        return;
    };
    let url = bc_cloud::sqlite_url(std::path::Path::new(&path));
    let count = |table: &'static str| {
        let path = path.clone();
        move || {
            let out = std::process::Command::new("sqlite3").arg(&path).arg(format!("select count(*) from {table}")).output().unwrap();
            String::from_utf8(out.stdout).unwrap().trim().to_string()
        }
    };
    let before: Vec<String> = ["sessions", "devices", "pairings"].iter().map(|t| count(t)()).collect();
    let db = connect(&url).await.unwrap();
    let after: Vec<String> = ["sessions", "devices", "pairings"].iter().map(|t| count(t)()).collect();
    assert_eq!(before, after, "sessions, devices and pairings survive the upgrade");
    let (identities,): (i64,) = sqlx::query_as("select count(*) from identities").fetch_one(&db).await.unwrap();
    let (accounts,): (i64,) = sqlx::query_as("select count(*) from accounts").fetch_one(&db).await.unwrap();
    assert_eq!(identities, accounts, "every existing account keeps its GitHub identity");
}

#[tokio::test]
async fn pairing_codes_reach_only_the_requester_and_lock_after_three_misses() {
    for (url, _dir) in databases().await {
        pairing_lockout_on(setup(&url).await).await;
    }
}

async fn pairing_lockout_on(env: Env) {
    let alice = env.login(1, "alice").await;
    let (pc, mac) = (SecretKey::generate(), SecretKey::generate());
    alice.register(&pc, "pc", DeviceKind::Windows).await.unwrap();
    alice.register(&mac, "studio", DeviceKind::Mac).await.unwrap();

    let request = alice.request_pairing(&pc.public().to_string(), "studio").await.unwrap();
    let code = request.code.clone().expect("the requester gets the code");
    assert_eq!(code.len(), 9, "ABCD-EFGH: {code}");
    let seen_by_mac = alice.pairings(&mac.public().to_string()).await.unwrap();
    assert!(seen_by_mac.iter().all(|p| p.code.is_none()), "the Mac must type the code, not read it");

    // Two misses leave the request; the right code typed loosely still works.
    for tries_left in ["2 tries left", "1 tries left"] {
        let err = alice.approve(request.id, "WRNG-CODE").await.unwrap_err();
        assert!(format!("{err:#}").contains(tries_left), "{err:#}");
    }
    let loose = code.to_lowercase().replace('-', " ");
    alice.approve(request.id, &loose).await.unwrap();

    // A new request: three misses cancel it, and the right code no longer works.
    alice.unpair(request.id).await.unwrap();
    let request = alice.request_pairing(&pc.public().to_string(), "studio").await.unwrap();
    let code = request.code.clone().unwrap();
    for _ in 0..2 {
        alice.approve(request.id, "WRNG-CODE").await.unwrap_err();
    }
    let err = alice.approve(request.id, "WRNG-CODE").await.unwrap_err();
    assert!(format!("{err:#}").contains("cancelled"), "{err:#}");
    assert!(alice.approve(request.id, &code).await.is_err(), "the request is gone");
}
