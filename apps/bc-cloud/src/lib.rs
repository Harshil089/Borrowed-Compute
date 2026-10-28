//! Cloud API for Borrowed Compute.
//!
//! Holds accounts, sessions, devices and pairings. It never sees job contents: jobs flow
//! device to device, end-to-end encrypted.
//!
//! The same service runs two ways: as `bc-cloud` on a server with Postgres, and inside the
//! Mac app with a SQLite file, so development needs no server at all. Queries are written
//! to run on both, and all times are Unix seconds computed here rather than in SQL.

pub mod auth;
pub mod devices;
pub mod error;
pub mod github;
pub mod google;
pub mod pairings;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use sqlx::any::AnyPoolOptions;
use sqlx::AnyPool;

pub use github::GitHub;
pub use google::Google;

/// A device counts as online if it sent a heartbeat this recently.
pub const ONLINE_WINDOW_SECS: i64 = 90;

#[derive(Clone)]
pub struct AppState {
    pub db: AnyPool,
    pub github: Arc<GitHub>,
    /// `None` when no Google client is configured; Google sign-in is then not offered.
    pub google: Option<Arc<Google>>,
    /// A relay advertised to devices; `None` for iroh's public relays.
    pub relay_url: Option<String>,
    /// One account only: once an account exists, sign-ins that would create another are
    /// refused. The Mac app's built-in cloud runs this way, because anyone who knows the
    /// Mac's code can reach it.
    pub single_owner: bool,
}

/// Google sign-in from `BC_GOOGLE_CLIENT_ID` and `BC_GOOGLE_CLIENT_SECRET`, falling back
/// to the given defaults. `None`, and no Google button, when either is missing.
pub fn google_client(default_id: &str, default_secret: Option<&str>) -> Option<Arc<Google>> {
    let id = std::env::var("BC_GOOGLE_CLIENT_ID").unwrap_or_else(|_| default_id.to_string());
    let secret = std::env::var("BC_GOOGLE_CLIENT_SECRET").ok().or(default_secret.map(str::to_string))?;
    (!id.is_empty() && !secret.is_empty()).then(|| Arc::new(Google::new(id, secret)))
}

pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Opens `database_url` (`postgres://…` or `sqlite://…`) and brings its schema up to date.
pub async fn connect(database_url: &str) -> anyhow::Result<AnyPool> {
    sqlx::any::install_default_drivers();
    let sqlite = database_url.starts_with("sqlite:");
    if sqlite {
        migrate_sqlite(database_url).await?;
    }
    let db = AnyPoolOptions::new()
        // SQLite allows one writer at a time; a single connection avoids "database is
        // locked" errors at the scale the Mac app runs at.
        .max_connections(if sqlite { 1 } else { 10 })
        .connect(database_url)
        .await
        .context("opening the database")?;
    if !sqlite {
        sqlx::migrate!("./migrations/postgres").run(&db).await?;
    }
    Ok(db)
}

/// Brings a SQLite database's schema up to date on a connection with foreign keys off.
/// Some migrations rebuild a table, and SQLite runs a table drop as a delete, which would
/// cascade into every row that refers to it. Foreign keys are checked once at the end.
async fn migrate_sqlite(database_url: &str) -> anyhow::Result<()> {
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::str::FromStr;
    let options = SqliteConnectOptions::from_str(database_url)?.foreign_keys(false);
    let pool = SqlitePoolOptions::new().max_connections(1).connect_with(options).await?;
    sqlx::migrate!("./migrations/sqlite").run(&pool).await?;
    let broken: Vec<(String,)> = sqlx::query_as("select \"table\" from pragma_foreign_key_check")
        .fetch_all(&pool)
        .await?;
    pool.close().await;
    if !broken.is_empty() {
        anyhow::bail!("database migration left broken references in {broken:?}");
    }
    Ok(())
}

/// A `sqlite://` URL for a database file, created if missing.
///
/// The `Any` driver parses the URL as a standard URL before SQLite sees it, which would
/// read the `C:` of a Windows path as a host and port. Percent-encoding every character
/// except unreserved ones makes the whole path pass through unchanged; sqlx decodes it
/// back to the original path.
pub fn sqlite_url(path: &std::path::Path) -> String {
    const KEEP: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
        .remove(b'-')
        .remove(b'.')
        .remove(b'_')
        .remove(b'~');
    let path = path.to_string_lossy();
    let encoded = percent_encoding::utf8_percent_encode(&path, KEEP);
    format!("sqlite://{encoded}?mode=rwc")
}

#[derive(serde::Serialize)]
struct Health {
    service: &'static str,
    version: &'static str,
}

async fn relay(axum::extract::State(state): axum::extract::State<AppState>) -> Json<bc_proto::cloud::RelayInfo> {
    Json(bc_proto::cloud::RelayInfo { url: state.relay_url.clone() })
}

async fn health() -> Json<Health> {
    Json(Health { service: "borrowed-compute-cloud", version: env!("CARGO_PKG_VERSION") })
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/auth/device", post(auth::start_device_flow))
        .route("/v1/auth/device/poll", post(auth::poll_device_flow))
        .route("/v1/auth/google/device", post(auth::start_google_flow))
        .route("/v1/auth/providers", get(auth::providers))
        .route("/v1/relay", get(relay))
        .route("/v1/auth/logout", post(auth::logout))
        .route("/v1/me", get(auth::me))
        .route("/v1/devices", get(devices::list).post(devices::register))
        .route("/v1/devices/{id}", delete(devices::remove))
        .route("/v1/devices/{id}/heartbeat", post(devices::heartbeat))
        .route("/v1/devices/{id}/paired", get(pairings::paired_clients))
        .route("/v1/pairings", get(pairings::list).post(pairings::create))
        .route("/v1/pairings/{id}", delete(pairings::remove))
        .route("/v1/pairings/{id}/approve", post(pairings::approve))
        .with_state(state)
}

/// Serves the API on `listener` until the task is dropped.
pub async fn serve(listener: tokio::net::TcpListener, state: AppState) -> std::io::Result<()> {
    axum::serve(listener, app(state)).await
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::str::FromStr;

    use sqlx::sqlite::SqliteConnectOptions;

    /// What the `Any` driver does: parse as a URL, then hand its text to SQLite.
    fn filename_after_any(url: &str) -> std::path::PathBuf {
        let parsed = url::Url::parse(url).unwrap();
        SqliteConnectOptions::from_str(parsed.as_str()).unwrap().get_filename().to_path_buf()
    }

    #[test]
    fn sqlite_url_keeps_windows_and_unix_paths() {
        for path in [
            r"C:\Users\me\AppData\Roaming\BorrowedCompute\cloud\cloud.db",
            r"C:\Users\me\Local Temp\.tmpAb12\cloud test.db",
            "/Users/me/Library/Application Support/BorrowedCompute/cloud/cloud.db",
            "/tmp/odd?name#100%.db",
        ] {
            let url = super::sqlite_url(Path::new(path));
            assert_eq!(filename_after_any(&url), Path::new(path), "{url}");
        }
    }
}
