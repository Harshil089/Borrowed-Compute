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
pub mod pairings;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use sqlx::any::AnyPoolOptions;
use sqlx::AnyPool;

pub use github::GitHub;

/// A device counts as online if it sent a heartbeat this recently.
pub const ONLINE_WINDOW_SECS: i64 = 90;

#[derive(Clone)]
pub struct AppState {
    pub db: AnyPool,
    pub github: Arc<GitHub>,
}

pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Opens `database_url` (`postgres://…` or `sqlite://…`) and brings its schema up to date.
pub async fn connect(database_url: &str) -> anyhow::Result<AnyPool> {
    sqlx::any::install_default_drivers();
    let sqlite = database_url.starts_with("sqlite:");
    let db = AnyPoolOptions::new()
        // SQLite allows one writer at a time; a single connection avoids "database is
        // locked" errors at the scale the Mac app runs at.
        .max_connections(if sqlite { 1 } else { 10 })
        .connect(database_url)
        .await
        .context("opening the database")?;
    if sqlite {
        sqlx::migrate!("./migrations/sqlite").run(&db).await?;
    } else {
        sqlx::migrate!("./migrations/postgres").run(&db).await?;
    }
    Ok(db)
}

/// A `sqlite://` URL for a database file, created if missing.
pub fn sqlite_url(path: &std::path::Path) -> String {
    // sqlx percent-decodes the path, so encode what a URL cannot carry as-is.
    let encoded = path.to_string_lossy().replace('%', "%25").replace(' ', "%20").replace('?', "%3F");
    format!("sqlite://{encoded}?mode=rwc")
}

#[derive(serde::Serialize)]
struct Health {
    service: &'static str,
    version: &'static str,
}

async fn health() -> Json<Health> {
    Json(Health { service: "borrowed-compute-cloud", version: env!("CARGO_PKG_VERSION") })
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/auth/device", post(auth::start_device_flow))
        .route("/v1/auth/device/poll", post(auth::poll_device_flow))
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
