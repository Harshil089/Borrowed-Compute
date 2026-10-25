//! Cloud API for Borrowed Compute.
//!
//! Holds accounts, sessions, devices and pairings. It never sees job contents: jobs flow
//! device to device, end-to-end encrypted.

pub mod auth;
pub mod devices;
pub mod error;
pub mod github;
pub mod pairings;

use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;
use sqlx::PgPool;

pub use github::GitHub;

/// A device counts as online if it sent a heartbeat this recently.
pub const ONLINE_WINDOW_SECS: i64 = 90;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub github: Arc<GitHub>,
}

pub async fn migrate(db: &PgPool) -> anyhow::Result<()> {
    sqlx::migrate!("./migrations").run(db).await?;
    Ok(())
}

pub fn app(state: AppState) -> Router {
    Router::new()
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
