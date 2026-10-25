use std::sync::Arc;

use anyhow::{Context, Result};
use bc_cloud::{connect, serve, AppState, GitHub};

/// Configuration comes from the environment:
///
/// - `BC_DATABASE_URL` (required): `postgres://…`, or `sqlite://<file>?mode=rwc`.
/// - `BC_GITHUB_CLIENT_ID` (required): OAuth app client id with device flow enabled.
/// - `BC_LISTEN`: address to bind, default `127.0.0.1:8080`.
#[tokio::main]
async fn main() -> Result<()> {
    let db_url = std::env::var("BC_DATABASE_URL").context("set BC_DATABASE_URL")?;
    let client_id = std::env::var("BC_GITHUB_CLIENT_ID").context("set BC_GITHUB_CLIENT_ID")?;
    let listen = std::env::var("BC_LISTEN").unwrap_or_else(|_| "127.0.0.1:8080".into());

    let db = connect(&db_url).await?;
    let state = AppState { db, github: Arc::new(GitHub::new(client_id)) };
    let listener = tokio::net::TcpListener::bind(&listen).await?;
    eprintln!("bc-cloud listening on http://{listen}");
    serve(listener, state).await?;
    Ok(())
}
