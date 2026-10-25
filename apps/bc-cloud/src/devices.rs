//! Device registry and presence.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use bc_proto::cloud::{register_message, DeviceInfo, DeviceKind, Heartbeat, RegisterDevice};
use iroh_base::{PublicKey, Signature};

use crate::auth::Account;
use crate::error::{ApiError, ApiResult};
use crate::{now, AppState, ONLINE_WINDOW_SECS};

/// How far a registration signature's timestamp may be from the cloud's clock.
const MAX_SKEW_SECS: u64 = 300;
const MAX_NAME: usize = 64;

type DeviceRow = (String, String, String, Option<i64>, Option<String>);

pub const DEVICE_SELECT: &str = "select endpoint_id, name, kind, last_seen, addr from devices";

pub fn to_info(row: DeviceRow, now: i64) -> DeviceInfo {
    let (endpoint_id, name, kind, last_seen, addr) = row;
    let ago = last_seen.map(|t| (now - t).max(0));
    DeviceInfo {
        endpoint_id,
        name,
        kind: DeviceKind::parse(&kind).unwrap_or(DeviceKind::Other),
        online: ago.is_some_and(|a| a < ONLINE_WINDOW_SECS),
        last_seen_secs_ago: ago,
        addr: addr.and_then(|a| serde_json::from_str(&a).ok()),
    }
}

pub async fn register(
    State(state): State<AppState>,
    account: Account,
    Json(req): Json<RegisterDevice>,
) -> ApiResult<(StatusCode, Json<DeviceInfo>)> {
    let name = req.name.trim();
    if name.is_empty() || name.len() > MAX_NAME {
        return Err(ApiError::bad_request(format!("device name must be 1 to {MAX_NAME} characters")));
    }
    verify_key_proof(&req, &account)?;

    let taken: Option<(String,)> = sqlx::query_as(
        "select endpoint_id from devices where account_id = $1 and name = $2 and endpoint_id <> $3",
    )
    .bind(account.id)
    .bind(name.to_string())
    .bind(req.endpoint_id.clone())
    .fetch_optional(&state.db)
    .await?;
    if taken.is_some() {
        return Err(ApiError::conflict(format!("another device is already named {name:?}")));
    }

    // Re-registering updates name and kind, but only for the account that owns the key.
    let owned: Option<(i64,)> = sqlx::query_as(
        "insert into devices (endpoint_id, account_id, name, kind, created_at) values ($1, $2, $3, $4, $5)
         on conflict (endpoint_id) do update set name = excluded.name, kind = excluded.kind
         where devices.account_id = excluded.account_id
         returning account_id",
    )
    .bind(req.endpoint_id.clone())
    .bind(account.id)
    .bind(name.to_string())
    .bind(req.kind.as_str())
    .bind(now())
    .fetch_optional(&state.db)
    .await?;
    if owned.is_none() {
        return Err(ApiError::conflict("this device key belongs to another account"));
    }

    let row: DeviceRow = sqlx::query_as(&format!("{DEVICE_SELECT} where endpoint_id = $1"))
        .bind(req.endpoint_id)
        .fetch_one(&state.db)
        .await?;
    Ok((StatusCode::CREATED, Json(to_info(row, now()))))
}

fn verify_key_proof(req: &RegisterDevice, account: &Account) -> ApiResult<()> {
    if (now() as u64).abs_diff(req.signed_at) > MAX_SKEW_SECS {
        return Err(ApiError::bad_request("registration timestamp is too far from server time"));
    }
    let key: PublicKey = req
        .endpoint_id
        .parse()
        .map_err(|_| ApiError::bad_request("endpoint_id is not a device public key"))?;
    let sig_bytes: [u8; 64] = hex::decode(&req.signature)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| ApiError::bad_request("signature must be 64 hex-encoded bytes"))?;
    let message = register_message(&req.endpoint_id, req.signed_at, &hex::encode(&account.token_hash));
    key.verify(&message, &Signature::from_bytes(&sig_bytes))
        .map_err(|_| ApiError::unauthorized("signature does not match the device key"))
}

pub async fn list(State(state): State<AppState>, account: Account) -> ApiResult<Json<Vec<DeviceInfo>>> {
    let rows: Vec<DeviceRow> =
        sqlx::query_as(&format!("{DEVICE_SELECT} where account_id = $1 order by name"))
            .bind(account.id)
            .fetch_all(&state.db)
            .await?;
    let now = now();
    Ok(Json(rows.into_iter().map(|r| to_info(r, now)).collect()))
}

pub async fn heartbeat(
    State(state): State<AppState>,
    account: Account,
    Path(id): Path<String>,
    Json(req): Json<Heartbeat>,
) -> ApiResult<StatusCode> {
    let addr = req.addr.map(|a| a.to_string());
    let done = sqlx::query(
        "update devices set last_seen = $1, addr = coalesce($2, addr)
         where endpoint_id = $3 and account_id = $4",
    )
    .bind(now())
    .bind(addr)
    .bind(id)
    .bind(account.id)
    .execute(&state.db)
    .await?;
    if done.rows_affected() == 0 {
        return Err(ApiError::not_found("device not registered to this account"));
    }
    Ok(StatusCode::NO_CONTENT)
}

pub async fn remove(
    State(state): State<AppState>,
    account: Account,
    Path(id): Path<String>,
) -> ApiResult<StatusCode> {
    let done = sqlx::query("delete from devices where endpoint_id = $1 and account_id = $2")
        .bind(id)
        .bind(account.id)
        .execute(&state.db)
        .await?;
    if done.rows_affected() == 0 {
        return Err(ApiError::not_found("device not registered to this account"));
    }
    Ok(StatusCode::NO_CONTENT)
}
