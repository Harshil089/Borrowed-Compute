//! Pairing: a client device asks to send jobs to a Mac, and the Mac owner approves it
//! by entering the code shown on both screens. v1 pairs only devices on one account.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use bc_proto::cloud::{
    ApprovePairing, CreatePairing, DeviceKind, PairedClients, PairingInfo, PairingStatus,
};
use serde::Deserialize;

use crate::auth::Account;
use crate::error::{ApiError, ApiResult};
use crate::AppState;

const CODE_MINUTES: i32 = 10;

type PairingRow = (i64, String, String, String, String, String, String, Option<i64>);

const PAIRING_SELECT: &str = "select p.id, p.client_id, c.name, p.mac_id, m.name, p.status, p.code,
        case when p.status = 'pending' then extract(epoch from p.expires_at - now())::bigint end
    from pairings p
    join devices c on c.endpoint_id = p.client_id
    join devices m on m.endpoint_id = p.mac_id";

fn to_info(row: PairingRow) -> PairingInfo {
    let (id, client_id, client_name, mac_id, mac_name, status, code, expires_in_secs) = row;
    let status = if status == "approved" { PairingStatus::Approved } else { PairingStatus::Pending };
    PairingInfo {
        id,
        client_id,
        client_name,
        mac_id,
        mac_name,
        status,
        code: (status == PairingStatus::Pending).then_some(code),
        expires_in_secs,
    }
}

async fn fetch(state: &AppState, id: i64) -> ApiResult<PairingInfo> {
    let row: PairingRow = sqlx::query_as(&format!("{PAIRING_SELECT} where p.id = $1"))
        .bind(id)
        .fetch_one(&state.db)
        .await?;
    Ok(to_info(row))
}

pub async fn create(
    State(state): State<AppState>,
    account: Account,
    Json(req): Json<CreatePairing>,
) -> ApiResult<(StatusCode, Json<PairingInfo>)> {
    let client: Option<(String,)> = sqlx::query_as(
        "select endpoint_id from devices where endpoint_id = $1 and account_id = $2",
    )
    .bind(&req.client)
    .bind(account.id)
    .fetch_optional(&state.db)
    .await?;
    let (client_id,) =
        client.ok_or_else(|| ApiError::not_found("this device is not registered; run login first"))?;

    let mac: Option<(String, String)> = sqlx::query_as(
        "select endpoint_id, kind from devices
         where account_id = $1 and (endpoint_id = $2 or name = $2)",
    )
    .bind(account.id)
    .bind(&req.mac)
    .fetch_optional(&state.db)
    .await?;
    let (mac_id, kind) = mac.ok_or_else(|| {
        ApiError::not_found(format!("no device named {:?} on this account", req.mac))
    })?;
    if DeviceKind::parse(&kind) != Some(DeviceKind::Mac) {
        return Err(ApiError::bad_request(format!("{:?} is not a Mac", req.mac)));
    }
    if mac_id == client_id {
        return Err(ApiError::bad_request("a device cannot pair with itself"));
    }

    // Keep an approved pairing or a still-valid code; otherwise issue a fresh code.
    let code = format!("{:06}", rand::random::<u32>() % 1_000_000);
    let (id,): (i64,) = sqlx::query_as(
        "insert into pairings (client_id, mac_id, code, status, expires_at)
         values ($1, $2, $3, 'pending', now() + make_interval(mins => $4))
         on conflict (client_id, mac_id) do update
           set code = case when pairings.status = 'pending' and pairings.expires_at <= now()
                           then excluded.code else pairings.code end,
               expires_at = case when pairings.status = 'pending' and pairings.expires_at <= now()
                                 then excluded.expires_at else pairings.expires_at end
         returning id",
    )
    .bind(&client_id)
    .bind(&mac_id)
    .bind(&code)
    .bind(CODE_MINUTES)
    .fetch_one(&state.db)
    .await?;
    Ok((StatusCode::CREATED, Json(fetch(&state, id).await?)))
}

#[derive(Deserialize)]
pub struct ListQuery {
    /// A device on the caller's account; lists pairings where it is client or Mac.
    pub device: String,
}

pub async fn list(
    State(state): State<AppState>,
    account: Account,
    Query(q): Query<ListQuery>,
) -> ApiResult<Json<Vec<PairingInfo>>> {
    let rows: Vec<PairingRow> = sqlx::query_as(&format!(
        "{PAIRING_SELECT}
         where (p.mac_id = $1 or p.client_id = $1) and m.account_id = $2
           and (p.status = 'approved' or p.expires_at > now())
         order by p.created_at"
    ))
    .bind(&q.device)
    .bind(account.id)
    .fetch_all(&state.db)
    .await?;
    Ok(Json(rows.into_iter().map(to_info).collect()))
}

pub async fn approve(
    State(state): State<AppState>,
    account: Account,
    Path(id): Path<i64>,
    Json(req): Json<ApprovePairing>,
) -> ApiResult<Json<PairingInfo>> {
    let row: Option<(String, String, bool)> = sqlx::query_as(
        "select p.status, p.code, p.expires_at > now() from pairings p
         join devices m on m.endpoint_id = p.mac_id
         where p.id = $1 and m.account_id = $2",
    )
    .bind(id)
    .bind(account.id)
    .fetch_optional(&state.db)
    .await?;
    let (status, code, fresh) = row.ok_or_else(|| ApiError::not_found("no such pairing request"))?;
    if status == "approved" {
        return Ok(Json(fetch(&state, id).await?));
    }
    if !fresh {
        return Err(ApiError::bad_request("pairing code expired; ask again from the other device"));
    }
    if code != req.code.trim() {
        return Err(ApiError::bad_request("pairing code does not match"));
    }
    sqlx::query("update pairings set status = 'approved' where id = $1")
        .bind(id)
        .execute(&state.db)
        .await?;
    Ok(Json(fetch(&state, id).await?))
}

pub async fn remove(
    State(state): State<AppState>,
    account: Account,
    Path(id): Path<i64>,
) -> ApiResult<StatusCode> {
    let done = sqlx::query(
        "delete from pairings p using devices m
         where p.id = $1 and m.endpoint_id = p.mac_id and m.account_id = $2",
    )
    .bind(id)
    .bind(account.id)
    .execute(&state.db)
    .await?;
    if done.rows_affected() == 0 {
        return Err(ApiError::not_found("no such pairing"));
    }
    Ok(StatusCode::NO_CONTENT)
}

/// Approved clients of a Mac. The agent polls this to decide who may send jobs.
pub async fn paired_clients(
    State(state): State<AppState>,
    account: Account,
    Path(mac_id): Path<String>,
) -> ApiResult<Json<PairedClients>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "select p.client_id from pairings p join devices m on m.endpoint_id = p.mac_id
         where p.mac_id = $1 and m.account_id = $2 and p.status = 'approved'
         order by p.client_id",
    )
    .bind(&mac_id)
    .bind(account.id)
    .fetch_all(&state.db)
    .await?;
    Ok(Json(PairedClients { clients: rows.into_iter().map(|(c,)| c).collect() }))
}
