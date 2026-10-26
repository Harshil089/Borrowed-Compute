//! Pairing: a client device asks to send jobs to a Mac, and the Mac owner approves it by
//! typing the code the client shows. Only the requesting client ever receives the code,
//! so approving proves the owner can see that client's screen. Three wrong codes delete
//! the request. v1 pairs only devices on one account.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use bc_proto::cloud::{
    ApprovePairing, CreatePairing, DeviceKind, PairedClients, PairingInfo, PairingStatus,
};
use serde::Deserialize;

use crate::auth::Account;
use crate::error::{ApiError, ApiResult};
use crate::{now, AppState};

const CODE_SECS: i64 = 10 * 60;
/// Wrong codes allowed before the request is deleted.
const MAX_ATTEMPTS: i64 = 3;
/// Letters and digits that cannot be mistaken for each other (no 0/O, 1/I/L).
const CODE_ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

/// Eight characters shown as `ABCD-EFGH`: about 40 bits, against three guesses.
fn new_code() -> String {
    let chars: String = (0..8)
        .map(|_| CODE_ALPHABET[rand::random::<usize>() % CODE_ALPHABET.len()] as char)
        .collect();
    format!("{}-{}", &chars[..4], &chars[4..])
}

/// Compares codes the way people type them: any case, with or without the dash.
fn normalize(code: &str) -> String {
    code.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
}

type PairingRow = (i64, String, String, String, String, String, String, i64);

const PAIRING_SELECT: &str = "select p.id, p.client_id, c.name, p.mac_id, m.name, p.status, p.code, p.expires_at
    from pairings p
    join devices c on c.endpoint_id = p.client_id
    join devices m on m.endpoint_id = p.mac_id";

fn to_info(row: PairingRow, now: i64) -> PairingInfo {
    let (id, client_id, client_name, mac_id, mac_name, status, code, expires_at) = row;
    let pending = status != "approved";
    PairingInfo {
        id,
        client_id,
        client_name,
        mac_id,
        mac_name,
        status: if pending { PairingStatus::Pending } else { PairingStatus::Approved },
        code: pending.then_some(code),
        expires_in_secs: pending.then_some(expires_at - now),
    }
}

async fn fetch(state: &AppState, id: i64) -> ApiResult<PairingInfo> {
    let row: PairingRow = sqlx::query_as(&format!("{PAIRING_SELECT} where p.id = $1"))
        .bind(id)
        .fetch_one(&state.db)
        .await?;
    Ok(to_info(row, now()))
}

pub async fn create(
    State(state): State<AppState>,
    account: Account,
    Json(req): Json<CreatePairing>,
) -> ApiResult<(StatusCode, Json<PairingInfo>)> {
    let client: Option<(String,)> =
        sqlx::query_as("select endpoint_id from devices where endpoint_id = $1 and account_id = $2")
            .bind(req.client.clone())
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
    .bind(req.mac.clone())
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
    let now = now();
    let code = new_code();
    let (id,): (i64,) = sqlx::query_as(
        "insert into pairings (client_id, mac_id, code, status, created_at, expires_at)
         values ($1, $2, $3, 'pending', $4, $5)
         on conflict (client_id, mac_id) do update
           set code = case when pairings.status = 'pending' and pairings.expires_at <= $4
                           then excluded.code else pairings.code end,
               attempts = case when pairings.status = 'pending' and pairings.expires_at <= $4
                               then 0 else pairings.attempts end,
               expires_at = case when pairings.status = 'pending' and pairings.expires_at <= $4
                                 then excluded.expires_at else pairings.expires_at end
         returning id",
    )
    .bind(client_id)
    .bind(mac_id)
    .bind(code)
    .bind(now)
    .bind(now + CODE_SECS)
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
    let now = now();
    let rows: Vec<PairingRow> = sqlx::query_as(&format!(
        "{PAIRING_SELECT}
         where (p.mac_id = $1 or p.client_id = $1) and m.account_id = $2
           and (p.status = 'approved' or p.expires_at > $3)
         order by p.created_at, p.id"
    ))
    .bind(q.device)
    .bind(account.id)
    .bind(now)
    .fetch_all(&state.db)
    .await?;
    // Codes go only to the device that asked (in `create`); the Mac must type it.
    Ok(Json(rows.into_iter().map(|r| PairingInfo { code: None, ..to_info(r, now) }).collect()))
}

pub async fn approve(
    State(state): State<AppState>,
    account: Account,
    Path(id): Path<i64>,
    Json(req): Json<ApprovePairing>,
) -> ApiResult<Json<PairingInfo>> {
    let row: Option<(String, String, i64)> = sqlx::query_as(
        "select p.status, p.code, p.expires_at from pairings p
         join devices m on m.endpoint_id = p.mac_id
         where p.id = $1 and m.account_id = $2",
    )
    .bind(id)
    .bind(account.id)
    .fetch_optional(&state.db)
    .await?;
    let (status, code, expires_at) = row.ok_or_else(|| ApiError::not_found("no such pairing request"))?;
    if status == "approved" {
        return Ok(Json(fetch(&state, id).await?));
    }
    if expires_at <= now() {
        return Err(ApiError::bad_request("pairing code expired; ask again from the other device"));
    }
    if normalize(&code) != normalize(&req.code) {
        let (attempts,): (i64,) =
            sqlx::query_as("update pairings set attempts = attempts + 1 where id = $1 returning attempts")
                .bind(id)
                .fetch_one(&state.db)
                .await?;
        if attempts >= MAX_ATTEMPTS {
            sqlx::query("delete from pairings where id = $1").bind(id).execute(&state.db).await?;
            return Err(ApiError::bad_request(
                "wrong code three times; the request was cancelled. Ask again from the other device",
            ));
        }
        return Err(ApiError::bad_request(format!(
            "that code does not match; {} tries left",
            MAX_ATTEMPTS - attempts
        )));
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
        "delete from pairings
         where id = $1 and mac_id in (select endpoint_id from devices where account_id = $2)",
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
    .bind(mac_id)
    .bind(account.id)
    .fetch_all(&state.db)
    .await?;
    Ok(Json(PairedClients { clients: rows.into_iter().map(|(c,)| c).collect() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unambiguous_and_typed_loosely() {
        let code = new_code();
        assert_eq!(code.len(), 9);
        assert!(code.chars().all(|c| c == '-' || CODE_ALPHABET.contains(&(c as u8))), "{code}");
        assert_eq!(normalize("abcd-efgh"), normalize("ABCDEFGH"));
        assert_eq!(normalize(" AbCd EfGh "), "ABCDEFGH");
    }
}
