//! Sign-in through GitHub device flow, and the session extractor for other routes.

use axum::extract::{FromRequestParts, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::Json;
use bc_proto::cloud::{DeviceFlowPoll, DeviceFlowStart, DeviceFlowStatus, Me, SESSION_PREFIX};
use sha2::{Digest, Sha256};

use crate::error::{ApiError, ApiResult};
use crate::github::Poll;
use crate::AppState;

const SESSION_DAYS: i32 = 90;

/// The signed-in account behind a request's bearer token.
#[derive(Debug, Clone)]
pub struct Account {
    pub id: i64,
    pub login: String,
    pub token_hash: Vec<u8>,
}

impl FromRequestParts<AppState> for Account {
    type Rejection = ApiError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, ApiError> {
        let token = parts
            .headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or_else(|| ApiError::unauthorized("sign in first: missing bearer token"))?;
        let token_hash = Sha256::digest(token.as_bytes()).to_vec();
        let row: Option<(i64, String)> = sqlx::query_as(
            "select a.id, a.github_login from sessions s join accounts a on a.id = s.account_id
             where s.token_hash = $1 and s.expires_at > now()",
        )
        .bind(&token_hash)
        .fetch_optional(&state.db)
        .await?;
        let (id, login) =
            row.ok_or_else(|| ApiError::unauthorized("session expired or invalid; sign in again"))?;
        Ok(Account { id, login, token_hash })
    }
}

pub async fn start_device_flow(State(state): State<AppState>) -> ApiResult<Json<DeviceFlowStart>> {
    let code = state.github.start().await?;
    let flow_id = random_hex(16);
    sqlx::query(
        "insert into device_flows (id, device_code, expires_at)
         values ($1, $2, now() + make_interval(secs => $3))",
    )
    .bind(&flow_id)
    .bind(&code.device_code)
    .bind(code.expires_in as f64)
    .execute(&state.db)
    .await?;
    // Old flows are useless after they expire.
    sqlx::query("delete from device_flows where expires_at < now()")
        .execute(&state.db)
        .await?;
    Ok(Json(DeviceFlowStart {
        flow_id,
        user_code: code.user_code,
        verification_uri: code.verification_uri,
        interval_secs: code.interval,
        expires_in_secs: code.expires_in,
    }))
}

pub async fn poll_device_flow(
    State(state): State<AppState>,
    Json(req): Json<DeviceFlowPoll>,
) -> ApiResult<Json<DeviceFlowStatus>> {
    let row: Option<(String,)> = sqlx::query_as(
        "select device_code from device_flows where id = $1 and expires_at > now()",
    )
    .bind(&req.flow_id)
    .fetch_optional(&state.db)
    .await?;
    let Some((device_code,)) = row else {
        return Ok(Json(DeviceFlowStatus::Expired));
    };

    let token = match state.github.poll(&device_code).await? {
        Poll::Pending => return Ok(Json(DeviceFlowStatus::Pending)),
        Poll::SlowDown(interval_secs) => return Ok(Json(DeviceFlowStatus::SlowDown { interval_secs })),
        Poll::Expired => return Ok(Json(DeviceFlowStatus::Expired)),
        Poll::Denied => return Ok(Json(DeviceFlowStatus::Denied)),
        Poll::Token(token) => token,
    };
    // The GitHub token is used once here and then dropped.
    let user = state.github.user(&token).await?;
    drop(token);

    let (account_id,): (i64,) = sqlx::query_as(
        "insert into accounts (github_id, github_login) values ($1, $2)
         on conflict (github_id) do update set github_login = excluded.github_login
         returning id",
    )
    .bind(user.id)
    .bind(&user.login)
    .fetch_one(&state.db)
    .await?;

    let session_token = format!("{SESSION_PREFIX}{}", random_hex(32));
    sqlx::query(
        "insert into sessions (token_hash, account_id, expires_at)
         values ($1, $2, now() + make_interval(days => $3))",
    )
    .bind(Sha256::digest(session_token.as_bytes()).to_vec())
    .bind(account_id)
    .bind(SESSION_DAYS)
    .execute(&state.db)
    .await?;
    sqlx::query("delete from device_flows where id = $1")
        .bind(&req.flow_id)
        .execute(&state.db)
        .await?;

    Ok(Json(DeviceFlowStatus::Complete { session_token, login: user.login }))
}

pub async fn me(account: Account) -> Json<Me> {
    Json(Me { login: account.login })
}

pub async fn logout(State(state): State<AppState>, account: Account) -> ApiResult<StatusCode> {
    sqlx::query("delete from sessions where token_hash = $1")
        .bind(&account.token_hash)
        .execute(&state.db)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

pub fn random_hex(bytes: usize) -> String {
    let data: Vec<u8> = (0..bytes).map(|_| rand::random::<u8>()).collect();
    hex::encode(data)
}
