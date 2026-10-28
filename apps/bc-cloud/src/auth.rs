//! Sign-in through the GitHub or Google device flow, account linking by verified email,
//! and the session extractor for other routes.

use axum::extract::{FromRequestParts, State};
use axum::http::request::Parts;
use axum::http::StatusCode;
use axum::Json;
use bc_proto::cloud::{DeviceFlowPoll, DeviceFlowStart, DeviceFlowStatus, Me, Provider, Providers, SESSION_PREFIX};
use sha2::{Digest, Sha256};

use crate::error::{ApiError, ApiResult};
use crate::github::{DeviceCode, Poll};
use crate::{now, AppState};

const SESSION_SECS: i64 = 90 * 24 * 3600;

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
            "select a.id, a.login from sessions s join accounts a on a.id = s.account_id
             where s.token_hash = $1 and s.expires_at > $2",
        )
        .bind(token_hash.clone())
        .bind(now())
        .fetch_optional(&state.db)
        .await?;
        let (id, login) =
            row.ok_or_else(|| ApiError::unauthorized("session expired or invalid; sign in again"))?;
        Ok(Account { id, login, token_hash })
    }
}

/// Starts GitHub sign-in. The original route, kept for clients that predate Google.
/// Sign-ins in progress at once. Each one asks GitHub or Google for a code with this
/// cloud's client id, so an unbounded number would let anyone use up its rate limit.
const MAX_PENDING_FLOWS: i64 = 20;

/// Refuses a new sign-in while too many are in progress.
async fn check_flow_room(state: &AppState) -> ApiResult<()> {
    let (pending,): (i64,) = sqlx::query_as("select count(*) from device_flows where expires_at > $1")
        .bind(now())
        .fetch_one(&state.db)
        .await?;
    if pending >= MAX_PENDING_FLOWS {
        return Err(ApiError::too_many("too many sign-ins in progress; try again in a few minutes"));
    }
    Ok(())
}

pub async fn start_device_flow(State(state): State<AppState>) -> ApiResult<Json<DeviceFlowStart>> {
    check_flow_room(&state).await?;
    let code = state.github.start().await?;
    record_flow(&state, Provider::Github, code).await
}

pub async fn start_google_flow(State(state): State<AppState>) -> ApiResult<Json<DeviceFlowStart>> {
    let google = state
        .google
        .as_ref()
        .ok_or_else(|| ApiError::not_found("Google sign-in is not set up on this cloud"))?;
    check_flow_room(&state).await?;
    let code = google.start().await?;
    record_flow(&state, Provider::Google, code).await
}

pub async fn providers(State(state): State<AppState>) -> Json<Providers> {
    Json(Providers { github: true, google: state.google.is_some() })
}

async fn record_flow(state: &AppState, provider: Provider, code: DeviceCode) -> ApiResult<Json<DeviceFlowStart>> {
    let flow_id = random_hex(16);
    let now = now();
    sqlx::query("insert into device_flows (id, device_code, expires_at, provider) values ($1, $2, $3, $4)")
        .bind(flow_id.clone())
        .bind(code.device_code.clone())
        .bind(now + code.expires_in as i64)
        .bind(provider.as_str())
        .execute(&state.db)
        .await?;
    // Old flows are useless after they expire.
    sqlx::query("delete from device_flows where expires_at < $1")
        .bind(now)
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

/// Who signed in, as the provider reports it. Provider tokens are used to build this and
/// then dropped; none are stored.
struct Identity {
    provider: Provider,
    subject: String,
    /// Shown as the account name when this identity creates the account.
    login: String,
    /// Verified addresses only, lowercased.
    emails: Vec<String>,
}

pub async fn poll_device_flow(
    State(state): State<AppState>,
    Json(req): Json<DeviceFlowPoll>,
) -> ApiResult<Json<DeviceFlowStatus>> {
    let row: Option<(String, String)> = sqlx::query_as(
        "select device_code, provider from device_flows where id = $1 and expires_at > $2",
    )
    .bind(req.flow_id.clone())
    .bind(now())
    .fetch_optional(&state.db)
    .await?;
    let Some((device_code, provider)) = row else {
        return Ok(Json(DeviceFlowStatus::Expired));
    };

    let poll = match provider.as_str() {
        "google" => match &state.google {
            Some(google) => google.poll(&device_code).await?,
            None => return Ok(Json(DeviceFlowStatus::Expired)),
        },
        _ => state.github.poll(&device_code).await?,
    };
    let token = match poll {
        Poll::Pending => return Ok(Json(DeviceFlowStatus::Pending)),
        Poll::SlowDown(interval_secs) => return Ok(Json(DeviceFlowStatus::SlowDown { interval_secs })),
        Poll::Expired => return Ok(Json(DeviceFlowStatus::Expired)),
        Poll::Denied => return Ok(Json(DeviceFlowStatus::Denied)),
        Poll::Token(token) => token,
    };

    let identity = match (provider.as_str(), &state.google) {
        ("google", Some(google)) => {
            let user = google.user(&token).await?;
            let email = user.email.filter(|_| user.email_verified).map(|e| e.to_lowercase());
            Identity {
                provider: Provider::Google,
                login: email.clone().unwrap_or_else(|| format!("google-{}", user.sub)),
                subject: user.sub,
                emails: email.into_iter().collect(),
            }
        }
        _ => {
            let user = state.github.user(&token).await?;
            let emails = state.github.verified_emails(&token).await?;
            Identity { provider: Provider::Github, subject: user.id.to_string(), login: user.login, emails }
        }
    };
    drop(token);

    let now = now();
    let (account_id, login) = account_for(&state, &identity, now).await?;

    let session_token = format!("{SESSION_PREFIX}{}", random_hex(32));
    sqlx::query(
        "insert into sessions (token_hash, account_id, created_at, expires_at) values ($1, $2, $3, $4)",
    )
    .bind(Sha256::digest(session_token.as_bytes()).to_vec())
    .bind(account_id)
    .bind(now)
    .bind(now + SESSION_SECS)
    .execute(&state.db)
    .await?;
    sqlx::query("delete from device_flows where id = $1")
        .bind(req.flow_id)
        .execute(&state.db)
        .await?;

    Ok(Json(DeviceFlowStatus::Complete { session_token, login }))
}

/// The account for a signed-in identity:
/// 1. the account this identity already belongs to;
/// 2. else the account of another identity with one of the same verified emails, which
///    this identity joins (Google and GitHub sign-ins of one person become one account);
/// 3. else a new account.
async fn account_for(state: &AppState, id: &Identity, now: i64) -> ApiResult<(i64, String)> {
    let email = id.emails.first().cloned();
    let known: Option<(i64,)> =
        sqlx::query_as("select account_id from identities where provider = $1 and subject = $2")
            .bind(id.provider.as_str())
            .bind(id.subject.clone())
            .fetch_optional(&state.db)
            .await?;
    let account_id = if let Some((account_id,)) = known {
        // Keep the stored address current; a provider may verify one later.
        if email.is_some() {
            sqlx::query("update identities set email = $1 where provider = $2 and subject = $3")
                .bind(email.clone())
                .bind(id.provider.as_str())
                .bind(id.subject.clone())
                .execute(&state.db)
                .await?;
        }
        if id.provider == Provider::Github {
            // GitHub logins can be renamed; the account name follows.
            sqlx::query("update accounts set login = $1 where id = $2")
                .bind(id.login.clone())
                .bind(account_id)
                .execute(&state.db)
                .await?;
        }
        account_id
    } else {
        let mut linked = None;
        for address in &id.emails {
            let row: Option<(i64,)> =
                sqlx::query_as("select account_id from identities where email = $1 order by created_at limit 1")
                    .bind(address.clone())
                    .fetch_optional(&state.db)
                    .await?;
            if let Some((account_id,)) = row {
                linked = Some(account_id);
                break;
            }
        }
        let account_id = match linked {
            Some(account_id) => account_id,
            None => {
                if state.single_owner {
                    let (accounts,): (i64,) =
                        sqlx::query_as("select count(*) from accounts").fetch_one(&state.db).await?;
                    if accounts > 0 {
                        return Err(ApiError::forbidden(
                            "this Mac's cloud belongs to another account; sign in with the account used on the Mac",
                        ));
                    }
                }
                let (account_id,): (i64,) =
                    sqlx::query_as("insert into accounts (login, created_at) values ($1, $2) returning id")
                        .bind(id.login.clone())
                        .bind(now)
                        .fetch_one(&state.db)
                        .await?;
                account_id
            }
        };
        sqlx::query(
            "insert into identities (provider, subject, account_id, email, created_at) values ($1, $2, $3, $4, $5)",
        )
        .bind(id.provider.as_str())
        .bind(id.subject.clone())
        .bind(account_id)
        .bind(email)
        .bind(now)
        .execute(&state.db)
        .await?;
        account_id
    };
    let (login,): (String,) = sqlx::query_as("select login from accounts where id = $1")
        .bind(account_id)
        .fetch_one(&state.db)
        .await?;
    Ok((account_id, login))
}

pub async fn me(account: Account) -> Json<Me> {
    Json(Me { login: account.login })
}

pub async fn logout(State(state): State<AppState>, account: Account) -> ApiResult<StatusCode> {
    sqlx::query("delete from sessions where token_hash = $1")
        .bind(account.token_hash)
        .execute(&state.db)
        .await?;
    Ok(StatusCode::NO_CONTENT)
}

pub fn random_hex(bytes: usize) -> String {
    let data: Vec<u8> = (0..bytes).map(|_| rand::random::<u8>()).collect();
    hex::encode(data)
}
