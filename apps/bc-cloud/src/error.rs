use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use bc_proto::cloud::ApiErrorBody;

#[derive(Debug)]
pub struct ApiError(pub StatusCode, pub String);

pub type ApiResult<T> = Result<T, ApiError>;

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::BAD_REQUEST, msg.into())
    }
    pub fn unauthorized(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::UNAUTHORIZED, msg.into())
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::NOT_FOUND, msg.into())
    }
    pub fn conflict(msg: impl Into<String>) -> Self {
        ApiError(StatusCode::CONFLICT, msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ApiErrorBody { error: self.1 })).into_response()
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(e: sqlx::Error) -> Self {
        eprintln!("database error: {e}");
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, "database error".into())
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        eprintln!("internal error: {e:#}");
        ApiError(StatusCode::BAD_GATEWAY, "upstream error".into())
    }
}
