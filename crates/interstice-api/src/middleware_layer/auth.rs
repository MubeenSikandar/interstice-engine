use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use crate::AppState;

pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract authorization header
    let auth_header = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    // For now, just check if header exists
    // In production, validate JWT or API key
    if let Some(auth) = auth_header {
        if auth.starts_with("Bearer ") {
            // Validate token
            let token = &auth[7..];
            if validate_token(token).await {
                return Ok(next.run(request).await);
            }
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

async fn validate_token(token: &str) -> bool {
    // TODO: Implement actual token validation
    // For now, accept any non-empty token
    !token.is_empty()
}

pub async fn api_key_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let api_key = request
        .headers()
        .get("x-api-key")
        .and_then(|h| h.to_str().ok());

    if let Some(key) = api_key {
        // Validate API key
        if validate_api_key(key).await {
            return Ok(next.run(request).await);
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

async fn validate_api_key(key: &str) -> bool {
    // TODO: Check against database
    key == std::env::var("API_KEY").unwrap_or_default()
}