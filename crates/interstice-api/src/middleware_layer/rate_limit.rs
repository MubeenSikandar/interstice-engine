// interstice-api/src/middleware_layer/rate_limit.rs

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::Serialize;
use tracing::warn;
use crate::AppState;

const DEFAULT_RATE_LIMIT: u32 = 100; // requests per minute
const DEFAULT_WINDOW_SECS: u64 = 60; // 1 minute window
const BURST_LIMIT: u32 = 10; // Allow burst of 10 requests

#[derive(Debug, Serialize)]
struct RateLimitError {
    error: String,
    retry_after: u64,
    limit: u32,
    remaining: u32,
}

/// Create rate limiting middleware for Slack
pub fn slack_rate_limit() -> tower::ServiceBuilder<tower::layer::util::Identity> {
    tower::ServiceBuilder::new()
}

/// General rate limiting middleware
pub async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    // Extract client identifier (IP or user ID)
    let client_id = extract_client_id(&request);
    
    // Get rate limit for this client
    let (limit, window) = get_rate_limit(&state, &client_id).await;
    
    // Check rate limit
    match check_rate_limit(&state, &client_id, limit, window).await {
        Ok(remaining) => {
            // Add rate limit headers to response
            let mut response = next.run(request).await;
            let headers = response.headers_mut();
            headers.insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Remaining", remaining.to_string().parse().unwrap());
            headers.insert("X-RateLimit-Reset", get_reset_time(window).to_string().parse().unwrap());
            
            Ok(response)
        }
        Err(retry_after) => {
            warn!("Rate limit exceeded for client: {}", client_id);
            Err(rate_limit_exceeded_response(limit, 0, retry_after))
        }
    }
}

/// Extract client identifier from request
fn extract_client_id(request: &Request) -> String {
    // Try to get authenticated user ID
    if let Some(auth_context) = request.extensions().get::<super::auth::AuthContext>() {
        return auth_context.user_id.clone();
    }
    
    // Fallback to IP address
    request
        .headers()
        .get("X-Forwarded-For")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or(s).trim().to_string())
        .or_else(|| {
            request
                .headers()
                .get("X-Real-IP")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Get rate limit for a client
async fn get_rate_limit(
    state: &Arc<AppState>,
    client_id: &str,
) -> (u32, u64) {
    // Check for custom rate limits in database
    if let Ok(Some(custom_limit)) = sqlx::query!(
        r#"
        SELECT rate_limit, window_seconds
        FROM rate_limit_overrides
        WHERE client_id = $1 AND active = true
        "#,
        client_id
    )
    .fetch_optional(&state.db)
    .await
    {
        return (
            custom_limit.rate_limit as u32,
            custom_limit.window_seconds as u64,
        );
    }
    
    // Check if user has API key with custom rate limit
    if let Ok(Some(api_key_limit)) = sqlx::query!(
        r#"
        SELECT rate_limit
        FROM api_keys
        WHERE id::text = $1 AND revoked = false
        "#,
        client_id
    )
    .fetch_optional(&state.db)
    .await
    {
        if let Some(limit) = api_key_limit.rate_limit {
            return (limit as u32, DEFAULT_WINDOW_SECS);
        }
    }
    
    // Return default rate limit
    (DEFAULT_RATE_LIMIT, DEFAULT_WINDOW_SECS)
}

/// Check rate limit using sliding window algorithm
async fn check_rate_limit(
    state: &Arc<AppState>,
    client_id: &str,
    limit: u32,
    window_secs: u64,
) -> Result<u32, u64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let window_start = now - window_secs;
    
    // Count requests in current window
    let count = sqlx::query!(
        r#"
        SELECT COUNT(*) as count
        FROM rate_limit_tracking
        WHERE client_id = $1 AND timestamp >= $2
        "#,
        client_id,
        window_start as i64
    )
    .fetch_one(&state.db)
    .await
    .map(|r| r.count.unwrap_or(0) as u32)
    .unwrap_or(0);
    
    // Check if limit exceeded
    if count >= limit {
        // Calculate retry after
        let oldest_request = sqlx::query!(
            r#"
            SELECT MIN(timestamp) as oldest
            FROM rate_limit_tracking
            WHERE client_id = $1 AND timestamp >= $2
            "#,
            client_id,
            window_start as i64
        )
        .fetch_one(&state.db)
        .await
        .ok()
        .and_then(|r| r.oldest)
        .unwrap_or(now as i64);
        
        let retry_after = (oldest_request as u64 + window_secs).saturating_sub(now);
        return Err(retry_after);
    }
    
    // Track this request
    let _ = sqlx::query!(
        r#"
        INSERT INTO rate_limit_tracking (client_id, timestamp)
        VALUES ($1, $2)
        "#,
        client_id,
        now as i64
    )
    .execute(&state.db)
    .await;
    
    // Clean up old entries (async, don't wait)
    let state_clone = state.clone();
    let client_id_clone = client_id.to_string();
    tokio::spawn(async move {
        let _ = sqlx::query!(
            r#"
            DELETE FROM rate_limit_tracking
            WHERE client_id = $1 AND timestamp < $2
            "#,
            client_id_clone,
            window_start as i64
        )
        .execute(&state_clone.db)
        .await;
    });
    
    Ok(limit - count - 1)
}

/// Get reset time for rate limit window
fn get_reset_time(window_secs: u64) -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + window_secs
}

/// Create rate limit exceeded response
fn rate_limit_exceeded_response(limit: u32, remaining: u32, retry_after: u64) -> Response {
    let mut response = (
        StatusCode::TOO_MANY_REQUESTS,
        Json(RateLimitError {
            error: "rate_limit_exceeded".to_string(),
            retry_after,
            limit,
            remaining,
        }),
    )
    .into_response();
    
    let headers = response.headers_mut();
    headers.insert("Retry-After", retry_after.to_string().parse().unwrap());
    headers.insert("X-RateLimit-Limit", limit.to_string().parse().unwrap());
    headers.insert("X-RateLimit-Remaining", remaining.to_string().parse().unwrap());
    
    response
}

/// IP-based rate limiting for public endpoints
pub async fn ip_rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    // Simple in-memory rate limiting using a static HashMap
    // In production, use Redis or similar
    use std::collections::HashMap;
    use std::sync::Mutex;
    
    static RATE_LIMITS: std::sync::OnceLock<Mutex<HashMap<String, (u32, SystemTime)>>> = 
        std::sync::OnceLock::new();
    
    let rate_limits = RATE_LIMITS.get_or_init(|| Mutex::new(HashMap::new()));
    
    let client_ip = extract_client_ip(&request);
    let now = SystemTime::now();
    
    let mut limits = rate_limits.lock().unwrap();
    let entry = limits.entry(client_ip.clone()).or_insert((0, now));
    
    // Reset counter if window expired
    if now.duration_since(entry.1).unwrap_or_default() > Duration::from_secs(60) {
        *entry = (1, now);
    } else {
        entry.0 += 1;
    }
    
    if entry.0 > 60 { // 60 requests per minute for public endpoints
        warn!("IP rate limit exceeded: {}", client_ip);
        return Err(rate_limit_exceeded_response(60, 0, 60));
    }
    
    Ok(next.run(request).await)
}

/// Extract client IP from request
fn extract_client_ip(request: &Request) -> String {
    request
        .headers()
        .get("X-Forwarded-For")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or(s).trim().to_string())
        .or_else(|| {
            request
                .headers()
                .get("X-Real-IP")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "127.0.0.1".to_string())
}