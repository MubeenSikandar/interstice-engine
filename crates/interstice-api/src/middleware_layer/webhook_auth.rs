// interstice-api/src/middleware_layer/webhook_auth.rs

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use tracing::{info, warn, error};
use crate::AppState;
use super::rate_limit::extract_client_ip;

type HmacSha256 = Hmac<Sha256>;

/// GitHub webhook signature verification middleware
pub fn github_signature_middleware() -> tower::ServiceBuilder<tower::layer::util::Identity> {
    tower::ServiceBuilder::new()
    // GitHub signature verification happens in the handler
}

/// Custom webhook authentication middleware  
pub fn custom_webhook_auth() -> tower::ServiceBuilder<tower::layer::util::Identity> {
    tower::ServiceBuilder::new()
    // Custom webhook auth happens in the handler
}

/// Verify GitHub webhook signature
pub async fn verify_github_signature(
    headers: &HeaderMap,
    body: &[u8],
    secret: &str,
) -> Result<(), StatusCode> {
    // Get signature from headers
    let signature = headers
        .get("X-Hub-Signature-256")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing GitHub signature header");
            StatusCode::UNAUTHORIZED
        })?;

    // Signature format: sha256=<signature>
    if !signature.starts_with("sha256=") {
        warn!("Invalid GitHub signature format");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let expected_signature = &signature[7..];

    // Calculate HMAC
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|e| {
            error!("Failed to create HMAC: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    mac.update(body);
    let result = mac.finalize();
    let calculated_signature = format!("{:x}", result.into_bytes());

    // Compare signatures (constant time)
    if !constant_time_compare(&calculated_signature, expected_signature) {
        warn!("GitHub signature verification failed");
        return Err(StatusCode::UNAUTHORIZED);
    }

    info!("GitHub signature verified successfully");
    Ok(())
}

/// Main webhook signature verification middleware
pub async fn verify_webhook_signature(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    // Apply webhook rate limiting first
    let client_ip = extract_client_ip(&request);
    let client_id = format!("webhook:{}", client_ip);
    
    // Check rate limit for webhooks
    let allowed = state.rate_limiter.check_rate_limit(&client_id).await;
    
    if !allowed {
        warn!(
            client_ip = %client_ip,
            path = %request.uri().path(),
            "Webhook rate limit exceeded"
        );
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }

    // For Slack webhooks, we use their specific verification
    if request.uri().path().contains("slack") {
        // Slack verification is handled in the handler
        return next.run(request).await;
    }

    // For other webhooks, use general verification
    let (parts, body) = request.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return StatusCode::BAD_REQUEST.into_response();
        }
    };

    // Verify based on webhook type
    let verification_result = if parts.headers.get("X-GitHub-Event").is_some() {
        // GitHub webhook
        if let Ok(secret) = std::env::var("GITHUB_WEBHOOK_SECRET") {
            verify_github_signature(&parts.headers, &body_bytes, &secret).await
        } else {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    } else {
        // Custom webhook
        if let Ok(secret) = std::env::var("WEBHOOK_SECRET") {
            verify_custom_webhook_signature(&parts.headers, &body_bytes, &secret).await
        } else {
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    };

    if let Err(status) = verification_result {
        return status.into_response();
    }

    // Reconstruct request with body
    let request = Request::from_parts(parts, axum::body::Body::from(body_bytes));
    next.run(request).await
}

/// Verify custom webhook signature using HMAC-SHA256
pub async fn verify_custom_webhook_signature(
    headers: &HeaderMap,
    body: &[u8],
    secret: &str,
) -> Result<(), StatusCode> {
    // Get signature from headers
    let signature = headers
        .get("X-Webhook-Signature")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing webhook signature header");
            StatusCode::UNAUTHORIZED
        })?;

    // Get timestamp for replay protection
    let timestamp = headers
        .get("X-Webhook-Timestamp")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i64>().ok())
        .ok_or_else(|| {
            warn!("Missing or invalid webhook timestamp");
            StatusCode::UNAUTHORIZED
        })?;

    // Check timestamp (5 minute window)
    let current_time = chrono::Utc::now().timestamp();
    if (current_time - timestamp).abs() > 300 {
        warn!("Webhook timestamp outside acceptable window");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Calculate HMAC with timestamp
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .map_err(|e| {
            error!("Failed to create HMAC: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    mac.update(format!("{}", timestamp).as_bytes());
    mac.update(body);
    let result = mac.finalize();
    let calculated_signature = format!("{:x}", result.into_bytes());

    // Compare signatures
    if !constant_time_compare(&calculated_signature, signature) {
        warn!("Webhook signature verification failed");
        return Err(StatusCode::UNAUTHORIZED);
    }

    info!("Webhook signature verified successfully");
    Ok(())
}

/// Constant time string comparison to prevent timing attacks
fn constant_time_compare(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut result = 0u8;

    for i in 0..a.len() {
        result |= a_bytes[i] ^ b_bytes[i];
    }

    result == 0
}