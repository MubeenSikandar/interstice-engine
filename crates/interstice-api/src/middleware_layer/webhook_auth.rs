// interstice-api/src/middleware_layer/webhook_auth.rs

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use tracing::{info, warn, error};
use crate::AppState;

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

/// Verify webhook IP whitelist
/// Verify webhook IP whitelist
pub async fn verify_webhook_ip(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP
    let client_ip = request
        .headers()
        .get("X-Forwarded-For")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(',').next().unwrap_or(s).trim())
        .or_else(|| {
            request
                .headers()
                .get("X-Real-IP")
                .and_then(|v| v.to_str().ok())
        })
        .unwrap_or("unknown");

    let is_whitelisted = sqlx::query!(
        r#"
        SELECT 1 as exists FROM webhook_ip_whitelist
        WHERE ip_address::text = $1
        "#,
        client_ip
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to check IP whitelist: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .is_some();

    if !is_whitelisted {
        warn!("Webhook request from non-whitelisted IP: {}", client_ip);
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

/// Verify webhook bearer token
pub async fn verify_webhook_bearer_token(
    headers: &HeaderMap,
    expected_token: &str,
) -> Result<(), StatusCode> {
    let auth_header = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing Authorization header");
            StatusCode::UNAUTHORIZED
        })?;

    if !auth_header.starts_with("Bearer ") {
        warn!("Invalid Authorization header format");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = &auth_header[7..];
    
    if !constant_time_compare(token, expected_token) {
        warn!("Invalid webhook bearer token");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(())
}

/// Webhook retry mechanism with exponential backoff
pub async fn retry_webhook(
    url: &str,
    payload: &serde_json::Value,
    secret: Option<&str>,
    max_retries: u32,
) -> Result<reqwest::Response, String> {
    let client = reqwest::Client::new();
    let mut attempt = 0;
    let mut delay = Duration::from_secs(1);

    loop {
        attempt += 1;
        
        let mut request = client
            .post(url)
            .json(payload)
            .timeout(Duration::from_secs(30));

        // Add signature if secret provided
        if let Some(secret) = secret {
            let timestamp = chrono::Utc::now().timestamp();
            let body = serde_json::to_vec(payload).unwrap();
            
            let mut mac = HmacSha256::new_from_slice(secret.as_bytes()).unwrap();
            mac.update(format!("{}", timestamp).as_bytes());
            mac.update(&body);
            let signature = format!("{:x}", mac.finalize().into_bytes());

            request = request
                .header("X-Webhook-Signature", signature)
                .header("X-Webhook-Timestamp", timestamp.to_string());
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                info!("Webhook delivered successfully to {}", url);
                return Ok(response);
            }
            Ok(response) => {
                warn!(
                    "Webhook delivery failed with status {}: attempt {}/{}",
                    response.status(),
                    attempt,
                    max_retries
                );
                
                if attempt >= max_retries {
                    return Err(format!("Webhook delivery failed after {} attempts", max_retries));
                }
            }
            Err(e) => {
                warn!(
                    "Webhook delivery error: {} - attempt {}/{}",
                    e,
                    attempt,
                    max_retries
                );
                
                if attempt >= max_retries {
                    return Err(format!("Webhook delivery failed: {}", e));
                }
            }
        }

        // Exponential backoff
        tokio::time::sleep(delay).await;
        delay = std::cmp::min(delay * 2, Duration::from_secs(60));
    }
}

use std::time::Duration;