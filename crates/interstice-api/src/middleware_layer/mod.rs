// interstice-api/src/middleware_layer/mod.rs

pub mod auth;
pub mod rate_limit;
pub mod webhook_auth;

use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use tower_http::cors::{CorsLayer, Any};
use std::time::Duration;
use uuid::Uuid;
use tracing::info;

// Re-export commonly used middleware
pub use auth::{auth_middleware, require_scope, require_role, require_workspace};
pub use rate_limit::{rate_limit_middleware, ip_rate_limit_middleware};

/// Create CORS layer for the application
pub fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .expose_headers([
            "X-Request-Id".parse().unwrap(),
            "X-RateLimit-Limit".parse().unwrap(),
            "X-RateLimit-Remaining".parse().unwrap(),
            "X-RateLimit-Reset".parse().unwrap(),
        ])
        .max_age(Duration::from_secs(3600))
}

/// Request ID middleware - adds a unique ID to each request
pub async fn request_id_middleware(
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let request_id = request
        .headers()
        .get("X-Request-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    // Add to extensions for use in handlers
    request.extensions_mut().insert(request_id.clone());

    // Add to response headers
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "X-Request-Id",
        request_id.parse().unwrap(),
    );

    Ok(response)
}

/// Logging middleware - logs all requests
pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let start = std::time::Instant::now();

    let response = next.run(request).await;
    let duration = start.elapsed();

    info!(
        method = %method,
        uri = %uri,
        status = %response.status(),
        duration_ms = %duration.as_millis(),
        "Request processed"
    );

    Ok(response)
}

/// Security headers middleware
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // Add security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'".parse().unwrap(),
    );
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );

    Ok(response)
}

/// Compression middleware configuration
pub fn compression_layer() -> tower_http::compression::CompressionLayer {
    tower_http::compression::CompressionLayer::new()
        .gzip(true)
        .br(true)
        .deflate(true)
}

/// Timeout middleware configuration
pub fn timeout_layer() -> tower::timeout::TimeoutLayer {
    tower::timeout::TimeoutLayer::new(Duration::from_secs(30))
}

/// Request body limit middleware
pub fn body_limit_layer() -> tower_http::limit::RequestBodyLimitLayer {
    tower_http::limit::RequestBodyLimitLayer::new(
        10 * 1024 * 1024, // 10MB default limit
    )
}