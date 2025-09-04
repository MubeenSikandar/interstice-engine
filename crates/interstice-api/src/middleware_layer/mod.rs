// interstice-api/src/middleware_layer/mod.rs

pub mod auth;
pub mod cors;
pub mod rate_limit;
pub mod webhook_auth;

use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use std::time::Duration;
use uuid::Uuid;
use tracing::{info, debug};
use tower::ServiceBuilder;

// Re-export middleware functions
pub use cors::cors_layer;

// Re-export auth middleware and helpers
pub use auth::{
    auth_middleware,
    require_scope,
    require_role,
    require_workspace,
    AuthContext,
    generate_jwt_token,
    generate_api_key,
    revoke_api_key,
    TokenType,
};

/// Create a comprehensive middleware stack for protected routes
pub fn protected_routes_middleware() -> ServiceBuilder<
    tower::layer::util::Stack<
        tower::timeout::TimeoutLayer,
        tower::layer::util::Stack<
            tower_http::limit::RequestBodyLimitLayer,
            tower::layer::util::Identity,
        >,
    >,
> {
    ServiceBuilder::new()
        .layer(body_limit_layer())
        .layer(timeout_layer())
}

/// Create a comprehensive middleware stack for public routes  
pub fn public_routes_middleware() -> ServiceBuilder<
    tower::layer::util::Stack<
        tower::timeout::TimeoutLayer,
        tower::layer::util::Identity,
    >,
> {
    ServiceBuilder::new()
        .layer(timeout_layer())
}

/// Request ID middleware - adds a unique ID to each request
pub async fn request_id_middleware(
    mut request: Request,
    next: Next,
) -> Response {
    let request_id = request
        .headers()
        .get("X-Request-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    debug!("Request ID: {}", request_id);

    // Add to extensions for use in handlers
    request.extensions_mut().insert(request_id.clone());

    // Add to response headers
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        "X-Request-Id",
        request_id.parse().unwrap(),
    );

    response
}

/// Logging middleware - logs all requests with detailed information
pub async fn logging_middleware(
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let path = uri.path().to_string();
    
    // Extract request ID if present
    let request_id = request.extensions()
        .get::<String>()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    
    // Extract user info if authenticated
    let user_id = request.extensions()
        .get::<auth::AuthContext>()
        .map(|ctx| ctx.user_id.clone())
        .unwrap_or_else(|| "anonymous".to_string());
    
    let start = std::time::Instant::now();

    let response = next.run(request).await;
    let duration = start.elapsed();
    let status = response.status();

    // Log with appropriate level based on status
    if status.is_server_error() {
        tracing::error!(
            method = %method,
            path = %path,
            status = %status,
            user_id = %user_id,
            request_id = %request_id,
            duration_ms = %duration.as_millis(),
            "Request failed with server error"
        );
    } else if status.is_client_error() {
        tracing::warn!(
            method = %method,
            path = %path,
            status = %status,
            user_id = %user_id,
            request_id = %request_id,
            duration_ms = %duration.as_millis(),
            "Request failed with client error"
        );
    } else {
        info!(
            method = %method,
            path = %path,
            status = %status,
            user_id = %user_id,
            request_id = %request_id,
            duration_ms = %duration.as_millis(),
            "Request processed successfully"
        );
    }

    response
}

/// Security headers middleware - adds comprehensive security headers
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // Core security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert("X-Permitted-Cross-Domain-Policies", "none".parse().unwrap());
    
    // Content Security Policy
    let csp = if cfg!(debug_assertions) {
        // More permissive in development
        "default-src 'self' http://localhost:*; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'"
    } else {
        // Strict in production
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    };
    headers.insert("Content-Security-Policy", csp.parse().unwrap());
    
    // HSTS (only in production)
    if !cfg!(debug_assertions) {
        headers.insert(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains; preload".parse().unwrap(),
        );
    }

    response
}

/// Compression middleware configuration with all algorithms
pub fn compression_layer() -> tower_http::compression::CompressionLayer {
    tower_http::compression::CompressionLayer::new()
        .gzip(true)
        .br(true)
        .deflate(true)
        .quality(tower_http::CompressionLevel::Default)
}

/// Timeout middleware configuration with configurable duration
pub fn timeout_layer() -> tower::timeout::TimeoutLayer {
    let timeout_secs = std::env::var("REQUEST_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    
    tower::timeout::TimeoutLayer::new(Duration::from_secs(timeout_secs))
}

/// Request body limit middleware with configurable limit
pub fn body_limit_layer() -> tower_http::limit::RequestBodyLimitLayer {
    let limit_mb = std::env::var("REQUEST_BODY_LIMIT_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10);
    
    tower_http::limit::RequestBodyLimitLayer::new(limit_mb * 1024 * 1024)
}

/// Create a comprehensive middleware stack for the entire application
pub fn create_middleware_stack() -> ServiceBuilder<
    tower::layer::util::Stack<
        tower_http::compression::CompressionLayer,
        tower::layer::util::Stack<
            tower_http::trace::TraceLayer<
                tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
            >,
            tower::layer::util::Identity,
        >,
    >,
> {
    ServiceBuilder::new()
        // Tracing/logging
        .layer(tower_http::trace::TraceLayer::new_for_http())
        // Response compression
        .layer(compression_layer())
}