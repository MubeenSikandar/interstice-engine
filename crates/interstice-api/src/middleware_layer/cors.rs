// interstice-api/src/middleware_layer/cors.rs

use axum::http::{
    header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, CACHE_CONTROL, PRAGMA},
    HeaderValue, HeaderName, Method,
};
use tower_http::cors::{CorsLayer, Any};
use tracing::{warn, info};
use std::time::Duration;

/// Production-ready CORS configuration with security considerations
pub fn cors_layer() -> CorsLayer {
    let is_production = std::env::var("ENVIRONMENT")
        .unwrap_or_else(|_| "development".to_string()) == "production";
    
    if is_production {
        production_cors()
    } else {
        development_cors()
    }
}

/// Strict CORS for production environment
fn production_cors() -> CorsLayer {
    let origins = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| {
            warn!("ALLOWED_ORIGINS not set in production, using restrictive defaults");
            "https://app.interstice.com".to_string()
        });
    
    let allowed_origins: Result<Vec<HeaderValue>, _> = origins
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            // Ensure HTTPS in production
            if !s.starts_with("https://") {
                warn!("Non-HTTPS origin in production: {}", s);
            }
            s.parse()
        })
        .collect();
    
    match allowed_origins {
        Ok(origins) => {
            info!("Production CORS configured with {} allowed origins", origins.len());
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([
                    Method::GET,
                    Method::POST,
                    Method::PUT,
                    Method::DELETE,
                    Method::OPTIONS,
                    Method::PATCH,
                ])
                .allow_headers([
                    CONTENT_TYPE,
                    AUTHORIZATION,
                    ACCEPT,
                    CACHE_CONTROL,
                    PRAGMA,
                    HeaderName::from_static("x-api-key"),
                    HeaderName::from_static("x-request-id"),
                    HeaderName::from_static("x-workspace-id"),
                ])
                .expose_headers([
                    HeaderName::from_static("x-request-id"),
                    HeaderName::from_static("x-rate-limit-limit"),
                    HeaderName::from_static("x-rate-limit-remaining"),
                    HeaderName::from_static("x-rate-limit-reset"),
                ])
                .allow_credentials(true)
                .max_age(Duration::from_secs(86400)) // 24 hours
        }
        Err(e) => {
            warn!("Invalid CORS origins, falling back to restrictive default: {}", e);
            CorsLayer::new()
                .allow_origin("https://app.interstice.com".parse::<HeaderValue>().unwrap())
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([CONTENT_TYPE, AUTHORIZATION])
                .allow_credentials(false)
                .max_age(Duration::from_secs(3600))
        }
    }
}

/// Permissive CORS for development environment
fn development_cors() -> CorsLayer {
    let origins = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| {
            "http://localhost:3000,http://localhost:5173,http://localhost:8080".to_string()
        });
    
    let allowed_origins: Vec<HeaderValue> = origins
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    
    info!("Development CORS configured with {} allowed origins", allowed_origins.len());
    
    let cors_layer = CorsLayer::new();
    
    let cors_layer = if allowed_origins.is_empty() {
        cors_layer.allow_origin(Any)
    } else {
        cors_layer.allow_origin(allowed_origins)
    };
    
    cors_layer
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
            Method::PATCH,
            Method::HEAD,
        ])
        .allow_headers([
            CONTENT_TYPE,
            AUTHORIZATION,
            ACCEPT,
            CACHE_CONTROL,
            PRAGMA,
            HeaderName::from_static("x-api-key"),
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("x-workspace-id"),
            HeaderName::from_static("x-forwarded-for"),
            HeaderName::from_static("x-real-ip"),
            HeaderName::from_static("user-agent"),
        ])
        .expose_headers([
            HeaderName::from_static("x-request-id"),
            HeaderName::from_static("x-rate-limit-limit"),
            HeaderName::from_static("x-rate-limit-remaining"),
            HeaderName::from_static("x-rate-limit-reset"),
            HeaderName::from_static("content-length"),
        ])
        .allow_credentials(true)
        .max_age(Duration::from_secs(300)) // 5 minutes
}
