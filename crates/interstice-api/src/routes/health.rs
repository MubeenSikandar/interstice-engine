//interstice-api/src/routes/health.rs
use axum::{routing::get, Router, Json};
use serde_json::json;
use std::sync::Arc;

pub fn health_routes() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/", get(health_check))
        .route("/ready", get(readiness_check))
}

async fn health_check() -> &'static str {
    "OK"
}

async fn readiness_check() -> Json<serde_json::Value> {
    // Check database, redis, etc.
    Json(json!({
        "status": "ready",
        "timestamp": chrono::Utc::now()
    }))
}