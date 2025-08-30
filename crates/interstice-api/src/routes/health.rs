use axum::{routing::get, Router, Json};
use serde_json::json;

pub fn health_routes() -> Router {
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