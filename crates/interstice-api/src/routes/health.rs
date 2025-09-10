//interstice-api/src/routes/health.rs
use axum::{extract::State, routing::get, Router, Json};
use serde_json::json;
use std::sync::Arc;

pub fn health_routes() -> Router<Arc<crate::AppState>> {
    Router::new()
        .route("/", get(health_check))
        .route("/ready", get(readiness_check))
        .route("/timeout-metrics", get(timeout_metrics))
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

/// Endpoint to get timeout metrics for monitoring
async fn timeout_metrics(State(state): State<Arc<crate::AppState>>) -> Json<serde_json::Value> {
    let metrics = state.timeout_manager.get_metrics().await;
    
    // Convert DashMap to HashMap for serialization
    let timeouts_by_operation: std::collections::HashMap<String, u64> = metrics.timeouts_by_operation
        .iter()
        .map(|entry| (entry.key().clone(), *entry.value()))
        .collect();
    
    Json(json!({
        "timeout_metrics": {
            "total_timeouts": metrics.total_timeouts,
            "timeout_rate": metrics.timeout_rate,
            "avg_request_duration_ms": metrics.avg_request_duration * 1000.0,
            "max_request_duration_ms": metrics.max_request_duration.as_millis(),
            "timeouts_by_operation": timeouts_by_operation,
        },
        "timestamp": chrono::Utc::now(),
        "description": "Production timeout monitoring metrics"
    }))
}