// middleware_layer/analytics.rs

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use interstice_core::{
    analytics::create_tagged_metric, types::MetricValue, WorkspaceId};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug};

use crate::AppState;

/// Analytics tracking middleware
/// Records API request metrics including duration, status codes, and endpoints
pub async fn analytics_tracking(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    // Skip analytics for health checks and metrics endpoints to avoid recursion
    let path = request.uri().path();
    if path.starts_with("/health") || path.starts_with("/api/v1/analytics") {
        return next.run(request).await;
    }
    
    let start = Instant::now();
    let method = request.method().clone();
    let endpoint = sanitize_path(path);
    
    // Extract workspace_id from headers or path
    let workspace_id = extract_workspace_id(&request);
    
    // Extract user_id if present (from auth headers or JWT)
    let user_id = extract_user_id(&request);
    
    // Process the request
    let response = next.run(request).await;
    
    // Record metrics after response
    let duration = start.elapsed();
    let status = response.status();
    
    // Only record if analytics is enabled
    if let Some(analytics) = &state.analytics {
        // Create the main request metric
        let mut event = create_tagged_metric(
            "api.request",
            workspace_id.clone(),
            MetricValue::Duration(duration),
            vec![
                format!("method:{}", method),
                format!("endpoint:{}", endpoint),
                format!("status:{}", status.as_u16()),
                format!("status_class:{}xx", status.as_u16() / 100),
            ],
        );
        
        // Add user context if available
        if let Some(uid) = user_id {
            event.user_id = Some(uid);
        }
        
        // Record the metric asynchronously
        let analytics_clone = analytics.clone();
        tokio::spawn(async move {
            if let Err(e) = analytics_clone.record_metric(event).await {
                debug!("Failed to record request metric: {}", e);
            }
        });
        
        // Record error metrics for 5xx responses
        if status.is_server_error() {
            record_error_metric(analytics.clone(), workspace_id, endpoint.clone(), status).await;
        }
        
        // Record slow request metrics (> 1 second)
        if duration.as_secs() > 1 {
            record_slow_request_metric(
                analytics.clone(),
                workspace_id,
                endpoint,
                duration,
                method.to_string(),
            ).await;
        }
    }
    
    response
}

/// Extract workspace_id from request
fn extract_workspace_id(request: &Request) -> WorkspaceId {
    // Try header first
    if let Some(header_value) = request.headers().get("X-Workspace-Id") {
        if let Ok(header_str) = header_value.to_str() {
            if let Ok(workspace_id) = header_str.parse() {
                return workspace_id;
            }
        }
    }
    
    // Try to extract from path (e.g., /api/v1/workspaces/{id}/...)
    let path = request.uri().path();
    if let Some(workspace_id) = extract_workspace_from_path(path) {
        return workspace_id;
    }
    
    // Default workspace
    WorkspaceId::new()
}

/// Extract user_id from request (if authenticated)
fn extract_user_id(request: &Request) -> Option<interstice_core::types::UserId> {
    // Try to get from header (set by auth middleware)
    request.headers()
        .get("X-User-Id")
        .and_then(|v| v.to_str().ok())
        .map(|s| interstice_core::types::UserId::new(s))
}

/// Extract workspace_id from path
fn extract_workspace_from_path(path: &str) -> Option<WorkspaceId> {
    // Pattern: /api/v1/workspaces/{workspace_id}/...
    let parts: Vec<&str> = path.split('/').collect();
    
    for (i, part) in parts.iter().enumerate() {
        if *part == "workspaces" && i + 1 < parts.len() {
            if let Ok(workspace_id) = parts[i + 1].parse() {
                return Some(workspace_id);
            }
        }
    }
    
    None
}

/// Sanitize path for metrics (remove IDs)
fn sanitize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    let mut sanitized = Vec::new();
    
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        
        // Check if this looks like a UUID or ID
        if is_likely_id(part) {
            // Replace with placeholder based on previous segment
            if i > 0 {
                let prev = parts[i - 1];
                sanitized.push(format!(":{}", singularize(prev)));
            } else {
                sanitized.push(":id".to_string());
            }
        } else {
            sanitized.push(part.to_string());
        }
    }
    
    format!("/{}", sanitized.join("/"))
}

/// Check if a path segment is likely an ID
fn is_likely_id(segment: &str) -> bool {
    // Check for UUID pattern
    if segment.len() == 36 && segment.chars().filter(|c| *c == '-').count() == 4 {
        return true;
    }
    
    // Check for numeric ID
    if segment.parse::<i64>().is_ok() {
        return true;
    }
    
    // Check for base64-like IDs (common in some systems)
    if segment.len() > 8 && segment.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return true;
    }
    
    false
}

/// Simple singularization for common cases
fn singularize(word: &str) -> &str {
    match word {
        "workspaces" => "workspace",
        "users" => "user",
        "artifacts" => "artifact",
        "outcomes" => "outcome",
        "webhooks" => "webhook",
        "messages" => "message",
        _ => word,
    }
}

/// Record error metrics
async fn record_error_metric(
    analytics: Arc<interstice_core::analytics::AnalyticsEngine>,
    workspace_id: WorkspaceId,
    endpoint: String,
    status: StatusCode,
) {
    let event = create_tagged_metric(
        "api.error",
        workspace_id,
        MetricValue::Integer(1),
        vec![
            format!("endpoint:{}", endpoint),
            format!("status:{}", status.as_u16()),
        ],
    );
    
    tokio::spawn(async move {
        if let Err(e) = analytics.record_metric(event).await {
            debug!("Failed to record error metric: {}", e);
        }
    });
}

/// Record slow request metrics
async fn record_slow_request_metric(
    analytics: Arc<interstice_core::analytics::AnalyticsEngine>,
    workspace_id: WorkspaceId,
    endpoint: String,
    duration: std::time::Duration,
    method: String,
) {
    let event = create_tagged_metric(
        "api.slow_request",
        workspace_id,
        MetricValue::Duration(duration),
        vec![
            format!("endpoint:{}", endpoint),
            format!("method:{}", method),
        ],
    );
    
    tokio::spawn(async move {
        if let Err(e) = analytics.record_metric(event).await {
            debug!("Failed to record slow request metric: {}", e);
        }
    });
}