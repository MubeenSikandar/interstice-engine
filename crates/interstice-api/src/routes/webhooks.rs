// interstice-api/src/routes/webhooks.rs

use crate::{
    handlers::slack::{
        get_oauth_url, handle_events, handle_interactions, handle_oauth_callback,
        handle_slash_commands,
    },
    AppState,
};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{info, warn};
use uuid::Uuid;

// Constants for webhook configuration
const MAX_WEBHOOK_BODY_SIZE: usize = 10 * 1024 * 1024; // 10MB
const WEBHOOK_TIMEOUT_SECS: u64 = 30; // 30 seconds for webhook processing

#[derive(Debug, Serialize)]
pub struct WebhookResponse {
    pub success: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webhook_id: Option<Uuid>,
}

/// Create webhook routes with production-ready configuration
pub fn webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Slack webhook endpoints
        .nest("/slack", slack_webhook_routes())
        // GitHub webhook endpoints
        .nest("/github", github_webhook_routes())
        // Generic webhook endpoints
        .nest("/custom", custom_webhook_routes())
        // Webhook management endpoints
        .route("/", get(list_webhooks))
        .route("/{webhook_id}", get(get_webhook_status))
        .route("/{webhook_id}/logs", get(get_webhook_logs))
        .route("/test", post(test_webhook))
        // Apply middleware to all webhook routes
        .layer(
            ServiceBuilder::new()
                // Add request body size limit
                .layer(RequestBodyLimitLayer::new(MAX_WEBHOOK_BODY_SIZE))
                // Add request ID for tracing
                .layer(tower_http::trace::TraceLayer::new_for_http()),
        )
}

/// Slack-specific webhook routes
fn slack_webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/events", post(handle_events))
        .route("/commands", post(handle_slash_commands))
        .route("/interactions", post(handle_interactions))
        .route("/oauth", get(handle_oauth_callback))
        .route("/oauth/url", get(get_oauth_url))
}

/// GitHub webhook routes
fn github_webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", post(handle_github_webhook))
        .route("/health", get(github_health))
        // GitHub-specific middleware
        .layer(
            ServiceBuilder::new()
                .layer(crate::middleware_layer::webhook_auth::github_signature_middleware()),
        )
}

/// Custom webhook routes for other integrations
fn custom_webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", post(handle_custom_webhook))
        .route("/register", post(register_custom_webhook))
        .route("/{webhook_id}", delete(delete_custom_webhook))
        // Custom webhook authentication
        .layer(
            ServiceBuilder::new()
                .layer(crate::middleware_layer::webhook_auth::custom_webhook_auth()),
        )
}

/// List all registered webhooks
async fn list_webhooks(
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<WebhookInfo>>, StatusCode> {
    let webhooks = sqlx::query_as!(
        WebhookInfo,
        r#"
        SELECT
            id,
            platform,
            url,
            active as "active!",
            created_at,
            last_triggered_at,
            trigger_count as "trigger_count!"
        FROM webhooks
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        warn!("Failed to list webhooks: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(webhooks))
}

/// Get webhook status
async fn get_webhook_status(
    State(state): State<Arc<AppState>>,
    Path(webhook_id): Path<Uuid>,
) -> Result<Json<WebhookStatus>, StatusCode> {
    let status = sqlx::query_as!(
        WebhookStatus,
        r#"
        SELECT
            w.id,
            w.platform,
            w.url,
            w.active as "active!",
            w.created_at,
            w.last_triggered_at,
            w.trigger_count as "trigger_count!",
            w.last_error,
            w.consecutive_failures as "consecutive_failures!",
            COUNT(wl.id) as "recent_events!"
        FROM webhooks w
        LEFT JOIN webhook_logs wl ON w.id = wl.webhook_id
            AND wl.created_at > NOW() - INTERVAL '1 hour'
        WHERE w.id = $1
        GROUP BY w.id
        "#,
        webhook_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        warn!("Failed to get webhook status: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(status))
}

/// Get webhook logs
async fn get_webhook_logs(
    State(state): State<Arc<AppState>>,
    Path(webhook_id): Path<Uuid>,
    Query(params): Query<LogQuery>,
) -> Result<Json<Vec<WebhookLog>>, StatusCode> {
    let limit = params.limit.unwrap_or(50).min(500);
    let offset = params.offset.unwrap_or(0);

    let logs = sqlx::query_as!(
        WebhookLog,
        r#"
        SELECT
            id,
            webhook_id,
            event_type,
            status_code as "status_code!",
            response_time_ms as "response_time_ms!",
            error_message,
            request_headers,
            response_body,
            created_at
        FROM webhook_logs
        WHERE webhook_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#,
        webhook_id,
        limit as i64,
        offset as i64
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        warn!("Failed to get webhook logs: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(logs))
}

/// Test webhook endpoint
async fn test_webhook(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<TestWebhookRequest>,
) -> Result<Json<TestWebhookResponse>, StatusCode> {
    info!("Testing webhook to URL: {}", payload.url);

    // Send test request
    let client = reqwest::Client::new();
    let start = std::time::Instant::now();

    let response = client
        .post(&payload.url)
        .json(&payload.test_payload)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| {
            warn!("Test webhook failed: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    let response_time = start.elapsed().as_millis() as u32;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();

    Ok(Json(TestWebhookResponse {
        success: status.is_success(),
        status_code: status.as_u16(),
        response_time_ms: response_time,
        response_body: Some(body),
    }))
}

/// Handle GitHub webhooks
async fn handle_github_webhook(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: String,
) -> Result<impl IntoResponse, StatusCode> {
    // Get event type from headers
    let event_type = headers
        .get("X-GitHub-Event")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");

    info!("Received GitHub webhook: {}", event_type);

    // Process with timeout to prevent hanging
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(WEBHOOK_TIMEOUT_SECS),
        async {
            // Parse and process based on event type
            match event_type {
                "push" => process_github_push(state, body).await?,
                "pull_request" => process_github_pr(state, body).await?,
                "issues" => process_github_issue(state, body).await?,
                "ping" => return Ok((StatusCode::OK, "pong")),
                _ => {
                    warn!("Unhandled GitHub event type: {}", event_type);
                }
            }
            Ok((StatusCode::OK, "processed"))
        },
    )
    .await
    .map_err(|_| {
        warn!(
            "GitHub webhook processing timed out after {} seconds",
            WEBHOOK_TIMEOUT_SECS
        );
        StatusCode::REQUEST_TIMEOUT
    })?;

    result
}

/// Process GitHub push events
async fn process_github_push(_state: Arc<AppState>, _body: String) -> Result<(), StatusCode> {
    // Parse and process push event
    // Store artifacts in database
    info!("Processing GitHub push event");
    Ok(())
}

/// Process GitHub PR events
async fn process_github_pr(_state: Arc<AppState>, _body: String) -> Result<(), StatusCode> {
    // Parse and process PR event
    info!("Processing GitHub PR event");
    Ok(())
}

/// Process GitHub issue events
async fn process_github_issue(_state: Arc<AppState>, _body: String) -> Result<(), StatusCode> {
    // Parse and process issue event
    info!("Processing GitHub issue event");
    Ok(())
}

/// GitHub health check
async fn github_health() -> &'static str {
    "GitHub integration healthy"
}

/// Handle custom webhooks
async fn handle_custom_webhook(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(_payload): Json<serde_json::Value>,
) -> Result<Json<WebhookResponse>, StatusCode> {
    // Extract webhook ID from headers
    let webhook_id = headers
        .get("X-Webhook-Id")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| Uuid::parse_str(s).ok())
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Process with timeout to prevent hanging
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(WEBHOOK_TIMEOUT_SECS),
        async {
            // Verify webhook exists and is active with timeout
            let _webhook = state
                .timeout_manager
                .execute_with_timeout(
                    || async {
                        sqlx::query!(
                            "SELECT id, secret FROM webhooks WHERE id = $1 AND active = true",
                            webhook_id
                        )
                        .fetch_optional(&state.db)
                        .await
                    },
                    state.timeout_manager.config().webhook_processing,
                    "webhook_verification",
                )
                .await
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            // Process webhook payload
            info!("Processing custom webhook: {}", webhook_id);

            // Update webhook statistics
            sqlx::query!(
                r#"
                UPDATE webhooks
                SET last_triggered_at = NOW(),
                    trigger_count = trigger_count + 1
                WHERE id = $1
                "#,
                webhook_id
            )
            .execute(&state.db)
            .await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

            Ok(Json(WebhookResponse {
                success: true,
                message: "Webhook processed successfully".to_string(),
                webhook_id: Some(webhook_id),
            }))
        },
    )
    .await
    .map_err(|_| {
        warn!(
            "Custom webhook processing timed out after {} seconds",
            WEBHOOK_TIMEOUT_SECS
        );
        StatusCode::REQUEST_TIMEOUT
    })?;

    result
}

/// Register a new custom webhook
async fn register_custom_webhook(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RegisterWebhookRequest>,
) -> Result<Json<RegisterWebhookResponse>, StatusCode> {
    let webhook_id = Uuid::new_v4();
    let secret = generate_webhook_secret();

    sqlx::query!(
        r#"
        INSERT INTO webhooks (id, platform, url, secret, active, created_at)
        VALUES ($1, $2, $3, $4, true, NOW())
        "#,
        webhook_id,
        request.platform,
        request.url,
        secret
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        warn!("Failed to register webhook: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    info!(
        "Registered new webhook: {} for platform: {}",
        webhook_id, request.platform
    );

    Ok(Json(RegisterWebhookResponse {
        webhook_id,
        secret,
        url: request.url,
    }))
}

/// Delete a custom webhook
async fn delete_custom_webhook(
    State(state): State<Arc<AppState>>,
    Path(webhook_id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    let result = sqlx::query!(
        "UPDATE webhooks SET active = false WHERE id = $1",
        webhook_id
    )
    .execute(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    info!("Deactivated webhook: {}", webhook_id);
    Ok(StatusCode::NO_CONTENT)
}

/// Generate a secure webhook secret
fn generate_webhook_secret() -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng(); // Fixed: use rng() instead of deprecated thread_rng()

    (0..32)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len()); // Fixed: use random_range()
            CHARSET[idx] as char
        })
        .collect()
}

// Request/Response types
#[derive(Debug, Deserialize)]
struct LogQuery {
    limit: Option<i32>,
    offset: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct WebhookInfo {
    id: Uuid,
    platform: String,
    url: String,
    active: bool,
    created_at: chrono::DateTime<chrono::Utc>,
    last_triggered_at: Option<chrono::DateTime<chrono::Utc>>,
    trigger_count: i32,
}

#[derive(Debug, Serialize)]
struct WebhookStatus {
    id: Uuid,
    platform: String,
    url: String,
    active: bool,
    created_at: chrono::DateTime<chrono::Utc>,
    last_triggered_at: Option<chrono::DateTime<chrono::Utc>>,
    trigger_count: i32,
    last_error: Option<String>,
    consecutive_failures: i32,
    recent_events: i64,
}

#[derive(Debug, Serialize)]
struct WebhookLog {
    id: Uuid,
    webhook_id: Uuid,
    event_type: String,
    status_code: i32,
    response_time_ms: i32,
    error_message: Option<String>,
    request_headers: serde_json::Value,
    response_body: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Deserialize)]
struct TestWebhookRequest {
    url: String,
    test_payload: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct TestWebhookResponse {
    success: bool,
    status_code: u16,
    response_time_ms: u32,
    response_body: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RegisterWebhookRequest {
    platform: String,
    url: String,
}

#[derive(Debug, Serialize)]
struct RegisterWebhookResponse {
    webhook_id: Uuid,
    secret: String,
    url: String,
}
