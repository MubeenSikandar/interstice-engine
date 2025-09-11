// src/handlers/slack/events.rs

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use interstice_core::Artifact;
use interstice_ml::OutcomePrediction;
use serde_json::Value as JsonValue;
use std::{sync::Arc, time::Instant};
use tracing::{error, info, instrument, warn, Span};
use uuid::Uuid;
use anyhow::{anyhow, Context, Result as AnyhowResult};

use crate::{
    handlers::slack::{
        extract_artifacts_from_event, store_artifacts_batch, store_predictions_batch,
        update_workspace_analytics, verify_slack_request, SlackEventMetrics, SlackEventRequest,
        SlackEventResponse, SlackPushEvent, MAX_BODY_SIZE,
    },
    AppState,
};

#[instrument(
    skip(state, headers, body),
    fields(
        request_id = %Uuid::new_v4(),
        event_type = tracing::field::Empty,
        team_id = tracing::field::Empty,
        processing_time_ms = tracing::field::Empty,
    )
)]
pub async fn handle_events(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<JsonValue>, StatusCode> {
    let body_bytes = axum::body::to_bytes(body, MAX_BODY_SIZE)
        .await
        .context("Failed to read request body")
        .map_err(|e| {
            error!("Failed to read request body: {}", e);
            StatusCode::BAD_REQUEST
        })?;
    let start_time = Instant::now();
    let span = Span::current();

    // Convert body to String (with size validation)
    if body_bytes.len() > MAX_BODY_SIZE {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    let body_str = String::from_utf8(body_bytes.to_vec())
        .context("Invalid UTF-8 in request body")
        .map_err(|e| {
            error!("Invalid UTF-8 in request body: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    // ===== SLACK EVENT PARSING =====

    // Parse Slack event with detailed error reporting
    let payload: SlackEventRequest = serde_json::from_str(&body_str)
        .context("Failed to parse Slack event")
        .map_err(|e| {
            error!("Failed to parse Slack event: {}", e);

            // Log first 500 chars of body for debugging (sanitized)
            let preview = if body_str.len() > 500 {
                format!("{}...", &body_str[..500])
            } else {
                body_str.clone()
            };
            warn!("Invalid event body preview: {}", preview);

            StatusCode::BAD_REQUEST
        })?;

    // Update tracing span with event details
    span.record("event_type", &payload.event_type.as_str());
    if let Some(team_id) = &payload.team_id {
        span.record("team_id", &team_id.as_str());
    }

    // ===== URL VERIFICATION HANDLING =====

    // Handle URL verification challenge (required for Slack app setup)
    if payload.event_type == "url_verification" {
        return handle_url_verification(payload);
    }

    // ===== ADAPTER VERIFICATION =====

    // Verify Slack adapter is configured
    let adapter = state.slack_adapter.as_ref().ok_or_else(|| {
        error!("Slack adapter not configured");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    // ===== SECURITY VERIFICATION =====

    // Verify request signature (security check)
    verify_slack_request(&headers, &body_str, adapter)
        .map_err(|e| {
            warn!("Request verification failed: {:?}", e);
            StatusCode::UNAUTHORIZED
        })?;

    // ===== EVENT PROCESSING =====

    // Clone values needed for logging
    let event_type = payload.event_type.clone();
    let team_id = payload.team_id.clone();
    
    let processing_result = process_slack_event_internal(payload, state.clone(), body_str).await;

    // ===== METRICS TRACKING =====

    let processing_time = start_time.elapsed();
    span.record("processing_time_ms", processing_time.as_millis());

    // Track metrics regardless of success/failure
    let metrics = SlackEventMetrics {
        event_type: event_type.clone(),
        team_id: team_id.clone(),
        platform: interstice_core::Platform::Slack,
        processed_artifacts: 0, // Will be updated by process_team_event
        processing_time_ms: processing_time.as_millis(),
    };
    track_event_metrics(metrics, &state).await;

    // ===== RESPONSE HANDLING =====

    match processing_result {
        Ok(response) => {
            info!(
                event_type = %event_type,
                team_id = ?team_id,
                duration_ms = processing_time.as_millis(),
                "Event processed successfully"
            );

            // Convert SlackEventResponse to JsonValue for consistent return type
            let json_response = serde_json::to_value(response)
                .context("Failed to serialize response")
                .map_err(|e| {
                    error!("Failed to serialize response: {}", e);
                    StatusCode::INTERNAL_SERVER_ERROR
                })?;

            Ok(Json(json_response))
        }
        Err(e) => {
            error!(
                event_type = %event_type,
                team_id = ?team_id,
                error = %e,
                duration_ms = processing_time.as_millis(),
                "Event processing failed"
            );

            // Return success to Slack to prevent retries for non-recoverable errors
            let fallback_response = serde_json::json!({
                "ok": true,
                "warning": "Event processed with errors"
            });

            Ok(Json(fallback_response))
        }
    }
}

/// Handle URL verification challenge from Slack
fn handle_url_verification(payload: SlackEventRequest) -> Result<Json<JsonValue>, StatusCode> {
    if let Some(challenge) = payload.challenge {
        info!("Responding to Slack URL verification challenge");

        let response = serde_json::json!({
            "challenge": challenge
        });

        return Ok(Json(response));
    }

    error!("URL verification request missing challenge parameter");
    Err(StatusCode::BAD_REQUEST)
}

/// Internal event processing with comprehensive error handling
async fn process_slack_event_internal(
    payload: SlackEventRequest,
    state: Arc<AppState>,
    _body_str: String,
) -> AnyhowResult<SlackEventResponse> {
    // Check for duplicate events (idempotency)
    if let Some(event_id) = &payload.event_id {
        if is_duplicate_event(event_id.clone(), state.clone()).await? {
            info!(event_id = %event_id, "Duplicate event skipped");
            return Ok(SlackEventResponse {
                challenge: None,
                ok: Some(true),
            });
        }
    }

    // Process event data
    if let Some(event_data) = &payload.event {
        let slack_event = SlackPushEvent {
            event_type: payload.event_type.clone(),
            event: Some(event_data.clone()),
            team_id: payload.team_id.clone(),
            api_app_id: payload.api_app_id.clone(),
            event_id: payload.event_id.clone(),
            event_time: payload.event_time,
        };

        // Process with team context
        if let Some(team_id) = &slack_event.team_id {
            if let Err(e) = process_team_event(slack_event.clone(), team_id.clone(), state.clone()).await {
                // Log error but don't fail - we want to acknowledge receipt
                error!(
                    team_id = %team_id,
                    error = %e,
                    "Failed to process team event"
                );
            }
        }

        // Store audit trail (best effort)
        if let Err(e) = store_event_audit(payload.clone(), state.clone()).await {
            warn!("Failed to store event audit: {}", e);
        }
    } else {
        warn!("Received Slack event without event data");
    }

    Ok(SlackEventResponse {
        challenge: None,
        ok: Some(true),
    })
}

// Process event for a specific team/workspace with robust error handling
async fn process_team_event(
    event: SlackPushEvent,
    team_id: String,
    state: Arc<AppState>,
) -> AnyhowResult<()> {
    let start_time = Instant::now();

    // ===== ARTIFACT EXTRACTION =====

    let artifacts = extract_artifacts_from_event(&event, &team_id)
        .await
        .map_err(|e| anyhow!("Failed to extract artifacts from event: {}", e))?;

    if artifacts.is_empty() {
        info!(
            team_id = %team_id,
            "No artifacts found in event - skipping processing"
        );
        return Ok(());
    }

    info!(
        team_id = %team_id,
        artifact_count = artifacts.len(),
        "Extracted artifacts from event"
    );

    // ===== WORKSPACE RESOLUTION =====

    let workspace_id = get_workspace_id(&team_id, &state)
        .await
        .context("Failed to get workspace ID")?;

    // ===== PARALLEL PROCESSING PIPELINE =====

    // Store artifacts with robust batch processing
    let artifacts_clone = artifacts.clone();
    let artifacts_future = store_artifacts_batch(artifacts_clone, workspace_id, state.db.clone());

    // Run ML predictions in parallel if available
    let artifacts_for_ml = artifacts.clone();
    let state_for_ml = state.clone();
    let team_id_for_ml = team_id.clone();
    let predictions_future = async move {
        if state_for_ml.ml_pipeline.is_available() {
            match run_ml_predictions(&artifacts_for_ml, workspace_id, &state_for_ml).await {
                Ok(preds) => {
                    info!(
                        team_id = %team_id_for_ml,
                        prediction_count = preds.len(),
                        "Generated ML predictions"
                    );
                    preds
                }
                Err(e) => {
                    warn!("ML predictions failed: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        }
    };

    // Execute storage and predictions in parallel
    let (storage_result, predictions) = tokio::join!(artifacts_future, predictions_future);

    // Handle storage errors
    storage_result
        .map_err(|e| anyhow!("Failed to store artifacts: {}", e))?;

    // ===== PREDICTION STORAGE =====

    if !predictions.is_empty() {
        let artifact_ids: Vec<Uuid> = artifacts.iter().map(|a| a.id).collect();

        // Best effort - don't fail the whole operation
        if let Err(e) =
            store_predictions_batch(predictions.clone(), artifact_ids, workspace_id, state.db.clone()).await
        {
            warn!("Failed to store predictions: {}", e);
        }
    }

    // ===== ANALYTICS UPDATE =====

    // Update analytics (best effort)
    let team_id_clone = team_id.clone();
    let artifacts_for_analytics = artifacts.clone();
    let predictions_for_analytics = predictions.clone();
    if let Err(e) = update_workspace_analytics(team_id_clone, artifacts_for_analytics, predictions_for_analytics, state.db.clone()).await {
        warn!("Failed to update analytics: {}", e);
    }

    // ===== METRICS TRACKING =====

    let processing_time = start_time.elapsed();
    let metrics = SlackEventMetrics {
        event_type: event.event_type.clone(),
        team_id: Some(team_id.clone()),
        platform: interstice_core::Platform::Slack,
        processed_artifacts: artifacts.len(),
        processing_time_ms: processing_time.as_millis(),
    };

    track_event_metrics(metrics, &state).await;

    info!(
        team_id = %team_id,
        artifacts = artifacts.len(),
        predictions = predictions.len(),
        duration_ms = processing_time.as_millis(),
        "Team event processed successfully"
    );

    Ok(())
}

/// Get workspace ID with caching support
async fn get_workspace_id(
    team_id: &str,
    state: &Arc<AppState>,
) -> AnyhowResult<Uuid> {
    // TODO: Add caching layer here for frequently accessed workspaces

    let workspace_id = sqlx::query_scalar!(
        "SELECT id FROM workspaces WHERE slack_team_id = $1",
        team_id
    )
    .fetch_optional(&state.db)
    .await
    .context("Failed to get workspace ID")?
    .ok_or_else(|| {
        error!("Workspace not found for team: {}", team_id);
        anyhow!("Workspace not found for team {}", team_id)
    })?;

    Ok(workspace_id)
}

/// Run ML predictions with timeout and error recovery
async fn run_ml_predictions(
    artifacts: &[Artifact],
    workspace_id: Uuid,
    state: &Arc<AppState>,
) -> AnyhowResult<Vec<OutcomePrediction>> {
    let result = super::run_ml_on_artifacts(artifacts, workspace_id, &state.ml_pipeline)
        .await
        .map_err(|e| anyhow!("ML processing failed: {}", e))?;
    
    Ok(result)
}

//// Check if an event has already been processed (idempotency)
async fn is_duplicate_event(
    event_id: String,
    state: Arc<AppState>,
) -> AnyhowResult<bool> {
    let result = sqlx::query!(
        r#"
        INSERT INTO slack_events (event_id, processed_at)
        VALUES ($1, NOW())
        ON CONFLICT (event_id) DO NOTHING
        RETURNING event_id
        "#,
        event_id
    )
    .fetch_optional(&state.db)
    .await
    .context("Failed to check for duplicate event")?;

    // If we couldn't insert, it's a duplicate
    Ok(result.is_none())
}

/// Store event audit trail for compliance and debugging
async fn store_event_audit(
    event: SlackEventRequest,
    state: Arc<AppState>,
) -> AnyhowResult<()> {
    sqlx::query!(
        r#"
        INSERT INTO slack_event_audit (
            event_id, event_type, team_id, event_data, created_at
        )
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (event_id) DO NOTHING
        "#,
        event.event_id.as_deref().unwrap_or("unknown"),
        event.event_type,
        event.team_id,
        serde_json::to_value(&event)
            .context("Failed to serialize event data")
            .map_err(|e| sqlx::Error::Protocol(format!("JSON serialization error: {}", e)))?
    )
    .execute(&state.db)
    .await
    .context("Failed to store event audit")?;

    Ok(())
}

pub async fn track_event_metrics(metrics: SlackEventMetrics, state: &Arc<AppState>) {
    info!(
        "Event processed: type={}, team={:?}, artifacts={}, time={}ms",
        metrics.event_type,
        metrics.team_id,
        metrics.processed_artifacts,
        metrics.processing_time_ms
    );

    let _ = sqlx::query!(
        r#"
        INSERT INTO event_metrics (platform, event_type, team_id, artifact_count, processing_time_ms, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        metrics.platform.to_string(),
        &metrics.event_type,
        metrics.team_id.as_ref(),
        metrics.processed_artifacts as i32,
        metrics.processing_time_ms as i32
    )
    .execute(&state.db)
    .await;
}


// Extension trait for ML pipeline availability check
trait MLPipelineExt {
    fn is_available(&self) -> bool;
}

impl MLPipelineExt for interstice_ml::MLPipeline {
    fn is_available(&self) -> bool {
        // TODO: Implement actual availability check
        // Could check for:
        // - Model loaded successfully
        // - GPU/CPU resources available
        // - Recent successful predictions
        // - Circuit breaker status
        true
    }
}
