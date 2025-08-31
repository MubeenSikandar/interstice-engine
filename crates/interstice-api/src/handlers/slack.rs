use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use interstice_adapters::SlackAdapter;
use interstice_core::{ProcessedArtifact, Platform};
use serde::{Deserialize, Serialize};
use slack_morphism::prelude::*;
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct SlackEventRequest {
    #[serde(rename = "type")]
    event_type: String,
    challenge: Option<String>,
    event: Option<serde_json::Value>,
    team_id: Option<String>,
    api_app_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SlackEventResponse {
    challenge: Option<String>,
    ok: Option<bool>,
}

/// Handle Slack Events API webhooks
pub async fn handle_events(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<SlackEventRequest>,
) -> Result<Json<SlackEventResponse>, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Handle URL verification challenge
    if payload.event_type == "url_verification" {
        if let Some(challenge) = payload.challenge {
            info!("Slack URL verification challenge received");
            return Ok(Json(SlackEventResponse {
                challenge: Some(challenge),
                ok: None,
            }));
        }
    }

    // Verify request signature
    if !verify_slack_signature(&headers, &payload, adapter).await {
        warn!("Invalid Slack signature");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Process the event
    match payload.event {
        Some(event_data) => {
            let slack_event: SlackPushEvent = serde_json::from_value(event_data)
                .map_err(|e| {
                    error!("Failed to parse Slack event: {}", e);
                    StatusCode::BAD_REQUEST
                })?;

            adapter.handle_event(slack_event).await.map_err(|e| {
                error!("Error processing Slack event: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            Ok(Json(SlackEventResponse {
                challenge: None,
                ok: Some(true),
            }))
        }
        None => {
            warn!("No event data in payload");
            Ok(Json(SlackEventResponse {
                challenge: None,
                ok: Some(false),
            }))
        }
    }
}

/// Handle Slack slash commands
pub async fn handle_slash_commands(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(command): Json<SlackCommandEvent>,
) -> Result<Json<SlackMessageContent>, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Verify the command signature
    if !verify_slack_signature(&headers, &command, adapter).await {
        warn!("Invalid Slack signature for slash command");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Process the slash command
    let response = adapter.handle_slash_command(command).await.map_err(|e| {
        error!("Error processing slash command: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(response))
}

/// Handle Slack interactive elements (button clicks, etc.)
pub async fn handle_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<SlackInteractionEvent>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Verify the interaction signature
    if !verify_slack_signature(&headers, &payload, adapter).await {
        warn!("Invalid Slack signature for interaction");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Process the interaction
    adapter.handle_interaction(payload).await.map_err(|e| {
        error!("Error processing interaction: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(serde_json::json!({ "ok": true })))
}

/// Verify Slack request signature
async fn verify_slack_signature<T: serde::Serialize>(
    headers: &HeaderMap,
    payload: &T,
    adapter: &SlackAdapter,
) -> bool {
    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    let signature = headers
        .get("x-slack-signature")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    let body = serde_json::to_string(payload).unwrap_or_default();

    adapter.verify_signature(timestamp, signature, &body)
}

/// Handle Slack OAuth callback
pub async fn handle_oauth_callback(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<SlackOAuthCallback>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    info!("Slack OAuth callback received for team: {}", payload.team_id);

    // In a real implementation, you would:
    // 1. Exchange the code for an access token
    // 2. Store the token securely
    // 3. Add the workspace to your database
    // 4. Send a welcome message

    Ok(Json(serde_json::json!({
        "ok": true,
        "message": "OAuth successful! Your workspace has been added to Interstice."
    })))
}

#[derive(Debug, Deserialize)]
pub struct SlackOAuthCallback {
    pub code: String,
    pub state: Option<String>,
    pub team_id: String,
}

/// Health check endpoint for Slack
pub async fn slack_health() -> &'static str {
    "Slack integration is healthy"
}
