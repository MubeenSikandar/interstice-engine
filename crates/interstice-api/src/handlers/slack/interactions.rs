// src/handlers/slack/interactions.rs

use axum::{extract::State, http::HeaderMap, Json};
use serde_json::Value as JsonValue;
use serde_urlencoded;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{error, info};
use tokio::spawn;
use tokio::time::Duration;

use crate::handlers::slack::{post_to_response_url, verify_slack_request, SlackInteractionEvent};
use crate::AppState;

pub async fn handle_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Result<Json<JsonValue>, axum::http::StatusCode> {
    let adapter = state.slack_adapter.as_ref().ok_or_else(|| {
        error!("Slack adapter not configured");
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    })?;

    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        error!("Failed to read body: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;
    
    let body_str = String::from_utf8(body_bytes.into()).map_err(|e| {
        error!("Failed to parse body as UTF-8: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;

    verify_slack_request(&headers, &body_str, adapter)?;

    let form: HashMap<String, String> = serde_urlencoded::from_str(&body_str).map_err(|e| {
        error!("Failed to parse interaction form: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;

    let payload_str = form.get("payload").ok_or_else(|| {
        error!("No payload field in interaction");
        axum::http::StatusCode::BAD_REQUEST
    })?;

    let payload: SlackInteractionEvent = serde_json::from_str(payload_str).map_err(|e| {
        error!("Failed to parse interaction payload: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;

    let response = match payload.event_type.as_str() {
        "block_actions" => {
            info!(
                "Block {} '{}' from user {} ({}) in channel {} ({})",
                payload.actions.iter().map(|a| a.action_id.clone()).collect::<Vec<String>>().join(", "),
                &payload.callback_id,
                &payload.user.id, 
                &payload.user.name, 
                &payload.channel.id,
                &payload.channel.name
            );
            
            match payload.callback_id.as_str() {
                "approve_task" => handle_task_approval(&payload, &state).await,
                "predict_outcome" => handle_prediction_request(&payload, &state).await,
                _ => {
                    if !payload.response_url.is_empty() {
                        let response_url_clone = payload.response_url.clone();
                        spawn(async move {
                            post_to_response_url(
                                &response_url_clone,
                                &serde_json::json!({
                                    "text": "Processing your request..."
                                })
                            ).await.ok();
                        });
                    }
                    
                    serde_json::json!({
                        "response_type": "ephemeral",
                        "text": "Action received"
                    })
                }
            }
        },
        "view_submission" => {
            if !payload.trigger_id.is_empty() {
                info!("Modal submission with trigger_id: {}", &payload.trigger_id);
            }
            
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Submission received"
            })
        },
        _ => serde_json::json!({
            "response_type": "ephemeral",
            "text": "Unknown interaction type"
        }),
    };

    Ok(Json(response))
}

async fn handle_task_approval(
    payload: &SlackInteractionEvent,
    _state: &Arc<AppState>,
) -> JsonValue {
    info!("Task approval from {} in channel {}", &payload.user.name, &payload.channel.name);
    
    serde_json::json!({
        "response_type": "ephemeral",
        "text": format!("Task approved by {}", &payload.user.name)
    })
}

async fn handle_prediction_request(
    payload: &SlackInteractionEvent,
    state: &Arc<AppState>,
) -> JsonValue {
    let response_url = payload.response_url.clone();
    let channel_name = payload.channel.name.clone();
    let state_clone = state.clone();
    
    spawn(async move {
        let result = state_clone.timeout_manager.execute_with_timeout(
            || async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                "Predictions completed"
            },
            state_clone.timeout_manager.config().slack_api,
            "slack_prediction",
        ).await;
        
        let message = match result {
            Ok(_) => format!("Predictions ready for #{}", channel_name),
            Err(_) => format!("Prediction timeout for #{} - please try again with a simpler request", channel_name),
        };
        
        post_to_response_url(&response_url, &serde_json::json!({
            "text": message
        })).await.ok();
    });
    
    serde_json::json!({
        "response_type": "ephemeral",
        "text": "Generating predictions..."
    })
}