// src/handlers/slack/utils.rs

use axum::{extract::State};
use http::StatusCode;
use interstice_adapters::{slack::SlackAdapter, PlatformAdapter};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{warn};

use crate::AppState;
use super::{SIGNATURE_VERSION, MAX_TIMESTAMP_AGE_SECS};

pub async fn slack_health(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    if state.slack_adapter.is_some() {
        (StatusCode::OK, "Slack integration is healthy")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Slack adapter not configured")
    }
}

pub fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

pub fn verify_slack_request(
    headers: &axum::http::HeaderMap,
    body: &str,
    adapter: &SlackAdapter,
) -> Result<(), StatusCode> {
    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing timestamp header");
            StatusCode::UNAUTHORIZED
        })?;

    let timestamp_num: i64 = timestamp.parse().map_err(|_| {
        warn!("Invalid timestamp format");
        StatusCode::UNAUTHORIZED
    })?;
    
    let current_time = chrono::Utc::now().timestamp();
    if (current_time - timestamp_num).abs() > MAX_TIMESTAMP_AGE_SECS {
        warn!("Request timestamp too old");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let signature = headers
        .get("x-slack-signature")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing signature header");
            StatusCode::UNAUTHORIZED
        })?;

    if !signature.starts_with(&format!("{}=", SIGNATURE_VERSION)) {
        warn!("Invalid signature version");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let mut headers_map = HashMap::new();
    headers_map.insert("x-slack-request-timestamp".to_string(), timestamp.to_string());
    headers_map.insert("x-slack-signature".to_string(), signature.to_string());
    
    if !adapter.verify_webhook(&headers_map, body.as_bytes()).unwrap_or(false) {
        warn!("Invalid Slack signature");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(())
}
impl Clone for super::SlackCommandEvent {
    fn clone(&self) -> Self {
        super::SlackCommandEvent {
            token: self.token.clone(),
            team_id: self.team_id.clone(),
            team_domain: self.team_domain.clone(),
            channel_id: self.channel_id.clone(),
            channel_name: self.channel_name.clone(),
            user_id: self.user_id.clone(),
            user_name: self.user_name.clone(),
            command: self.command.clone(),
            text: self.text.clone(),
            response_url: self.response_url.clone(),
            trigger_id: self.trigger_id.clone(),
        }
    }
}
