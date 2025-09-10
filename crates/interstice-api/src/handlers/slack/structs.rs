// src/handlers/slack/structs.rs

use interstice_core::Platform;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SlackEventRequest {
    #[serde(rename = "type")]
    pub event_type: String,
    pub challenge: Option<String>,
    pub event: Option<JsonValue>,
    pub event_id: Option<String>,
    pub event_time: Option<i64>,
    pub team_id: Option<String>,
    pub api_app_id: Option<String>,
    #[serde(skip_serializing)]
    pub authed_users: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct SlackEventResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct SlackCommandEvent {
    pub token: String,
    pub team_id: String,
    pub team_domain: String,
    pub channel_id: String,
    pub channel_name: String,
    pub user_id: String,
    pub user_name: String,
    pub command: String,
    pub text: String,
    pub response_url: String,
    pub trigger_id: String,
}

#[derive(Debug, Deserialize)]
pub struct SlackInteractionEvent {
    pub event_type: String,
    pub user: SlackUser,
    pub channel: SlackChannel,
    pub actions: Vec<SlackAction>,
    pub callback_id: String,
    pub response_url: String,
    pub trigger_id: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SlackPushEvent {
    pub event_type: String,
    pub event: Option<JsonValue>,
    pub team_id: Option<String>,
    pub api_app_id: Option<String>,
    pub event_id: Option<String>,
    pub event_time: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SlackEvent {
    pub event_type: String,
    pub text: Option<String>,
    pub user: Option<String>,
    pub channel: Option<String>,
    pub ts: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SlackUser {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct SlackChannel {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct SlackAction {
    pub action_id: String,
    pub _value: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SlackOAuthCallback {
    pub code: String,
    pub state: Option<String>,
    #[serde(skip)]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SlackOAuthResponse {
    pub ok: bool,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace_id: Option<Uuid>,
}

#[derive(Debug, Serialize)]
pub struct SlackEventMetrics {
    pub event_type: String,
    pub team_id: Option<String>,
    pub platform: Platform,
    pub processed_artifacts: usize,
    pub processing_time_ms: u128,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SlackOAuthTokenResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub access_token: Option<String>,
    pub token_type: Option<String>,
    pub scope: Option<String>,
    pub bot_user_id: Option<String>,
    pub app_id: Option<String>,
    pub team: Option<SlackTeamInfo>,
    pub enterprise: Option<SlackEnterpriseInfo>,
    pub authed_user: Option<SlackAuthedUser>,
    pub is_enterprise_install: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SlackTeamInfo {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SlackEnterpriseInfo {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SlackAuthedUser {
    pub id: String,
    pub scope: Option<String>,
    pub access_token: Option<String>,
    pub token_type: Option<String>,
}

#[derive(Debug)]
pub struct WorkspaceConfig {
    pub workspace_id: Uuid,
    pub team_id: String,
    pub team_name: String,
    pub enterprise_id: Option<String>,
    pub enterprise_name: Option<String>,
    pub is_enterprise: bool,
    pub access_token: String,
    pub token_type: String,
}

#[derive(Debug)]
pub struct WorkspacePatterns {
    pub peak_hour: String,
    pub most_active_channel: String,
    pub avg_daily_artifacts: f64,
    pub common_artifact_type: String,
}

#[derive(Debug)]
pub struct WorkspaceStatistics {
    pub weekly_artifacts: i64,
    pub weekly_commands: i64,
    pub active_users: i64,
    pub monthly_artifacts: i64,
    pub monthly_predictions: i64,
    pub success_rate: f64,
}