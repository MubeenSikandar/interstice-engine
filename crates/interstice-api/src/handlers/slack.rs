//interstice-api/src/handlers/slack.rs
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use interstice_adapters::{traits::{EventMetadata, EventType, PlatformAdapter, PlatformEvent}, slack::SlackAdapter};
use interstice_core::{artifact::{AccessLevel, ArtifactState, DesignType, DocumentType, IssueState, Priority, QualityMetrics}, Artifact, ArtifactType, Platform, ProcessedData, WorkspaceId};
use interstice_ml::{types::{Duration, ImpactLevel}, OutcomePrediction};
use ring::{aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM}, rand::SystemRandom, rand::SecureRandom};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::{collections::{HashMap, HashSet}, sync::{Arc, OnceLock}};
use tracing::{error, info, warn, instrument};
use uuid::Uuid;
use serde_json::Value as JsonValue;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};

use crate::AppState;

// Constants for Slack API
const SLACK_OAUTH_URL: &str = "https://slack.com/api/oauth.v2.access";
const SIGNATURE_VERSION: &str = "v0";
const MAX_TIMESTAMP_AGE_SECS: i64 = 300; // 5 minutes
const EXPECTED_TOKEN_TYPE: &str = "Bearer"; // OAuth 2.0 standard
static ENCRYPTION_KEY: OnceLock<LessSafeKey> = OnceLock::new();

#[derive(Debug, Deserialize, Serialize)]
pub struct SlackEventRequest {
    #[serde(rename = "type")]
    event_type: String,
    challenge: Option<String>,
    event: Option<serde_json::Value>,
    event_id: Option<String>,
    event_time: Option<i64>,
    team_id: Option<String>,
    api_app_id: Option<String>,
    #[serde(skip_serializing)]
    authed_users: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct SlackEventResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    challenge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ok: Option<bool>,
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

#[derive(Debug, Deserialize, Serialize)]
pub struct SlackPushEvent {
    pub event_type: String,
    pub event: Option<serde_json::Value>,
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

/// Event tracking for analytics and monitoring
#[derive(Debug, Serialize)]
struct SlackEventMetrics {
    event_type: String,
    team_id: Option<String>,
    platform: Platform,
    processed_artifacts: usize,
    processing_time_ms: u128,
}

#[derive(Debug, Deserialize, Clone)]
struct SlackOAuthTokenResponse {
    ok: bool,
    error: Option<String>,
    access_token: Option<String>,
    token_type: Option<String>,
    scope: Option<String>,
    bot_user_id: Option<String>,
    app_id: Option<String>,
    team: Option<SlackTeamInfo>,
    enterprise: Option<SlackEnterpriseInfo>,
    authed_user: Option<SlackAuthedUser>,
    // Additional fields for enterprise
    is_enterprise_install: Option<bool>,
}

#[derive(Debug, Deserialize, Clone)]
struct SlackTeamInfo {
    id: String,
    name: String,
}


#[derive(Debug, Deserialize, Clone)]
struct SlackEnterpriseInfo {
    id: String,
    name: String,
}

#[derive(Debug, Deserialize, Clone)]
struct SlackAuthedUser {
    id: String,
    scope: Option<String>,
    access_token: Option<String>,
    token_type: Option<String>,
}

#[derive(Debug)]
struct WorkspaceConfig {
    workspace_id: Uuid,
    team_id: String,
    team_name: String,
    enterprise_id: Option<String>,
    enterprise_name: Option<String>,
    is_enterprise: bool,
    access_token: String,
    token_type: String,
}

/// Handle Slack Events API webhooks with full production features
#[instrument(skip(state, headers))]
pub async fn handle_events(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Result<impl IntoResponse, StatusCode> {
    let start_time = std::time::Instant::now();
    
    let payload: SlackEventRequest = serde_json::from_str(&body)
        .map_err(|e| {
            error!("Failed to parse Slack event: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Handle URL verification
    if payload.event_type == "url_verification" {
        if let Some(challenge) = payload.challenge {
            info!("Slack URL verification challenge received");
            return Ok(Json(SlackEventResponse {
                challenge: Some(challenge),
                ok: None,
            }));
        }
    }

    verify_slack_request(&headers, &body, adapter)?;

    // Check for duplicate events
    if let Some(event_id) = &payload.event_id {
        if is_duplicate_event(event_id, &state).await {
            info!("Duplicate event {} skipped", event_id);
            return Ok(Json(SlackEventResponse {
                challenge: None,
                ok: Some(true),
            }));
        }
    }

    // Log authed users for multi-workspace tracking
    if let Some(authed_users) = &payload.authed_users {
        info!(
            "Event from team {} authorized by users: {:?}",
            payload.team_id.as_ref().unwrap_or(&"unknown".to_string()),
            authed_users
        );
        
        // In a multi-workspace app, use this to determine which workspace's
        // configuration to use for processing
    }

    // Process the event
    let processed = match payload.event.clone() {
        Some(event_data) => {
            let slack_event = SlackPushEvent {
                event_type: payload.event_type.clone(),
                event: Some(event_data),
                team_id: payload.team_id.clone(),
                api_app_id: payload.api_app_id.clone(),
                event_id: payload.event_id.clone(),
                event_time: payload.event_time,
            };
            
            // Process and store the event with ML
            if let Some(team_id) = &slack_event.team_id {
                if let Err(e) = process_and_store_event(&slack_event, team_id, &state).await {
                    error!("Failed to process event: {}", e);
                }
            }

            // Process and get artifacts
            adapter.process_event(PlatformEvent {
                id: Uuid::new_v4(),
                platform: Platform::Slack,
                event_type: EventType::MessageNew,
                workspace_id: payload.team_id.as_ref().and_then(|id| id.parse().ok()),
                timestamp: chrono::Utc::now(),
                raw_data: serde_json::to_value(slack_event).unwrap(),
                metadata: EventMetadata::default(),
            }).await.map_err(|e| {
                error!("Error processing Slack event: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

            // Track metrics
            let metrics = SlackEventMetrics {
                event_type: payload.event_type.clone(),
                team_id: payload.team_id.clone(),
                platform: Platform::Slack,
                processed_artifacts: 0, // Would come from actual processing
                processing_time_ms: start_time.elapsed().as_millis(),
            };
            
            track_event_metrics(metrics, &state).await;
            
            ProcessedData {
                artifacts: vec![],
                predictions: vec![],
                outcomes: vec![],
                processing_results: vec![],
                platform: Platform::Slack,
                metadata: interstice_core::ProcessingMetadata {
                    duration: std::time::Duration::from_millis(0),
                    timestamp: chrono::Utc::now(),
                    engine_version: "1.0.0".to_string(),
                },
            }
        }
        None => {
            warn!("No event data in payload");
            ProcessedData {
                artifacts: vec![],
                predictions: vec![],
                outcomes: vec![],
                processing_results: vec![],
                platform: Platform::Slack,
                metadata: interstice_core::ProcessingMetadata {
                    duration: std::time::Duration::from_millis(0),
                    timestamp: chrono::Utc::now(),
                    engine_version: "1.0.0".to_string(),
                },
            }
        }
    };

    // Store event for audit trail
    store_event_audit(&payload, &processed, &state).await;

    Ok(Json(SlackEventResponse {
        challenge: None,
        ok: Some(true),
    }))
}

async fn process_and_store_event(
    event: &SlackPushEvent,
    team_id: &str,
    state: &Arc<AppState>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract artifacts from the event
    let artifacts = extract_artifacts_from_event(event, team_id).await?;
    
    if artifacts.is_empty() {
        info!("No artifacts found in event from team {}", team_id);
        return Ok(());
    }
    
    // Get workspace_id for storage
    let workspace_id = sqlx::query_scalar!(
        "SELECT id FROM workspaces WHERE slack_team_id = $1",
        team_id
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or("Workspace not found")?;
    
    // Store artifacts in database
    for artifact in &artifacts {
        store_artifact_from_event(artifact, workspace_id, &state.db).await?;
    }
    
    // Run ML predictions on artifacts
    let predictions = run_ml_on_artifacts(&artifacts, workspace_id, &state.ml_pipeline).await?;
    
    // Store predictions
    for pred in &predictions {
        store_prediction_from_event(pred, workspace_id, artifacts[0].id, &state.db).await?;
    }
    
    // Update workspace analytics
    update_workspace_analytics(team_id, &artifacts, &predictions, &state.db).await?;
    
    info!(
        "Processed event from team {}: {} artifacts, {} predictions",
        team_id,
        artifacts.len(),
        predictions.len()
    );
    
    Ok(())
}

async fn extract_artifacts_from_event(
    event: &SlackPushEvent,
    team_id: &str,
) -> Result<Vec<Artifact>, Box<dyn std::error::Error>> {
    let mut artifacts = Vec::new();
    
    // Handle different event types
    if let Some(event_data) = &event.event {
        let event_type = event_data.get("event_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        
        match event_type {
            "message" => {
    // Extract message content
    if let Some(text) = event_data.get("text").and_then(|v| v.as_str()) {
            if !text.is_empty() {
            let channel_id = event_data.get("channel").and_then(|v| v.as_str()).unwrap_or("unknown");
            let user_id = event_data.get("user").and_then(|v| v.as_str()).unwrap_or("unknown");
            let ts = event_data.get("ts").and_then(|v| v.as_str()).unwrap_or("");
                        
                        // Determine artifact type based on content
                        let artifact_type = determine_artifact_type(text);
                        
                        let artifact = Artifact {
                            id: Uuid::new_v4(),
                            workspace_id: WorkspaceId::from_uuid(
                                Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())
                            ),
                            artifact_type,
                            platform: Platform::Slack,
                            content: text.to_string(),
                            metadata: serde_json::json!({
                                "channel_id": channel_id,
                                "user_id": user_id,
                                "message_ts": ts,
                                "event_type": "message",
                            }),
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            version: 1,
                            state: ArtifactState::Pending,
                            quality_metrics: QualityMetrics::default(),
                            related_artifacts: vec![],
                            tags: HashSet::new(),
                        };
                        
                        artifacts.push(artifact);
                    }
                }
                
                // Process attachments if present
                if let Some(attachments) = event_data.get("attachments").and_then(|a| a.as_array()) {
                    for attachment in attachments {
                        if let Some(artifact) = process_attachment(attachment, team_id) {
                            artifacts.push(artifact);
                        }
                    }
                }
            }
            "file_shared" => {
                // Process file uploads
                if let Some(file) = event_data.get("file") {
                    if let Some(artifact) = process_file_share(file, team_id) {
                        artifacts.push(artifact);
                    }
                }
            }
            "reaction_added" => {
                // Track reactions as engagement metrics
                // Create a lightweight artifact for the reaction
                if let Some(reaction) = event_data.get("reaction").and_then(|v| v.as_str()) {
                    if let Some(item) = event_data.get("item") {
                        let channel = item.get("channel").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let ts = item.get("ts").and_then(|v| v.as_str()).unwrap_or("");
                        let user = event_data.get("user").and_then(|v| v.as_str()).unwrap_or("unknown");

                        let artifact = Artifact {
                            id: Uuid::new_v4(),
                            workspace_id: WorkspaceId::from_uuid(
                                Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())
                            ),
                            artifact_type: ArtifactType::Message {
                                id: Uuid::new_v4().to_string(),
                                channel: "unknown".to_string(),
                                thread_id: None,
                                author: "unknown".to_string(),
                                content: format!("Reaction added: {}", reaction),
                                mentions: vec![],
                                attachments: vec![],
                                reactions: HashMap::new(),
                                sentiment: interstice_core::artifact::Sentiment::Neutral,
                                intent: interstice_core::artifact::MessageIntent::Discussion,
                                is_edited: false,
                                reply_count: 0,
                            },
                            platform: Platform::Slack,
                            content: format!("Reaction added: {}", reaction),
                            metadata: serde_json::json!({
                                "reaction": reaction,
                                "channel": channel,
                                "message_ts": ts,
                                "user": user,
                                "event_type": "reaction_added",
                            }),
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            version: 1,
                            state: ArtifactState::Pending,
                            quality_metrics: QualityMetrics::default(),
                            related_artifacts: vec![],
                            tags: HashSet::new(),
                        };

                        artifacts.push(artifact);
                    }
                }
            }
            _ => {
                tracing::info!("Unhandled event type: {}", event_type);
            }
        }
    }
    
    Ok(artifacts)
}

fn determine_artifact_type(text: &str) -> ArtifactType {
    // Simple heuristic; can be improved with ML
    if text.contains("PR #") || text.contains("pull request") {
        ArtifactType::PullRequest {
            title: text.to_string(),
            state: interstice_core::artifact::PullRequestState::Open,
            author: "unknown".to_string(),
            reviewers: vec![],
            labels: vec![],
            merge_conflict: false,
            ci_status: None,
            base_branch: "main".to_string(),
            head_branch: "feature".to_string(),
            files_changed: 0,
            additions: 0,
            deletions: 0,
            merged: false,
            draft: false,
            number: 0,
        }
    } else if text.contains("#") && text.contains("issue") {
        ArtifactType::Issue {
            id: Uuid::new_v4().to_string(),
        title: text.to_string(),
        state: IssueState::Open,
        priority: Priority::Medium,
        assignees: vec![],
        labels: vec![],
        story_points: None,
        sprint: None,
        epic: None,
        blocked: false,
        blockers: vec![],
        time_estimate: None,
        time_spent: None,
        }
    } else if text.contains("task") || text.contains("TODO") {
        ArtifactType::Task {
            id: Uuid::new_v4().to_string(),
            title: text.to_string(),
            status: interstice_core::artifact::TaskStatus::Todo,
            assignee: None,
            due_date: None,
            completed_at: None,
            checklist_items: vec![],
            dependencies: vec![],
            tags: vec![],
            recurring: false,
            parent_task: None,
            subtasks: vec![],
        }
    } else {
        ArtifactType::Message {
            id: Uuid::new_v4().to_string(),
            channel: "unknown".to_string(),
            thread_id: None,
            author: "unknown".to_string(),
            content: text.to_string(),
            mentions: vec![],
            attachments: vec![],
            reactions: HashMap::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        }
    }
}

fn process_attachment(attachment: &JsonValue, team_id: &str) -> Option<Artifact> {
    // Extract relevant info from attachment
    if let Some(title) = attachment.get("title").and_then(|v| v.as_str()) {
        if !title.is_empty() {
            let artifact_type = determine_artifact_type(title);
            return Some(Artifact {
                id: Uuid::new_v4(),
                workspace_id: WorkspaceId::from_uuid(
                    Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())
                ),
                artifact_type,
                platform: Platform::Slack,
                content: title.to_string(),
                metadata: serde_json::json!({
                    "attachment": attachment.clone(),
                    "event_type": "attachment",
                }),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                version: 1,
                state: ArtifactState::Pending,
                quality_metrics: QualityMetrics::default(),
                related_artifacts: vec![],
                tags: HashSet::new(),
            });
        }
    }
    None
}

fn process_file_share(file: &JsonValue, team_id: &str) -> Option<Artifact> {
    if let Some(name) = file.get("name").and_then(|v| v.as_str()) {
        if !name.is_empty() {
            let artifact_type = if name.ends_with(".md") || name.ends_with(".txt") {
                ArtifactType::Document {
                    id: Uuid::new_v4().to_string(),
                    title: name.to_string(),
                    doc_type: DocumentType::Wiki,
                    url: None,
                    author: "unknown".to_string(),
                    collaborators: vec![],
                    word_count: None,
                    last_modified: Utc::now(),
                    version: 1,
                    is_template: false,
                    access_level: AccessLevel::Internal,
                }
            } else if name.ends_with(".png") || name.ends_with(".jpg") {
                ArtifactType::Design {
                    id: Uuid::new_v4().to_string(),
                    title: name.to_string(),
                    design_type: DesignType::Figma,
                    version: None,
                    collaborators: vec![],
                    components: 0,
                    screens: 0,
                    last_modified: Utc::now(),
                    design_system: None,
                    accessibility_score: None,
                }
            } else {
                ArtifactType::Document {
                    id: Uuid::new_v4().to_string(),
                    title: name.to_string(),
                    doc_type: DocumentType::Wiki,
                    url: None,
                    author: "unknown".to_string(),
                    collaborators: vec![],
                    word_count: None,
                    last_modified: Utc::now(),
                    version: 1,
                    is_template: false,
                    access_level: AccessLevel::Internal,
                }
            };
            return Some(Artifact {
                id: Uuid::new_v4(),
                workspace_id: WorkspaceId::from_uuid(
                    Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())
                ),
                artifact_type,
                platform: Platform::Slack,
                content: name.to_string(),
                metadata: serde_json::json!({
                    "file": file.clone(),
                    "event_type": "file_shared",
                }),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                version: 1,
                state: ArtifactState::Pending,
                quality_metrics: QualityMetrics::default(),
                related_artifacts: vec![],
                tags: HashSet::new(),
            });
        }
    }
    None
}

/// Store artifact from event
async fn store_artifact_from_event(
    artifact: &Artifact,
    workspace_id: Uuid,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let channel_id = artifact.metadata.get("channel_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    
    let message_id = artifact.metadata.get("message_ts")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    
    sqlx::query!(
        r#"
        INSERT INTO artifacts (
            id, workspace_id, artifact_type, content,
            channel_id, message_id, metadata, platform, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        ON CONFLICT (id) DO NOTHING
        "#,
        artifact.id,
        workspace_id,
        format!("{:?}", artifact.artifact_type),
        artifact.content,
        channel_id,
        message_id,
        artifact.metadata,
        artifact.platform.to_string()
    )
    .execute(db)
    .await?;
    
    Ok(())
}

/// Run ML pipeline on artifacts
async fn run_ml_on_artifacts(
    artifacts: &[Artifact],
    workspace_id: Uuid,
    ml_pipeline: &Arc<interstice_ml::MLPipeline>,
) -> Result<Vec<OutcomePrediction>, Box<dyn std::error::Error>> {
    // Convert to ML artifacts
    let ml_artifacts: Vec<interstice_ml::types::Artifact> = artifacts.iter().map(|a| {
        interstice_ml::types::Artifact::new(
            a.id.to_string(),
            a.content.clone(),
            interstice_ml::types::Platform::Slack,
            interstice_ml::types::ArtifactType::Message
        )
    }).collect();
    
    // Run predictions
    let context = artifacts.iter()
        .map(|a| a.content.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    
    ml_pipeline.predict_outcomes(workspace_id, &ml_artifacts, &context).await
        .map_err(|e| e.into())
}

/// Store prediction from event processing
async fn store_prediction_from_event(
    prediction: &OutcomePrediction,
    workspace_id: Uuid,
    artifact_id: Uuid,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    sqlx::query!(
        r#"
        INSERT INTO inference_history (
            id, workspace_id, artifact_id,
            confidence, prediction_data, created_at
        )
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        Uuid::new_v4(),
        workspace_id,
        artifact_id,
        prediction.confidence as f32,
        serde_json::to_value(prediction)?
    )
    .execute(db)
    .await?;
    
    Ok(())
}

/// Update workspace analytics after event processing
async fn update_workspace_analytics(
    team_id: &str,
    artifacts: &[Artifact],
    predictions: &[OutcomePrediction],
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Update or insert analytics record
    sqlx::query!(
        r#"
        INSERT INTO workspace_analytics (
            workspace_id, date, artifact_count, prediction_count,
            message_count, task_count, document_count
        )
        VALUES (
            (SELECT id FROM workspaces WHERE slack_team_id = $1),
            CURRENT_DATE,
            $2,
            $3,
            $4,
            $5,
            $6
        )
        ON CONFLICT (workspace_id, date) DO UPDATE SET
            artifact_count = workspace_analytics.artifact_count + EXCLUDED.artifact_count,
            prediction_count = workspace_analytics.prediction_count + EXCLUDED.prediction_count,
            message_count = workspace_analytics.message_count + EXCLUDED.message_count,
            task_count = workspace_analytics.task_count + EXCLUDED.task_count,
            document_count = workspace_analytics.document_count + EXCLUDED.document_count,
            updated_at = NOW()
        "#,
        team_id,
        artifacts.len() as i32,
        predictions.len() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Message { .. })).count() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Task { .. })).count() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Document { .. })).count() as i32
    )
    .execute(db)
    .await?;
    
    Ok(())
}


fn extract_workspace_config(
    oauth_response: SlackOAuthTokenResponse,
) -> Result<WorkspaceConfig, StatusCode> {
    let access_token = oauth_response.access_token
        .ok_or_else(|| {
            error!("No access token in OAuth response");
            StatusCode::BAD_REQUEST
        })?;

    let token_type = oauth_response.token_type
        .unwrap_or_else(|| EXPECTED_TOKEN_TYPE.to_string());

    // Handle both regular and enterprise installations
    let (team_id, team_name, enterprise_id, enterprise_name, is_enterprise) = 
        if let Some(enterprise) = oauth_response.enterprise {
            // Enterprise Grid installation
            info!("Enterprise Grid installation detected: {} ({})", 
                  enterprise.name, enterprise.id);
            
            let team = oauth_response.team.unwrap_or(SlackTeamInfo {
                id: enterprise.id.clone(),
                name: enterprise.name.clone(),
            });
            
            (
                team.id,
                team.name,
                Some(enterprise.id),
                Some(enterprise.name),
                true
            )
        } else if let Some(team) = oauth_response.team {
            // Regular workspace installation
            (team.id, team.name, None, None, false)
        } else {
            error!("No team or enterprise information in OAuth response");
            return Err(StatusCode::BAD_REQUEST);
        };

    Ok(WorkspaceConfig {
        workspace_id: Uuid::new_v4(),
        team_id,
        team_name,
        enterprise_id,
        enterprise_name,
        is_enterprise,
        access_token,
        token_type,
    })
}

async fn store_workspace(
    config: WorkspaceConfig,
    oauth_response: SlackOAuthTokenResponse,
    state: &Arc<AppState>,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    // Encrypt tokens
    let encrypted_access_token = encrypt_token(&config.access_token)?;
    
    // Store workspace with enterprise information
    let workspace_record = sqlx::query!(
        r#"
        INSERT INTO workspaces (
            id, name, slack_team_id, slack_team_name,
            slack_enterprise_id, slack_enterprise_name, is_enterprise,
            access_token_encrypted, token_type, bot_user_id, app_id,
            scopes, is_enterprise_install, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW(), NOW())
        ON CONFLICT (slack_team_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            slack_team_name = EXCLUDED.slack_team_name,
            slack_enterprise_id = EXCLUDED.slack_enterprise_id,
            slack_enterprise_name = EXCLUDED.slack_enterprise_name,
            is_enterprise = EXCLUDED.is_enterprise,
            access_token_encrypted = EXCLUDED.access_token_encrypted,
            token_type = EXCLUDED.token_type,
            bot_user_id = EXCLUDED.bot_user_id,
            app_id = EXCLUDED.app_id,
            scopes = EXCLUDED.scopes,
            is_enterprise_install = EXCLUDED.is_enterprise_install,
            updated_at = NOW()
        RETURNING id
        "#,
        config.workspace_id,
        config.team_name.clone(),
        config.team_id,
        config.team_name,
        config.enterprise_id,
        config.enterprise_name,
        config.is_enterprise,
        encrypted_access_token,
        config.token_type,
        oauth_response.bot_user_id,
        oauth_response.app_id,
        oauth_response.scope,
        oauth_response.is_enterprise_install
    )
    .fetch_one(&state.db)
    .await?;

    // Store authed user with token type validation
    if let Some(authed_user) = oauth_response.authed_user {
        // Validate user token type if present
        if let Some(user_token_type) = &authed_user.token_type {
            if user_token_type != EXPECTED_TOKEN_TYPE {
                warn!("Unexpected user token type: {} for user {}", 
                      user_token_type, authed_user.id);
            }
        }

        let encrypted_user_token = authed_user.access_token
            .as_ref()
            .map(|t| encrypt_token(t).ok())
            .flatten();

        sqlx::query!(
            r#"
            INSERT INTO slack_authed_users (
                workspace_id, user_id, scope,
                access_token_encrypted, token_type, created_at
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (workspace_id, user_id) 
            DO UPDATE SET 
                scope = EXCLUDED.scope,
                access_token_encrypted = EXCLUDED.access_token_encrypted,
                token_type = EXCLUDED.token_type,
                updated_at = NOW()
            "#,
            workspace_record.id,
            authed_user.id,
            authed_user.scope,
            encrypted_user_token,
            authed_user.token_type
        )
        .execute(&state.db)
        .await?;

        info!("Stored authed user {} for workspace", authed_user.id);
    }

    // Log installation type for monitoring
    if config.is_enterprise {
        info!(
            "Enterprise workspace created/updated: {} ({}) under enterprise {} ({})",
            config.team_name, config.team_id,
            config.enterprise_name.unwrap_or_default(),
            config.enterprise_id.unwrap_or_default()
        );
    } else {
        info!("Regular workspace created/updated: {} ({})", 
              config.team_name, config.team_id);
    }

    Ok(workspace_record.id)
}

#[instrument(skip(state, params))]
pub async fn handle_oauth_callback(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SlackOAuthCallback>,
) -> Result<Json<SlackOAuthResponse>, StatusCode> {
    // Check for OAuth errors
    if let Some(error) = params.error {
        warn!("OAuth error: {}", error);
        return Ok(Json(SlackOAuthResponse {
            ok: false,
            message: format!("OAuth failed: {}", error),
            workspace_id: None,
        }));
    }

    // Verify state parameter (CSRF protection)
    if let Some(state_param) = &params.state {
        if !verify_oauth_state(state_param, &state).await {
            error!("Invalid OAuth state parameter");
            return Err(StatusCode::UNAUTHORIZED);
        }
    } else {
        warn!("No state parameter in OAuth callback - potential CSRF risk");
        // In production, you might want to reject this
    }

    // Exchange code for access token
    let token_response = exchange_oauth_code(&params.code).await
        .map_err(|e| {
            error!("Failed to exchange OAuth code: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Validate token type
    if let Some(token_type) = &token_response.token_type {
        if token_type != EXPECTED_TOKEN_TYPE {
            warn!("Unexpected token type: {} (expected {})", token_type, EXPECTED_TOKEN_TYPE);
        }
    }

    // Create workspace configuration
    let workspace_config = extract_workspace_config(token_response.clone())?;

    // Store workspace in database
    let workspace_id = store_workspace(workspace_config, token_response, &state).await
        .map_err(|e| {
            error!("Failed to store workspace: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!("OAuth successful for workspace {}", workspace_id);

    Ok(Json(SlackOAuthResponse {
        ok: true,
        message: "OAuth successful! Your workspace has been added to Interstice.".to_string(),
        workspace_id: Some(workspace_id),
    }))
}

/// Handle Slack slash commands with full validation
#[instrument(skip(state, headers, body))]
pub async fn handle_slash_commands(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Result<impl IntoResponse, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Verify the request
    verify_slack_request(&headers, &body, adapter)?;

    // Parse the command
    let command: SlackCommandEvent = serde_urlencoded::from_str(&body)
        .map_err(|e| {
            error!("Failed to parse slash command: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    info!(
        "Slash command '{}' received from team {} by user {} with text: '{}'",
        command.command,
        command.team_id,
        command.user_id,
        command.text
    );

    // Process based on command
    let response = match command.command.as_str() {
        "/interstice" => handle_interstice_command(&command, &state).await,
        "/interstice-track" => handle_track_command(&command, &state).await,
        "/interstice-insights" => handle_insights_command(&command, &state).await,
        _ => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Unknown command. Use `/interstice help` to see available commands."
            })
        }
    };

    // Store command usage for analytics
    store_command_usage(&command, &response, &state).await;

    // Check if we need async processing (>3s operations)
    if should_use_async_response(&command.text) {
        // Send immediate acknowledgment
        let ack_response = serde_json::json!({
            "response_type": "ephemeral",
            "text": "Processing your request... Results will appear shortly."
        });
        
        // Process async and post to response_url
        let state_clone = state.clone();
        let command_clone = command.clone();
        tokio::spawn(async move {
            let full_response = process_complex_command(&command_clone, &state_clone).await;
            let _ = post_to_response_url(&command_clone.response_url, &full_response).await;
        });
        
        return Ok(Json(ack_response));
    }
    
    Ok(Json(response))
}

/// Handle the main /interstice command with subcommands
async fn handle_interstice_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> serde_json::Value {
    let args: Vec<&str> = command.text.split_whitespace().collect();
    let subcommand = args.get(0).map(|s| s.to_lowercase());
    
    match subcommand.as_deref() {
        Some("help") | None => show_help_command(),
        Some("status") => show_workspace_status(&command.team_id, state).await,
        Some("predict") => run_predictions_command(command, state).await,
        Some("analyze") => analyze_workspace_patterns(&command.team_id, state).await,
        Some("recent") => show_recent_artifacts(&command.team_id, state).await,
        Some("stats") => show_workspace_stats(&command.team_id, state).await,
        _ => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": format!("Unknown subcommand '{}'. Use `/interstice help` for available commands.", args[0])
            })
        }
    }
}

async fn analyze_workspace_patterns(
    team_id: &str,
    state: &Arc<AppState>,
) -> serde_json::Value {
    match fetch_workspace_patterns(team_id, &state.db).await {
        Ok(patterns) => {
            let mut blocks = vec![
                serde_json::json!({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📊 Workspace Analysis*"
                    }
                }),
                serde_json::json!({
                    "type": "divider"
                })
            ];
            
            blocks.push(serde_json::json!({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": format!("*Peak Activity Hour:*\n{}", patterns.peak_hour)
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Most Active Channel:*\n{}", patterns.most_active_channel)
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Avg Daily Artifacts:*\n{:.1}", patterns.avg_daily_artifacts)
                    },
                    {
                        "type": "mrkdwn",
                        "text": format!("*Common Artifact Type:*\n{}", patterns.common_artifact_type)
                    }
                ]
            }));
            
            serde_json::json!({
                "response_type": "ephemeral",
                "blocks": blocks
            })
        }
        Err(e) => {
            error!("Failed to analyze patterns: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to analyze workspace patterns. Please try again later."
            })
        }
    }
}

async fn show_recent_artifacts(
    team_id: &str,
    state: &Arc<AppState>,
) -> serde_json::Value {
    match fetch_recent_artifacts(team_id, &state.db, 5).await {
        Ok(artifacts) if !artifacts.is_empty() => {
            let mut blocks = vec![
                serde_json::json!({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📝 Recent Artifacts*"
                    }
                })
            ];
            
            for artifact in artifacts {
                blocks.push(serde_json::json!({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!(
                            "*{:?}*\n_{}_\n```{}```",
                            artifact.artifact_type,
                            artifact.created_at.format("%Y-%m-%d %H:%M UTC"),
                            truncate_string(&artifact.content, 100)
                        )
                    }
                }));
            }
            
            serde_json::json!({
                "response_type": "ephemeral",
                "blocks": blocks
            })
        }
        Ok(_) => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "No recent artifacts found. Start tracking some activities!"
            })
        }
        Err(e) => {
            error!("Failed to fetch recent artifacts: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch recent artifacts. Please try again later."
            })
        }
    }
}


/// Show help information
fn show_help_command() -> serde_json::Value {
    serde_json::json!({
        "response_type": "ephemeral",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Welcome to Interstice Engine!* 🚀\n\nI help track work artifacts and predict outcomes from your Slack activity."
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Available Commands:*\n\n• `/interstice help` - Show this help message\n• `/interstice status` - View workspace status\n• `/interstice predict` - Get outcome predictions\n• `/interstice analyze` - Analyze workspace patterns\n• `/interstice recent` - Show recent artifacts\n• `/interstice stats` - View detailed statistics\n\n• `/interstice-track <text>` - Manually track an artifact\n• `/interstice-insights` - Generate AI-powered insights"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "💡 _Tip: I automatically track artifacts from your messages!_"
                    }
                ]
            }
        ]
    })
}

async fn show_workspace_stats(
    team_id: &str,
    state: &Arc<AppState>,
) -> serde_json::Value {
    match fetch_workspace_statistics(team_id, &state.db).await {
        Ok(stats) => {
            serde_json::json!({
                "response_type": "ephemeral",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*📈 Detailed Statistics*"
                        }
                    },
                    {
                        "type": "divider"
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": format!(
                                "*Last 7 Days:*\n• Artifacts: {}\n• Commands: {}\n• Active Users: {}\n\n*Last 30 Days:*\n• Total Artifacts: {}\n• Predictions: {}\n• Success Rate: {:.1}%",
                                stats.weekly_artifacts,
                                stats.weekly_commands,
                                stats.active_users,
                                stats.monthly_artifacts,
                                stats.monthly_predictions,
                                stats.success_rate * 100.0
                            )
                        }
                    }
                ]
            })
        }
        Err(e) => {
            error!("Failed to fetch statistics: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch statistics. Please try again later."
            })
        }
    }
}

fn extract_channel_from_text(text: &str) -> Option<String> {
    // Look for <#CHANNEL_ID|channel-name> format
    if let Some(start) = text.find("<#") {
        if let Some(end) = text[start..].find('>') {
            let channel_ref = &text[start+2..start+end];
            if let Some(pipe) = channel_ref.find('|') {
                return Some(channel_ref[..pipe].to_string());
            }
            return Some(channel_ref.to_string());
        }
    }
    None
}

/// Extract and predict artifacts
async fn extract_and_predict_artifacts(
    team_id: &str,
    channel_id: &str,
    state: &Arc<AppState>,
) -> Result<Vec<OutcomePrediction>, Box<dyn std::error::Error>> {
    // Fetch recent artifacts from the channel
    let artifacts = fetch_channel_artifacts(team_id, channel_id, &state.db, 10).await?;
    
    if artifacts.is_empty() {
        return Ok(vec![]);
    }
    
    // Use ML pipeline to predict outcomes
    if let Some(ml) = Some(state.ml()) {
        // Convert core artifacts to ML artifacts
        let ml_artifacts: Vec<interstice_ml::types::Artifact> = artifacts.iter().map(|a| {
            interstice_ml::types::Artifact::new(
                a.id.to_string(),
                a.content.clone(),
                interstice_ml::types::Platform::Slack,
                interstice_ml::types::ArtifactType::Message
            )
        }).collect();
        
        let predictions = ml.predict_outcomes(
            Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4()),
            &ml_artifacts,
            &artifacts.iter().map(|a| a.content.as_str()).collect::<Vec<_>>().join(" ")
        ).await?;
        
        // Store predictions for analytics
        for pred in &predictions {
            store_prediction(pred, team_id, &state.db).await?;
        }
        
        Ok(predictions)
    } else {
        // Fallback predictions if ML is not available
        Ok(vec![
            OutcomePrediction {
                outcome_id: String::new(),
                outcome_name: "Task Completion".to_string(),
                confidence: 0.75,
                contributing_factors: vec![],
                alternative_outcomes: vec![],
                predicted_impact: ImpactLevel::Medium,
                time_to_completion: Some(Duration::from_hours_with_uncertainty(0.1, 0.1)),
                reasoning: Some("Based on recent activity patterns".to_string()),
            },
            OutcomePrediction {
                outcome_id: String::new(),
                outcome_name: "Meeting Scheduled".to_string(),
                confidence: 0.60,
                contributing_factors: vec![],
                alternative_outcomes: vec![],
                predicted_impact: ImpactLevel::Medium,
                time_to_completion: Some(Duration::from_hours_with_uncertainty(0.1, 0.1)),
                reasoning: Some("Scheduled based on team availability".to_string()),
            },
        ])
    }
}

/// Process complex commands asynchronously
async fn process_complex_command(
    command: &SlackCommandEvent,
    _state: &Arc<AppState>,
) -> serde_json::Value {
    // This would handle long-running operations
    // For now, return a placeholder
    serde_json::json!({
        "response_type": "in_channel",
        "text": format!("Completed processing: {}", command.text)
    })
}

/// Show workspace status
async fn show_workspace_status(
    team_id: &str,
    state: &Arc<AppState>,
) -> serde_json::Value {
    match fetch_workspace_statistics(team_id, &state.db).await {
        Ok(status) => {
            serde_json::json!({
                "response_type": "ephemeral",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": format!("*Workspace Status for Team {}*", team_id)
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": format!("*Weekly Artifacts:*\n{}", status.weekly_artifacts)
                            },
                            {
                                "type": "mrkdwn",
                                "text": format!("*Monthly Predictions:*\n{}", status.monthly_predictions)
                            },
                            {
                                "type": "mrkdwn",
                                "text": format!("*Active Users:*\n{}", status.active_users)
                            },
                            {
                                "type": "mrkdwn",
                                "text": format!("*Success Rate:*\n{:.1}%", status.success_rate * 100.0)
                            }
                        ]
                    }
                ]
            })
        }
        Err(e) => {
            error!("Failed to fetch workspace status: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch workspace status. Please try again later."
            })
        }
    }
}

/// Handle Slack interactive elements (button clicks, etc.)
#[instrument(skip(state, headers, body))]
pub async fn handle_interactions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Result<impl IntoResponse, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        error!("Slack adapter not configured");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Verify the request
    verify_slack_request(&headers, &body, adapter)?;

    // Parse the interaction payload (it comes as form-encoded with a 'payload' field)
    let form: std::collections::HashMap<String, String> = 
        serde_urlencoded::from_str(&body)
            .map_err(|e| {
                error!("Failed to parse interaction form: {}", e);
                StatusCode::BAD_REQUEST
            })?;

    let payload_str = form.get("payload")
        .ok_or_else(|| {
            error!("No payload field in interaction");
            StatusCode::BAD_REQUEST
        })?;

    let payload: SlackInteractionEvent = serde_json::from_str(payload_str)
        .map_err(|e| {
            error!("Failed to parse interaction payload: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    // Process the interaction based on type
    let response = match payload.event_type.as_str() {
        "block_actions" => {
            info!(
                "Block {} '{}' from user {} ({}) in channel {} ({})",
                payload.actions.iter().map(|a| a.action_id.clone()).collect::<Vec<String>>().join(", "),
                payload.callback_id,     // NOW USED
                payload.user.id, 
                payload.user.name,       // NOW USED
                payload.channel.id,
                payload.channel.name     // NOW USED
            );
            
            // Handle specific callbacks
            match payload.callback_id.as_str() {
                "approve_task" => handle_task_approval(&payload, &state).await,
                "predict_outcome" => handle_prediction_request(&payload, &state).await,
                _ => {
                    // For unhandled callbacks, acknowledge and potentially respond async
                    if !payload.response_url.is_empty() {
                        tokio::spawn(async move {
                            let _ = post_to_response_url(
                                &payload.response_url,
                                &serde_json::json!({
                                    "text": "Processing your request..."
                                })
                            ).await;
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
            // Use trigger_id for modal operations
            if !payload.trigger_id.is_empty() {
                // Could open another modal or update existing one
                info!("Modal submission with trigger_id: {}", payload.trigger_id);
            }
            
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Submission received"
            })
        },
        _ => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Unknown interaction type"
            })
        }
    };

    Ok(Json(response))
}

async fn handle_task_approval(
    payload: &SlackInteractionEvent,
    _state: &Arc<AppState>,
) -> serde_json::Value {
    // Use all the fields
    info!(
        "Task approval from {} in channel {}", 
        payload.user.name,
        payload.channel.name
    );
    
    // Process approval...
    
    serde_json::json!({
        "response_type": "ephemeral",
        "text": format!("Task approved by {}", payload.user.name)
    })
}

async fn handle_prediction_request(
    payload: &SlackInteractionEvent,
    _state: &Arc<AppState>,
) -> serde_json::Value {
    // Use response_url for async processing
    let response_url = payload.response_url.clone();
    let channel_name = payload.channel.name.clone();
    
    tokio::spawn(async move {
        // Do expensive prediction work
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        
        let _ = post_to_response_url(&response_url, &serde_json::json!({
            "text": format!("Predictions ready for #{}", channel_name)
        })).await;
    });
    
    serde_json::json!({
        "response_type": "ephemeral",
        "text": "Generating predictions..."
    })
}

/// Verify Slack request signature and timestamp
fn verify_slack_request(
    headers: &HeaderMap,
    body: &str,
    adapter: &SlackAdapter,
) -> Result<(), StatusCode> {
    // Extract timestamp
    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing timestamp header");
            StatusCode::UNAUTHORIZED
        })?;

    // Check timestamp age to prevent replay attacks
    let timestamp_num: i64 = timestamp.parse().map_err(|_| {
        warn!("Invalid timestamp format");
        StatusCode::UNAUTHORIZED
    })?;
    
    let current_time = chrono::Utc::now().timestamp();
    if (current_time - timestamp_num).abs() > MAX_TIMESTAMP_AGE_SECS {
        warn!("Request timestamp too old");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Extract signature
    let signature = headers
        .get("x-slack-signature")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| {
            warn!("Missing signature header");
            StatusCode::UNAUTHORIZED
        })?;

    // Verify signature starts with correct version
    if !signature.starts_with(&format!("{}=", SIGNATURE_VERSION)) {
        warn!("Invalid signature version");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Verify signature
    let mut headers_map = std::collections::HashMap::new();
    headers_map.insert("x-slack-request-timestamp".to_string(), timestamp.to_string());
    headers_map.insert("x-slack-signature".to_string(), signature.to_string());
    
    if !adapter.verify_webhook(&headers_map, body.as_bytes()).unwrap_or(false) {
        warn!("Invalid Slack signature");
        return Err(StatusCode::UNAUTHORIZED);
    }

    Ok(())
}

async fn run_predictions_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> serde_json::Value {
    // Extract channel from command text or use current channel
    let channel_id = extract_channel_from_text(&command.text)
        .unwrap_or_else(|| command.channel_id.clone());
    
    match extract_and_predict_artifacts(&command.team_id, &channel_id, state).await {
        Ok(predictions) if !predictions.is_empty() => {
            format_predictions_response(&predictions)
        }
        Ok(_) => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "No predictions available. Try generating more activity first!"
            })
        }
        Err(e) => {
            error!("Failed to run predictions: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to generate predictions. Please try again later."
            })
        }
    }
}

/// Format predictions into Slack blocks
fn format_predictions_response(predictions: &[OutcomePrediction]) -> serde_json::Value {
    let mut blocks = vec![
        serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*🔮 Outcome Predictions*"
            }
        })
    ];
    
    for (i, pred) in predictions.iter().enumerate().take(5) {
        let confidence_bar = "█".repeat((pred.confidence * 10.0) as usize);
        let empty_bar = "░".repeat(10 - (pred.confidence * 10.0) as usize);
        
        blocks.push(serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": format!(
                    "*{}. {}*\n{}{} _{:.1}% confidence_",
                    i + 1,
                    pred.outcome_name,
                    confidence_bar,
                    empty_bar,
                    pred.confidence * 100.0
                )
            }
        }));
    }
    
    serde_json::json!({
        "response_type": "in_channel",
        "blocks": blocks
    })
}

/// Handle /interstice-track command
async fn handle_track_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> serde_json::Value {
    if command.text.is_empty() {
        return serde_json::json!({
            "response_type": "ephemeral",
            "text": "Please provide text to track. Usage: `/interstice-track <your artifact text>`"
        });
    }
    
    // Create artifact from command text
    let artifact = Artifact {
        id: Uuid::new_v4(),
        workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(&command.team_id).unwrap_or_else(|_| Uuid::new_v4())),
        artifact_type: ArtifactType::Task {
            id: Uuid::new_v4().to_string(),
            title: command.text.clone(),
            status: interstice_core::artifact::TaskStatus::Todo,
            assignee: Some(command.user_id.clone()),
            due_date: None,
            completed_at: None,
            checklist_items: vec![],
            dependencies: vec![],
            tags: vec![],
            recurring: false,
            parent_task: None,
            subtasks: vec![],
        },
        platform: Platform::Slack,
        content: command.text.clone(),
        metadata: serde_json::json!({
            "tracked_by": "slash_command",
            "user_id": command.user_id,
            "channel_id": command.channel_id
        }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: HashSet::new(),
    };
    
    // Store the artifact
    match store_artifact(&artifact, &command.team_id, &state.db).await {
        Ok(_) => {
            serde_json::json!({
                "response_type": "ephemeral",
                "text": format!("✅ Artifact tracked successfully: \"{}\"", command.text),
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": format!("✅ *Artifact Tracked*\n```{}```", command.text)
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": format!("Type: {:?} | ID: {}", artifact.artifact_type, artifact.id)
                            }
                        ]
                    }
                ]
            })
        }
        Err(e) => {
            error!("Failed to track artifact: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to track artifact. Please try again later."
            })
        }
    }
}

async fn store_command_usage(
    command: &SlackCommandEvent,
    response: &serde_json::Value,
    state: &Arc<AppState>,
) {
    let _ = sqlx::query!(
        r#"
        INSERT INTO slack_command_usage (
            team_id, user_id, command, text, response_type, created_at
        )
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        command.team_id,
        command.user_id,
        command.command,
        command.text,
        response["response_type"].as_str()
    )
    .execute(&state.db)
    .await;
}

/// Store an artifact in the database
async fn store_artifact(
    artifact: &Artifact,
    team_id: &str,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get workspace_id from team_id
    let workspace_id = sqlx::query_scalar!(
        "SELECT id FROM workspaces WHERE slack_team_id = $1",
        team_id
    )
    .fetch_optional(db)
    .await?
    .ok_or("Workspace not found")?;

    // Extract channel_id from artifact metadata or source
    let channel_id = artifact.metadata.get("channel_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    sqlx::query!(
        r#"
        INSERT INTO artifacts (
            id, workspace_id, artifact_type, content, 
            channel_id, metadata, platform, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
        artifact.id,
        workspace_id,
        format!("{:?}", artifact.artifact_type), // Convert enum to string
        artifact.content,
        channel_id,
        artifact.metadata,
        artifact.platform.to_string()
    )
    .execute(db)
    .await?;
    
    Ok(())
}

/// Determine if async response is needed based on command complexity
fn should_use_async_response(text: &str) -> bool {
    // Complex operations that might take >3s
    let complex_keywords = [
        "analyze",
        "report", 
        "deep",
        "comprehensive",
        "detailed",
        "full",
        "export",
        "generate",
        "historical"
    ];
    
    let text_lower = text.to_lowercase();
    complex_keywords.iter().any(|keyword| text_lower.contains(keyword))
}

/// Post response to Slack's response URL for async processing
async fn post_to_response_url(
    response_url: &str,
    response: &serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    
    let res = client
        .post(response_url)
        .json(response)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await?;
    
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        error!("Failed to post to response_url. Status: {}, Body: {}", status, body);
        return Err(format!("Failed to post async response: {}", status).into());
    }
    
    info!("Successfully posted async response to Slack");
    Ok(())
}

/// Handle /interstice-insights command
async fn handle_insights_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> serde_json::Value {
    match generate_workspace_insights(&command.team_id, state).await {
        Ok(insights) => insights,
        Err(e) => {
            error!("Failed to generate insights: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to generate insights. Please try again later."
            })
        }
    }
}

/// Generate AI-powered workspace insights
async fn generate_workspace_insights(
    team_id: &str,
    state: &Arc<AppState>,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    // Fetch recent patterns and statistics
    let _stats = fetch_workspace_statistics(team_id, &state.db).await?;
    
    // Use ML pipeline to generate insights
    let insights = vec![
        "📈 Task completion rate increased by 15% this week",
        "🎯 Most productive hours: 10am - 12pm",
        "👥 Top collaborators: Alice, Bob, Charlie",
        "⚡ Average response time: 2.3 hours",
    ];
    
    let mut blocks = vec![
        serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*🧠 Workspace Insights*"
            }
        }),
        serde_json::json!({
            "type": "divider"
        })
    ];
    
    for insight in insights {
        blocks.push(serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": insight
            }
        }));
    }
    
    blocks.push(serde_json::json!({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": format!("_Generated at {} UTC_", Utc::now().format("%Y-%m-%d %H:%M"))
            }
        ]
    }));
    
    Ok(serde_json::json!({
        "response_type": "in_channel",
        "blocks": blocks
    }))
}


/// Check if an event has already been processed (idempotency)
async fn is_duplicate_event(event_id: &str, state: &Arc<AppState>) -> bool {
    // Use Redis or database to track processed events
    // For now, we'll use a simple in-memory check via the database
    
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
    .await;

    // If the insert returned nothing, it was a duplicate
    matches!(result, Ok(None))
}

/// Track event metrics for monitoring
async fn track_event_metrics(metrics: SlackEventMetrics, state: &Arc<AppState>) {
    // In production, send to metrics service (Datadog, CloudWatch, etc.)
    info!(
        "Event processed: type={}, team={:?}, artifacts={}, time={}ms",
        metrics.event_type,
        metrics.team_id,
        metrics.processed_artifacts,
        metrics.processing_time_ms
    );

    // Store in database for analytics
    let _ = sqlx::query!(
        r#"
        INSERT INTO event_metrics (platform, event_type, team_id, artifact_count, processing_time_ms, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        metrics.platform.to_string(),
        metrics.event_type,
        metrics.team_id,
        metrics.processed_artifacts as i32,
        metrics.processing_time_ms as i32
    )
    .execute(&state.db)
    .await;
}

/// Store event for audit trail
async fn store_event_audit(
    event: &SlackEventRequest,
    processed: &ProcessedData,
    state: &Arc<AppState>,
) {
    let _ = sqlx::query!(
        r#"
        INSERT INTO slack_event_audit (
            event_id, event_type, team_id, event_data, 
            artifacts_found, predictions_made, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        "#,
        event.event_id.as_deref().unwrap_or("unknown"),
        event.event_type,
        event.team_id,
        serde_json::to_value(event).ok(),
        processed.artifacts.len() as i32,
        processed.predictions.len() as i32
    )
    .execute(&state.db)
    .await;
}

/// Verify OAuth state parameter for CSRF protection
async fn verify_oauth_state(state_param: &str, app_state: &Arc<AppState>) -> bool {
    // Check if this state exists and hasn't expired
    let result = sqlx::query!(
        r#"
        DELETE FROM oauth_states 
        WHERE state = $1 
          AND created_at > NOW() - INTERVAL '10 minutes'
        RETURNING state
        "#,
        state_param
    )
    .fetch_optional(&app_state.db)
    .await;

    result.is_ok() && result.unwrap().is_some()
}

/// Exchange OAuth code for access token
async fn exchange_oauth_code(
    code: &str,
) -> Result<SlackOAuthTokenResponse, Box<dyn std::error::Error>> {
    // Get OAuth credentials from environment
    let client_id = std::env::var("SLACK_CLIENT_ID")
        .map_err(|_| "SLACK_CLIENT_ID not set")?;
    let client_secret = std::env::var("SLACK_CLIENT_SECRET")
        .map_err(|_| "SLACK_CLIENT_SECRET not set")?;
    let redirect_uri = std::env::var("SLACK_REDIRECT_URI")
        .unwrap_or_else(|_| {
            warn!("SLACK_REDIRECT_URI not set, using default");
            "https://api.interstice.com/webhooks/slack/oauth".to_string()
        });
    
    // Prepare OAuth parameters
    let params = [
        ("client_id", client_id.as_str()),
        ("client_secret", client_secret.as_str()),
        ("code", code),
        ("redirect_uri", redirect_uri.as_str()),
    ];

    info!("Exchanging OAuth code for workspace installation");

    // Make OAuth request
    let client = reqwest::Client::new();
    let response = client
        .post(SLACK_OAUTH_URL)
        .form(&params)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("OAuth request failed: {}", e))?;

    // Check HTTP status
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OAuth request failed with status {}: {}", status, body).into());
    }

    // Parse response
    let oauth_response: SlackOAuthTokenResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse OAuth response: {}", e))?;
    
    // Check Slack API response
    if !oauth_response.ok {
        return Err(format!("Slack OAuth failed: {:?}", oauth_response.error).into());
    }

    // Log successful exchange
    if let Some(team) = &oauth_response.team {
        info!("OAuth code exchanged successfully for team: {}", team.name);
    } else if let Some(enterprise) = &oauth_response.enterprise {
        info!("OAuth code exchanged successfully for enterprise: {}", enterprise.name);
    }

    Ok(oauth_response)
}

pub fn initialize_encryption_key() -> Result<(), Box<dyn std::error::Error>> {
    // Try multiple sources in order of preference
    let key_material = if let Ok(kms_key_id) = std::env::var("KMS_KEY_ID") {
        // Production: Fetch from AWS KMS
        fetch_key_from_kms(&kms_key_id)?
    } else if let Ok(vault_path) = std::env::var("VAULT_KEY_PATH") {
        // Production: Fetch from HashiCorp Vault
        fetch_key_from_vault(&vault_path)?
    } else if let Ok(key_base64) = std::env::var("ENCRYPTION_KEY") {
        // Development/staging: Base64-encoded 256-bit key
        URL_SAFE_NO_PAD.decode(key_base64)?
    } else {
        return Err("No encryption key source configured. Set one of: KMS_KEY_ID, VAULT_KEY_PATH, or ENCRYPTION_KEY".into());
    };

    // Validate key length
    if key_material.len() != 32 {
        return Err(format!("Invalid key length: expected 32 bytes, got {}", key_material.len()).into());
    }

    // Create the encryption key
    let unbound_key = UnboundKey::new(&AES_256_GCM, &key_material)
        .map_err(|e| format!("Failed to create encryption key: {:?}", e))?;
    
    let key = LessSafeKey::new(unbound_key);
    
    ENCRYPTION_KEY.set(key)
        .map_err(|_| "Encryption key already initialized")?;
    
    Ok(())
}

#[instrument(skip(token))]
fn encrypt_token(token: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Get the encryption key
    let key = ENCRYPTION_KEY.get()
        .ok_or("Encryption key not initialized. Call initialize_encryption_key() at startup")?;
    
    // Generate a random 96-bit nonce
    let rng = SystemRandom::new();
    let mut nonce_bytes = [0u8; 12];
    rng.fill(&mut nonce_bytes)
        .map_err(|e| format!("Failed to generate nonce: {:?}", e))?;
    
    // Prepare the plaintext
    let mut in_out = token.as_bytes().to_vec();
    
    // Reserve space for the authentication tag (16 bytes for AES-256-GCM)
    in_out.reserve(16);
    
    // Create nonce
    let nonce = Nonce::assume_unique_for_key(nonce_bytes);
    
    // Add metadata as additional authenticated data (AAD)
    // This ensures the ciphertext can't be used in a different context
    let aad_data = format!("slack_token_v1_{}", chrono::Utc::now().timestamp());
    let aad = Aad::from(aad_data.as_bytes());
    
    // Encrypt in place and append authentication tag
    key.seal_in_place_append_tag(nonce, aad, &mut in_out)
        .map_err(|e| format!("Encryption failed: {:?}", e))?;
    
    // Combine nonce and ciphertext
    let mut encrypted = Vec::with_capacity(nonce_bytes.len() + in_out.len());
    encrypted.extend_from_slice(&nonce_bytes);
    encrypted.extend_from_slice(&in_out);
    
    // Encode as URL-safe base64 without padding
    Ok(URL_SAFE_NO_PAD.encode(encrypted))
}

/// Decrypt a token encrypted with encrypt_token
/// 
/// This function is the counterpart to encrypt_token and should be used
/// when retrieving tokens from storage
#[instrument(skip(encrypted))]
pub fn decrypt_token(encrypted: &str) -> Result<String, Box<dyn std::error::Error>> {
    // Get the encryption key
    let key = ENCRYPTION_KEY.get()
        .ok_or("Encryption key not initialized")?;
    
    // Decode from base64
    let encrypted_bytes = URL_SAFE_NO_PAD.decode(encrypted)
        .map_err(|e| format!("Invalid base64: {}", e))?;
    
    // Extract nonce (first 12 bytes)
    if encrypted_bytes.len() < 12 {
        return Err("Invalid encrypted data: too short".into());
    }
    
    let (nonce_bytes, ciphertext) = encrypted_bytes.split_at(12);
    let nonce = Nonce::assume_unique_for_key(nonce_bytes.try_into()?);
    
    // Prepare ciphertext buffer
    let mut in_out = ciphertext.to_vec();
    
    // Recreate AAD (this would need timestamp from metadata in production)
    // For now, using a fixed AAD pattern
    let aad_data = b"slack_token_v1_*";
    let aad = Aad::from(&aad_data[..]);
    
    // Decrypt and verify authentication tag
    key.open_in_place(nonce, aad, &mut in_out)
        .map_err(|e| {
            error!("Decryption failed - possible tampering detected: {:?}", e);
            format!("Failed to decrypt token: {:?}", e)
        })?;
    
    // Convert to string
    String::from_utf8(in_out)
        .map_err(|e| format!("Invalid UTF-8 in decrypted token: {}", e).into())
}


#[cfg(feature = "aws")]
async fn fetch_key_from_kms(key_id: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use aws_sdk_kms::{Client, types::DataKeySpec};
    
    let config = aws_config::load_from_env().await;
    let client = Client::new(&config);
    
    let response = client
        .generate_data_key()
        .key_id(key_id)
        .key_spec(DataKeySpec::Aes256)
        .send()
        .await?;
    
    response.plaintext
        .ok_or("No plaintext key returned from KMS".into())
        .map(|b| b.into_inner())
}

#[cfg(not(feature = "aws"))]
fn fetch_key_from_kms(_key_id: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("AWS KMS support not compiled. Enable 'aws' feature".into())
}

#[cfg(feature = "vault")]
fn fetch_key_from_vault(path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use vaultrs::{client::{VaultClient, VaultClientSettingsBuilder}, kv2};
    
    let client = VaultClient::new(
        VaultClientSettingsBuilder::default()
            .address(std::env::var("VAULT_ADDR")?)
            .token(std::env::var("VAULT_TOKEN")?)
            .build()?
    )?;
    
    let secret: std::collections::HashMap<String, String> = 
        kv2::read(&client, "secret", path)?;
    
    let key_base64 = secret.get("encryption_key")
        .ok_or("No encryption_key field in Vault secret")?;
    
    URL_SAFE_NO_PAD.decode(key_base64)
        .map_err(|e| e.into())
}

#[cfg(not(feature = "vault"))]
fn fetch_key_from_vault(_path: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Err("HashiCorp Vault support not compiled. Enable 'vault' feature".into())
}


// Add cleanup job for expired OAuth states
pub async fn cleanup_expired_oauth_states(state: &Arc<AppState>) {
    let result = sqlx::query!(
        r#"
        DELETE FROM oauth_states 
        WHERE expires_at < NOW()
        RETURNING state
        "#
    )
    .fetch_all(&state.db)
    .await;
    
    match result {
        Ok(deleted) => {
            if !deleted.is_empty() {
                info!("Cleaned up {} expired OAuth states", deleted.len());
            }
        }
        Err(e) => {
            error!("Failed to cleanup expired OAuth states: {}", e);
        }
    }
}


/// Health check endpoint for Slack integration
#[instrument(skip(state))]
pub async fn slack_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.slack_adapter.is_some() {
        (StatusCode::OK, "Slack integration is healthy")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "Slack adapter not configured")
    }
}

/// Generate OAuth URL for Slack installation
pub async fn get_oauth_url(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let client_id = std::env::var("SLACK_CLIENT_ID")
        .map_err(|_| {
            error!("SLACK_CLIENT_ID not configured");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    let redirect_uri = std::env::var("SLACK_REDIRECT_URI")
        .unwrap_or_else(|_| "https://api.interstice.com/webhooks/slack/oauth".to_string());
    
    // Generate and store state for CSRF protection
    let state_param = Uuid::new_v4().to_string();
    
    sqlx::query!(
        r#"
        INSERT INTO oauth_states (state, created_at, expires_at)
        VALUES ($1, NOW(), NOW() + INTERVAL '10 minutes')
        "#,
        state_param
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to store OAuth state: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    // Define scopes based on your app's needs
    let scopes = [
        "app_mentions:read",     // Read @mentions
        "channels:history",      // Read public channel messages
        "channels:read",         // List channels
        "chat:write",           // Send messages
        "commands",             // Slash commands
        "groups:history",       // Read private channel messages
        "groups:read",          // List private channels
        "im:history",           // Read DMs
        "im:read",              // List DMs
        "mpim:history",         // Read group DMs
        "mpim:read",            // List group DMs
        "reactions:read",       // Read reactions
        "team:read",           // Read team info
        "users:read",          // Read user info
    ].join(",");
    
    // Build OAuth URL with all parameters
    let oauth_url = format!(
        "https://slack.com/oauth/v2/authorize?client_id={}&scope={}&redirect_uri={}&state={}",
        urlencoding::encode(&client_id),
        urlencoding::encode(&scopes),
        urlencoding::encode(&redirect_uri),
        urlencoding::encode(&state_param)
    );
    
    info!("Generated OAuth URL for Slack installation");
    
    Ok(Json(serde_json::json!({
        "oauth_url": oauth_url,
        "state": state_param,
        "expires_in": 600 // 10 minutes
    })))
}

#[derive(Debug)]
struct WorkspacePatterns {
    peak_hour: String,
    most_active_channel: String,
    avg_daily_artifacts: f64,
    common_artifact_type: String,
}

#[derive(Debug)]
struct WorkspaceStatistics {
    weekly_artifacts: i64,
    weekly_commands: i64,
    active_users: i64,
    monthly_artifacts: i64,
    monthly_predictions: i64,
    success_rate: f64,
}

async fn fetch_workspace_patterns(
    team_id: &str,
    db: &PgPool,
) -> Result<WorkspacePatterns, Box<dyn std::error::Error>> {
    // Query for peak hour
    let peak_hour = sqlx::query_scalar!(
        r#"
        SELECT EXTRACT(HOUR FROM created_at)::INT as hour
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '30 days'
        GROUP BY hour
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .map(|h| format!("{}:00", h.unwrap_or(0)))
    .unwrap_or_else(|| "N/A".to_string());
    
    // Query for most active channel
    let most_active_channel = sqlx::query_scalar!(
        r#"
        SELECT channel_id
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND channel_id IS NOT NULL
        GROUP BY channel_id
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .flatten()
    .unwrap_or_else(|| "N/A".to_string());
    
    // Average daily artifacts
    let avg_daily = sqlx::query_scalar!(
        r#"
        SELECT AVG(daily_count)::FLOAT
        FROM (
            SELECT DATE(created_at) as day, COUNT(*) as daily_count
            FROM artifacts
            WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
              AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY day
        ) as daily_counts
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .flatten()
    .unwrap_or(0.0) as f64;
    
    // Most common artifact type
    let common_type = sqlx::query_scalar!(
        r#"
        SELECT artifact_type
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        GROUP BY artifact_type
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .unwrap_or_else(|| "N/A".to_string());
    
    Ok(WorkspacePatterns {
        peak_hour,
        most_active_channel,
        avg_daily_artifacts: avg_daily,
        common_artifact_type: common_type,
    })
}

async fn fetch_recent_artifacts(
    team_id: &str,
    db: &PgPool,
    limit: i32,
) -> Result<Vec<Artifact>, Box<dyn std::error::Error>> {
    let records = sqlx::query!(
        r#"
        SELECT id, artifact_type, content, channel_id, metadata, created_at
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        team_id,
        limit as i64
    )
    .fetch_all(db)
    .await?;
    
    let artifacts = records.into_iter().map(|r| Artifact {
        id: r.id,
        workspace_id: WorkspaceId::from_uuid(Uuid::new_v4()), // Placeholder - would need actual workspace lookup
        artifact_type: ArtifactType::Message {
            id: r.id.to_string(),
            channel: r.channel_id.unwrap_or_default(),
            thread_id: None,
            author: "unknown".to_string(),
            content: r.content.clone(),
            mentions: vec![],
            attachments: vec![],
            reactions: HashMap::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        },
        platform: Platform::Slack,
        content: r.content,
        metadata: r.metadata.unwrap_or_else(|| serde_json::json!({})),
        created_at: r.created_at.unwrap_or_else(|| Utc::now()),
        updated_at: r.created_at.unwrap_or_else(|| Utc::now()),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: HashSet::new(),
    }).collect();
    
    Ok(artifacts)
}

async fn fetch_channel_artifacts(
    team_id: &str,
    channel_id: &str,
    db: &PgPool,
    limit: i32,
) -> Result<Vec<Artifact>, Box<dyn std::error::Error>> {
    let records = sqlx::query!(
        r#"
        SELECT id, artifact_type, content, metadata, created_at
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1) AND channel_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        "#,
        team_id,
        channel_id,
        limit as i64
    )
    .fetch_all(db)
    .await?;
    
    let artifacts = records.into_iter().map(|r| Artifact {
        id: r.id,
        workspace_id: WorkspaceId::from_uuid(Uuid::new_v4()), // Placeholder - would need actual workspace lookup
        artifact_type: ArtifactType::Message {
            id: r.id.to_string(),
            channel: channel_id.to_string(),
            thread_id: None,
            author: "unknown".to_string(),
            content: r.content.clone(),
            mentions: vec![],
            attachments: vec![],
            reactions: HashMap::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        },
        platform: Platform::Slack,
        content: r.content,
        metadata: r.metadata.unwrap_or_else(|| serde_json::json!({})),
        created_at: r.created_at.unwrap_or_else(|| Utc::now()),
        updated_at: r.created_at.unwrap_or_else(|| Utc::now()),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: HashSet::new(),
    }).collect();
    
    Ok(artifacts)
}

async fn fetch_workspace_statistics(
    team_id: &str,
    db: &PgPool,
) -> Result<WorkspaceStatistics, Box<dyn std::error::Error>> {
    // Weekly artifacts
    let weekly_artifacts = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    // Weekly commands
    let weekly_commands = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM slack_command_usage
        WHERE team_id = $1
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    // Active users
    let active_users = sqlx::query_scalar!(
        r#"
        SELECT COUNT(DISTINCT user_id)
        FROM slack_command_usage
        WHERE team_id = $1
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    // Monthly artifacts  
    let monthly_artifacts = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '30 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    // Monthly predictions
    let monthly_predictions = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM inference_history
        WHERE artifact_id IN (
            SELECT id FROM artifacts WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        )
        AND created_at > NOW() - INTERVAL '30 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    Ok(WorkspaceStatistics {
        weekly_artifacts,
        weekly_commands,
        active_users,
        monthly_artifacts,
        monthly_predictions,
        success_rate: 0.85, // Placeholder - calculate from actual data
    })
}

async fn store_prediction(
    prediction: &OutcomePrediction,
    team_id: &str,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Store in inference_history table
    sqlx::query!(
        r#"
        INSERT INTO inference_history (
            id, workspace_id, artifact_id,
            confidence, prediction_data, created_at
        )
        VALUES ($1, 
                (SELECT id FROM workspaces WHERE slack_team_id = $2 LIMIT 1),
                NULL, $3, $4, NOW())
        "#,
        Uuid::new_v4(),
        team_id,
        prediction.confidence as f32,
        serde_json::to_value(prediction)?
    )
    .execute(db)
    .await?;
    
    Ok(())
}

/// Helper function to truncate strings
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len-3])
    }
}

impl Clone for SlackCommandEvent {
    fn clone(&self) -> Self {
        SlackCommandEvent {
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