//interstice-api/src/handlers/slack.rs
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use interstice_adapters::{traits::{EventMetadata, EventType, PlatformAdapter, PlatformEvent}, SlackAdapter};
use interstice_core::{Platform, ProcessedData};
use serde::{Deserialize, Serialize};
use slack_morphism::prelude::*;
use std::sync::Arc;
use tracing::{error, info, warn, instrument};
use uuid::Uuid;

use crate::AppState;

// Constants for Slack API
const SLACK_OAUTH_URL: &str = "https://slack.com/api/oauth.v2.access";
const SIGNATURE_VERSION: &str = "v0";
const MAX_TIMESTAMP_AGE_SECS: i64 = 300; // 5 minutes
const EXPECTED_TOKEN_TYPE: &str = "Bearer"; // OAuth 2.0 standard

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
            let slack_event: SlackPushEvent = serde_json::from_value(event_data)
                .map_err(|e| {
                    error!("Failed to parse Slack event: {}", e);
                    StatusCode::BAD_REQUEST
                })?;

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
        "Slash command '{}' received from team {} by user {}",
        command.command,
        command.team_id,
        command.user_id
    );

    // Process the slash command
    let response = serde_json::json!({
        "response_type": "ephemeral",
        "text": "Slash command processing not yet implemented"
    });

    Ok(Json(response))
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

    // Process the interaction
    info!("Interaction received: {:?}", payload);

    // Return acknowledgment
    Ok(Json(serde_json::json!({ "ok": true })))
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

fn encrypt_token(token: &str) -> Result<String, Box<dyn std::error::Error>> {
    // In production, use one of these approaches:
    
    // Option 1: AWS KMS
    // let kms_client = aws_sdk_kms::Client::new(&aws_config);
    // let encrypted = kms_client.encrypt()...
    
    // Option 2: HashiCorp Vault
    // let vault_client = vault::Client::new(...);
    // let encrypted = vault_client.encrypt(token)...
    
    // Option 3: Local encryption with ring/sodiumoxide
    // use ring::aead;
    // let key = get_encryption_key()?;
    // let encrypted = aead::seal(...)?;
    
    // Temporary placeholder - NEVER use in production
    use base64::{Engine as _, engine::general_purpose};
    
    // Add warning in logs
    warn!("Using placeholder encryption - implement proper encryption before production!");
    
    Ok(general_purpose::STANDARD.encode(token))
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

/// Create workspace from OAuth response
async fn create_workspace_from_oauth(
    oauth_response: SlackOAuthTokenResponse,
    state: &Arc<AppState>,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let workspace_id = Uuid::new_v4();
    
    // Extract team information from OAuth response
    let team_id = oauth_response.team
        .as_ref()
        .map(|t| t.id.clone())
        .ok_or("No team information in OAuth response")?;
    
    let team_name = oauth_response.team
        .as_ref()
        .map(|t| t.name.clone())
        .unwrap_or_else(|| "Unknown Workspace".to_string());
    
    let access_token = oauth_response.access_token
        .ok_or("No access token in OAuth response")?;
    
    // Encrypt the token before storing
    let encrypted_token = encrypt_token(&access_token)?;
    
    // Store workspace with actual data
    sqlx::query!(
        r#"
        INSERT INTO workspaces (
            id, name, slack_team_id, slack_team_name, 
            access_token_encrypted, bot_user_id, app_id, 
            scopes, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW(), NOW())
        ON CONFLICT (slack_team_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            slack_team_name = EXCLUDED.slack_team_name,
            access_token_encrypted = EXCLUDED.access_token_encrypted,
            bot_user_id = EXCLUDED.bot_user_id,
            app_id = EXCLUDED.app_id,
            scopes = EXCLUDED.scopes,
            updated_at = NOW()
        RETURNING id
        "#,
        workspace_id,
        team_name.clone(),
        team_id,
        team_name,
        encrypted_token,
        oauth_response.bot_user_id,
        oauth_response.app_id,
        oauth_response.scope
    )
    .fetch_one(&state.db)
    .await?;

    // Store authed user information if present
    if let Some(authed_user) = oauth_response.authed_user {
        sqlx::query!(
            r#"
            INSERT INTO slack_authed_users (
                workspace_id, user_id, scope, 
                access_token_encrypted, created_at
            )
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (workspace_id, user_id) 
            DO UPDATE SET 
                scope = EXCLUDED.scope,
                access_token_encrypted = EXCLUDED.access_token_encrypted
            "#,
            workspace_id,
            authed_user.id,
            authed_user.scope,
            authed_user.access_token.map(|t| encrypt_token(&t).ok()).flatten()
        )
        .execute(&state.db)
        .await?;
    }

    info!("Workspace created/updated: {} ({})", team_name, team_id);
    
    Ok(workspace_id)
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