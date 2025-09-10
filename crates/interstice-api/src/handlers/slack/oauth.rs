// src/handlers/slack/oauth.rs

use axum::{extract::{Query, State}, Json};
use http::StatusCode;
use reqwest::Client;
use std::sync::Arc;
use tracing::{error, info, warn};
use uuid::Uuid;
use urlencoding::encode;

use crate::{handlers::slack::{encrypt_token, SlackOAuthCallback, SlackOAuthResponse, SlackOAuthTokenResponse, WorkspaceConfig, EXPECTED_TOKEN_TYPE, SLACK_OAUTH_URL}, AppState};

pub fn extract_workspace_config(
    oauth_response: SlackOAuthTokenResponse,
) -> Result<WorkspaceConfig, StatusCode> {
    let access_token = oauth_response.access_token.ok_or_else(|| {
        error!("No access token in OAuth response");
        StatusCode::BAD_REQUEST
    })?;

    let token_type = oauth_response.token_type.unwrap_or_else(|| EXPECTED_TOKEN_TYPE.to_string());

    let (team_id, team_name, enterprise_id, enterprise_name, is_enterprise) = 
        if let Some(enterprise) = oauth_response.enterprise {
            info!("Enterprise Grid installation detected: {} ({})", enterprise.name, enterprise.id);
            let team = oauth_response.team.unwrap_or(super::SlackTeamInfo {
                id: enterprise.id.clone(),
                name: enterprise.name.clone(),
            });
            (team.id, team.name, Some(enterprise.id), Some(enterprise.name), true)
        } else if let Some(team) = oauth_response.team {
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

pub async fn store_workspace(
    config: WorkspaceConfig,
    oauth_response: SlackOAuthTokenResponse,
    state: &Arc<AppState>,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let encrypted_access_token = encrypt_token(&config.access_token)?;
    
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
        &config.team_id,
        &config.team_name,
        config.enterprise_id,
        config.enterprise_name,
        config.is_enterprise,
        encrypted_access_token,
        &config.token_type,
        oauth_response.bot_user_id,
        oauth_response.app_id,
        oauth_response.scope,
        oauth_response.is_enterprise_install
    )
    .fetch_one(&state.db)
    .await?;

    if let Some(authed_user) = oauth_response.authed_user {
        if let Some(user_token_type) = &authed_user.token_type {
            if user_token_type != EXPECTED_TOKEN_TYPE {
                warn!("Unexpected user token type: {} for user {}", user_token_type, &authed_user.id);
            }
        }

        let encrypted_user_token = authed_user.access_token.as_ref().map(|t| encrypt_token(t).ok()).flatten();

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
            &authed_user.id,
            authed_user.scope,
            encrypted_user_token,
            authed_user.token_type
        )
        .execute(&state.db)
        .await?;

        info!("Stored authed user {} for workspace", &authed_user.id);
    }

    if config.is_enterprise {
        info!(
            "Enterprise workspace created/updated: {} ({}) under enterprise {} ({})",
            &config.team_name, &config.team_id,
            config.enterprise_name.as_deref().unwrap_or(""),
            config.enterprise_id.as_deref().unwrap_or("")
        );
    } else {
        info!("Regular workspace created/updated: {} ({})", &config.team_name, &config.team_id);
    }

    Ok(workspace_record.id)
}

pub async fn handle_oauth_callback(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SlackOAuthCallback>,
) -> Result<Json<SlackOAuthResponse>, StatusCode> {
    if let Some(error) = params.error {
        warn!("OAuth error: {}", error);
        return Ok(Json(SlackOAuthResponse {
            ok: false,
            message: format!("OAuth failed: {}", error),
            workspace_id: None,
        }));
    }

    if let Some(state_param) = &params.state {
        if !verify_oauth_state(state_param, &state).await {
            error!("Invalid OAuth state parameter");
            return Err(StatusCode::UNAUTHORIZED);
        }
    } else {
        warn!("No state parameter in OAuth callback - potential CSRF risk");
    }

    let token_response = exchange_oauth_code(&params.code).await
        .map_err(|e| {
            error!("Failed to exchange OAuth code: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if let Some(token_type) = &token_response.token_type {
        if token_type != EXPECTED_TOKEN_TYPE {
            warn!("Unexpected token type: {} (expected {})", token_type, EXPECTED_TOKEN_TYPE);
        }
    }

    let workspace_config = extract_workspace_config(token_response.clone())?;

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

async fn verify_oauth_state(state_param: &str, app_state: &Arc<AppState>) -> bool {
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

    matches!(result, Ok(Some(_)))
}

async fn exchange_oauth_code(
    code: &str,
) -> Result<SlackOAuthTokenResponse, Box<dyn std::error::Error>> {
    let client_id = std::env::var("SLACK_CLIENT_ID").map_err(|_| "SLACK_CLIENT_ID not set")?;
    let client_secret = std::env::var("SLACK_CLIENT_SECRET").map_err(|_| "SLACK_CLIENT_SECRET not set")?;
    let redirect_uri = std::env::var("SLACK_REDIRECT_URI").unwrap_or_else(|_| {
        warn!("SLACK_REDIRECT_URI not set, using default");
        "https://api.interstice.com/webhooks/slack/oauth".to_string()
    });
    
    let params = [
        ("client_id", client_id.as_str()),
        ("client_secret", client_secret.as_str()),
        ("code", code),
        ("redirect_uri", redirect_uri.as_str()),
    ];

    info!("Exchanging OAuth code for workspace installation");

    let client = Client::new();
    let response = client
        .post(SLACK_OAUTH_URL)
        .form(&params)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| format!("OAuth request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("OAuth request failed with status {}: {}", status, body).into());
    }

    let oauth_response: SlackOAuthTokenResponse = response.json().await.map_err(|e| format!("Failed to parse OAuth response: {}", e))?;
    
    if !oauth_response.ok {
        return Err(format!("Slack OAuth failed: {:?}", oauth_response.error).into());
    }

    if let Some(team) = &oauth_response.team {
        info!("OAuth code exchanged successfully for team: {}", team.name);
    } else if let Some(enterprise) = &oauth_response.enterprise {
        info!("OAuth code exchanged successfully for enterprise: {}", enterprise.name);
    }

    Ok(oauth_response)
}

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
        Ok(deleted) if !deleted.is_empty() => info!("Cleaned up {} expired OAuth states", deleted.len()),
        Err(e) => error!("Failed to cleanup expired OAuth states: {}", e),
        _ => (),
    }
}

pub async fn get_oauth_url(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let client_id = std::env::var("SLACK_CLIENT_ID").map_err(|_| {
        error!("SLACK_CLIENT_ID not configured");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let redirect_uri = std::env::var("SLACK_REDIRECT_URI").unwrap_or_else(|_| "https://api.interstice.com/webhooks/slack/oauth".to_string());
    
    let state_param = Uuid::new_v4().to_string();
    
    sqlx::query!(
        r#"
        INSERT INTO oauth_states (state, created_at, expires_at)
        VALUES ($1, NOW(), NOW() + INTERVAL '10 minutes')
        "#,
        &state_param
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to store OAuth state: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let scopes = [
        "app_mentions:read", "channels:history", "channels:read", "chat:write", "commands", 
        "groups:history", "groups:read", "im:history", "im:read", "mpim:history", "mpim:read", 
        "reactions:read", "team:read", "users:read"
    ].join(",");
    
    let oauth_url = format!(
        "https://slack.com/oauth/v2/authorize?client_id={}&scope={}&redirect_uri={}&state={}",
        encode(&client_id),
        encode(&scopes),
        encode(&redirect_uri),
        encode(&state_param)
    );
    
    info!("Generated OAuth URL for Slack installation");
    
    Ok(Json(serde_json::json!({
        "oauth_url": oauth_url,
        "state": state_param,
        "expires_in": 600
    })))
}