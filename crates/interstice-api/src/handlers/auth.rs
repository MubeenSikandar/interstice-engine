// interstice-api/src/handlers/auth.rs

use axum::{extract::{Path, Query, State}, http::StatusCode, Extension, Json};
use chrono::NaiveDateTime;
use interstice_core::types::PaginationParams;
use serde::{Deserialize, Serialize};
use crate::middleware_layer::auth::{generate_api_key, generate_jwt_token, revoke_api_key, AuthContext, TokenType};
use std::sync::Arc;
use crate::AppState;
use uuid::Uuid;
use bcrypt::{hash, verify, DEFAULT_COST};

#[derive(Deserialize)]
pub struct LoginRequest {
    email: String,
    password: String,
}

#[derive(Serialize)]
pub struct LoginResponse {
    access_token: String,
    refresh_token: String,
    token_type: String,
    expires_in: i64,
    user: UserInfo,
}

#[derive(Serialize)]
pub struct UserInfo {
    id: String,
    email: String,
    workspace_id: Option<Uuid>,
    roles: Vec<String>,
}

pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, StatusCode> {
    // Fetch user from database
    let user = sqlx::query!(
        r#"
        SELECT id, email, password_hash, workspace_id, roles
        FROM users
        WHERE email = $1
        "#,
        payload.email
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    .ok_or(StatusCode::UNAUTHORIZED)?;
    
    // Verify password
    verify(&payload.password, &user.password_hash)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    // Generate tokens
    let access_token = generate_jwt_token(
        &user.id.to_string(),
        user.workspace_id,
        vec!["artifacts:read".to_string(), "artifacts:write".to_string()],
        user.roles.clone().unwrap_or_default(),
        TokenType::Access,
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let refresh_token = generate_jwt_token(
        &user.id.to_string(),
        user.workspace_id,
        vec!["refresh".to_string()],
        user.roles.clone().unwrap_or_default(),
        TokenType::Refresh,
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        token_type: "Bearer".to_string(),
        expires_in: 900, // 15 minutes
        user: UserInfo {
            id: user.id.to_string(),
            email: user.email,
            workspace_id: user.workspace_id,
            roles: user.roles.unwrap_or_default(),
        },
    }))
}

#[derive(Deserialize)]
pub struct RegisterRequest {
    email: String,
    password: String,
    workspace_name: Option<String>,
}

pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RegisterRequest>,
) -> Result<Json<LoginResponse>, StatusCode> {
    // Hash password
    let password_hash = hash(&payload.password, DEFAULT_COST)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Create workspace if needed
    let workspace_id = if let Some(workspace_name) = payload.workspace_name {
        let workspace_id = Uuid::new_v4();
        sqlx::query!(
            "INSERT INTO workspaces (id, name, created_at) VALUES ($1, $2, NOW())",
            workspace_id,
            workspace_name
        )
        .execute(&state.db)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        Some(workspace_id)
    } else {
        None
    };
    
    // Create user
    let user_id = Uuid::new_v4();
    sqlx::query!(
        r#"
        INSERT INTO users (id, email, password_hash, workspace_id, roles, created_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        user_id,
        payload.email,
        password_hash,
        workspace_id,
        &vec!["user".to_string()]
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        if e.to_string().contains("duplicate") {
            StatusCode::CONFLICT
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    
    // Generate tokens
    let access_token = generate_jwt_token(
        &user_id.to_string(),
        workspace_id,
        vec!["artifacts:read".to_string(), "artifacts:write".to_string()],
        vec!["user".to_string()],
        TokenType::Access,
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let refresh_token = generate_jwt_token(
        &user_id.to_string(),
        workspace_id,
        vec!["refresh".to_string()],
        vec!["user".to_string()],
        TokenType::Refresh,
    ).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        token_type: "Bearer".to_string(),
        expires_in: 900,
        user: UserInfo {
            id: user_id.to_string(),
            email: payload.email,
            workspace_id,
            roles: vec!["user".to_string()],
        },
    }))
}

#[derive(Deserialize)]
pub struct CreateApiKeyRequest {
    name: String,
    scopes: Vec<String>,
    expires_in_days: Option<i64>,
}

#[derive(Serialize)]
pub struct CreateApiKeyResponse {
    api_key: String,
    key_id: Uuid,
    name: String,
    expires_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub async fn create_api_key(
    State(state): State<Arc<AppState>>,
    auth: Extension<AuthContext>,
    Json(payload): Json<CreateApiKeyRequest>,
) -> Result<Json<CreateApiKeyResponse>, StatusCode> {
    let workspace_id = auth.workspace_id.ok_or(StatusCode::BAD_REQUEST)?;
    
    let (api_key, key_id) = generate_api_key(
        &state,
        workspace_id,
        payload.name.clone(),
        payload.scopes,
        payload.expires_in_days,
    ).await?;
    
    let expires_at = payload.expires_in_days
        .map(|days| chrono::Utc::now() + chrono::Duration::days(days));
    
    Ok(Json(CreateApiKeyResponse {
        api_key,
        key_id,
        name: payload.name,
        expires_at,
    }))
}

pub async fn revoke_key(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<Uuid>,
) -> Result<StatusCode, StatusCode> {
    revoke_api_key(&state, key_id).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn list_users(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<UserResponse>>, StatusCode> {
    let offset = ((params.page - 1) * params.per_page) as i64;
    let limit = params.per_page as i64;
    
    let users = sqlx::query!(
        r#"
        SELECT id, email, workspace_id, roles, created_at
        FROM users
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        "#,
        limit,
        offset
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list users: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let response: Vec<UserResponse> = users
        .into_iter()
        .map(|u| UserResponse {
            id: u.id,
            email: u.email,
            workspace_id: u.workspace_id,
            roles: u.roles.unwrap_or_default(),
            created_at: u.created_at,
        })
        .collect();
    
    Ok(Json(response))
}

// Add UserResponse type
#[derive(Serialize)]
pub struct UserResponse {
    id: Uuid,
    email: String,
    workspace_id: Option<Uuid>,
    roles: Vec<String>,
    created_at: NaiveDateTime,
}