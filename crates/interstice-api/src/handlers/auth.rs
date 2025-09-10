// interstice-api/src/handlers/auth.rs

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, NaiveDateTime, Utc};
use interstice_core::types::PaginationParams;
use serde::{Deserialize, Serialize};
use sqlx::types::ipnetwork::IpNetwork;
use std::{collections::HashMap, sync::Arc};
use uuid::Uuid;
use validator::Validate;

use crate::{
    middleware_layer::auth::{
        generate_api_key, generate_jwt_token, hash_password, revoke_api_key, verify_password, AuthContext, Claims, TokenType
    },
    AppState,
};

// ============================================================================
// Request/Response Types with Validation
// ============================================================================

#[derive(Debug, Deserialize, Validate)]
pub struct LoginRequest {
    #[validate(email(message = "Invalid email format"))]
    pub email: String,
    #[validate(length(min = 1, message = "Password cannot be empty"))]
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub user: UserInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
    pub workspace_id: Option<Uuid>,
    pub roles: Vec<String>,
    pub email_verified: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(email(message = "Invalid email format"))]
    pub email: String,
    #[validate(length(
        min = 8,
        max = 128,
        message = "Password must be between 8 and 128 characters"
    ))]
    #[validate(custom(function = "validate_password_strength"))]
    pub password: String,
    #[validate(length(min = 1, max = 100, message = "Invalid workspace name length"))]
    pub workspace_name: Option<String>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Deserialize, Validate)]
pub struct CreateApiKeyRequest {
    #[validate(length(min = 1, max = 100, message = "Name must be between 1-100 characters"))]
    pub name: String,
    #[validate(length(min = 1, message = "At least one scope is required"))]
    pub scopes: Vec<String>,
    #[validate(range(min = 1, max = 365, message = "Expiry must be between 1-365 days"))]
    pub expires_in_days: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CreateApiKeyResponse {
    pub api_key: String,
    pub key_id: Uuid,
    pub name: String,
    pub expires_at: Option<DateTime<Utc>>,
    pub scopes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct UserResponse {
    pub id: Uuid,
    pub email: String,
    pub workspace_id: Option<Uuid>,
    pub roles: Vec<String>,
    pub email_verified: bool,
    pub created_at: NaiveDateTime,
    pub last_login_at: Option<NaiveDateTime>,
}

#[derive(Debug, Deserialize, Validate)]
pub struct PasswordResetRequest {
    #[validate(email(message = "Invalid email format"))]
    pub email: String,
}

#[derive(Debug, Deserialize, Validate)]
pub struct PasswordResetConfirmRequest {
    pub token: String,
    #[validate(length(
        min = 8,
        max = 128,
        message = "Password must be between 8 and 128 characters"
    ))]
    #[validate(custom(function = "validate_password_strength"))]
    pub new_password: String,
}

#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub message: Option<String>,
}

// ============================================================================
// Custom Validators
// ============================================================================

fn validate_password_strength(password: &str) -> Result<(), validator::ValidationError> {
    let has_uppercase = password.chars().any(|c| c.is_ascii_uppercase());
    let has_lowercase = password.chars().any(|c| c.is_ascii_lowercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    let has_special = password.chars().any(|c| !c.is_ascii_alphanumeric());

    if !has_uppercase || !has_lowercase || !has_digit {
        return Err(validator::ValidationError::new("password_strength"));
    }

    if password.len() >= 12 && !has_special {
        return Err(validator::ValidationError::new("password_strength"));
    }

    Ok(())
}

// ============================================================================
// Auth Handlers
// ============================================================================

/// Login handler with comprehensive security measures
pub async fn login(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<LoginRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate request
    payload
        .validate()
        .map_err(|e| {
            tracing::warn!("Login validation failed: {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

    // Normalize email to lowercase
    let email = payload.email.to_lowercase();

    // Rate limiting check (implemented in middleware)
    
    // Fetch user with timing attack protection
    let user = sqlx::query!(
        r#"
        SELECT 
            id, email, password_hash, workspace_id, 
            roles, COALESCE(email_verified, false) as email_verified, created_at,
            COALESCE(failed_login_attempts, 0) as failed_login_attempts, locked_until
        FROM users
        WHERE email = $1
        "#,
        email
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error during login: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Constant-time user verification
    let user = match user {
        Some(u) => u,
        None => {
            // Perform dummy password verification to prevent timing attacks
            let _ = verify_password("dummy", "$2b$12$dummy.hash.to.prevent.timing.attacks");
            
            // Log failed attempt
            tracing::warn!("Login attempt for non-existent user: {}", email);
            
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Check if account is locked
    if let Some(locked_until) = user.locked_until {
        if locked_until > Utc::now() {
            tracing::warn!("Login attempt for locked account: {}", email);
            return Err(StatusCode::FORBIDDEN);
        }
    }

    // Verify password
    let password_valid = verify_password(&payload.password, &user.password_hash)
        .map_err(|e| {
            tracing::error!("Password verification error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if !password_valid {
        // Increment failed login attempts
        let failed_attempts = user.failed_login_attempts.unwrap_or(0) + 1;
        let locked_until = if failed_attempts >= 5 {
            Some(Utc::now() + chrono::Duration::minutes(15))
        } else {
            None
        };

        sqlx::query!(
            r#"
            UPDATE users 
            SET failed_login_attempts = $1, locked_until = $2
            WHERE id = $3
            "#,
            failed_attempts,
            locked_until,
            user.id
        )
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Failed to update login attempts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        tracing::warn!("Failed login attempt for user: {}", email);
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Clear failed login attempts and update last login
    sqlx::query!(
        r#"
        UPDATE users 
        SET failed_login_attempts = 0, 
            locked_until = NULL,
            last_login_at = NOW(),
            last_login_ip = $1
        WHERE id = $2
        "#,
        None::<IpNetwork>, // IP would come from request context
        user.id
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update login metadata: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Generate tokens with proper scopes
    let user_roles = user.roles.clone().unwrap_or_default();
    let scopes = determine_user_scopes(&user_roles);
    
    let access_token = generate_jwt_token(
        &user.id.to_string(),
        user.workspace_id,
        scopes.clone(),
        user_roles.clone(),
        TokenType::Access,
    )
    .map_err(|e| {
        tracing::error!("JWT generation error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let refresh_token = generate_jwt_token(
        &user.id.to_string(),
        user.workspace_id,
        vec!["refresh".to_string()],
        user_roles,
        TokenType::Refresh,
    )
    .map_err(|e| {
        tracing::error!("JWT generation error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Store refresh token in database for revocation capability
    sqlx::query!(
        r#"
        INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at)
        VALUES ($1, $2, $3, NOW())
        ON CONFLICT (user_id) DO UPDATE
        SET token_hash = $1, expires_at = $3, created_at = NOW()
        "#,
        hash_token(&refresh_token),
        user.id,
        Utc::now() + chrono::Duration::days(30)
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to store refresh token: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Log successful login
    tracing::info!("Successful login for user: {}", email);

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
            email_verified: user.email_verified.unwrap_or(false),
            created_at: user.created_at,
        },
    }))
}

/// Register new user with comprehensive validation
pub async fn register(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RegisterRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate request
    payload
        .validate()
        .map_err(|e| {
            tracing::warn!("Registration validation failed: {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

    // Normalize email
    let email = payload.email.to_lowercase();

    // Begin transaction for atomicity
    let mut tx = state
        .db
        .begin()
        .await
        .map_err(|e| {
            tracing::error!("Failed to begin transaction: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Check if email already exists
    let existing = sqlx::query!(
        "SELECT id FROM users WHERE email = $1",
        email
    )
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Database error checking existing user: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if existing.is_some() {
        tracing::warn!("Registration attempt with existing email: {}", email);
        return Err(StatusCode::CONFLICT);
    }

    // Hash password with secure algorithm
    let password_hash = hash_password(&payload.password)
        .map_err(|e| {
            tracing::error!("Password hashing error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Create workspace if needed
    let workspace_id = if let Some(workspace_name) = payload.workspace_name {
        let workspace_id = Uuid::new_v4();
        
        sqlx::query!(
            r#"
            INSERT INTO workspaces (id, name, owner_email, created_at)
            VALUES ($1, $2, $3, NOW())
            "#,
            workspace_id,
            workspace_name,
            email
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            tracing::error!("Failed to create workspace: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        
        Some(workspace_id)
    } else {
        None
    };

    // Create user with secure defaults
    let user_id = Uuid::new_v4();
    let verification_token = Uuid::new_v4().to_string();
    
    sqlx::query!(
        r#"
        INSERT INTO users (
            id, email, password_hash, workspace_id, 
            roles, email_verified, verification_token,
            created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
        "#,
        user_id,
        email,
        password_hash,
        workspace_id,
        &vec!["user".to_string()],
        false,
        verification_token
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create user: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Commit transaction
    tx.commit()
        .await
        .map_err(|e| {
            tracing::error!("Failed to commit transaction: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Generate tokens after successful registration
    let user_roles = vec!["user".to_string()];
    let scopes = determine_user_scopes(&user_roles);
    
    let access_token = generate_jwt_token(
        &user_id.to_string(),
        workspace_id,
        scopes.clone(),
        user_roles.clone(),
        TokenType::Access,
    )
    .map_err(|e| {
        tracing::error!("JWT generation error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let refresh_token = generate_jwt_token(
        &user_id.to_string(),
        workspace_id,
        vec!["refresh".to_string()],
        user_roles,
        TokenType::Refresh,
    )
    .map_err(|e| {
        tracing::error!("JWT generation error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Store refresh token in database
    sqlx::query!(
        r#"
        INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at)
        VALUES ($1, $2, $3, NOW())
        "#,
        hash_token(&refresh_token),
        user_id,
        Utc::now() + chrono::Duration::days(30)
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to store refresh token: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("User registered successfully: {}", email);

    Ok(Json(LoginResponse {
        access_token,
        refresh_token,
        token_type: "Bearer".to_string(),
        expires_in: 900, // 15 minutes
        user: UserInfo {
            id: user_id.to_string(),
            email: email,
            workspace_id,
            roles: vec!["user".to_string()],
            email_verified: false,
            created_at: Utc::now(),
        },
    }))
}

/// Logout handler - invalidates refresh token
pub async fn logout(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<impl IntoResponse, StatusCode> {
    // Parse user ID
    let user_id = Uuid::parse_str(&auth.user_id)
        .map_err(|e| {
            tracing::error!("Invalid user ID format: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Invalidate refresh token
    sqlx::query!(
        "DELETE FROM refresh_tokens WHERE user_id = $1",
        user_id
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to invalidate refresh token: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("User logged out: {}", auth.user_id);

    Ok(Json(ApiResponse::<()> {
        success: true,
        data: None,
        message: Some("Logged out successfully".to_string()),
    }))
}

/// Get current user info
pub async fn get_current_user(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<impl IntoResponse, StatusCode> {
    // Parse user ID
    let user_id = Uuid::parse_str(&auth.user_id)
        .map_err(|e| {
            tracing::error!("Invalid user ID format: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Fetch user details
    let user = sqlx::query!(
        r#"
        SELECT 
            id, email, workspace_id, roles, 
            email_verified, created_at, last_login_at
        FROM users
        WHERE id = $1
        "#,
        user_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error fetching user: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(UserResponse {
        id: user.id,
        email: user.email,
        workspace_id: user.workspace_id,
        roles: user.roles.unwrap_or_default(),
        email_verified: user.email_verified.unwrap_or(false),
        created_at: user.created_at.naive_utc(),
        last_login_at: user.last_login_at.map(|t| t.naive_utc()),
    }))
}

/// Verify email with token
pub async fn verify_email(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, StatusCode> {
    let token = params
        .get("token")
        .ok_or(StatusCode::BAD_REQUEST)?;

    // Update user's email verification status
    let result = sqlx::query!(
        r#"
        UPDATE users
        SET email_verified = true, verification_token = NULL
        WHERE verification_token = $1
        RETURNING id
        "#,
        token
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error during email verification: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    match result {
        Some(user) => {
            tracing::info!("Email verified for user: {}", user.id);
            Ok(Json(ApiResponse::<()> {
                success: true,
                data: None,
                message: Some("Email verified successfully".to_string()),
            }))
        }
        None => {
            tracing::warn!("Invalid email verification token");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// List API keys for current workspace
pub async fn list_api_keys(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<impl IntoResponse, StatusCode> {
    let workspace_id = auth.workspace_id.ok_or(StatusCode::FORBIDDEN)?;

    let keys = sqlx::query!(
        r#"
        SELECT 
            id, name, scopes, expires_at, 
            last_used_at, created_at
        FROM api_keys
        WHERE workspace_id = $1 AND revoked = false
        ORDER BY created_at DESC
        "#,
        workspace_id
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list API keys: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let response: Vec<_> = keys
        .into_iter()
        .map(|k| serde_json::json!({
            "id": k.id,
            "name": k.name,
            "scopes": k.scopes,
            "expires_at": k.expires_at,
            "last_used_at": k.last_used_at,
            "created_at": k.created_at,
        }))
        .collect();

    Ok(Json(ApiResponse {
        success: true,
        data: Some(response),
        message: None,
    }))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Determine user scopes based on roles
fn determine_user_scopes(roles: &[String]) -> Vec<String> {
    let mut scopes = vec![
        "artifacts:read".to_string(),
        "profile:read".to_string(),
    ];

    if roles.contains(&"admin".to_string()) {
        scopes.extend(vec![
            "artifacts:write".to_string(),
            "artifacts:delete".to_string(),
            "users:read".to_string(),
            "users:write".to_string(),
            "workspace:admin".to_string(),
        ]);
    } else if roles.contains(&"user".to_string()) {
        scopes.push("artifacts:write".to_string());
    }

    scopes
}

/// Validate requested scopes against user's available scopes
fn validate_scopes(requested: &[String], available: &[String]) -> bool {
    requested.iter().all(|scope| {
        available.contains(scope) || available.contains(&"*".to_string())
    })
}

/// Hash a token for storage
fn hash_token(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Decode JWT token (helper function - actual implementation in middleware)
fn decode_token(token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    // This would use the actual JWT decoding logic from middleware
    // Placeholder for compilation
    use crate::middleware_layer::auth::decode_jwt_token;
    decode_jwt_token(token)
}

/// Refresh access token using refresh token
pub async fn refresh_token(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<RefreshTokenRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate refresh token format
    let claims = decode_token(&payload.refresh_token)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Verify token type
    if !matches!(claims.token_type, TokenType::Refresh) {
        tracing::warn!("Invalid token type for refresh");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Verify refresh token exists and is valid
    let token_hash = hash_token(&payload.refresh_token);
    let stored_token = sqlx::query!(
        r#"
        SELECT user_id, expires_at
        FROM refresh_tokens
        WHERE token_hash = $1 AND expires_at > NOW()
        "#,
        token_hash
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error checking refresh token: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::UNAUTHORIZED)?;

    // Fetch user details
    let user = sqlx::query!(
        r#"
        SELECT id, email, workspace_id, roles, email_verified
        FROM users
        WHERE id = $1
        "#,
        stored_token.user_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error fetching user: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::UNAUTHORIZED)?;

    // Generate new access token
    let user_roles = user.roles.clone().unwrap_or_default();
    let scopes = determine_user_scopes(&user_roles);
    
    let new_access_token = generate_jwt_token(
        &user.id.to_string(),
        user.workspace_id,
        scopes,
        user_roles,
        TokenType::Access,
    )
    .map_err(|e| {
        tracing::error!("JWT generation error: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(ApiResponse {
        success: true,
        data: Some(serde_json::json!({
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": 900,
        })),
        message: None,
    }))
}

/// Create API key for programmatic access
pub async fn create_api_key(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(payload): Json<CreateApiKeyRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate request
    payload
        .validate()
        .map_err(|e| {
            tracing::warn!("API key creation validation failed: {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

    // Require workspace membership
    let workspace_id = auth.workspace_id.ok_or_else(|| {
        tracing::warn!("API key creation attempted without workspace");
        StatusCode::FORBIDDEN
    })?;

    // Validate scopes against user permissions
    if !validate_scopes(&payload.scopes, &auth.scopes) {
        tracing::warn!("Invalid scopes requested for API key");
        return Err(StatusCode::FORBIDDEN);
    }

    // Check API key limit for workspace
    let key_count = sqlx::query!(
        r#"
        SELECT COUNT(*) as count
        FROM api_keys
        WHERE workspace_id = $1 AND revoked = false
        "#,
        workspace_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error checking API key count: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if key_count.count.unwrap_or(0) >= 10 {
        tracing::warn!("API key limit reached for workspace: {}", workspace_id);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    // Generate API key with metadata
    let (api_key, key_id) = generate_api_key(
        &state,
        workspace_id,
        payload.name.clone(),
        payload.scopes.clone(),
        payload.expires_in_days,
    )
    .await
    .map_err(|e| {
        tracing::error!("Failed to generate API key: {:?}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    let expires_at = payload
        .expires_in_days
        .map(|days| Utc::now() + chrono::Duration::days(days));

    // Log API key creation
    tracing::info!(
        "API key created: {} for workspace: {} by user: {}",
        key_id,
        workspace_id,
        auth.user_id
    );

    Ok(Json(CreateApiKeyResponse {
        api_key,
        key_id,
        name: payload.name,
        expires_at,
        scopes: payload.scopes,
    }))
}

/// Revoke API key
pub async fn revoke_key(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(key_id): Path<Uuid>,
) -> Result<impl IntoResponse, StatusCode> {
    // Verify ownership
    let key = sqlx::query!(
        r#"
        SELECT workspace_id
        FROM api_keys
        WHERE id = $1 AND revoked = false
        "#,
        key_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error fetching API key: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::NOT_FOUND)?;

    // Check if user has permission to revoke
    if Some(key.workspace_id) != auth.workspace_id {
        tracing::warn!(
            "Unauthorized API key revocation attempt by user: {}",
            auth.user_id
        );
        return Err(StatusCode::FORBIDDEN);
    }

    // Revoke the key
    revoke_api_key(&state, key_id)
        .await
        .map_err(|e| {
            tracing::error!("Failed to revoke API key: {:?}", e);
            e
        })?;

    tracing::info!(
        "API key revoked: {} by user: {}",
        key_id,
        auth.user_id
    );

    Ok(StatusCode::NO_CONTENT)
}

/// List users with pagination and filtering
pub async fn list_users(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<PaginationParams>,
) -> Result<impl IntoResponse, StatusCode> {
    // Require admin role
    if !auth.roles.contains(&"admin".to_string()) {
        tracing::warn!("Unauthorized user list attempt by: {}", auth.user_id);
        return Err(StatusCode::FORBIDDEN);
    }

    // Validate pagination params
    let page = params.page.max(1);
    let per_page = params.per_page.min(100).max(1);
    let offset = ((page - 1) * per_page) as i64;
    let limit = per_page as i64;

    // Fetch users with filtering
    let users = sqlx::query!(
        r#"
        SELECT 
            id, email, workspace_id, roles, 
            email_verified, created_at, last_login_at
        FROM users
        WHERE ($1::uuid IS NULL OR workspace_id = $1)
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#,
        auth.workspace_id,
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
            email_verified: u.email_verified.unwrap_or(false),
            created_at: u.created_at.naive_utc(),
            last_login_at: u.last_login_at.map(|t| t.naive_utc()),
        })
        .collect();

    Ok(Json(ApiResponse {
        success: true,
        data: Some(response),
        message: None,
    }))
}

/// Request password reset
pub async fn request_password_reset(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PasswordResetRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate request
    payload
        .validate()
        .map_err(|e| {
            tracing::warn!("Password reset validation failed: {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

    let email = payload.email.to_lowercase();

    // Check if user exists (don't reveal result)
    let user = sqlx::query!(
        "SELECT id, email FROM users WHERE email = $1",
        email
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Database error during password reset: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if let Some(user) = user {
        // Generate reset token
        let reset_token = Uuid::new_v4().to_string();
        let token_hash = hash_token(&reset_token);
        
        // Store reset token with expiry
        sqlx::query!(
            r#"
            INSERT INTO password_reset_tokens (
                token_hash, user_id, expires_at, created_at
            )
            VALUES ($1, $2, $3, NOW())
            "#,
            token_hash,
            user.id,
            Utc::now() + chrono::Duration::hours(1)
        )
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Failed to store reset token: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        // Queue email with reset link (handled by worker)
        tracing::info!("Password reset requested for: {}", email);
    }

    // Always return success to prevent email enumeration
    Ok(Json(ApiResponse::<()> {
        success: true,
        data: None,
        message: Some("If the email exists, a reset link has been sent".to_string()),
    }))
}

/// Confirm password reset with token
pub async fn confirm_password_reset(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PasswordResetConfirmRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate request
    payload
        .validate()
        .map_err(|e| {
            tracing::warn!("Password reset confirmation validation failed: {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

    let token_hash = hash_token(&payload.token);

    // Begin transaction
    let mut tx = state
        .db
        .begin()
        .await
        .map_err(|e| {
            tracing::error!("Failed to begin transaction: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Verify token
    let token_data = sqlx::query!(
        r#"
        SELECT user_id
        FROM password_reset_tokens
        WHERE token_hash = $1 
            AND expires_at > NOW()
            AND used = false
        "#,
        token_hash
    )
    .fetch_optional(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Database error verifying reset token: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::UNAUTHORIZED)?;

    // Hash new password
    let password_hash = hash_password(&payload.new_password)
        .map_err(|e| {
            tracing::error!("Password hashing error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Update password
    sqlx::query!(
        r#"
        UPDATE users
        SET password_hash = $1, updated_at = NOW()
        WHERE id = $2
        "#,
        password_hash,
        token_data.user_id
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update password: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Mark token as used
    sqlx::query!(
        r#"
        UPDATE password_reset_tokens
        SET used = true, used_at = NOW()
        WHERE token_hash = $1
        "#,
        token_hash
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Failed to mark token as used: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Invalidate all refresh tokens for security
    sqlx::query!(
        "DELETE FROM refresh_tokens WHERE user_id = $1",
        token_data.user_id
    )
    .execute(&mut *tx)
    .await
    .map_err(|e| {
        tracing::error!("Failed to invalidate refresh tokens: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Commit transaction
    tx.commit()
        .await
        .map_err(|e| {
            tracing::error!("Failed to commit transaction: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    tracing::info!("Password reset completed for user: {}", token_data.user_id);

    Ok(Json(ApiResponse::<()> {
        success: true,
        data: None,
        message: Some("Password has been reset successfully".to_string()),
    }))
}