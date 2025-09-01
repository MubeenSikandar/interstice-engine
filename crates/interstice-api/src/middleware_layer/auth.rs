// interstice-api/src/middleware_layer/auth.rs

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
use tracing::{info, warn, error};
use crate::AppState;

// JWT Configuration
const JWT_SECRET_ENV: &str = "JWT_SECRET";
const JWT_ISSUER: &str = "interstice-api";
const JWT_AUDIENCE: &str = "interstice-client";
const ACCESS_TOKEN_DURATION_MINS: i64 = 15;
const REFRESH_TOKEN_DURATION_DAYS: i64 = 30;
const API_KEY_HEADER: &str = "X-API-Key";
const REQUEST_ID_HEADER: &str = "X-Request-Id";

// Token types
#[derive(Debug, Serialize, Deserialize)]
pub enum TokenType {
    Access,
    Refresh,
    ApiKey,
}

// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,           // Subject (user ID)
    pub exp: i64,              // Expiration time
    pub iat: i64,              // Issued at
    pub nbf: i64,              // Not before
    pub iss: String,           // Issuer
    pub aud: String,           // Audience
    pub jti: String,           // JWT ID (unique identifier)
    pub token_type: TokenType,
    pub workspace_id: Option<Uuid>,
    pub scopes: Vec<String>,
    pub email: Option<String>,
    pub roles: Vec<String>,
}

// API Key structure
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ApiKey {
    pub id: Uuid,
    pub key_hash: String,
    pub name: String,
    pub workspace_id: Uuid,
    pub scopes: Vec<String>,
    pub rate_limit: Option<u32>,
    pub expires_at: Option<DateTime<Utc>>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub revoked: bool,
}

// Authentication context passed to handlers
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub workspace_id: Option<Uuid>,
    pub scopes: Vec<String>,
    pub roles: Vec<String>,
    pub auth_method: AuthMethod,
    pub request_id: String,
}

#[derive(Debug, Clone)]
pub enum AuthMethod {
    Jwt,
    ApiKey,
    ServiceAccount,
}

// Error response structure
#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
    pub request_id: String,
}

/// Main authentication middleware for protected routes
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {  // Changed from Result<Response, Response>
    let request_id = extract_or_generate_request_id(&request);
    
    // Try different authentication methods in order
    let auth_context = match authenticate_request(&state, &request, &request_id).await {
        Ok(context) => context,
        Err(_) => return Err(StatusCode::UNAUTHORIZED),  // Changed to return StatusCode
    };

    // Log successful authentication
    info!(
        request_id = %request_id,
        user_id = %auth_context.user_id,
        auth_method = ?auth_context.auth_method,
        "Request authenticated successfully"
    );

    // Add auth context to request extensions
    request.extensions_mut().insert(auth_context.clone());
    
    // Track API usage
    track_api_usage(&state, &auth_context).await;

    // Continue with the request
    Ok(next.run(request).await)
}


/// Authenticate the request using various methods
async fn authenticate_request(
    state: &Arc<AppState>,
    request: &Request,
    request_id: &str,
) -> Result<AuthContext, StatusCode> {  // Changed from Result<AuthContext, Response>
    // Try JWT authentication first
    if let Some(auth_header) = request.headers().get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                let token = &auth_str[7..];
                return authenticate_jwt(state, token, request_id).await;
            }
        }
    }

    // Try API key authentication
    if let Some(api_key_header) = request.headers().get(API_KEY_HEADER) {
        if let Ok(api_key) = api_key_header.to_str() {
            return authenticate_api_key(state, api_key, request_id).await;
        }
    }

    // Try service account authentication (for internal services)
    if let Some(service_header) = request.headers().get("X-Service-Account") {
        if let Ok(service_token) = service_header.to_str() {
            return authenticate_service_account(state, service_token, request_id).await;
        }
    }

    // No valid authentication found
    Err(StatusCode::UNAUTHORIZED)
}


/// Authenticate using JWT
async fn authenticate_jwt(
    state: &Arc<AppState>,
    token: &str,
    request_id: &str,
) -> Result<AuthContext, StatusCode> {  // Changed from Result<AuthContext, Response>
    // Get JWT secret from environment
    let secret = std::env::var(JWT_SECRET_ENV)
        .map_err(|_| {
            error!("JWT_SECRET not configured");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Decode and validate JWT
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::new(jsonwebtoken::Algorithm::HS256),
    )
    .map_err(|e| {
        warn!("JWT validation failed: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    let claims = token_data.claims;

    // Check if token is revoked
    if is_token_revoked(state, &claims.jti).await {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Create auth context
    Ok(AuthContext {
        user_id: claims.sub,
        workspace_id: claims.workspace_id,
        scopes: claims.scopes,
        roles: claims.roles,
        auth_method: AuthMethod::Jwt,
        request_id: request_id.to_string(),
    })
}


/// Authenticate using API key
async fn authenticate_api_key(
    state: &Arc<AppState>,
    api_key: &str,
    request_id: &str,
) -> Result<AuthContext, StatusCode> {  // Changed from Result<AuthContext, Response>
    // Hash the API key for comparison
    let key_hash = hash_api_key(api_key);

    // Look up API key in database
    let key_data = sqlx::query!(
        r#"
        SELECT 
            id, key_hash, name, workspace_id, 
            scopes, rate_limit, expires_at, last_used_at, 
            created_at, revoked
        FROM api_keys
        WHERE key_hash = $1 AND revoked = false
        "#,
        key_hash
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        error!("Database error during API key lookup: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::UNAUTHORIZED)?;

    // Convert database row to ApiKey struct
    let key_data = ApiKey {
        id: key_data.id,
        key_hash: key_data.key_hash,
        name: key_data.name,
        workspace_id: key_data.workspace_id,
        scopes: key_data.scopes.unwrap_or_default(),
        rate_limit: key_data.rate_limit.map(|r| r as u32),
        expires_at: key_data.expires_at,
        last_used_at: key_data.last_used_at,
        created_at: key_data.created_at,
        revoked: key_data.revoked.unwrap_or(false),
    };

    // Check if key is expired
    if let Some(expires_at) = key_data.expires_at {
        if expires_at < Utc::now() {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    // Update last used timestamp
    let _ = sqlx::query!(
        "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
        key_data.id
    )
    .execute(&state.db)
    .await;

    // Create auth context
    Ok(AuthContext {
        user_id: key_data.id.to_string(),
        workspace_id: Some(key_data.workspace_id),
        scopes: key_data.scopes,
        roles: vec!["api_key".to_string()],
        auth_method: AuthMethod::ApiKey,
        request_id: request_id.to_string(),
    })
}

/// Authenticate service account (for internal services)
async fn authenticate_service_account(
    state: &Arc<AppState>,
    service_token: &str,
    request_id: &str,
) -> Result<AuthContext, StatusCode> {  // Changed from Result<AuthContext, Response>
    // Verify service token against configured services
    let service_secret = std::env::var("SERVICE_SECRET")
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    if service_token != service_secret {
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Create auth context for service account
    Ok(AuthContext {
        user_id: "service-account".to_string(),
        workspace_id: None,
        scopes: vec!["*".to_string()], // Full access for internal services
        roles: vec!["service".to_string()],
        auth_method: AuthMethod::ServiceAccount,
        request_id: request_id.to_string(),
    })
}

/// Generate a new JWT token
pub fn generate_jwt_token(
    user_id: &str,
    workspace_id: Option<Uuid>,
    scopes: Vec<String>,
    roles: Vec<String>,
    token_type: TokenType,
) -> Result<String, jsonwebtoken::errors::Error> {
    let secret = std::env::var(JWT_SECRET_ENV)
        .expect("JWT_SECRET must be set");

    let duration = match token_type {
        TokenType::Access => Duration::minutes(ACCESS_TOKEN_DURATION_MINS),
        TokenType::Refresh => Duration::days(REFRESH_TOKEN_DURATION_DAYS),
        _ => Duration::minutes(ACCESS_TOKEN_DURATION_MINS),
    };

    let now = Utc::now();
    let expiration = now + duration;

    let claims = Claims {
        sub: user_id.to_string(),
        exp: expiration.timestamp(),
        iat: now.timestamp(),
        nbf: now.timestamp(),
        iss: JWT_ISSUER.to_string(),
        aud: JWT_AUDIENCE.to_string(),
        jti: Uuid::new_v4().to_string(),
        token_type,
        workspace_id,
        scopes,
        email: None,
        roles,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
}

/// Generate a new API key
pub async fn generate_api_key(
    state: &Arc<AppState>,
    workspace_id: Uuid,
    name: String,
    scopes: Vec<String>,
    expires_in_days: Option<i64>,
) -> Result<(String, Uuid), StatusCode> {
    // Generate random API key
    let api_key = generate_random_key();
    let key_hash = hash_api_key(&api_key);
    let key_id = Uuid::new_v4();

    let expires_at = expires_in_days.map(|days| Utc::now() + Duration::days(days));

    // Store in database
    sqlx::query!(
        r#"
        INSERT INTO api_keys (
            id, key_hash, name, workspace_id, 
            scopes, expires_at, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        "#,
        key_id,
        key_hash,
        name,
        workspace_id,
        &scopes,
        expires_at
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to store API key: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok((api_key, key_id))
}

/// Revoke an API key
pub async fn revoke_api_key(
    state: &Arc<AppState>,
    key_id: Uuid,
) -> Result<(), StatusCode> {
    sqlx::query!(
        "UPDATE api_keys SET revoked = true WHERE id = $1",
        key_id
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to revoke API key: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(())
}

/// Check if a JWT token is revoked
async fn is_token_revoked(state: &Arc<AppState>, jti: &str) -> bool {
    sqlx::query!(
        "SELECT 1 as exists FROM revoked_tokens WHERE jti = $1",
        jti
    )
    .fetch_optional(&state.db)
    .await
    .ok()
    .flatten()
    .is_some()
}

/// Track API usage for rate limiting and analytics
async fn track_api_usage(state: &Arc<AppState>, auth_context: &AuthContext) {
    let _ = sqlx::query!(
        r#"
        INSERT INTO api_usage (
            user_id, workspace_id, auth_method, request_id, created_at
        )
        VALUES ($1, $2, $3, $4, NOW())
        "#,
        auth_context.user_id,
        auth_context.workspace_id,
        format!("{:?}", auth_context.auth_method),
        auth_context.request_id
    )
    .execute(&state.db)
    .await;
}

/// Generate a random API key
fn generate_random_key() -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    
    let key: String = (0..32)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    
    format!("ik_{}", key) // Prefix for easy identification
}

/// Hash an API key for storage
fn hash_api_key(key: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract or generate request ID
fn extract_or_generate_request_id(request: &Request) -> String {
    request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

/// Middleware for requiring specific scopes
pub async fn require_scope(
    scope: &'static str,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Check if user has required scope
    if !auth_context.scopes.contains(&scope.to_string()) 
        && !auth_context.scopes.contains(&"*".to_string()) {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

/// Middleware for requiring specific roles
pub async fn require_role(
    role: &'static str,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Check if user has required role
    if !auth_context.roles.contains(&role.to_string()) {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

/// Middleware for requiring workspace membership
pub async fn require_workspace(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_context = request
        .extensions()
        .get::<AuthContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Service accounts bypass workspace checks
    if matches!(auth_context.auth_method, AuthMethod::ServiceAccount) {
        return Ok(next.run(request).await);
    }

    // Ensure user has a workspace
    if auth_context.workspace_id.is_none() {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}