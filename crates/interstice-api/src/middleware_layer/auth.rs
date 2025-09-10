// interstice-api/src/middleware_layer/auth.rs

use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use sqlx::types::ipnetwork::IpNetwork;
use std::{collections::HashMap};
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::AppState;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_issuer: String,
    pub jwt_audience: String,
    pub access_token_duration_mins: i64,
    pub refresh_token_duration_days: i64,
    pub api_key_header: String,
    pub request_id_header: String,
    pub service_secret: Option<String>,
    pub max_login_attempts: i32,
    pub lockout_duration_mins: i64,
}

impl AuthConfig {
   pub fn from_env() -> Self {
        Self {
            jwt_secret: env::var("JWT_SECRET")
                .expect("JWT_SECRET must be set"),
            jwt_issuer: env::var("JWT_ISSUER")
                .unwrap_or_else(|_| "interstice-api".to_string()),
            jwt_audience: env::var("JWT_AUDIENCE")
                .unwrap_or_else(|_| "interstice-client".to_string()),
            access_token_duration_mins: env::var("ACCESS_TOKEN_DURATION_MINS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(15),
            refresh_token_duration_days: env::var("REFRESH_TOKEN_DURATION_DAYS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30),
            api_key_header: env::var("API_KEY_HEADER")
                .unwrap_or_else(|_| "X-API-Key".to_string()),
            request_id_header: env::var("REQUEST_ID_HEADER")
                .unwrap_or_else(|_| "X-Request-Id".to_string()),
            service_secret: env::var("SERVICE_SECRET").ok(),
            max_login_attempts: env::var("MAX_LOGIN_ATTEMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5),
            lockout_duration_mins: env::var("LOCKOUT_DURATION_MINS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(15),
        }
    }
}

pub static AUTH_CONFIG: Lazy<AuthConfig> = Lazy::new(AuthConfig::from_env);

// ============================================================================
// Core Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    Access,
    Refresh,
    ApiKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    Jwt,
    ApiKey,
    ServiceAccount,
}

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

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub workspace_id: Option<Uuid>,
    pub scopes: Vec<String>,
    pub roles: Vec<String>,
    pub auth_method: AuthMethod,
    pub request_id: String,
    pub ip_address: Option<String>,
}

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

#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
    pub request_id: String,
}

// ============================================================================
// Rate Limiting
// ============================================================================

pub struct RateLimiter {
    requests: Arc<RwLock<HashMap<String, Vec<std::time::Instant>>>>,
    max_requests: usize,
    window_duration: std::time::Duration,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_duration: std::time::Duration) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_requests,
            window_duration,
        }
    }

    pub async fn check_rate_limit(&self, key: &str) -> bool {
        let now = std::time::Instant::now();
        let mut requests = self.requests.write().await;

        let entry = requests.entry(key.to_string()).or_insert_with(Vec::new);

        // Remove old requests outside the window
        entry.retain(|&time| now.duration_since(time) < self.window_duration);

        // Check if under limit
        if entry.len() >= self.max_requests {
            false
        } else {
            entry.push(now);
            true
        }
    }
    
    pub async fn reset(&self, key: &str) {
        let mut requests = self.requests.write().await;
        requests.remove(key);
    }
}

// Global rate limiter instances
static GLOBAL_RATE_LIMITER: Lazy<RateLimiter> = Lazy::new(|| {
    RateLimiter::new(
        1000,  // 1000 requests
        std::time::Duration::from_secs(3600),  // per hour
    )
});

// ============================================================================
// Main Authentication Middleware
// ============================================================================

/// Primary authentication middleware for protected routes
pub async fn auth_middleware(
    state: Arc<AppState>, 
    request: Request, 
    next: Next
) -> Response {
    let request_id = extract_or_generate_request_id(&request);
    let ip_address = extract_ip_address(&request);

    // Apply global rate limiting
    if !GLOBAL_RATE_LIMITER.check_rate_limit(&ip_address).await {
        warn!(
            request_id = %request_id,
            ip = %ip_address,
            "Global rate limit exceeded"
        );
        return StatusCode::TOO_MANY_REQUESTS.into_response();
    }

    // Authenticate the request
    let auth_context = match authenticate_request(&state, request.headers(), &request_id, &ip_address).await {
        Ok(context) => context,
        Err(status) => {
            warn!(
                request_id = %request_id,
                status = ?status,
                "Authentication failed"
            );
            return create_auth_error_response(status, "Authentication failed", &request_id);
        }
    };

    // Log successful authentication
    info!(
        request_id = %request_id,
        user_id = %auth_context.user_id,
        auth_method = ?auth_context.auth_method,
        "Request authenticated successfully"
    );

    // Add auth context to request extensions
    let mut request = request;
    request.extensions_mut().insert(auth_context.clone());

    // Track API usage asynchronously
    let state_clone = state.clone();
    let auth_clone = auth_context.clone();
    tokio::spawn(async move {
        track_api_usage(&state_clone, &auth_clone).await;
    });

    // Continue with the request
    next.run(request).await
}

/// Authenticate the request using various methods
async fn authenticate_request(
    state: &Arc<AppState>,
    headers: &axum::http::HeaderMap,
    request_id: &str,
    ip_address: &str,
) -> Result<AuthContext, StatusCode> {
    // Try JWT authentication first
    if let Some(auth_header) = headers.get(header::AUTHORIZATION) {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                let token = &auth_str[7..];
                return authenticate_jwt(state, token, request_id, ip_address).await;
            }
        }
    }

    // Try API key authentication
    if let Some(api_key_header) = headers.get(&*AUTH_CONFIG.api_key_header) {
        if let Ok(api_key) = api_key_header.to_str() {
            return authenticate_api_key(state, api_key, request_id, ip_address).await;
        }
    }

    // Try service account authentication
    if let Some(service_header) = headers.get("X-Service-Account") {
        if let Ok(service_token) = service_header.to_str() {
            return authenticate_service_account(state, service_token, request_id, ip_address).await;
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

/// Authenticate using JWT
async fn authenticate_jwt(
    state: &Arc<AppState>,
    token: &str,
    request_id: &str,
    ip_address: &str,
) -> Result<AuthContext, StatusCode> {
    // Decode and validate JWT
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(AUTH_CONFIG.jwt_secret.as_bytes()),
        &Validation::new(jsonwebtoken::Algorithm::HS256),
    )
    .map_err(|e| {
        warn!("JWT validation failed: {}", e);
        StatusCode::UNAUTHORIZED
    })?;

    let claims = token_data.claims;

    // Validate token type
    if matches!(claims.token_type, TokenType::Refresh) {
        warn!("Refresh token used for API access");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Check if token is revoked
    if is_token_revoked(state, &claims.jti).await {
        warn!("Revoked token used: {}", claims.jti);
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
        ip_address: Some(ip_address.to_string()),
    })
}

/// Authenticate using API key
async fn authenticate_api_key(
    state: &Arc<AppState>,
    api_key: &str,
    request_id: &str,
    ip_address: &str,
) -> Result<AuthContext, StatusCode> {
    // Validate API key format
    if !api_key.starts_with("ik_") || api_key.len() != 35 {
        warn!("Invalid API key format");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Hash the API key for comparison
    let key_hash = hash_api_key(api_key);

    // Look up API key in database with caching consideration
    let key_data = sqlx::query!(
        r#"
        SELECT 
            id, key_hash, name, workspace_id, 
            COALESCE(scopes, ARRAY[]::text[]) as scopes, rate_limit, expires_at, last_used_at, 
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
        revoked: key_data.revoked.unwrap_or_default(),
    };

    // Check if key is expired
    if let Some(expires_at) = key_data.expires_at {
        if expires_at < Utc::now() {
            warn!("Expired API key used: {}", key_data.id);
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    // Apply API key specific rate limiting
    if let Some(rate_limit) = key_data.rate_limit {
        let limiter = RateLimiter::new(
            rate_limit as usize,
            std::time::Duration::from_secs(3600),
        );
        if !limiter.check_rate_limit(&key_data.id.to_string()).await {
            warn!("API key rate limit exceeded: {}", key_data.id);
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }
    }

    // Update last used timestamp asynchronously
    let db = state.db.clone();
    let key_id = key_data.id;
    tokio::spawn(async move {
        let _ = sqlx::query!(
            "UPDATE api_keys SET last_used_at = NOW() WHERE id = $1",
            key_id
        )
        .execute(&db)
        .await;
    });

    // Create auth context
    Ok(AuthContext {
        user_id: key_data.id.to_string(),
        workspace_id: Some(key_data.workspace_id),
        scopes: key_data.scopes,
        roles: vec!["api_key".to_string()],
        auth_method: AuthMethod::ApiKey,
        request_id: request_id.to_string(),
        ip_address: Some(ip_address.to_string()),
    })
}

/// Authenticate service account for internal services
async fn authenticate_service_account(
    _state: &Arc<AppState>,
    service_token: &str,
    request_id: &str,
    ip_address: &str,
) -> Result<AuthContext, StatusCode> {
    // Verify service token
    let service_secret = AUTH_CONFIG
        .service_secret
        .as_ref()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Use constant-time comparison to prevent timing attacks
    if !constant_time_compare(service_token.as_bytes(), service_secret.as_bytes()) {
        warn!("Invalid service account token");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Create auth context for service account
    Ok(AuthContext {
        user_id: "service-account".to_string(),
        workspace_id: None,
        scopes: vec!["*".to_string()],
        roles: vec!["service".to_string()],
        auth_method: AuthMethod::ServiceAccount,
        request_id: request_id.to_string(),
        ip_address: Some(ip_address.to_string()),
    })
}

// ============================================================================
// Token Management
// ============================================================================

/// Generate a new JWT token
pub fn generate_jwt_token(
    user_id: &str,
    workspace_id: Option<Uuid>,
    scopes: Vec<String>,
    roles: Vec<String>,
    token_type: TokenType,
) -> Result<String, jsonwebtoken::errors::Error> {
    let duration = match token_type {
        TokenType::Access => Duration::minutes(AUTH_CONFIG.access_token_duration_mins),
        TokenType::Refresh => Duration::days(AUTH_CONFIG.refresh_token_duration_days),
        _ => Duration::minutes(AUTH_CONFIG.access_token_duration_mins),
    };

    let now = Utc::now();
    let expiration = now + duration;

    let claims = Claims {
        sub: user_id.to_string(),
        exp: expiration.timestamp(),
        iat: now.timestamp(),
        nbf: now.timestamp(),
        iss: AUTH_CONFIG.jwt_issuer.clone(),
        aud: AUTH_CONFIG.jwt_audience.clone(),
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
        &EncodingKey::from_secret(AUTH_CONFIG.jwt_secret.as_bytes()),
    )
}

/// Decode JWT token
pub fn decode_jwt_token(token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(AUTH_CONFIG.jwt_secret.as_bytes()),
        &Validation::new(jsonwebtoken::Algorithm::HS256),
    )?;
    Ok(token_data.claims)
}

// ============================================================================
// API Key Management
// ============================================================================

/// Generate a new API key
pub async fn generate_api_key(
    state: &Arc<AppState>,
    workspace_id: Uuid,
    name: String,
    scopes: Vec<String>,
    expires_in_days: Option<i64>,
) -> Result<(String, Uuid), StatusCode> {
    // Generate cryptographically secure API key
    let api_key = generate_secure_key();
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

    info!("API key created: {} for workspace: {}", key_id, workspace_id);

    Ok((api_key, key_id))
}

/// Revoke an API key
pub async fn revoke_api_key(
    state: &Arc<AppState>,
    key_id: Uuid,
) -> Result<(), StatusCode> {
    let result = sqlx::query!(
        "UPDATE api_keys SET revoked = true, revoked_at = NOW() WHERE id = $1",
        key_id
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        error!("Failed to revoke API key: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    info!("API key revoked: {}", key_id);
    Ok(())
}

// ============================================================================
// Password Management
// ============================================================================

/// Hash password using bcrypt
pub fn hash_password(password: &str) -> Result<String, bcrypt::BcryptError> {
    bcrypt::hash(password, bcrypt::DEFAULT_COST)
}

/// Verify password against hash
pub fn verify_password(password: &str, hash: &str) -> Result<bool, bcrypt::BcryptError> {
    bcrypt::verify(password, hash)
}

// ============================================================================
// Middleware Functions
// ============================================================================

/// Middleware for requiring specific roles
pub async fn require_role(
    role: &'static str,
    request: Request,
    next: Next,
) -> Response {
    let auth_context = match request.extensions().get::<AuthContext>() {
        Some(ctx) => ctx,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    // Check if user has required role
    if !auth_context.roles.contains(&role.to_string()) {
        warn!(
            user_id = %auth_context.user_id,
            required_role = %role,
            "Insufficient role"
        );
        return StatusCode::FORBIDDEN.into_response();
    }

    next.run(request).await
}

/// Middleware for requiring workspace membership
pub async fn require_workspace(
    State(_state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let auth_context = match request.extensions().get::<AuthContext>() {
        Some(ctx) => ctx,
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };

    // Service accounts bypass workspace checks
    if matches!(auth_context.auth_method, AuthMethod::ServiceAccount) {
        return next.run(request).await;
    }

    // Ensure user has a workspace
    if auth_context.workspace_id.is_none() {
        warn!(
            user_id = %auth_context.user_id,
            "No workspace membership"
        );
        return StatusCode::FORBIDDEN.into_response();
    }

    next.run(request).await
}

// ============================================================================
// Helper Functions
// ============================================================================

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

/// Track API usage for analytics and rate limiting
async fn track_api_usage(state: &Arc<AppState>, auth_context: &AuthContext) {
    let _ = sqlx::query!(
        r#"
        INSERT INTO api_usage (
            user_id, workspace_id, auth_method, 
            request_id, ip_address, created_at
        )
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        auth_context.user_id,
        auth_context.workspace_id,
        format!("{:?}", auth_context.auth_method),
        auth_context.request_id,
        auth_context.ip_address.as_ref().and_then(|ip| ip.parse::<IpNetwork>().ok())
    )
    .execute(&state.db)
    .await;
}

/// Generate a cryptographically secure API key
fn generate_secure_key() -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = rand::rng();
    
    let key: String = (0..32)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect();
    
    format!("ik_{}", key)
}

/// Hash an API key using SHA-256
fn hash_api_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract or generate request ID
fn extract_or_generate_request_id(request: &Request) -> String {
    request
        .headers()
        .get(&*AUTH_CONFIG.request_id_header)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

/// Extract IP address from request
fn extract_ip_address(request: &Request) -> String {
    // Check X-Forwarded-For header first (for reverse proxy setups)
    if let Some(forwarded) = request.headers().get("X-Forwarded-For") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            // Take the first IP in the chain
            if let Some(ip) = forwarded_str.split(',').next() {
                return ip.trim().to_string();
            }
        }
    }

    // Check X-Real-IP header
    if let Some(real_ip) = request.headers().get("X-Real-IP") {
        if let Ok(ip) = real_ip.to_str() {
            return ip.to_string();
        }
    }

    // Default to unknown
    "unknown".to_string()
}

/// Create error response with proper formatting
fn create_auth_error_response(
    status: StatusCode,
    message: &str,
    request_id: &str,
) -> Response {
    let error = AuthError {
        error: status.to_string(),
        message: message.to_string(),
        request_id: request_id.to_string(),
    };

    (
        status,
        axum::Json(error),
    ).into_response()
}

/// Constant-time string comparison to prevent timing attacks
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare(b"hello", b"hello"));
        assert!(!constant_time_compare(b"hello", b"world"));
        assert!(!constant_time_compare(b"hello", b"hello!"));
    }

    #[test]
    fn test_generate_secure_key() {
        let key1 = generate_secure_key();
        let key2 = generate_secure_key();
        
        assert!(key1.starts_with("ik_"));
        assert_eq!(key1.len(), 35);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_hash_api_key() {
        let key = "ik_test123";
        let hash1 = hash_api_key(key);
        let hash2 = hash_api_key(key);
        
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(2, std::time::Duration::from_secs(1));
        
        assert!(limiter.check_rate_limit("test").await);
        assert!(limiter.check_rate_limit("test").await);
        assert!(!limiter.check_rate_limit("test").await);
        
        // Different key should work
        assert!(limiter.check_rate_limit("other").await);
        
        // After waiting, should work again
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        assert!(limiter.check_rate_limit("test").await);
    }

    #[test]
    fn test_jwt_generation_and_validation() {
        let token = generate_jwt_token(
            "user123",
            Some(Uuid::new_v4()),
            vec!["read".to_string()],
            vec!["user".to_string()],
            TokenType::Access,
        ).unwrap();

        let claims = decode_jwt_token(&token).unwrap();
        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.roles, vec!["user".to_string()]);
    }
}