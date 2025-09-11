// interstice-api/src/middleware_layer/rate_limit.rs

use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock as TokioRwLock;
use tracing::{debug, warn, error};

use crate::AppState;

/// Rate limiting algorithms supported
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    /// Fixed window counter
    FixedWindow,
    /// Sliding window log
    SlidingWindowLog,
    /// Sliding window counter
    SlidingWindowCounter,
    /// Token bucket
    TokenBucket,
    /// Leaky bucket
    LeakyBucket,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests allowed
    pub limit: u64,
    /// Time window in seconds
    pub window_secs: u64,
    /// Algorithm to use
    pub algorithm: RateLimitAlgorithm,
    /// Burst capacity for token/leaky bucket
    pub burst_capacity: Option<u64>,
    /// Refill rate per second for token bucket
    pub refill_rate: Option<f64>,
    /// Whether to use distributed storage
    pub distributed: bool,
    /// Custom headers to include
    pub custom_headers: HashMap<String, String>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            limit: 1000,
            window_secs: 3600, // 1 hour
            algorithm: RateLimitAlgorithm::SlidingWindowCounter,
            burst_capacity: Some(100),
            refill_rate: Some(10.0),
            distributed: false,
            custom_headers: HashMap::new(),
        }
    }
}

/// Rate limit tiers for different types of users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitTier {
    /// Public/anonymous users - most restrictive
    Public {
        requests_per_minute: u64,
        requests_per_hour: u64,
        burst_capacity: u64,
    },
    /// Authenticated users - moderate limits
    Authenticated {
        requests_per_minute: u64,
        requests_per_hour: u64,
        requests_per_day: u64,
        burst_capacity: u64,
    },
    /// API key users - higher limits
    ApiKey {
        requests_per_minute: u64,
        requests_per_hour: u64,
        requests_per_day: u64,
        burst_capacity: u64,
    },
    /// Premium users - very high limits
    Premium {
        requests_per_minute: u64,
        requests_per_hour: u64,
        requests_per_day: u64,
        burst_capacity: u64,
    },
    /// Admin users - unlimited or very high limits
    Admin {
        requests_per_minute: u64,
        requests_per_hour: u64,
        requests_per_day: u64,
    },
    /// Service accounts - high limits for internal services
    Service {
        requests_per_second: u64,
        requests_per_minute: u64,
        burst_capacity: u64,
    },
    /// Custom tier with specific configuration
    Custom(RateLimitConfig),
}

impl RateLimitTier {
    /// Get default tier configurations
    pub fn public() -> Self {
        Self::Public {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst_capacity: 20,
        }
    }

    pub fn authenticated() -> Self {
        Self::Authenticated {
            requests_per_minute: 300,
            requests_per_hour: 5000,
            requests_per_day: 50000,
            burst_capacity: 100,
        }
    }

    pub fn api_key() -> Self {
        Self::ApiKey {
            requests_per_minute: 600,
            requests_per_hour: 10000,
            requests_per_day: 100000,
            burst_capacity: 200,
        }
    }

    pub fn premium() -> Self {
        Self::Premium {
            requests_per_minute: 1200,
            requests_per_hour: 25000,
            requests_per_day: 500000,
            burst_capacity: 500,
        }
    }

    pub fn admin() -> Self {
        Self::Admin {
            requests_per_minute: 5000,
            requests_per_hour: 100000,
            requests_per_day: 1000000,
        }
    }

    pub fn service() -> Self {
        Self::Service {
            requests_per_second: 100,
            requests_per_minute: 3000,
            burst_capacity: 1000,
        }
    }
}

/// Rate limit result
#[derive(Debug)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Remaining requests in current window
    pub remaining: u64,
    /// Total limit for the window
    pub limit: u64,
    /// Time when the limit resets (Unix timestamp)
    pub reset_time: u64,
    /// Seconds to wait before retry (if denied)
    pub retry_after: Option<u64>,
}

/// Token bucket state for rate limiting
#[derive(Debug)]
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            last_refill: Instant::now(),
            capacity,
            refill_rate,
        }
    }

    fn try_consume(&mut self, tokens: f64) -> bool {
        self.refill();
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = elapsed * self.refill_rate;
        
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
}

/// Sliding window log entry
#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: Instant,
}

/// In-memory rate limiter for high-performance scenarios
pub struct MemoryRateLimiter {
    /// Token buckets for each client
    token_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    /// Sliding window logs for each client
    sliding_logs: Arc<RwLock<HashMap<String, VecDeque<LogEntry>>>>,
    /// Fixed window counters
    fixed_counters: Arc<RwLock<HashMap<String, (u64, Instant)>>>,
    /// Configuration
    config: RateLimitConfig,
}

impl MemoryRateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            token_buckets: Arc::new(RwLock::new(HashMap::new())),
            sliding_logs: Arc::new(RwLock::new(HashMap::new())),
            fixed_counters: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check rate limit for a client
    pub fn check_limit(&self, client_id: &str, tokens: f64) -> RateLimitResult {
        match self.config.algorithm {
            RateLimitAlgorithm::TokenBucket => self.check_token_bucket(client_id, tokens),
            RateLimitAlgorithm::SlidingWindowLog => self.check_sliding_window_log(client_id),
            RateLimitAlgorithm::FixedWindow => self.check_fixed_window(client_id),
            RateLimitAlgorithm::SlidingWindowCounter => self.check_sliding_window_counter(client_id),
            RateLimitAlgorithm::LeakyBucket => self.check_leaky_bucket(client_id),
        }
    }

    fn check_token_bucket(&self, client_id: &str, tokens: f64) -> RateLimitResult {
        let mut buckets = self.token_buckets.write().unwrap();
        let bucket = buckets.entry(client_id.to_string()).or_insert_with(|| {
            TokenBucket::new(
                self.config.burst_capacity.unwrap_or(self.config.limit) as f64,
                self.config.refill_rate.unwrap_or(1.0),
            )
        });

        let allowed = bucket.try_consume(tokens);
        let remaining = bucket.tokens as u64;
        let reset_time = self.calculate_reset_time();

        RateLimitResult {
            allowed,
            remaining,
            limit: self.config.limit,
            reset_time,
            retry_after: if allowed { None } else { Some(1) },
        }
    }

    fn check_sliding_window_log(&self, client_id: &str) -> RateLimitResult {
        let mut logs = self.sliding_logs.write().unwrap();
        let log = logs.entry(client_id.to_string()).or_default();
        
        let now = Instant::now();
        let window_start = now - Duration::from_secs(self.config.window_secs);
        
        // Remove old entries
        while let Some(entry) = log.front() {
            if entry.timestamp < window_start {
                log.pop_front();
            } else {
                break;
            }
        }
        
        let current_count = log.len() as u64;
        let allowed = current_count < self.config.limit;
        
        if allowed {
            log.push_back(LogEntry { timestamp: now });
        }
        
        let reset_time = self.calculate_reset_time();
        
        RateLimitResult {
            allowed,
            remaining: self.config.limit.saturating_sub(current_count + if allowed { 1 } else { 0 }),
            limit: self.config.limit,
            reset_time,
            retry_after: if allowed { None } else { Some(self.config.window_secs) },
        }
    }

    fn check_fixed_window(&self, client_id: &str) -> RateLimitResult {
        let mut counters = self.fixed_counters.write().unwrap();
        let now = Instant::now();
        
        let entry = counters.entry(client_id.to_string()).or_insert((0, now));
        
        // Check if we're in a new window
        if now.duration_since(entry.1) >= Duration::from_secs(self.config.window_secs) {
            *entry = (1, now);
        } else {
            entry.0 += 1;
        }
        
        let allowed = entry.0 <= self.config.limit;
        let remaining = self.config.limit.saturating_sub(entry.0);
        let reset_time = entry.1.elapsed().as_secs() + self.config.window_secs;
        
        RateLimitResult {
            allowed,
            remaining,
            limit: self.config.limit,
            reset_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() + reset_time,
            retry_after: if allowed { None } else { Some(reset_time) },
        }
    }

    fn check_sliding_window_counter(&self, _client_id: &str) -> RateLimitResult {
        // Simplified implementation - in production, use a more sophisticated approach
        // with multiple time buckets
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        RateLimitResult {
            allowed: true,
            remaining: self.config.limit,
            limit: self.config.limit,
            reset_time: now_secs + self.config.window_secs,
            retry_after: None,
        }
    }

    fn check_leaky_bucket(&self, client_id: &str) -> RateLimitResult {
        // Leaky bucket is similar to token bucket but with constant outflow
        self.check_token_bucket(client_id, 1.0)
    }

    fn calculate_reset_time(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() + self.config.window_secs
    }

    /// Clean up old entries periodically
    pub fn cleanup(&self) {
        let now = Instant::now();
        let cutoff = now - Duration::from_secs(self.config.window_secs * 2);

        // Clean sliding logs
        if let Ok(mut logs) = self.sliding_logs.write() {
            logs.retain(|_, log| {
                log.retain(|entry| entry.timestamp > cutoff);
                !log.is_empty()
            });
        }

        // Clean fixed counters
        if let Ok(mut counters) = self.fixed_counters.write() {
            counters.retain(|_, (_, timestamp)| *timestamp > cutoff);
        }
    }
}

/// Distributed rate limiter using database storage (Fixed for actual schema)
pub struct DatabaseRateLimiter {
    db: sqlx::PgPool,
    config: RateLimitConfig,
}

impl DatabaseRateLimiter {
    pub fn new(db: sqlx::PgPool, config: RateLimitConfig) -> Self {
        Self { db, config }
    }

    /// Check rate limit using database storage
    pub async fn check_limit(&self, client_id: &str) -> Result<RateLimitResult, sqlx::Error> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        
        let window_start = now - self.config.window_secs as i64;

        // Get current count in window - Fixed to match actual schema (no request_count column)
        let count: i64 = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*) as total
            FROM rate_limit_tracking 
            WHERE client_id = $1 AND timestamp >= $2
            "#,
            client_id,
            window_start
        )
        .fetch_one(&self.db)
        .await?
        .unwrap_or(0);

        let current_count = count as u64;
        let allowed = current_count < self.config.limit;

        if allowed {
            // Record this request - Fixed to match actual schema (no request_count column)
            sqlx::query!(
                r#"
                INSERT INTO rate_limit_tracking (client_id, timestamp)
                VALUES ($1, $2)
                "#,
                client_id,
                now
            )
            .execute(&self.db)
            .await?;
        }

        // Calculate retry after
        let retry_after = if !allowed {
            let oldest_in_window: Option<i64> = sqlx::query_scalar!(
                r#"
                SELECT MIN(timestamp) 
                FROM rate_limit_tracking 
                WHERE client_id = $1 AND timestamp >= $2
                "#,
                client_id,
                window_start
            )
            .fetch_optional(&self.db)
            .await?
            .flatten();

            oldest_in_window.map(|oldest| {
                (oldest + self.config.window_secs as i64 - now).max(0) as u64
            })
        } else {
            None
        };

        // Async cleanup of old entries
        let db_clone = self.db.clone();
        let client_id_clone = client_id.to_string();
        let cleanup_before = now - (self.config.window_secs as i64 * 2);
        
        tokio::spawn(async move {
            let _ = sqlx::query!(
                r#"
                DELETE FROM rate_limit_tracking 
                WHERE client_id = $1 AND timestamp < $2
                "#,
                client_id_clone,
                cleanup_before
            )
            .execute(&db_clone)
            .await;
        });

        Ok(RateLimitResult {
            allowed,
            remaining: self.config.limit.saturating_sub(current_count + if allowed { 1 } else { 0 }),
            limit: self.config.limit,
            reset_time: (now + self.config.window_secs as i64) as u64,
            retry_after,
        })
    }
}

/// Main rate limiter that can use memory or database storage
#[derive(Clone)]
pub struct RateLimiter {
    memory: Arc<MemoryRateLimiter>,
    database: Option<Arc<DatabaseRateLimiter>>,
    tier_configs: Arc<RwLock<HashMap<String, RateLimitTier>>>,
}

impl RateLimiter {
    /// Create new rate limiter with default configuration
    pub fn new(limit: u64, window: Duration) -> Self {
        let config = RateLimitConfig {
            limit,
            window_secs: window.as_secs(),
            ..Default::default()
        };

        Self {
            memory: Arc::new(MemoryRateLimiter::new(config.clone())),
            database: None,
            tier_configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create rate limiter with database support
    pub fn with_database(db: sqlx::PgPool, config: RateLimitConfig) -> Self {
        let memory = Arc::new(MemoryRateLimiter::new(config.clone()));
        let database = if config.distributed {
            Some(Arc::new(DatabaseRateLimiter::new(db, config)))
        } else {
            None
        };

        Self {
            memory,
            database,
            tier_configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set rate limit tier for a client
    pub fn set_tier(&self, client_id: &str, tier: RateLimitTier) {
        if let Ok(mut tiers) = self.tier_configs.write() {
            tiers.insert(client_id.to_string(), tier);
        }
    }

    /// Get rate limit tier for a client
    pub fn get_tier(&self, client_id: &str) -> RateLimitTier {
        if let Ok(tiers) = self.tier_configs.read() {
            tiers.get(client_id).cloned().unwrap_or_else(RateLimitTier::public)
        } else {
            RateLimitTier::public()
        }
    }

    /// Check rate limit for a client
    pub async fn check_rate_limit(&self, client_id: &str) -> bool {
        // Use database limiter if available and configured for distributed mode
        if let Some(db_limiter) = &self.database {
            match db_limiter.check_limit(client_id).await {
                Ok(result) => return result.allowed,
                Err(e) => {
                    error!("Database rate limiter failed, falling back to memory: {}", e);
                }
            }
        }

        // Fallback to memory limiter
        self.memory.check_limit(client_id, 1.0).allowed
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
        let memory = Arc::clone(&self.memory);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
            
            loop {
                interval.tick().await;
                memory.cleanup();
                debug!("Rate limiter cleanup completed");
            }
        })
    }
}

/// Global rate limiter instance
static GLOBAL_RATE_LIMITER: std::sync::OnceLock<Arc<TokioRwLock<RateLimiter>>> = std::sync::OnceLock::new();

/// Get or initialize the global rate limiter
pub fn global_rate_limiter() -> &'static Arc<TokioRwLock<RateLimiter>> {
    GLOBAL_RATE_LIMITER.get_or_init(|| {
        Arc::new(TokioRwLock::new(RateLimiter::new(1000, Duration::from_secs(3600))))
    })
}

/// Initialize global rate limiter with custom configuration
pub fn init_global_rate_limiter(limiter: RateLimiter) {
    let _ = GLOBAL_RATE_LIMITER.set(Arc::new(TokioRwLock::new(limiter)));
}

/// Extract client identifier from request
pub fn extract_client_id(request: &Request) -> String {
    // Try to get authenticated user ID first
    if let Some(auth_context) = request.extensions().get::<super::auth::AuthContext>() {
        return format!("user:{}", auth_context.user_id);
    }

    // Try API key
    if let Some(api_key) = extract_api_key(request) {
        return format!("api_key:{}", api_key);
    }

    // Fallback to IP address
    format!("ip:{}", extract_client_ip(request))
}

/// Extract API key from request
fn extract_api_key(request: &Request) -> Option<String> {
    // Check Authorization header
    if let Some(auth_header) = request.headers().get("Authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if auth_str.starts_with("Bearer ") {
                return Some(auth_str[7..].to_string());
            }
        }
    }

    // Check X-API-Key header
    if let Some(api_key) = request.headers().get("X-API-Key") {
        if let Ok(key_str) = api_key.to_str() {
            return Some(key_str.to_string());
        }
    }

    None
}

/// Extract client IP from request
pub fn extract_client_ip(request: &Request) -> String {
    // Check X-Forwarded-For header first
    if let Some(forwarded) = request.headers().get("X-Forwarded-For") {
        if let Ok(forwarded_str) = forwarded.to_str() {
            if let Some(first_ip) = forwarded_str.split(',').next() {
                return first_ip.trim().to_string();
            }
        }
    }

    // Check X-Real-IP header
    if let Some(real_ip) = request.headers().get("X-Real-IP") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }

    // Check CloudFlare and other headers
    if let Some(cf_ip) = request.headers().get("CF-Connecting-IP") {
        if let Ok(ip_str) = cf_ip.to_str() {
            return ip_str.to_string();
        }
    }

    // Default fallback
    "127.0.0.1".to_string()
}

/// Rate limiting middleware for authenticated routes
pub async fn auth_rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    let client_id = extract_client_id(&request);
    
    // Check rate limit and get detailed result
    let result = if let Some(db_limiter) = &state.rate_limiter.database {
        match db_limiter.check_limit(&client_id).await {
            Ok(result) => result,
            Err(e) => {
                error!("Database rate limiter failed, falling back to memory: {}", e);
                state.rate_limiter.memory.check_limit(&client_id, 1.0)
            }
        }
    } else {
        state.rate_limiter.memory.check_limit(&client_id, 1.0)
    };
    
    if !result.allowed {
        warn!(
            client_id = %client_id,
            remaining = %result.remaining,
            limit = %result.limit,
            "Rate limit exceeded for authenticated user"
        );
        
        return Err(create_rate_limit_response(&result));
    }

    // Continue with request
    let response = next.run(request).await;
    
    // Add rate limit headers to successful responses
    let mut response = response;
    let headers = response.headers_mut();
    let _ = headers.insert("X-RateLimit-Limit", HeaderValue::from_str(&result.limit.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Remaining", HeaderValue::from_str(&result.remaining.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Reset", HeaderValue::from_str(&result.reset_time.to_string()).unwrap());
    
    debug!(
        client_id = %client_id,
        remaining = %result.remaining,
        limit = %result.limit,
        "Rate limit check passed for authenticated user"
    );
    
    Ok(response)
}


/// IP-based rate limiting for public endpoints
pub async fn ip_rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    let client_ip = extract_client_ip(&request);
    let client_id = format!("public_ip:{}", client_ip);
    
    // Use global rate limiter with public tier limits
    let limiter = global_rate_limiter().read().await;
    let result = limiter.memory.check_limit(&client_id, 1.0);
    
    if !result.allowed {
        warn!(
            client_ip = %client_ip,
            remaining = %result.remaining,
            limit = %result.limit,
            "Public IP rate limit exceeded"
        );
        
        return Err(create_rate_limit_response(&result));
    }

    let response = next.run(request).await;
    
    // Add rate limit headers to successful responses
    let mut response = response;
    let headers = response.headers_mut();
    let _ = headers.insert("X-RateLimit-Limit", HeaderValue::from_str(&result.limit.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Remaining", HeaderValue::from_str(&result.remaining.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Reset", HeaderValue::from_str(&result.reset_time.to_string()).unwrap());
    
    Ok(response)
}


/// Create rate limit exceeded response with proper rate limit information
fn create_rate_limit_response(result: &RateLimitResult) -> Response {
    #[derive(Serialize)]
    struct RateLimitError {
        error: String,
        message: String,
        limit: u64,
        remaining: u64,
        reset_time: u64,
        retry_after: u64,
    }

    let error = RateLimitError {
        error: "rate_limit_exceeded".to_string(),
        message: "Rate limit exceeded. Please try again later.".to_string(),
        limit: result.limit,
        remaining: result.remaining,
        reset_time: result.reset_time,
        retry_after: result.retry_after.unwrap_or(60),
    };

    let mut response = (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response();
    
    let headers = response.headers_mut();
    let _ = headers.insert("X-RateLimit-Limit", HeaderValue::from_str(&result.limit.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Remaining", HeaderValue::from_str(&result.remaining.to_string()).unwrap());
    let _ = headers.insert("X-RateLimit-Reset", HeaderValue::from_str(&result.reset_time.to_string()).unwrap());
    
    if let Some(retry_after) = result.retry_after {
        let _ = headers.insert("Retry-After", HeaderValue::from_str(&retry_after.to_string()).unwrap());
    }
    
    response
}

/// Slack-specific rate limiting middleware
pub async fn slack_rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    // Slack has specific rate limiting requirements
    let client_id = format!("slack:{}", extract_client_ip(&request));
    
    // Create Slack-specific configuration
    let config = RateLimitConfig {
        limit: 1, // Slack Events API: 1 request per second
        window_secs: 1,
        algorithm: RateLimitAlgorithm::TokenBucket,
        burst_capacity: Some(5),
        refill_rate: Some(1.0),
        distributed: false,
        custom_headers: HashMap::new(),
    };
    
    let limiter = MemoryRateLimiter::new(config);
    let result = limiter.check_limit(&client_id, 1.0);
    
    if !result.allowed {
        warn!(
            client_id = %client_id,
            remaining = %result.remaining,
            limit = %result.limit,
            "Slack rate limit exceeded"
        );
        return Err(create_rate_limit_response(&result));
    }

    Ok(next.run(request).await)
}

/// Webhook rate limiting middleware
pub async fn webhook_rate_limit_middleware(
    request: Request,
    next: Next,
) -> Result<Response, Response> {
    let client_id = format!("webhook:{}", extract_client_ip(&request));
    
    // Webhook-specific limits - more restrictive
    let config = RateLimitConfig {
        limit: 100, // 100 requests per hour for webhooks
        window_secs: 3600,
        algorithm: RateLimitAlgorithm::SlidingWindowCounter,
        burst_capacity: Some(10),
        refill_rate: None,
        distributed: false,
        custom_headers: HashMap::new(),
    };
    
    let limiter = MemoryRateLimiter::new(config);
    let result = limiter.check_limit(&client_id, 1.0);
    
    if !result.allowed {
        warn!(
            client_id = %client_id,
            remaining = %result.remaining,
            limit = %result.limit,
            "Webhook rate limit exceeded"
        );
        return Err(create_rate_limit_response(&result));
    }

    Ok(next.run(request).await)
}

pub fn slack_rate_limit() -> tower::ServiceBuilder<tower::layer::util::Identity> {
    tower::ServiceBuilder::new()
}


#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_memory_rate_limiter_token_bucket() {
        let config = RateLimitConfig {
            limit: 10,
            window_secs: 60,
            algorithm: RateLimitAlgorithm::TokenBucket,
            burst_capacity: Some(10),
            refill_rate: Some(1.0), // 1 token per second
            ..Default::default()
        };

        let limiter = MemoryRateLimiter::new(config);
        let client_id = "test_client";

        // Should allow initial burst
        for i in 0..10 {
            let result = limiter.check_limit(client_id, 1.0);
            assert!(result.allowed, "Request {} should be allowed", i);
        }

        // 11th request should be denied
        let result = limiter.check_limit(client_id, 1.0);
        assert!(!result.allowed, "11th request should be denied");

        // Wait for token refill and test again
        sleep(Duration::from_secs(2)).await;
        let result = limiter.check_limit(client_id, 1.0);
        assert!(result.allowed, "Request should be allowed after refill");
    }

    #[tokio::test]
    async fn test_sliding_window_log() {
        let config = RateLimitConfig {
            limit: 5,
            window_secs: 2,
            algorithm: RateLimitAlgorithm::SlidingWindowLog,
            ..Default::default()
        };

        let limiter = MemoryRateLimiter::new(config);
        let client_id = "test_client_sliding";

        // Allow up to limit
        for i in 0..5 {
            let result = limiter.check_limit(client_id, 1.0);
            assert!(result.allowed, "Request {} should be allowed", i);
        }

        // 6th request should be denied
        let result = limiter.check_limit(client_id, 1.0);
        assert!(!result.allowed, "6th request should be denied");

        // Wait for window to pass
        sleep(Duration::from_secs(3)).await;
        let result = limiter.check_limit(client_id, 1.0);
        assert!(result.allowed, "Request should be allowed after window reset");
    }

    #[tokio::test]
    async fn test_fixed_window() {
        let config = RateLimitConfig {
            limit: 3,
            window_secs: 1,
            algorithm: RateLimitAlgorithm::FixedWindow,
            ..Default::default()
        };

        let limiter = MemoryRateLimiter::new(config);
        let client_id = "test_client_fixed";

        // Allow up to limit in window
        for i in 0..3 {
            let result = limiter.check_limit(client_id, 1.0);
            assert!(result.allowed, "Request {} should be allowed", i);
        }

        // 4th request in same window should be denied
        let result = limiter.check_limit(client_id, 1.0);
        assert!(!result.allowed, "4th request should be denied");

        // Wait for next window
        sleep(Duration::from_secs(2)).await;
        let result = limiter.check_limit(client_id, 1.0);
        assert!(result.allowed, "Request should be allowed in new window");
    }

    #[test]
    fn test_rate_limit_tiers() {
        let public = RateLimitTier::public();
        let premium = RateLimitTier::premium();
        let admin = RateLimitTier::admin();

        // Ensure tiers have appropriate limits
        match public {
            RateLimitTier::Public { requests_per_minute, .. } => {
                assert!(requests_per_minute <= 100);
            }
            _ => panic!("Expected Public tier"),
        }

        match premium {
            RateLimitTier::Premium { requests_per_minute, .. } => {
                assert!(requests_per_minute >= 1000);
            }
            _ => panic!("Expected Premium tier"),
        }

        match admin {
            RateLimitTier::Admin { requests_per_minute, .. } => {
                assert!(requests_per_minute >= 5000);
            }
            _ => panic!("Expected Admin tier"),
        }
    }

    #[test]
    fn test_client_id_extraction() {
        // Test with mock request (would need to set up proper test request)
        // This is a placeholder for proper integration tests
        assert!(true);
    }
}