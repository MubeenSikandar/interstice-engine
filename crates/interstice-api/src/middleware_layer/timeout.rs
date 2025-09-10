// interstice-api/src/middleware_layer/timeout.rs

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{Semaphore, broadcast},
    time::timeout,
};
use tracing::{debug, error, info, instrument, warn, Span};
use uuid::Uuid;

use crate::AppState;

// ==================== Configuration ====================

/// Timeout configuration for different operation types with adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Default request timeout
    pub default_request: Duration,
    /// Database operation timeout
    pub database: Duration,
    /// Individual database query timeout
    pub database_query: Duration,
    /// ML pipeline operations timeout
    pub ml_pipeline: Duration,
    /// ML inference timeout
    pub ml_inference: Duration,
    /// Slack handler timeout
    pub slack_handler: Duration,
    /// Slack API call timeout
    pub slack_api: Duration,
    /// Webhook processing timeout
    pub webhook_processing: Duration,
    /// File upload timeout
    pub file_upload: Duration,
    /// Analytics processing timeout
    pub analytics: Duration,
    /// Auth operations timeout
    pub auth: Duration,
    /// Health check timeout
    pub health_check: Duration,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Adaptive timeout configuration
    pub adaptive: AdaptiveTimeoutConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Time window for counting failures
    pub failure_window: Duration,
    /// How long to wait before attempting to close circuit
    pub recovery_timeout: Duration,
    /// Number of successful requests to close circuit
    pub success_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTimeoutConfig {
    /// Enable adaptive timeout adjustments
    pub enabled: bool,
    /// Minimum allowed timeout
    pub min_timeout: Duration,
    /// Maximum allowed timeout
    pub max_timeout: Duration,
    /// Percentile to use for timeout calculation (e.g., 99 for p99)
    pub target_percentile: u8,
    /// How often to recalculate timeouts
    pub recalculation_interval: Duration,
    /// Number of samples to keep for percentile calculation
    pub sample_window_size: usize,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_request: Duration::from_secs(30),
            database: Duration::from_secs(10),
            database_query: Duration::from_secs(5),
            ml_pipeline: Duration::from_secs(60),
            ml_inference: Duration::from_secs(30),
            slack_handler: Duration::from_secs(45),
            slack_api: Duration::from_secs(15),
            webhook_processing: Duration::from_secs(20),
            file_upload: Duration::from_secs(300),
            analytics: Duration::from_secs(10),
            auth: Duration::from_secs(5),
            health_check: Duration::from_secs(2),
            circuit_breaker: CircuitBreakerConfig::default(),
            adaptive: AdaptiveTimeoutConfig::default(),
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 3,
        }
    }
}

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_timeout: Duration::from_secs(1),
            max_timeout: Duration::from_secs(300),
            target_percentile: 99,
            recalculation_interval: Duration::from_secs(60),
            sample_window_size: 1000,
        }
    }
}

impl TimeoutConfig {
    /// Load timeout configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        macro_rules! load_duration {
            ($field:ident, $env_var:expr, $default:expr) => {
                config.$field = Duration::from_secs(
                    std::env::var($env_var)
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or($default)
                );
            };
        }
        
        load_duration!(default_request, "TIMEOUT_DEFAULT_REQUEST_SECS", 30);
        load_duration!(database, "TIMEOUT_DATABASE_SECS", 10);
        load_duration!(database_query, "TIMEOUT_DATABASE_QUERY_SECS", 5);
        load_duration!(ml_pipeline, "TIMEOUT_ML_PIPELINE_SECS", 60);
        load_duration!(ml_inference, "TIMEOUT_ML_INFERENCE_SECS", 30);
        load_duration!(slack_handler, "TIMEOUT_SLACK_HANDLER_SECS", 45);
        load_duration!(slack_api, "TIMEOUT_SLACK_API_SECS", 15);
        load_duration!(webhook_processing, "TIMEOUT_WEBHOOK_PROCESSING_SECS", 20);
        load_duration!(file_upload, "TIMEOUT_FILE_UPLOAD_SECS", 300);
        load_duration!(analytics, "TIMEOUT_ANALYTICS_SECS", 10);
        load_duration!(auth, "TIMEOUT_AUTH_SECS", 5);
        load_duration!(health_check, "TIMEOUT_HEALTH_CHECK_SECS", 2);
        
        // Load circuit breaker config
        if let Ok(val) = std::env::var("CIRCUIT_BREAKER_ENABLED") {
            if val.to_lowercase() == "true" {
                config.circuit_breaker.failure_threshold = std::env::var("CIRCUIT_BREAKER_FAILURE_THRESHOLD")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5);
            }
        }
        
        // Load adaptive timeout config
        config.adaptive.enabled = std::env::var("ADAPTIVE_TIMEOUT_ENABLED")
            .ok()
            .map(|s| s.to_lowercase() == "true")
            .unwrap_or(true);
        
        config
    }

    /// Get production-optimized timeout configuration
    pub fn production() -> Self {
        Self {
            default_request: Duration::from_secs(25),
            database: Duration::from_secs(8),
            database_query: Duration::from_secs(3),
            ml_pipeline: Duration::from_secs(45),
            ml_inference: Duration::from_secs(20),
            slack_handler: Duration::from_secs(30),
            slack_api: Duration::from_secs(10),
            webhook_processing: Duration::from_secs(15),
            file_upload: Duration::from_secs(180),
            analytics: Duration::from_secs(5),
            auth: Duration::from_secs(3),
            health_check: Duration::from_secs(1),
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 3,
                failure_window: Duration::from_secs(30),
                recovery_timeout: Duration::from_secs(15),
                success_threshold: 2,
            },
            adaptive: AdaptiveTimeoutConfig {
                enabled: true,
                min_timeout: Duration::from_millis(500),
                max_timeout: Duration::from_secs(120),
                target_percentile: 95,
                recalculation_interval: Duration::from_secs(30),
                sample_window_size: 500,
            },
        }
    }

    /// Get development-friendly timeout configuration
    pub fn development() -> Self {
        Self {
            default_request: Duration::from_secs(120),
            database: Duration::from_secs(30),
            database_query: Duration::from_secs(15),
            ml_pipeline: Duration::from_secs(300),
            ml_inference: Duration::from_secs(180),
            slack_handler: Duration::from_secs(180),
            slack_api: Duration::from_secs(60),
            webhook_processing: Duration::from_secs(90),
            file_upload: Duration::from_secs(600),
            analytics: Duration::from_secs(30),
            auth: Duration::from_secs(15),
            health_check: Duration::from_secs(5),
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 10,
                failure_window: Duration::from_secs(120),
                recovery_timeout: Duration::from_secs(60),
                success_threshold: 5,
            },
            adaptive: AdaptiveTimeoutConfig {
                enabled: false,
                min_timeout: Duration::from_secs(5),
                max_timeout: Duration::from_secs(600),
                target_percentile: 99,
                recalculation_interval: Duration::from_secs(120),
                sample_window_size: 100,
            },
        }
    }
}

// ==================== Circuit Breaker ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicU32>,
    success_count: Arc<AtomicU32>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    last_state_change: Arc<RwLock<Instant>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicU32::new(0)),
            success_count: Arc::new(AtomicU32::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            last_state_change: Arc::new(RwLock::new(Instant::now())),
            config,
        }
    }

    pub fn record_success(&self) {
        let current_state = *self.state.read();
        
        match current_state {
            CircuitState::HalfOpen => {
                let count = self.success_count.fetch_add(1, Ordering::SeqCst) + 1;
                if count >= self.config.success_threshold {
                    self.close_circuit();
                }
            }
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    pub fn record_failure(&self) {
        let now = Instant::now();
        let mut last_failure = self.last_failure_time.write();
        
        // Check if we're within the failure window
        if let Some(last) = *last_failure {
            if now.duration_since(last) > self.config.failure_window {
                self.failure_count.store(0, Ordering::SeqCst);
            }
        }
        
        *last_failure = Some(now);
        drop(last_failure);
        
        let failures = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        
        if failures >= self.config.failure_threshold {
            self.open();
        }
    }

    pub fn should_allow_request(&self) -> bool {
        let current_state = *self.state.read();
        
        match current_state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let last_change = *self.last_state_change.read();
                if Instant::now().duration_since(last_change) > self.config.recovery_timeout {
                    self.half_open();
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    fn open(&self) {
        let mut state = self.state.write();
        if *state != CircuitState::Open {
            *state = CircuitState::Open;
            *self.last_state_change.write() = Instant::now();
            warn!("Circuit breaker opened");
        }
    }

    fn close_circuit(&self) {
        let mut state = self.state.write();
        if *state != CircuitState::Closed {
            *state = CircuitState::Closed;
            *self.last_state_change.write() = Instant::now();
            self.failure_count.store(0, Ordering::SeqCst);
            self.success_count.store(0, Ordering::SeqCst);
            info!("Circuit breaker closed");
        }
    }

    fn half_open(&self) {
        let mut state = self.state.write();
        if *state != CircuitState::HalfOpen {
            *state = CircuitState::HalfOpen;
            *self.last_state_change.write() = Instant::now();
            self.success_count.store(0, Ordering::SeqCst);
            info!("Circuit breaker half-open");
        }
    }

    pub fn current_state(&self) -> CircuitState {
        *self.state.read()
    }
}

// ==================== Metrics ====================

#[derive(Debug, Clone, Serialize)]
pub struct TimeoutMetrics {
    pub total_requests: u64,
    pub total_timeouts: u64,
    #[serde(serialize_with = "serialize_dashmap")]
    pub timeouts_by_operation: DashMap<String, u64>,
    pub avg_request_duration: f64,
    pub max_request_duration: Duration,
    pub min_request_duration: Duration,
    pub timeout_rate: f64,
    pub p50_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub circuit_breaker_state: CircuitState,
    pub active_requests: usize,
}

fn serialize_dashmap<S>(dashmap: &DashMap<String, u64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeMap;
    let mut map = serializer.serialize_map(Some(dashmap.len()))?;
    for entry in dashmap.iter() {
        map.serialize_entry(entry.key(), entry.value())?;
    }
    map.end()
}

#[derive(Debug)]
struct DurationSample {
    duration: Duration,
    timestamp: Instant,
    operation: String,
}

// ==================== Timeout Manager ====================

pub struct TimeoutManager {
    config: Arc<RwLock<TimeoutConfig>>,
    metrics: Arc<RwLock<TimeoutMetrics>>,
    circuit_breakers: Arc<DashMap<String, Arc<CircuitBreaker>>>,
    duration_samples: Arc<RwLock<VecDeque<DurationSample>>>,
    active_requests: Arc<AtomicUsize>,
    request_semaphore: Arc<Semaphore>,
    shutdown_signal: broadcast::Sender<()>,
    total_requests: Arc<AtomicU64>,
    total_timeouts: Arc<AtomicU64>,
}

impl TimeoutManager {
    pub fn new(config: TimeoutConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        
        let manager = Self {
            config: Arc::new(RwLock::new(config.clone())),
            metrics: Arc::new(RwLock::new(TimeoutMetrics {
                total_requests: 0,
                total_timeouts: 0,
                timeouts_by_operation: DashMap::new(),
                avg_request_duration: 0.0,
                max_request_duration: Duration::from_secs(0),
                min_request_duration: Duration::from_secs(u64::MAX),
                timeout_rate: 0.0,
                p50_duration: Duration::from_secs(0),
                p95_duration: Duration::from_secs(0),
                p99_duration: Duration::from_secs(0),
                circuit_breaker_state: CircuitState::Closed,
                active_requests: 0,
            })),
            circuit_breakers: Arc::new(DashMap::new()),
            duration_samples: Arc::new(RwLock::new(VecDeque::with_capacity(
                config.adaptive.sample_window_size,
            ))),
            active_requests: Arc::new(AtomicUsize::new(0)),
            request_semaphore: Arc::new(Semaphore::new(1000)), // Max concurrent requests
            shutdown_signal: shutdown_tx,
            total_requests: Arc::new(AtomicU64::new(0)),
            total_timeouts: Arc::new(AtomicU64::new(0)),
        };
        
        // Start adaptive timeout recalculation task if enabled
        if config.adaptive.enabled {
            manager.start_adaptive_timeout_task();
        }
        
        // Start metrics aggregation task
        manager.start_metrics_aggregation_task();
        
        manager
    }

    /// Execute an operation with timeout and circuit breaker
    #[instrument(skip(self, operation), fields(operation_name = %operation_name))]
    pub async fn execute_with_timeout<T, F, Fut>(
        &self,
        operation: F,
        timeout_duration: Duration,
        operation_name: &str,
    ) -> Result<T, TimeoutError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        // Check circuit breaker
        let circuit_breaker = self.get_or_create_circuit_breaker(operation_name);
        if !circuit_breaker.should_allow_request() {
            return Err(TimeoutError::CircuitBreakerOpen {
                operation: operation_name.to_string(),
            });
        }
        
        // Acquire semaphore permit for rate limiting
        let _permit = self.request_semaphore.acquire().await
            .map_err(|_| TimeoutError::RateLimited {
                operation: operation_name.to_string(),
            })?;
        
        // Track active requests
        self.active_requests.fetch_add(1, Ordering::SeqCst);
        let _guard = ActiveRequestGuard {
            counter: Arc::clone(&self.active_requests),
        };
        
        let start = Instant::now();
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        
        // Get adaptive timeout if enabled
        let actual_timeout = self.get_adaptive_timeout(operation_name, timeout_duration).await;
        
        match timeout(actual_timeout, operation()).await {
            Ok(result) => {
                let duration = start.elapsed();
                self.record_success(operation_name, duration).await;
                circuit_breaker.record_success();
                Ok(result)
            }
            Err(_) => {
                let duration = start.elapsed();
                self.record_timeout(operation_name, actual_timeout, duration).await;
                circuit_breaker.record_failure();
                
                Err(TimeoutError::OperationTimeout {
                    operation: operation_name.to_string(),
                    timeout_duration: actual_timeout,
                    actual_duration: duration,
                })
            }
        }
    }

    /// Get or create circuit breaker for an operation
    fn get_or_create_circuit_breaker(&self, operation_name: &str) -> Arc<CircuitBreaker> {
        self.circuit_breakers
            .entry(operation_name.to_string())
            .or_insert_with(|| {
                let config = self.config.read();
                Arc::new(CircuitBreaker::new(config.circuit_breaker.clone()))
            })
            .clone()
    }

    /// Get adaptive timeout based on historical data
    async fn get_adaptive_timeout(&self, operation_name: &str, base_timeout: Duration) -> Duration {
        let config = self.config.read();
        if !config.adaptive.enabled {
            return base_timeout;
        }
        
        let samples = self.duration_samples.read();
        let operation_samples: Vec<_> = samples
            .iter()
            .filter(|s| s.operation == operation_name)
            .map(|s| s.duration)
            .collect();
        
        if operation_samples.len() < 10 {
            return base_timeout; // Not enough data
        }
        
        let percentile_duration = calculate_percentile(&operation_samples, config.adaptive.target_percentile);
        
        // Add 20% buffer to the percentile
        let adaptive_timeout = percentile_duration.mul_f32(1.2);
        
        // Clamp to min/max bounds
        adaptive_timeout
            .max(config.adaptive.min_timeout)
            .min(config.adaptive.max_timeout)
    }

    /// Record successful operation
    async fn record_success(&self, operation_name: &str, duration: Duration) {
        // Update samples
        {
            let mut samples = self.duration_samples.write();
            if samples.len() >= self.config.read().adaptive.sample_window_size {
                samples.pop_front();
            }
            samples.push_back(DurationSample {
                duration,
                timestamp: Instant::now(),
                operation: operation_name.to_string(),
            });
        }
        
        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.total_requests += 1;
        
        // Update min/max
        if duration < metrics.min_request_duration {
            metrics.min_request_duration = duration;
        }
        if duration > metrics.max_request_duration {
            metrics.max_request_duration = duration;
        }
        
        // Update average (exponential moving average)
        let alpha = 0.1; // Smoothing factor
        metrics.avg_request_duration = 
            alpha * duration.as_secs_f64() + (1.0 - alpha) * metrics.avg_request_duration;
        
        debug!(
            operation = operation_name,
            duration = ?duration,
            "Operation completed successfully"
        );
    }

    /// Record timeout occurrence
    async fn record_timeout(&self, operation_name: &str, timeout_duration: Duration, actual_duration: Duration) {
        self.total_timeouts.fetch_add(1, Ordering::SeqCst);
        
        // Update timeout count for specific operation
        self.metrics.read()
            .timeouts_by_operation
            .entry(operation_name.to_string())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        
        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.total_timeouts += 1;
        metrics.timeout_rate = 
            metrics.total_timeouts as f64 / metrics.total_requests.max(1) as f64;
        
        error!(
            operation = operation_name,
            timeout_duration = ?timeout_duration,
            actual_duration = ?actual_duration,
            total_timeouts = metrics.total_timeouts,
            "Operation timed out"
        );
    }

    /// Start adaptive timeout recalculation task
    fn start_adaptive_timeout_task(&self) {
        let config = Arc::clone(&self.config);
        let samples = Arc::clone(&self.duration_samples);
        let mut shutdown_rx = self.shutdown_signal.subscribe();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                config.read().adaptive.recalculation_interval
            );
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Recalculate percentiles
                        let samples_vec: Vec<_> = samples.read()
                            .iter()
                            .map(|s| s.duration)
                            .collect();
                        
                        if samples_vec.len() >= 50 {
                            debug!("Recalculating adaptive timeouts with {} samples", samples_vec.len());
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Shutting down adaptive timeout task");
                        break;
                    }
                }
            }
        });
    }

    /// Start metrics aggregation task
    fn start_metrics_aggregation_task(&self) {
        let metrics = Arc::clone(&self.metrics);
        let samples = Arc::clone(&self.duration_samples);
        let total_requests = Arc::clone(&self.total_requests);
        let total_timeouts = Arc::clone(&self.total_timeouts);
        let active_requests = Arc::clone(&self.active_requests);
        let mut shutdown_rx = self.shutdown_signal.subscribe();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let samples_vec: Vec<_> = samples.read()
                            .iter()
                            .map(|s| s.duration)
                            .collect();
                        
                        if !samples_vec.is_empty() {
                            let mut metrics = metrics.write();
                            metrics.p50_duration = calculate_percentile(&samples_vec, 50);
                            metrics.p95_duration = calculate_percentile(&samples_vec, 95);
                            metrics.p99_duration = calculate_percentile(&samples_vec, 99);
                            metrics.total_requests = total_requests.load(Ordering::SeqCst);
                            metrics.total_timeouts = total_timeouts.load(Ordering::SeqCst);
                            metrics.active_requests = active_requests.load(Ordering::SeqCst);
                            
                            if metrics.total_requests > 0 {
                                metrics.timeout_rate = 
                                    metrics.total_timeouts as f64 / metrics.total_requests as f64;
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Shutting down metrics aggregation task");
                        break;
                    }
                }
            }
        });
    }

    /// Get current metrics snapshot
    pub async fn get_metrics(&self) -> TimeoutMetrics {
        self.metrics.read().clone()
    }

    /// Reset all metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write();
        *metrics = TimeoutMetrics {
            total_requests: 0,
            total_timeouts: 0,
            timeouts_by_operation: DashMap::new(),
            avg_request_duration: 0.0,
            max_request_duration: Duration::from_secs(0),
            min_request_duration: Duration::from_secs(u64::MAX),
            timeout_rate: 0.0,
            p50_duration: Duration::from_secs(0),
            p95_duration: Duration::from_secs(0),
            p99_duration: Duration::from_secs(0),
            circuit_breaker_state: CircuitState::Closed,
            active_requests: 0,
        };
        
        self.total_requests.store(0, Ordering::SeqCst);
        self.total_timeouts.store(0, Ordering::SeqCst);
        self.duration_samples.write().clear();
        
        info!("All timeout metrics have been reset");
    }

    /// Update configuration dynamically
    pub async fn update_config(&self, new_config: TimeoutConfig) {
        *self.config.write() = new_config;
        info!("Timeout configuration updated");
    }

    /// Get current configuration
    pub fn config(&self) -> TimeoutConfig {
        self.config.read().clone()
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) {
        info!("Initiating timeout manager shutdown");
        let _ = self.shutdown_signal.send(());
        
        // Wait for active requests to complete (with timeout)
        let shutdown_timeout = Duration::from_secs(30);
        let start = Instant::now();
        
        while self.active_requests.load(Ordering::SeqCst) > 0 {
            if start.elapsed() > shutdown_timeout {
                warn!("Shutdown timeout exceeded, forcing shutdown");
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        info!("Timeout manager shutdown complete");
    }
}

// Guard to automatically decrement active request counter
struct ActiveRequestGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

// ==================== Error Types ====================

#[derive(Debug, Clone, Serialize, Error)]
pub enum TimeoutError {
    #[error("Operation '{operation}' timed out after {timeout_duration:?} (actual: {actual_duration:?})")]
    OperationTimeout {
        operation: String,
        timeout_duration: Duration,
        actual_duration: Duration,
    },
    
    #[error("Circuit breaker is open for operation '{operation}'")]
    CircuitBreakerOpen {
        operation: String,
    },
    
    #[error("Rate limit exceeded for operation '{operation}'")]
    RateLimited {
        operation: String,
    },
}

unsafe impl Send for TimeoutError {}
unsafe impl Sync for TimeoutError {}

// ==================== Middleware ====================

/// Global timeout middleware for all HTTP requests
pub async fn timeout_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    
    let span = Span::current();
    span.record("request_id", &request_id.as_str());
    
    let path = request.uri().path().to_string();
    let method = request.method().clone();
    
    // Get timeout manager from app state
    let timeout_manager = &state.timeout_manager;
    
    // Determine timeout based on route
    let timeout_duration = determine_request_timeout(&path, &timeout_manager.config());
    
    let start = Instant::now();
    
    match timeout(timeout_duration, next.run(request)).await {
        Ok(response) => {
            let duration = start.elapsed();
            
            // Add timing headers
            let mut response = response;
            response.headers_mut().insert(
                "x-response-time",
                duration.as_millis().to_string().parse().unwrap(),
            );
            response.headers_mut().insert(
                "x-request-id",
                request_id.parse().unwrap(),
            );
            
            debug!(
                request_id = %request_id,
                method = %method,
                path = %path,
                duration = ?duration,
                status = response.status().as_u16(),
                "Request completed successfully"
            );
            
            response
        }
        Err(_) => {
            let duration = start.elapsed();
            
            warn!(
                request_id = %request_id,
                method = %method,
                path = %path,
                duration = ?duration,
                timeout = ?timeout_duration,
                "Request timed out"
            );
            
            create_timeout_response(&request_id, timeout_duration, &path)
        }
    }
}

/// Determine appropriate timeout based on request path
fn determine_request_timeout(path: &str, config: &TimeoutConfig) -> Duration {
    match path {
        p if p.starts_with("/health") || p.starts_with("/metrics") => config.health_check,
        p if p.starts_with("/api/v1/auth") || p.starts_with("/auth") => config.auth,
        p if p.starts_with("/api/v1/slack") => config.slack_handler,
        p if p.starts_with("/api/v1/ml/inference") => config.ml_inference,
        p if p.starts_with("/api/v1/ml") => config.ml_pipeline,
        p if p.starts_with("/api/v1/analytics") => config.analytics,
        p if p.starts_with("/webhooks") => config.webhook_processing,
        p if p.contains("/upload") || p.contains("/file") => config.file_upload,
        p if p.starts_with("/api/v1/db") => config.database,
        _ => config.default_request,
    }
}

/// Create standardized timeout response
fn create_timeout_response(request_id: &str, timeout_duration: Duration, path: &str) -> Response {
    use axum::response::Json;
    
    #[derive(Serialize)]
    struct TimeoutResponse {
        error: String,
        error_code: String,
        message: String,
        request_id: String,
        path: String,
        timeout_duration_ms: u128,
        timestamp: String,
        retry_after_seconds: u64,
    }
    
    let retry_after = (timeout_duration.as_secs() as f64 * 1.5).ceil() as u64;
    
    let response = TimeoutResponse {
        error: "request_timeout".to_string(),
        error_code: "TIMEOUT_ERROR".to_string(),
        message: format!(
            "Request timed out after {} seconds. The server is under heavy load or the operation is taking longer than expected. Please retry after {} seconds.",
            timeout_duration.as_secs(),
            retry_after
        ),
        request_id: request_id.to_string(),
        path: path.to_string(),
        timeout_duration_ms: timeout_duration.as_millis(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        retry_after_seconds: retry_after,
    };
    
    let mut response = (StatusCode::REQUEST_TIMEOUT, Json(response)).into_response();
    
    // Add retry-after header
    response.headers_mut().insert(
        "retry-after",
        retry_after.to_string().parse().unwrap(),
    );
    response.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap(),
    );
    
    response
}

// ==================== Utility Functions ====================

/// Calculate percentile from a slice of durations
fn calculate_percentile(durations: &[Duration], percentile: u8) -> Duration {
    if durations.is_empty() {
        return Duration::from_secs(0);
    }
    
    let mut sorted: Vec<_> = durations.to_vec();
    sorted.sort_unstable();
    
    let index = ((percentile as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

// ==================== Macros ====================

/// Execute database operations with timeout and automatic retry
#[macro_export]
macro_rules! db_timeout {
    ($timeout_manager:expr, $operation:expr) => {{
        use std::time::Duration;
        
        let mut retries = 0;
        const MAX_RETRIES: u32 = 3;
        let mut _last_error = None;
        
        loop {
            match $timeout_manager
                .execute_with_timeout(
                    || async { $operation },
                    $timeout_manager.config().database_query,
                    "database_query",
                )
                .await
            {
                Ok(result) => break Ok(result),
                Err(e) => {
                    retries += 1;
                    _last_error = Some(e);
                    
                    if retries >= MAX_RETRIES {
                        break Err(sqlx::Error::PoolTimedOut);
                    }
                    
                    // Exponential backoff
                    let backoff = Duration::from_millis(100 * (2_u64.pow(retries - 1)));
                    tokio::time::sleep(backoff).await;
                }
            }
        }
    }};
    ($timeout_manager:expr, $operation:expr, $operation_name:expr) => {{
        $timeout_manager
            .execute_with_timeout(
                || async { $operation },
                $timeout_manager.config().database,
                $operation_name,
            )
            .await
            .map_err(|e| sqlx::Error::Protocol(e.to_string()))
    }};
}

/// Execute ML operations with timeout
#[macro_export]
macro_rules! ml_timeout {
    ($timeout_manager:expr, $operation:expr, $operation_name:expr) => {{
        $timeout_manager
            .execute_with_timeout(
                || async { $operation },
                $timeout_manager.config().ml_inference,
                $operation_name,
            )
            .await
            .map_err(|e| anyhow::anyhow!("ML operation '{}' failed: {}", $operation_name, e))
    }};
    ($timeout_manager:expr, $operation:expr) => {{
        ml_timeout!($timeout_manager, $operation, "ml_inference")
    }};
}

/// Execute Slack operations with timeout
#[macro_export]
macro_rules! slack_timeout {
    ($timeout_manager:expr, $operation:expr) => {{
        $timeout_manager
            .execute_with_timeout(
                || async { $operation },
                $timeout_manager.config().slack_api,
                "slack_api_call",
            )
            .await
            .map_err(|e| anyhow::anyhow!("Slack API call failed: {}", e))
    }};
    ($timeout_manager:expr, $operation:expr, $custom_timeout:expr) => {{
        $timeout_manager
            .execute_with_timeout(
                || async { $operation },
                $custom_timeout,
                "slack_custom_operation",
            )
            .await
            .map_err(|e| anyhow::anyhow!("Slack operation failed: {}", e))
    }};
}

/// Execute webhook operations with timeout
#[macro_export]
macro_rules! webhook_timeout {
    ($timeout_manager:expr, $operation:expr) => {{
        $timeout_manager
            .execute_with_timeout(
                || async { $operation },
                $timeout_manager.config().webhook_processing,
                "webhook_processing",
            )
            .await
            .map_err(|e| anyhow::anyhow!("Webhook processing failed: {}", e))
    }};
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_timeout_manager_success() {
        let config = TimeoutConfig::default();
        let manager = TimeoutManager::new(config);
        
        let result = manager
            .execute_with_timeout(
                || async { 42 },
                Duration::from_millis(100),
                "test_operation",
            )
            .await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_timeouts, 0);
        assert_eq!(metrics.total_requests, 1);
    }

    #[tokio::test]
    async fn test_timeout_manager_timeout() {
        let config = TimeoutConfig::default();
        let manager = TimeoutManager::new(config);
        
        let result = manager
            .execute_with_timeout(
                || async {
                    sleep(Duration::from_millis(200)).await;
                    42
                },
                Duration::from_millis(50),
                "test_timeout",
            )
            .await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TimeoutError::OperationTimeout { .. }));
        
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_timeouts, 1);
        assert_eq!(metrics.total_requests, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_failures() {
        let mut config = TimeoutConfig::default();
        config.circuit_breaker.failure_threshold = 3;
        
        let manager = TimeoutManager::new(config);
        
        // Cause 3 timeouts to open circuit
        for _ in 0..3 {
            let _ = manager
                .execute_with_timeout(
                    || async {
                        sleep(Duration::from_millis(200)).await;
                        42
                    },
                    Duration::from_millis(50),
                    "test_circuit",
                )
                .await;
        }
        
        // Next request should fail immediately due to open circuit
        let result = manager
            .execute_with_timeout(
                || async { 42 },
                Duration::from_millis(100),
                "test_circuit",
            )
            .await;
        
        assert!(matches!(result.unwrap_err(), TimeoutError::CircuitBreakerOpen { .. }));
    }

    #[tokio::test]
    async fn test_adaptive_timeout() {
        let mut config = TimeoutConfig::default();
        config.adaptive.enabled = true;
        config.adaptive.sample_window_size = 10;
        
        let manager = TimeoutManager::new(config);
        
        // Generate some sample data
        for i in 0..15 {
            let duration = Duration::from_millis(10 + i * 5);
            manager.record_success("test_adaptive", duration).await;
        }
        
        // Check that adaptive timeout is calculated
        let timeout = manager
            .get_adaptive_timeout("test_adaptive", Duration::from_secs(1))
            .await;
        
        // Should be based on percentile of recorded durations
        assert!(timeout < Duration::from_secs(1));
        assert!(timeout > Duration::from_millis(10));
    }

    #[tokio::test]
    async fn test_metrics_calculation() {
        let config = TimeoutConfig::default();
        let manager = TimeoutManager::new(config);
        
        // Generate mixed results
        for i in 0..10 {
            if i % 3 == 0 {
                // Timeout
                let _ = manager
                    .execute_with_timeout(
                        || async {
                            sleep(Duration::from_millis(200)).await;
                            42
                        },
                        Duration::from_millis(50),
                        "test_metrics",
                    )
                    .await;
            } else {
                // Success
                let _ = manager
                    .execute_with_timeout(
                        || async { 42 },
                        Duration::from_millis(100),
                        "test_metrics",
                    )
                    .await;
            }
        }
        
        // Allow time for metrics aggregation
        sleep(Duration::from_millis(100)).await;
        
        let metrics = manager.get_metrics().await;
        assert!(metrics.total_requests > 0);
        assert!(metrics.total_timeouts > 0);
        assert!(metrics.timeout_rate > 0.0 && metrics.timeout_rate < 1.0);
    }

    #[test]
    fn test_timeout_config_from_env() {
        std::env::set_var("TIMEOUT_DEFAULT_REQUEST_SECS", "45");
        std::env::set_var("TIMEOUT_DATABASE_SECS", "15");
        std::env::set_var("ADAPTIVE_TIMEOUT_ENABLED", "false");
        
        let config = TimeoutConfig::from_env();
        assert_eq!(config.default_request, Duration::from_secs(45));
        assert_eq!(config.database, Duration::from_secs(15));
        assert!(!config.adaptive.enabled);
        
        // Cleanup
        std::env::remove_var("TIMEOUT_DEFAULT_REQUEST_SECS");
        std::env::remove_var("TIMEOUT_DATABASE_SECS");
        std::env::remove_var("ADAPTIVE_TIMEOUT_ENABLED");
    }

    #[test]
    fn test_percentile_calculation() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
            Duration::from_millis(60),
            Duration::from_millis(70),
            Duration::from_millis(80),
            Duration::from_millis(90),
            Duration::from_millis(100),
        ];
        
        let p50 = calculate_percentile(&durations, 50);
        assert_eq!(p50, Duration::from_millis(50));
        
        let p95 = calculate_percentile(&durations, 95);
        assert_eq!(p95, Duration::from_millis(100));
        
        let p99 = calculate_percentile(&durations, 99);
        assert_eq!(p99, Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_concurrent_requests() {
        let config = TimeoutConfig::default();
        let manager = Arc::new(TimeoutManager::new(config));
        
        let mut handles = vec![];
        
        for i in 0..10 {
            let manager_clone = Arc::clone(&manager);
            let handle = tokio::spawn(async move {
                manager_clone
                    .execute_with_timeout(
                        || async move {
                            sleep(Duration::from_millis(10)).await;
                            i
                        },
                        Duration::from_millis(100),
                        &format!("concurrent_{}", i),
                    )
                    .await
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
        
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.total_requests, 10);
        assert_eq!(metrics.total_timeouts, 0);
    }

    #[tokio::test]
    async fn test_graceful_shutdown() {
        let config = TimeoutConfig::default();
        let manager = Arc::new(TimeoutManager::new(config));
        
        // Start some operations
        let manager_clone = Arc::clone(&manager);
        let handle = tokio::spawn(async move {
            let _ = manager_clone
                .execute_with_timeout(
                    || async {
                        sleep(Duration::from_millis(100)).await;
                        42
                    },
                    Duration::from_secs(1),
                    "shutdown_test",
                )
                .await;
        });
        
        // Give it time to start
        sleep(Duration::from_millis(10)).await;
        
        // Shutdown
        manager.shutdown().await;
        
        // Operation should complete
        let _ = handle.await;
    }
}