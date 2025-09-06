//! # Interstice Platform Adapters
//! 
//! Production-ready adapter system for integrating with various work platforms.
//! Provides a unified interface for platform communication with advanced features
//! including health monitoring, rate limiting, and intelligent routing.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};


use dashmap::DashMap;
use futures::future::{join_all};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

pub mod traits;
pub mod slack;

// Platform-specific adapters
#[cfg(feature = "slack")]
pub use slack::SlackAdapter;

#[cfg(feature = "github")]
pub mod github;
#[cfg(feature = "github")]
pub use github::GitHubAdapter;

#[cfg(feature = "teams")]
pub mod teams;
#[cfg(feature = "teams")]
pub use teams::TeamsAdapter;

#[cfg(feature = "jira")]
pub mod jira;
#[cfg(feature = "jira")]
pub use jira::JiraAdapter;

// Re-export core traits
pub use traits::{
    AdapterCapabilities, AdapterError, AdapterMetadata, AuthCredentials, AuthToken,
    ChannelInfo, ConfigSchema, EventType, ExtendedAdapter, HealthState, HealthStatus,
    HistoryParams, ItemId, ItemResponse, ItemType, PlatformAdapter, PlatformEvent,
    PlatformResponse, RateLimitStatus, ResponseContent, ResponseOptions, ResponseTarget,
    SearchQuery, SearchResults, Subscription, SubscriptionHandle, UpdateItemRequest,
    UserInfo,
};

use interstice_core::{Platform, ProcessedData, Result as CoreResult};

/// Adapter manager errors
#[derive(Error, Debug)]
pub enum ManagerError {
    /// Adapter not found for platform
    #[error("No adapter registered for platform: {0}")]
    AdapterNotFound(Platform),
    
    /// Adapter registration failed
    #[error("Failed to register adapter: {0}")]
    RegistrationFailed(String),
    
    /// Health check failed
    #[error("Health check failed for {platform}: {message}")]
    HealthCheckFailed {
        /// Platform that failed
        platform: Platform,
        /// Error message
        message: String,
    },
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Multiple adapters error
    #[error("Multiple adapters registered for platform: {0}")]
    DuplicateAdapter(Platform),
}

/// Advanced adapter manager with intelligent routing and monitoring
pub struct AdapterManager {
    /// Registered adapters
    adapters: Arc<DashMap<Platform, Arc<dyn PlatformAdapter>>>,
    
    /// Health status cache
    health_cache: Arc<DashMap<Platform, CachedHealth>>,
    
    /// Performance metrics
    metrics: Arc<ManagerMetrics>,
    
    /// Configuration
    config: ManagerConfig,
    
    /// Rate limiter for global operations
    rate_limiter: Arc<GlobalRateLimiter>,
    
    /// Circuit breakers per platform
    circuit_breakers: Arc<DashMap<Platform, CircuitBreaker>>,
    
    /// Event bus for cross-adapter communication
    event_bus: Arc<EventBus>,
}

impl AdapterManager {
    /// Create a new adapter manager with default configuration
    pub fn new() -> Self {
        Self::with_config(ManagerConfig::default())
    }
    
    /// Create a new adapter manager with custom configuration
    pub fn with_config(config: ManagerConfig) -> Self {
        let manager = Self {
            adapters: Arc::new(DashMap::new()),
            health_cache: Arc::new(DashMap::new()),
            metrics: Arc::new(ManagerMetrics::new()),
            config: config.clone(),
            rate_limiter: Arc::new(GlobalRateLimiter::new(
                config.global_rate_limit,
                Duration::from_secs(60),
            )),
            circuit_breakers: Arc::new(DashMap::new()),
            event_bus: Arc::new(EventBus::new()),
        };
        
        // Start background tasks
        manager.start_background_tasks();
        
        manager
    }
    
    /// Register a platform adapter
    #[instrument(skip(self, adapter))]
    pub fn register(&self, adapter: Arc<dyn PlatformAdapter>) -> Result<(), ManagerError> {
        let platform = adapter.platform();
        
        // Check for duplicates
        if self.adapters.contains_key(&platform) && !self.config.allow_override {
            return Err(ManagerError::DuplicateAdapter(platform));
        }
        
        // Initialize circuit breaker
        self.circuit_breakers.insert(
            platform,
            CircuitBreaker::new(self.config.circuit_breaker_config.clone()),
        );
        
        // Register adapter
        self.adapters.insert(platform, adapter.clone());
        
        // Log registration
        info!("Registered adapter for platform: {:?}", platform);
        self.metrics.increment_registration(platform);
        
        // Notify event bus
        self.event_bus.publish(ManagerEvent::AdapterRegistered { platform });
        
        Ok(())
    }
    
    /// Unregister a platform adapter
    pub fn unregister(&self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        let adapter = self.adapters.remove(&platform).map(|(_, v)| v);
        
        if adapter.is_some() {
            self.health_cache.remove(&platform);
            self.circuit_breakers.remove(&platform);
            self.event_bus.publish(ManagerEvent::AdapterUnregistered { platform });
            info!("Unregistered adapter for platform: {:?}", platform);
        }
        
        adapter
    }
    
    /// Get an adapter for a specific platform
    pub fn get(&self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.get(&platform).map(|entry| entry.clone())
    }
    
    /// Get adapter with circuit breaker protection
    #[instrument(skip(self))]
    pub async fn get_with_protection(
        &self,
        platform: Platform,
    ) -> Result<Arc<dyn PlatformAdapter>, ManagerError> {
        // Check circuit breaker
        if let Some(breaker) = self.circuit_breakers.get(&platform) {
            if !breaker.is_closed() {
                self.metrics.increment_circuit_open(platform);
                return Err(ManagerError::HealthCheckFailed {
                    platform,
                    message: "Circuit breaker is open".to_string(),
                });
            }
        }
        
        // Get adapter
        self.get(platform)
            .ok_or_else(|| ManagerError::AdapterNotFound(platform))
    }
    
    /// List all registered platforms
    pub fn platforms(&self) -> Vec<Platform> {
        self.adapters
            .iter()
            .map(|entry| *entry.key())
            .collect()
    }
    
    /// Get the count of registered adapters
    pub fn len(&self) -> usize {
        self.adapters.len()
    }
    
    /// Check if there are no adapters registered
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
    
    /// Get all adapter metadata
    pub fn metadata(&self) -> HashMap<Platform, AdapterMetadata> {
        self.adapters
            .iter()
            .map(|entry| (*entry.key(), entry.value().metadata()))
            .collect()
    }
    
    /// Perform health check on all adapters
    #[instrument(skip(self))]
    pub async fn health_check_all(&self) -> HashMap<Platform, HealthStatus> {
        let futures: Vec<_> = self
            .adapters
            .iter()
            .map(|entry| {
                let platform = *entry.key();
                let adapter = entry.clone();
                async move {
                    let health = match adapter.health_check().await {
                        Ok(status) => status,
                        Err(e) => HealthStatus {
                            status: HealthState::Unhealthy,
                            message: Some(e.to_string()),
                            last_successful_event: None,
                            error_count: 0,
                            metrics: HashMap::new(),
                        },
                    };
                    
                    // Update cache
                    self.update_health_cache(platform, &health);
                    
                    // Update circuit breaker
                    self.update_circuit_breaker(platform, &health);
                    
                    (platform, health)
                }
            })
            .collect();
        
        let results = join_all(futures).await;
        results.into_iter().collect()
    }
    
    /// Get cached health status
    pub fn get_health(&self, platform: Platform) -> Option<HealthStatus> {
        self.health_cache
            .get(&platform)
            .and_then(|cached| {
                if cached.is_valid(self.config.health_cache_ttl) {
                    Some(cached.status.clone())
                } else {
                    None
                }
            })
    }
    
    /// Process event with automatic platform detection and routing
    #[instrument(skip(self, event))]
    pub async fn process_event(&self, event: PlatformEvent) -> CoreResult<ProcessedData> {
        let platform = event.platform;
        
        // Get adapter with protection
        let adapter = self
            .get_with_protection(platform)
            .await
            .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        // Check rate limit
        self.rate_limiter
            .acquire()
            .await
            .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        // Process with metrics
        let start = Instant::now();
        let result = adapter.process_event(event).await;
        let duration = start.elapsed();
        
        // Record metrics
        self.metrics.record_processing(platform, duration, result.is_ok());
        
        // Update circuit breaker
        if let Some(mut breaker) = self.circuit_breakers.get_mut(&platform) {
            if result.is_ok() {
                breaker.on_success();
            } else {
                breaker.on_failure();
            }
        }
        
        result
    }
    
    /// Send response to the appropriate platform
    pub async fn send_response(
        &self,
        platform: Platform,
        response: PlatformResponse,
    ) -> CoreResult<()> {
        let adapter = self
            .get_with_protection(platform)
            .await
            .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        adapter.send_response(response).await
    }
    
    /// Broadcast a response to multiple platforms
    pub async fn broadcast(
        &self,
        platforms: Vec<Platform>,
        response: PlatformResponse,
    ) -> HashMap<Platform, CoreResult<()>> {
        let futures: Vec<_> = platforms
            .into_iter()
            .map(|platform| {
                let response = response.clone();
                async move {
                    let result = self.send_response(platform, response).await;
                    (platform, result)
                }
            })
            .collect();
        
        join_all(futures).await.into_iter().collect()
    }
    
    /// Find the best adapter for a given capability
    pub fn find_by_capability(&self, capability: &str) -> Vec<Platform> {
        self.adapters
            .iter()
            .filter_map(|entry| {
                let metadata = entry.value().metadata();
                if self.has_capability(&metadata.capabilities, capability) {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> ManagerMetricsSnapshot {
        self.metrics.snapshot()
    }
    
    /// Subscribe to manager events
    pub fn subscribe(&self) -> EventSubscription {
        self.event_bus.subscribe()
    }
    
    /// Update health cache
    fn update_health_cache(&self, platform: Platform, health: &HealthStatus) {
        self.health_cache.insert(
            platform,
            CachedHealth {
                status: health.clone(),
                cached_at: Instant::now(),
            },
        );
    }
    
    /// Update circuit breaker based on health
    fn update_circuit_breaker(&self, platform: Platform, health: &HealthStatus) {
        if let Some(mut breaker) = self.circuit_breakers.get_mut(&platform) {
            match health.status {
                HealthState::Healthy => breaker.on_success(),
                HealthState::Degraded => {
                    // Don't trip on degraded, but don't reset either
                }
                HealthState::Unhealthy | HealthState::Unknown => breaker.on_failure(),
            }
        }
    }
    
    /// Check if capabilities include a specific capability
    fn has_capability(&self, capabilities: &AdapterCapabilities, capability: &str) -> bool {
        match capability {
            "real_time" => capabilities.real_time,
            "webhooks" => capabilities.webhooks,
            "polling" => capabilities.polling,
            "bidirectional" => capabilities.bidirectional,
            "file_upload" => capabilities.file_upload,
            "rich_formatting" => capabilities.rich_formatting,
            "threading" => capabilities.threading,
            "reactions" => capabilities.reactions,
            "search" => capabilities.search,
            "user_presence" => capabilities.user_presence,
            "custom_fields" => capabilities.custom_fields,
            "bulk_operations" => capabilities.bulk_operations,
            _ => false,
        }
    }
    
    /// Start background maintenance tasks
    fn start_background_tasks(&self) {
        // Health check task
        if self.config.enable_health_checks {
            let manager = self.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(manager.config.health_check_interval);
                loop {
                    interval.tick().await;
                    let _ = manager.health_check_all().await;
                }
            });
        }
        
        // Metrics aggregation task
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                metrics.aggregate();
            }
        });
        
        // Circuit breaker reset task
        let breakers = self.circuit_breakers.clone();
        let config = self.config.circuit_breaker_config.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.reset_timeout);
            loop {
                interval.tick().await;
                for mut breaker in breakers.iter_mut() {
                    breaker.try_reset();
                }
            }
        });
    }
}

impl Clone for AdapterManager {
    fn clone(&self) -> Self {
        Self {
            adapters: self.adapters.clone(),
            health_cache: self.health_cache.clone(),
            metrics: self.metrics.clone(),
            config: self.config.clone(),
            rate_limiter: self.rate_limiter.clone(),
            circuit_breakers: self.circuit_breakers.clone(),
            event_bus: self.event_bus.clone(),
        }
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerConfig {
    /// Allow overriding existing adapters
    pub allow_override: bool,
    
    /// Enable automatic health checks
    pub enable_health_checks: bool,
    
    /// Health check interval
    pub health_check_interval: Duration,
    
    /// Health cache TTL
    pub health_cache_ttl: Duration,
    
    /// Global rate limit (requests per minute)
    pub global_rate_limit: usize,
    
    /// Circuit breaker configuration
    pub circuit_breaker_config: CircuitBreakerConfig,
    
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Maximum concurrent operations
    pub max_concurrent_operations: usize,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            allow_override: false,
            enable_health_checks: true,
            health_check_interval: Duration::from_secs(300), // 5 minutes
            health_cache_ttl: Duration::from_secs(60),
            global_rate_limit: 1000,
            circuit_breaker_config: CircuitBreakerConfig::default(),
            enable_metrics: true,
            max_concurrent_operations: 100,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    
    /// Success threshold to close circuit
    pub success_threshold: u32,
    
    /// Timeout before attempting reset
    pub reset_timeout: Duration,
    
    /// Half-open test requests
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            reset_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

/// Circuit breaker implementation
struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    config: CircuitBreakerConfig,
    failure_count: Arc<std::sync::atomic::AtomicU32>,
    success_count: Arc<std::sync::atomic::AtomicU32>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            config,
            failure_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            success_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
        }
    }
    
    fn is_closed(&self) -> bool {
        futures::executor::block_on(async {
            matches!(*self.state.read().await, CircuitState::Closed)
        })
    }
    
    fn on_success(&mut self) {
        self.success_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        futures::executor::block_on(async {
            let mut state = self.state.write().await;
            if matches!(*state, CircuitState::HalfOpen) {
                let count = self.success_count.load(std::sync::atomic::Ordering::Relaxed);
                if count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failure_count.store(0, std::sync::atomic::Ordering::Relaxed);
                    self.success_count.store(0, std::sync::atomic::Ordering::Relaxed);
                }
            }
        });
    }
    
    fn on_failure(&mut self) {
        let count = self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        
        futures::executor::block_on(async {
            *self.last_failure_time.write().await = Some(Instant::now());
            
            let mut state = self.state.write().await;
            if count >= self.config.failure_threshold && matches!(*state, CircuitState::Closed) {
                *state = CircuitState::Open;
            }
        });
    }
    
    fn try_reset(&mut self) {
        futures::executor::block_on(async {
            let last_failure = *self.last_failure_time.read().await;
            if let Some(last_time) = last_failure {
                if last_time.elapsed() > self.config.reset_timeout {
                    let mut state = self.state.write().await;
                    if matches!(*state, CircuitState::Open) {
                        *state = CircuitState::HalfOpen;
                        self.success_count.store(0, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
        });
    }
}

#[derive(Debug, Clone)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Cached health status
struct CachedHealth {
    status: HealthStatus,
    cached_at: Instant,
}

impl CachedHealth {
    fn is_valid(&self, ttl: Duration) -> bool {
        self.cached_at.elapsed() < ttl
    }
}

/// Global rate limiter
struct GlobalRateLimiter {
    semaphore: Arc<Semaphore>,
}

impl GlobalRateLimiter {
    fn new(rate_per_minute: usize, _window: Duration) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(rate_per_minute)),
        }
    }
    
    async fn acquire(&self) -> Result<(), String> {
        self.semaphore
            .acquire()
            .await
            .map(|permit| {
                std::mem::forget(permit); // Don't hold the permit
                ()
            })
            .map_err(|e| e.to_string())
    }
}

/// Manager metrics
struct ManagerMetrics {
    registrations: Arc<DashMap<Platform, u64>>,
    processing_times: Arc<DashMap<Platform, Vec<Duration>>>,
    success_count: Arc<DashMap<Platform, u64>>,
    error_count: Arc<DashMap<Platform, u64>>,
    circuit_opens: Arc<DashMap<Platform, u64>>,
}

impl ManagerMetrics {
    fn new() -> Self {
        Self {
            registrations: Arc::new(DashMap::new()),
            processing_times: Arc::new(DashMap::new()),
            success_count: Arc::new(DashMap::new()),
            error_count: Arc::new(DashMap::new()),
            circuit_opens: Arc::new(DashMap::new()),
        }
    }
    
    fn increment_registration(&self, platform: Platform) {
        self.registrations
            .entry(platform)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
    
    fn record_processing(&self, platform: Platform, duration: Duration, success: bool) {
        self.processing_times
            .entry(platform)
            .and_modify(|times| {
                times.push(duration);
                if times.len() > 1000 {
                    times.drain(0..100);
                }
            })
            .or_insert_with(|| vec![duration]);
        
        if success {
            self.success_count
                .entry(platform)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        } else {
            self.error_count
                .entry(platform)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }
    
    fn increment_circuit_open(&self, platform: Platform) {
        self.circuit_opens
            .entry(platform)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }
    
    fn aggregate(&self) {
        // Periodic aggregation logic
        debug!("Aggregating metrics");
    }
    
    fn snapshot(&self) -> ManagerMetricsSnapshot {
        let mut platform_metrics = HashMap::new();
        
        for platform in self.registrations.iter() {
            let platform_key = *platform.key();
            
            let avg_processing_time = self
                .processing_times
                .get(&platform_key)
                .map(|times| {
                    if times.is_empty() {
                        Duration::from_secs(0)
                    } else {
                        let sum: Duration = times.iter().sum();
                        sum / times.len() as u32
                    }
                })
                .unwrap_or_default();
            
            platform_metrics.insert(
                platform_key,
                PlatformMetrics {
                    registrations: *platform.value(),
                    success_count: self.success_count.get(&platform_key).map(|v| *v).unwrap_or(0),
                    error_count: self.error_count.get(&platform_key).map(|v| *v).unwrap_or(0),
                    avg_processing_time,
                    circuit_opens: self.circuit_opens.get(&platform_key).map(|v| *v).unwrap_or(0),
                },
            );
        }
        
        ManagerMetricsSnapshot {
            total_platforms: self.registrations.len(),
            platform_metrics,
        }
    }
}

/// Manager metrics snapshot
#[derive(Debug, Clone, Serialize)]
pub struct ManagerMetricsSnapshot {
    /// Total number of platforms
    pub total_platforms: usize,
    
    /// Per-platform metrics
    pub platform_metrics: HashMap<Platform, PlatformMetrics>,
}

/// Platform-specific metrics
#[derive(Debug, Clone, Serialize)]
pub struct PlatformMetrics {
    /// Number of registrations
    pub registrations: u64,
    
    /// Success count
    pub success_count: u64,
    
    /// Error count
    pub error_count: u64,
    
    /// Average processing time
    pub avg_processing_time: Duration,
    
    /// Circuit breaker opens
    pub circuit_opens: u64,
}

/// Event bus for manager events
struct EventBus {
    subscribers: Arc<DashMap<Uuid, tokio::sync::mpsc::UnboundedSender<ManagerEvent>>>,
}

impl EventBus {
    fn new() -> Self {
        Self {
            subscribers: Arc::new(DashMap::new()),
        }
    }
    
    fn subscribe(&self) -> EventSubscription {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let id = Uuid::new_v4();
        self.subscribers.insert(id, tx);
        
        EventSubscription { id, receiver: rx }
    }
    
    fn publish(&self, event: ManagerEvent) {
        for subscriber in self.subscribers.iter() {
            let _ = subscriber.send(event.clone());
        }
    }
}

/// Event subscription handle
pub struct EventSubscription {
    id: Uuid,
    receiver: tokio::sync::mpsc::UnboundedReceiver<ManagerEvent>,
}

impl EventSubscription {
    /// Receive next event
    pub async fn recv(&mut self) -> Option<ManagerEvent> {
        self.receiver.recv().await
    }
}

/// Manager events
#[derive(Debug, Clone, Serialize)]
pub enum ManagerEvent {
    /// Adapter registered
    AdapterRegistered { platform: Platform },
    
    /// Adapter unregistered
    AdapterUnregistered { platform: Platform },
    
    /// Health status changed
    HealthStatusChanged {
        platform: Platform,
        old_status: HealthState,
        new_status: HealthState,
    },
    
    /// Circuit breaker opened
    CircuitBreakerOpened { platform: Platform },
    
    /// Circuit breaker closed
    CircuitBreakerClosed { platform: Platform },
    
    /// Rate limit exceeded
    RateLimitExceeded,
}

/// Adapter builder for fluent configuration
pub struct AdapterBuilder {
    adapters: Vec<Arc<dyn PlatformAdapter>>,
    config: ManagerConfig,
}

impl AdapterBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            config: ManagerConfig::default(),
        }
    }
    
    /// Add an adapter
    pub fn with_adapter(mut self, adapter: Arc<dyn PlatformAdapter>) -> Self {
        self.adapters.push(adapter);
        self
    }
    
    /// Set configuration
    pub fn with_config(mut self, config: ManagerConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Build the manager
    pub fn build(self) -> Result<AdapterManager, ManagerError> {
        let manager = AdapterManager::with_config(self.config);
        
        for adapter in self.adapters {
            manager.register(adapter)?;
        }
        
        Ok(manager)
    }
}

impl Default for AdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}