//! # Analytics Module
//! 
//! Production-ready analytics engine for the INTERSTICE-ENGINE WorkOS.
//! Provides comprehensive metrics collection, aggregation, and analysis capabilities
//! for outcome tracking, performance monitoring, and behavioral insights.
//interstice-core/src/analytics.rs
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{RwLock};
use tracing::{debug, error, info, instrument};

use crate::error::CoreError;
use crate::outcome::OutcomeId;
use crate::storage::StorageBackend;
use crate::types::{MetricValue, TimeRange, UserId, WorkspaceId};

/// Result type for analytics operations
pub type AnalyticsResult<T> = Result<T, AnalyticsError>;

/// Analytics-specific error types
#[derive(Error, Debug)]
pub enum AnalyticsError {
    #[error("Metric not found: {0}")]
    MetricNotFound(String),
    
    #[error("Invalid time range: {0}")]
    InvalidTimeRange(String),
    
    #[error("Aggregation error: {0}")]
    AggregationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] CoreError),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Lock poisoned")]
    LockPoisoned,
    
    #[error("Rate limit exceeded for metric: {0}")]
    RateLimitExceeded(String),
}

/// Core analytics engine with advanced capabilities
pub struct AnalyticsEngine {
    /// Storage backend for persistent metrics
    storage: Arc<dyn StorageBackend>,
    
    /// In-memory metric buffer for high-performance writes
    buffer: Arc<RwLock<MetricBuffer>>,
    
    /// Real-time aggregators for different metric types
    aggregators: Arc<RwLock<HashMap<String, Box<dyn Aggregator>>>>,
    
    /// Time-series data manager
    time_series: Arc<TimeSeriesManager>,
    
    /// Anomaly detection system
    anomaly_detector: Arc<AnomalyDetector>,
    
    /// Rate limiter for metric ingestion
    rate_limiter: Arc<RateLimiter>,
    
    /// Configuration
    config: AnalyticsConfig,
}

/// Configuration for the analytics engine
#[derive(Debug, Clone, Deserialize)]
pub struct AnalyticsConfig {
    /// Maximum buffer size before flush
    pub buffer_size: usize,
    
    /// Buffer flush interval
    pub flush_interval: Duration,
    
    /// Enable real-time anomaly detection
    pub enable_anomaly_detection: bool,
    
    /// Retention period for metrics
    pub retention_period: Duration,
    
    /// Maximum events per second per metric
    pub rate_limit: u32,
    
    /// Enable metric compression
    pub enable_compression: bool,
    
    /// Sampling rate for high-frequency metrics
    pub sampling_rate: f64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10_000,
            flush_interval: Duration::from_secs(30),
            enable_anomaly_detection: true,
            retention_period: Duration::from_secs(90 * 24 * 3600), // 90 days
            rate_limit: 1000,
            enable_compression: true,
            sampling_rate: 1.0,
        }
    }
}

/// Metric event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEvent {
    /// Unique metric identifier
    pub metric_id: String,
    
    /// Workspace context
    pub workspace_id: WorkspaceId,
    
    /// Optional user context
    pub user_id: Option<UserId>,
    
    /// Metric value
    pub value: MetricValue,
    
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Associated outcome ID if applicable
    pub outcome_id: Option<OutcomeId>,
    
    /// Event tags for filtering
    pub tags: Vec<String>,
}

/// In-memory buffer for high-performance metric ingestion
struct MetricBuffer {
    events: VecDeque<MetricEvent>,
    size_limit: usize,
    last_flush: Instant,
}

impl MetricBuffer {
    fn new(size_limit: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(size_limit),
            size_limit,
            last_flush: Instant::now(),
        }
    }
    
    fn push(&mut self, event: MetricEvent) -> bool {
        if self.events.len() >= self.size_limit {
            return false;
        }
        self.events.push_back(event);
        true
    }
    
    fn should_flush(&self, interval: Duration) -> bool {
        self.events.len() >= self.size_limit || 
        self.last_flush.elapsed() >= interval
    }
    
    fn flush(&mut self) -> Vec<MetricEvent> {
        self.last_flush = Instant::now();
        self.events.drain(..).collect()
    }
}

/// Trait for metric aggregation strategies
#[async_trait]
pub trait Aggregator: Send + Sync {
    /// Aggregate a new metric event
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()>;
    
    /// Get current aggregation result
    async fn result(&self) -> AnalyticsResult<AggregationResult>;
    
    /// Reset the aggregator
    async fn reset(&mut self);
    
    /// Get aggregator type
    fn aggregator_type(&self) -> AggregatorType;
}

/// Types of aggregation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregatorType {
    Sum,
    Average,
    Count,
    Min,
    Max,
    Percentile(f64),
    StandardDeviation,
    Custom(String),
}

/// Aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    pub aggregator_type: AggregatorType,
    pub value: f64,
    pub sample_count: u64,
    pub time_range: TimeRange,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Time-series data manager for efficient storage and retrieval
pub struct TimeSeriesManager {
    /// Time-bucketed data storage
    buckets: Arc<RwLock<HashMap<String, TimeBucket>>>,
    
    /// Bucket duration
    bucket_duration: Duration,
    
    /// Compression settings
    compression_enabled: bool,
}

/// Time bucket for organizing time-series data
#[derive(Debug, Clone)]
struct TimeBucket {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    data_points: Vec<DataPoint>,
    compressed: bool,
}

/// Individual data point in time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: Vec<String>,
}

/// Anomaly detection system
pub struct AnomalyDetector {
    /// Statistical models for each metric
    models: Arc<RwLock<HashMap<String, StatisticalModel>>>,
    
    /// Detected anomalies
    anomalies: Arc<RwLock<Vec<Anomaly>>>,
    
    /// Configuration
    sensitivity: f64,
}

/// Statistical model for anomaly detection
struct StatisticalModel {
    mean: f64,
    std_dev: f64,
    sample_count: u64,
    last_values: VecDeque<f64>,
}

impl StatisticalModel {
    fn new() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            sample_count: 0,
            last_values: VecDeque::with_capacity(100),
        }
    }
    
    fn update(&mut self, value: f64) {
        self.last_values.push_back(value);
        if self.last_values.len() > 100 {
            self.last_values.pop_front();
        }
        
        // Update running statistics
        self.sample_count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.sample_count as f64;
        let delta2 = value - self.mean;
        self.std_dev = ((self.std_dev.powi(2) * (self.sample_count - 1) as f64 + delta * delta2) 
            / self.sample_count as f64).sqrt();
    }
    
    fn is_anomaly(&self, value: f64, sensitivity: f64) -> bool {
        if self.sample_count < 10 {
            return false; // Not enough data
        }
        
        let z_score = (value - self.mean).abs() / self.std_dev;
        z_score > sensitivity
    }
}

/// Detected anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub metric_id: String,
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub expected_range: (f64, f64),
    pub severity: AnomalySeverity,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Anomaly severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Rate limiter for metric ingestion
pub struct RateLimiter {
    limits: Arc<RwLock<HashMap<String, RateLimit>>>,
}

struct RateLimit {
    max_per_second: u32,
    window_start: Instant,
    count: u32,
}

impl AnalyticsEngine {
    /// Create a new analytics engine
    pub async fn new(
        storage: Arc<dyn StorageBackend>,
        config: AnalyticsConfig,
    ) -> AnalyticsResult<Self> {
        let buffer = Arc::new(RwLock::new(MetricBuffer::new(config.buffer_size)));
        let aggregators = Arc::new(RwLock::new(HashMap::new()));
        
        let time_series = Arc::new(TimeSeriesManager {
            buckets: Arc::new(RwLock::new(HashMap::new())),
            bucket_duration: Duration::from_secs(3600), // 1 hour buckets
            compression_enabled: config.enable_compression,
        });
        
        let anomaly_detector = Arc::new(AnomalyDetector {
            models: Arc::new(RwLock::new(HashMap::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
            sensitivity: 3.0, // 3 standard deviations
        });
        
        let rate_limiter = Arc::new(RateLimiter {
            limits: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let engine = Self {
            storage,
            buffer,
            aggregators,
            time_series,
            anomaly_detector,
            rate_limiter,
            config,
        };
        
        // Start background flush task
        engine.start_flush_task().await;
        
        Ok(engine)
    }
    
    /// Record a metric event
    #[instrument(skip(self))]
    pub async fn record_metric(&self, mut event: MetricEvent) -> AnalyticsResult<()> {
        // Check rate limit
        if !self.check_rate_limit(&event.metric_id).await? {
            return Err(AnalyticsError::RateLimitExceeded(event.metric_id.clone()));
        }
        
        // Apply sampling if configured
        if self.should_sample() {
            debug!("Metric sampled out: {}", event.metric_id);
            return Ok(());
        }
        
        // Enrich event with system metadata
        event.metadata.insert(
            "recorded_at".to_string(),
            serde_json::Value::String(Utc::now().to_rfc3339()),
        );
        
        // Buffer the event
        {
            let mut buffer = self.buffer.write().await;
            if !buffer.push(event.clone()) {
                // Buffer full, trigger immediate flush
                self.flush_buffer().await?;
                buffer.push(event.clone());
            }
        }
        
        // Update aggregators
        self.update_aggregators(&event).await?;
        
        // Check for anomalies if enabled
        if self.config.enable_anomaly_detection {
            self.detect_anomaly(&event).await?;
        }
        
        // Update time series
        self.update_time_series(&event).await?;
        
        Ok(())
    }
    
    /// Query metrics with advanced filtering
    #[instrument(skip(self))]
    pub async fn query_metrics(
        &self,
        query: MetricQuery,
    ) -> AnalyticsResult<QueryResult> {
        info!("Executing metric query: {:?}", query);
        
        // Fetch from storage
        let stored_metrics = self.storage
            .query_metrics(&query)
            .await
            .map_err(|e| AnalyticsError::StorageError(e))?;
        
        // Apply additional filters and transformations
        let processed = self.process_query_results(stored_metrics, &query).await?;
        let total_count = processed.len();
        
        Ok(QueryResult {
            metrics: processed,
            query: query.clone(),
            execution_time: Duration::from_millis(0), // TODO: Track actual time
            total_count,
        })
    }
    
    /// Get real-time dashboard data
    pub async fn get_dashboard_metrics(
        &self,
        workspace_id: WorkspaceId,
    ) -> AnalyticsResult<DashboardMetrics> {
        let now = Utc::now();
        let last_24h = TimeRange {
            start: now - chrono::Duration::hours(24),
            end: now,
        };
        
        // Aggregate key metrics
        let total_events = self.get_event_count(workspace_id, &last_24h).await?;
        let active_users = self.get_active_users(workspace_id, &last_24h).await?;
        let outcome_completion_rate = self.get_outcome_completion_rate(workspace_id, &last_24h).await?;
        let average_response_time = self.get_avg_response_time(workspace_id, &last_24h).await?;
        
        // Get recent anomalies
        let anomalies = self.get_recent_anomalies(workspace_id, 10).await?;
        
        // Get trending metrics
        let trending = self.get_trending_metrics(workspace_id, 5).await?;
        
        Ok(DashboardMetrics {
            workspace_id,
            time_range: last_24h,
            total_events,
            active_users,
            outcome_completion_rate,
            average_response_time,
            anomalies,
            trending_metrics: trending,
            generated_at: now,
        })
    }
    
    /// Export analytics data
    pub async fn export_analytics(
        &self,
        workspace_id: WorkspaceId,
        format: ExportFormat,
        time_range: TimeRange,
    ) -> AnalyticsResult<Vec<u8>> {
        let query = MetricQuery {
            workspace_id: Some(workspace_id),
            time_range: Some(time_range),
            ..Default::default()
        };
        
        let result = self.query_metrics(query).await?;
        
        match format {
            ExportFormat::Json => {
                serde_json::to_vec_pretty(&result)
                    .map_err(|e| AnalyticsError::SerializationError(e))
            }
            ExportFormat::Csv => {
                self.export_as_csv(result).await
            }
            ExportFormat::Parquet => {
                self.export_as_parquet(result).await
            }
        }
    }
    
    // Private helper methods
    
    async fn start_flush_task(&self) {
        let buffer = Arc::clone(&self.buffer);
        let storage = Arc::clone(&self.storage);
        let interval = self.config.flush_interval;
        
        tokio::spawn(async move {
            let mut flush_interval = tokio::time::interval(interval);
            
            loop {
                flush_interval.tick().await;
                
                let should_flush = {
                    let buffer = buffer.read().await;
                    buffer.should_flush(interval)
                };
                
                if should_flush {
                    if let Err(e) = Self::flush_buffer_static(buffer.clone(), storage.clone()).await {
                        error!("Failed to flush buffer: {}", e);
                    }
                }
            }
        });
    }
    
    async fn flush_buffer(&self) -> AnalyticsResult<()> {
        Self::flush_buffer_static(Arc::clone(&self.buffer), Arc::clone(&self.storage)).await
    }
    
    async fn flush_buffer_static(
        buffer: Arc<RwLock<MetricBuffer>>,
        storage: Arc<dyn StorageBackend>,
    ) -> AnalyticsResult<()> {
        let events = {
            let mut buffer = buffer.write().await;
            buffer.flush()
        };
        
        if !events.is_empty() {
            info!("Flushing {} events to storage", events.len());
            storage.store_metrics(events).await
                .map_err(|e| AnalyticsError::StorageError(e))?;
        }
        
        Ok(())
    }
    
    async fn check_rate_limit(&self, metric_id: &str) -> AnalyticsResult<bool> {
        let mut limits = self.rate_limiter.limits.write().await;
        let now = Instant::now();
        
        let limit = limits.entry(metric_id.to_string())
            .or_insert_with(|| RateLimit {
                max_per_second: self.config.rate_limit,
                window_start: now,
                count: 0,
            });
        
        // Reset window if needed
        if limit.window_start.elapsed() >= Duration::from_secs(1) {
            limit.window_start = now;
            limit.count = 0;
        }
        
        if limit.count >= limit.max_per_second {
            return Ok(false);
        }
        
        limit.count += 1;
        Ok(true)
    }
    
    fn should_sample(&self) -> bool {
        if self.config.sampling_rate >= 1.0 {
            return false;
        }
        
        rand::random::<f64>() > self.config.sampling_rate
    }
    
    async fn update_aggregators(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        let mut aggregators = self.aggregators.write().await;
        
        for (_, aggregator) in aggregators.iter_mut() {
            aggregator.aggregate(event).await?;
        }
        
        Ok(())
    }
    
    async fn detect_anomaly(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(value) = event.value {
            let mut models = self.anomaly_detector.models.write().await;
            let model = models.entry(event.metric_id.clone())
                .or_insert_with(StatisticalModel::new);
            
            if model.is_anomaly(value, self.anomaly_detector.sensitivity) {
                let anomaly = Anomaly {
                    metric_id: event.metric_id.clone(),
                    timestamp: event.timestamp,
                    value,
                    expected_range: (
                        model.mean - 3.0 * model.std_dev,
                        model.mean + 3.0 * model.std_dev,
                    ),
                    severity: self.calculate_anomaly_severity(value, model),
                    metadata: event.metadata.clone(),
                };
                
                let mut anomalies = self.anomaly_detector.anomalies.write().await;
                anomalies.push(anomaly);
                
                // Keep only recent anomalies (last 1000)
                if anomalies.len() > 1000 {
                    anomalies.drain(0..100);
                }
            }
            
            model.update(value);
        }
        
        Ok(())
    }
    
    fn calculate_anomaly_severity(&self, value: f64, model: &StatisticalModel) -> AnomalySeverity {
        let z_score = (value - model.mean).abs() / model.std_dev;
        
        if z_score > 6.0 {
            AnomalySeverity::Critical
        } else if z_score > 4.5 {
            AnomalySeverity::High
        } else if z_score > 3.5 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }
    
    async fn update_time_series(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(value) = event.value {
            let data_point = DataPoint {
                timestamp: event.timestamp,
                value,
                tags: event.tags.clone(),
            };
            
            let bucket_key = self.get_bucket_key(&event.metric_id, event.timestamp);
            let mut buckets = self.time_series.buckets.write().await;
            
            let bucket = buckets.entry(bucket_key).or_insert_with(|| {
                let start = event.timestamp;
                TimeBucket {
                    start_time: start,
                    end_time: start + chrono::Duration::from_std(self.time_series.bucket_duration).unwrap(),
                    data_points: Vec::new(),
                    compressed: false,
                }
            });
            
            bucket.data_points.push(data_point);
            
            // Compress old buckets if needed
            if self.time_series.compression_enabled && 
               bucket.data_points.len() > 1000 && 
               !bucket.compressed {
                // TODO: Implement compression
                bucket.compressed = true;
            }
        }
        
        Ok(())
    }
    
    fn get_bucket_key(&self, metric_id: &str, timestamp: DateTime<Utc>) -> String {
        let bucket_index = timestamp.timestamp() / self.time_series.bucket_duration.as_secs() as i64;
        format!("{}:{}", metric_id, bucket_index)
    }
    
    async fn process_query_results(
        &self,
        metrics: Vec<MetricEvent>,
        query: &MetricQuery,
    ) -> AnalyticsResult<Vec<MetricEvent>> {
        let mut result = metrics;
        
        // Apply additional filters
        if let Some(ref tags) = query.tags {
            result = result.into_iter()
                .filter(|m| tags.iter().any(|t| m.tags.contains(t)))
                .collect();
        }
        
        // Sort if requested
        if let Some(ref sort) = query.sort_by {
            result.sort_by(|a, b| {
                match sort.as_str() {
                    "timestamp" => a.timestamp.cmp(&b.timestamp),
                    "value" => {
                        let a_val = a.value.as_float().unwrap_or(0.0);
                        let b_val = b.value.as_float().unwrap_or(0.0);
                        a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    _ => std::cmp::Ordering::Equal,
                }
            });
        }
        
        // Apply limit
        if let Some(limit) = query.limit {
            result.truncate(limit);
        }
        
        Ok(result)
    }
    
    async fn get_event_count(
        &self,
        workspace_id: WorkspaceId,
        time_range: &TimeRange,
    ) -> AnalyticsResult<u64> {
        let query = MetricQuery {
            workspace_id: Some(workspace_id),
            time_range: Some(time_range.clone()),
            ..Default::default()
        };
        
        let result = self.query_metrics(query).await?;
        Ok(result.total_count as u64)
    }
    
    async fn get_active_users(
        &self,
        workspace_id: WorkspaceId,
        time_range: &TimeRange,
    ) -> AnalyticsResult<u64> {
        // TODO: Implement unique user counting
        Ok(0)
    }
    
    async fn get_outcome_completion_rate(
        &self,
        workspace_id: WorkspaceId,
        time_range: &TimeRange,
    ) -> AnalyticsResult<f64> {
        // TODO: Implement outcome completion calculation
        Ok(0.0)
    }
    
    async fn get_avg_response_time(
        &self,
        workspace_id: WorkspaceId,
        time_range: &TimeRange,
    ) -> AnalyticsResult<Duration> {
        // TODO: Implement average response time calculation
        Ok(Duration::from_millis(0))
    }
    
    async fn get_recent_anomalies(
        &self,
        workspace_id: WorkspaceId,
        limit: usize,
    ) -> AnalyticsResult<Vec<Anomaly>> {
        let anomalies = self.anomaly_detector.anomalies.read().await;
        let filtered: Vec<Anomaly> = anomalies.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect();
        Ok(filtered)
    }
    
    async fn get_trending_metrics(
        &self,
        workspace_id: WorkspaceId,
        limit: usize,
    ) -> AnalyticsResult<Vec<TrendingMetric>> {
        // TODO: Implement trending metrics calculation
        Ok(Vec::new())
    }
    
    async fn export_as_csv(&self, result: QueryResult) -> AnalyticsResult<Vec<u8>> {
        // TODO: Implement CSV export
        Ok(Vec::new())
    }
    
    async fn export_as_parquet(&self, result: QueryResult) -> AnalyticsResult<Vec<u8>> {
        // TODO: Implement Parquet export
        Ok(Vec::new())
    }
}

/// Query structure for metrics retrieval
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricQuery {
    pub workspace_id: Option<WorkspaceId>,
    pub user_id: Option<UserId>,
    pub metric_ids: Option<Vec<String>>,
    pub time_range: Option<TimeRange>,
    pub tags: Option<Vec<String>>,
    pub outcome_ids: Option<Vec<OutcomeId>>,
    pub aggregation: Option<AggregatorType>,
    pub group_by: Option<Vec<String>>,
    pub sort_by: Option<String>,
    pub limit: Option<usize>,
}

/// Query result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub metrics: Vec<MetricEvent>,
    pub query: MetricQuery,
    pub execution_time: Duration,
    pub total_count: usize,
}

/// Dashboard metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub workspace_id: WorkspaceId,
    pub time_range: TimeRange,
    pub total_events: u64,
    pub active_users: u64,
    pub outcome_completion_rate: f64,
    pub average_response_time: Duration,
    pub anomalies: Vec<Anomaly>,
    pub trending_metrics: Vec<TrendingMetric>,
    pub generated_at: DateTime<Utc>,
}

/// Trending metric information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingMetric {
    pub metric_id: String,
    pub name: String,
    pub current_value: f64,
    pub previous_value: f64,
    pub change_percentage: f64,
    pub trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Up,
    Down,
    Stable,
}

/// Export format options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Parquet,
}

// Implement standard aggregators

/// Sum aggregator
pub struct SumAggregator {
    sum: f64,
    count: u64,
    time_range: TimeRange,
}

#[async_trait]
impl Aggregator for SumAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(val) = event.value {
            self.sum += val;
            self.count += 1;
        }
        Ok(())
    }
    
    async fn result(&self) -> AnalyticsResult<AggregationResult> {
        Ok(AggregationResult {
            aggregator_type: AggregatorType::Sum,
            value: self.sum,
            sample_count: self.count,
            time_range: self.time_range.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
    
    fn aggregator_type(&self) -> AggregatorType {
        AggregatorType::Sum
    }
}

/// Average aggregator
pub struct AverageAggregator {
    sum: f64,
    count: u64,
    time_range: TimeRange,
}

#[async_trait]
impl Aggregator for AverageAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(val) = event.value {
            self.sum += val;
            self.count += 1;
        }
        Ok(())
    }
    
    async fn result(&self) -> AnalyticsResult<AggregationResult> {
        let avg = if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        };
        
        Ok(AggregationResult {
            aggregator_type: AggregatorType::Average,
            value: avg,
            sample_count: self.count,
            time_range: self.time_range.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
    
    fn aggregator_type(&self) -> AggregatorType {
        AggregatorType::Average
    }
}

/// Percentile aggregator with reservoir sampling
pub struct PercentileAggregator {
    percentile: f64,
    reservoir: Vec<f64>,
    reservoir_size: usize,
    total_seen: u64,
    time_range: TimeRange,
}

impl PercentileAggregator {
    pub fn new(percentile: f64, reservoir_size: usize, time_range: TimeRange) -> Self {
        Self {
            percentile,
            reservoir: Vec::with_capacity(reservoir_size),
            reservoir_size,
            total_seen: 0,
            time_range,
        }
    }
    
    fn calculate_percentile(&self) -> f64 {
        if self.reservoir.is_empty() {
            return 0.0;
        }
        
        let mut sorted = self.reservoir.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((self.percentile / 100.0) * (sorted.len() - 1) as f64) as usize;
        sorted[index]
    }
}

#[async_trait]
impl Aggregator for PercentileAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(val) = event.value {
            self.total_seen += 1;
            
            // Reservoir sampling algorithm
            if self.reservoir.len() < self.reservoir_size {
                self.reservoir.push(val);
            } else {
                let index = fastrand::usize(..) % self.total_seen as usize;
                if index < self.reservoir_size {
                    self.reservoir[index] = val;
                }
            }
        }
        Ok(())
    }
    
    async fn result(&self) -> AnalyticsResult<AggregationResult> {
        Ok(AggregationResult {
            aggregator_type: AggregatorType::Percentile(self.percentile),
            value: self.calculate_percentile(),
            sample_count: self.total_seen,
            time_range: self.time_range.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn reset(&mut self) {
        self.reservoir.clear();
        self.total_seen = 0;
    }
    
    fn aggregator_type(&self) -> AggregatorType {
        AggregatorType::Percentile(self.percentile)
    }
}

/// Builder pattern for constructing analytics engine
pub struct AnalyticsEngineBuilder {
    storage: Option<Arc<dyn StorageBackend>>,
    config: AnalyticsConfig,
    custom_aggregators: Vec<(String, Box<dyn Aggregator>)>,
}

impl AnalyticsEngineBuilder {
    pub fn new() -> Self {
        Self {
            storage: None,
            config: AnalyticsConfig::default(),
            custom_aggregators: Vec::new(),
        }
    }
    
    pub fn with_storage(mut self, storage: Arc<dyn StorageBackend>) -> Self {
        self.storage = Some(storage);
        self
    }
    
    pub fn with_config(mut self, config: AnalyticsConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }
    
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.config.flush_interval = interval;
        self
    }
    
    pub fn with_anomaly_detection(mut self, enabled: bool) -> Self {
        self.config.enable_anomaly_detection = enabled;
        self
    }
    
    pub fn with_custom_aggregator(mut self, name: String, aggregator: Box<dyn Aggregator>) -> Self {
        self.custom_aggregators.push((name, aggregator));
        self
    }
    
    pub async fn build(self) -> AnalyticsResult<AnalyticsEngine> {
        let storage = self.storage.ok_or_else(|| {
            AnalyticsError::StorageError(CoreError::Configuration(
                "Storage backend required".to_string()
            ))
        })?;
        
        let engine = AnalyticsEngine::new(storage, self.config).await?;
        
        // Add custom aggregators
        for (name, aggregator) in self.custom_aggregators {
            engine.aggregators.write().await.insert(name, aggregator);
        }
        
        Ok(engine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::storage::{StorageBackend, WorkspaceStats, CleanupStats, ArtifactFilters};
    use crate::outcome::{Outcome, OutcomeFilters, OutcomeId};
    use crate::artifact::Artifact;
    use crate::types::SystemEvent;
    use uuid::Uuid;
    use serde_json::Value as JsonValue;
    
    // Mock storage backend for testing
    struct MockStorage;
    
    #[async_trait]
    impl StorageBackend for MockStorage {
        async fn store_metrics(&self, _events: Vec<MetricEvent>) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn query_metrics(&self, _query: &MetricQuery) -> Result<Vec<MetricEvent>, CoreError> {
            Ok(Vec::new())
        }

        async fn delete_old_metrics(&self, _before: DateTime<Utc>) -> Result<u64, CoreError> {
            Ok(0)
        }
        
        async fn store_outcome(&self, _outcome: Outcome) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn get_outcome(&self, _id: OutcomeId) -> Result<Option<Outcome>, CoreError> {
            Ok(None)
        }
        
        async fn update_outcome(&self, _outcome: Outcome) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn delete_outcome(&self, _id: OutcomeId) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn query_outcomes(&self, _workspace_id: WorkspaceId, _filters: Option<OutcomeFilters>) -> Result<Vec<Outcome>, CoreError> {
            Ok(Vec::new())
        }
        
        async fn store_artifact(&self, _artifact: Artifact) -> Result<Uuid, CoreError> {
            Ok(Uuid::new_v4())
        }
        
        async fn get_artifact(&self, _id: Uuid) -> Result<Option<Artifact>, CoreError> {
            Ok(None)
        }
        
        async fn query_artifacts(&self, _workspace_id: WorkspaceId, _filters: Option<ArtifactFilters>) -> Result<Vec<Artifact>, CoreError> {
            Ok(Vec::new())
        }
        
        async fn delete_artifact(&self, _id: Uuid) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn link_artifact_outcome(&self, _artifact_id: Uuid, _outcome_id: OutcomeId, _confidence: f64, _metadata: Option<JsonValue>) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn store_event(&self, _event: SystemEvent) -> Result<(), CoreError> {
            Ok(())
        }
        
        async fn get_workspace_stats(&self, _workspace_id: WorkspaceId) -> Result<WorkspaceStats, CoreError> {
            Ok(WorkspaceStats {
                workspace_id: WorkspaceId::new(),
                total_artifacts: 0,
                total_outcomes: 0,
                completed_outcomes: 0,
                total_metrics: 0,
                recent_activity: 0,
                mapped_work_percentage: 0.0,
                created_at: Utc::now(),
            })
        }
        
        async fn health_check(&self) -> Result<bool, CoreError> {
            Ok(true)
        }
        
        async fn cleanup_expired_data(&self) -> Result<CleanupStats, CoreError> {
            Ok(CleanupStats {
                metrics_deleted: 0,
                artifacts_deleted: 0,
                tokens_deleted: 0,
                events_deleted: 0,
                timestamp: Utc::now(),
            })
        }
    }
    
    #[tokio::test]
    async fn test_analytics_engine_creation() {
        let storage = Arc::new(MockStorage);
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_metric_recording() {
        let storage = Arc::new(MockStorage);
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await.unwrap();
        
        let event = MetricEvent {
            metric_id: "test_metric".to_string(),
            workspace_id: WorkspaceId::new(),
            user_id: None,
            value: MetricValue::Float(42.0),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            outcome_id: None,
            tags: vec!["test".to_string()],
        };
        
        let result = engine.record_metric(event).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let mut config = AnalyticsConfig::default();
        config.rate_limit = 2; // Very low limit for testing
        
        let storage = Arc::new(MockStorage);
        let engine = AnalyticsEngine::new(storage, config).await.unwrap();
        
        // Should succeed for first few events
        for i in 0..2 {
            let event = MetricEvent {
                metric_id: "rate_test".to_string(),
                workspace_id: WorkspaceId::new(),
                user_id: None,
                value: MetricValue::Float(i as f64),
                timestamp: Utc::now(),
                metadata: HashMap::new(),
                outcome_id: None,
                tags: vec![],
            };
            
            let result = engine.record_metric(event).await;
            assert!(result.is_ok());
        }
        
        // Should fail on exceeding rate limit
        let event = MetricEvent {
            metric_id: "rate_test".to_string(),
            workspace_id: WorkspaceId::new(),
            user_id: None,
            value: MetricValue::Float(99.0),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            outcome_id: None,
            tags: vec![],
        };
        
        let result = engine.record_metric(event).await;
        assert!(matches!(result, Err(AnalyticsError::RateLimitExceeded(_))));
    }
    
    #[tokio::test]
    async fn test_aggregators() {
        let mut sum_agg = SumAggregator {
            sum: 0.0,
            count: 0,
            time_range: TimeRange {
                start: Utc::now() - chrono::Duration::hours(1),
                end: Utc::now(),
            },
        };
        
        for i in 1..=10 {
            let event = MetricEvent {
                metric_id: "sum_test".to_string(),
                workspace_id: WorkspaceId::new(),
                user_id: None,
                value: MetricValue::Float(i as f64),
                timestamp: Utc::now(),
                metadata: HashMap::new(),
                outcome_id: None,
                tags: vec![],
            };
            
            sum_agg.aggregate(&event).await.unwrap();
        }
        
        let result = sum_agg.result().await.unwrap();
        assert_eq!(result.value, 55.0); // Sum of 1..10
        assert_eq!(result.sample_count, 10);
    }
    
    #[tokio::test]
    async fn test_anomaly_detection() {
        let mut model = StatisticalModel::new();
        
        // Feed normal values
        for _ in 0..100 {
            model.update(50.0 + rand::random::<f64>() * 10.0 - 5.0);
        }
        
        // Test normal value
        assert!(!model.is_anomaly(52.0, 3.0));
        
        // Test anomaly
        assert!(model.is_anomaly(500.0, 3.0));
    }
    
    #[tokio::test]
    async fn test_builder_pattern() {
        let storage = Arc::new(MockStorage);
        
        let engine = AnalyticsEngineBuilder::new()
            .with_storage(storage)
            .with_buffer_size(5000)
            .with_flush_interval(Duration::from_secs(60))
            .with_anomaly_detection(true)
            .build()
            .await;
        
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert_eq!(engine.config.buffer_size, 5000);
        assert_eq!(engine.config.flush_interval, Duration::from_secs(60));
        assert!(engine.config.enable_anomaly_detection);
    }
}