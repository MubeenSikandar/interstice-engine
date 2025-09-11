//! # Analytics Module - Complete Implementation
//! 
//! Production-ready analytics engine for the INTERSTICE-ENGINE WorkOS.
//! Provides comprehensive metrics collection, aggregation, and analysis capabilities
//! for outcome tracking, performance monitoring, and behavioral insights.

use std::collections::{HashMap, VecDeque, BTreeMap};
use csv::Writer;

/// Engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatistics {
    pub buffered_events: usize,
    pub active_aggregators: usize,
    pub time_series_buckets: usize,
    pub detected_anomalies: usize,
    pub trained_models: usize,
    pub compression_enabled: bool,
    pub anomaly_detection_enabled: bool,
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

// ============================================================================
// Standard Aggregator Implementations
// ============================================================================

/// Sum aggregator
pub struct SumAggregator {
    sum: f64,
    count: u64,
    time_range: TimeRange,
}

impl SumAggregator {
    pub fn new(time_range: TimeRange) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            time_range,
        }
    }
}

#[async_trait]
impl Aggregator for SumAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(val) = event.value {
            self.sum += val;
            self.count += 1;
        } else if let MetricValue::Integer(val) = event.value {
            self.sum += val as f64;
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

impl AverageAggregator {
    pub fn new(time_range: TimeRange) -> Self {
        Self {
            sum: 0.0,
            count: 0,
            time_range,
        }
    }
}

#[async_trait]
impl Aggregator for AverageAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(val) = event.value {
            self.sum += val;
            self.count += 1;
        } else if let MetricValue::Integer(val) = event.value {
            self.sum += val as f64;
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

/// Min/Max aggregator
pub struct MinMaxAggregator {
    min: Option<f64>,
    max: Option<f64>,
    count: u64,
    time_range: TimeRange,
    track_min: bool,
}

impl MinMaxAggregator {
    pub fn new_min(time_range: TimeRange) -> Self {
        Self {
            min: None,
            max: None,
            count: 0,
            time_range,
            track_min: true,
        }
    }
    
    pub fn new_max(time_range: TimeRange) -> Self {
        Self {
            min: None,
            max: None,
            count: 0,
            time_range,
            track_min: false,
        }
    }
}

#[async_trait]
impl Aggregator for MinMaxAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        let value = match event.value {
            MetricValue::Float(val) => val,
            MetricValue::Integer(val) => val as f64,
            _ => return Ok(()),
        };
        
        self.min = Some(self.min.map_or(value, |m| m.min(value)));
        self.max = Some(self.max.map_or(value, |m| m.max(value)));
        self.count += 1;
        
        Ok(())
    }
    
    async fn result(&self) -> AnalyticsResult<AggregationResult> {
        let value = if self.track_min {
            self.min.unwrap_or(0.0)
        } else {
            self.max.unwrap_or(0.0)
        };
        
        Ok(AggregationResult {
            aggregator_type: if self.track_min { 
                AggregatorType::Min 
            } else { 
                AggregatorType::Max 
            },
            value,
            sample_count: self.count,
            time_range: self.time_range.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn reset(&mut self) {
        self.min = None;
        self.max = None;
        self.count = 0;
    }
    
    fn aggregator_type(&self) -> AggregatorType {
        if self.track_min {
            AggregatorType::Min
        } else {
            AggregatorType::Max
        }
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
        let value = match event.value {
            MetricValue::Float(val) => val,
            MetricValue::Integer(val) => val as f64,
            _ => return Ok(()),
        };
        
        self.total_seen += 1;
        
        // Reservoir sampling algorithm
        if self.reservoir.len() < self.reservoir_size {
            self.reservoir.push(value);
        } else {
            let index = rand::random::<u32>() as usize % self.total_seen as usize;
            if index < self.reservoir_size {
                self.reservoir[index] = value;
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

/// Standard deviation aggregator
pub struct StdDevAggregator {
    sum: f64,
    sum_of_squares: f64,
    count: u64,
    time_range: TimeRange,
}

impl StdDevAggregator {
    pub fn new(time_range: TimeRange) -> Self {
        Self {
            sum: 0.0,
            sum_of_squares: 0.0,
            count: 0,
            time_range,
        }
    }
}

#[async_trait]
impl Aggregator for StdDevAggregator {
    async fn aggregate(&mut self, event: &MetricEvent) -> AnalyticsResult<()> {
        let value = match event.value {
            MetricValue::Float(val) => val,
            MetricValue::Integer(val) => val as f64,
            _ => return Ok(()),
        };
        
        self.sum += value;
        self.sum_of_squares += value * value;
        self.count += 1;
        
        Ok(())
    }
    
    async fn result(&self) -> AnalyticsResult<AggregationResult> {
        let std_dev = if self.count > 1 {
            let mean = self.sum / self.count as f64;
            let variance = (self.sum_of_squares / self.count as f64) - (mean * mean);
            variance.sqrt()
        } else {
            0.0
        };
        
        Ok(AggregationResult {
            aggregator_type: AggregatorType::StandardDeviation,
            value: std_dev,
            sample_count: self.count,
            time_range: self.time_range.clone(),
            metadata: HashMap::new(),
        })
    }
    
    async fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_of_squares = 0.0;
        self.count = 0;
    }
    
    fn aggregator_type(&self) -> AggregatorType {
        AggregatorType::StandardDeviation
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
    
    pub fn with_retention_period(mut self, period: Duration) -> Self {
        self.config.retention_period = period;
        self
    }
    
    pub fn with_rate_limit(mut self, limit: u32) -> Self {
        self.config.rate_limit = limit;
        self
    }
    
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.config.enable_compression = enabled;
        self
    }
    
    pub fn with_sampling_rate(mut self, rate: f64) -> Self {
        self.config.sampling_rate = rate.clamp(0.0, 1.0);
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

impl Default for AnalyticsEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create standard aggregators for common metrics
pub async fn create_standard_aggregators(time_range: TimeRange) -> HashMap<String, Box<dyn Aggregator>> {
    let mut aggregators: HashMap<String, Box<dyn Aggregator>> = HashMap::new();
    
    aggregators.insert(
        "sum".to_string(),
        Box::new(SumAggregator::new(time_range.clone())),
    );
    
    aggregators.insert(
        "average".to_string(),
        Box::new(AverageAggregator::new(time_range.clone())),
    );
    
    aggregators.insert(
        "min".to_string(),
        Box::new(MinMaxAggregator::new_min(time_range.clone())),
    );
    
    aggregators.insert(
        "max".to_string(),
        Box::new(MinMaxAggregator::new_max(time_range.clone())),
    );
    
    aggregators.insert(
        "p50".to_string(),
        Box::new(PercentileAggregator::new(50.0, 1000, time_range.clone())),
    );
    
    aggregators.insert(
        "p95".to_string(),
        Box::new(PercentileAggregator::new(95.0, 1000, time_range.clone())),
    );
    
    aggregators.insert(
        "p99".to_string(),
        Box::new(PercentileAggregator::new(99.0, 1000, time_range.clone())),
    );
    
    aggregators.insert(
        "stddev".to_string(),
        Box::new(StdDevAggregator::new(time_range)),
    );
    
    aggregators
}

/// Helper to create metric event
pub fn create_metric_event(
    metric_id: impl Into<String>,
    workspace_id: WorkspaceId,
    value: MetricValue,
) -> MetricEvent {
    MetricEvent {
        metric_id: metric_id.into(),
        workspace_id,
        user_id: None,
        value,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
        outcome_id: None,
        tags: Vec::new(),
    }
}

/// Helper to create metric event with tags
pub fn create_tagged_metric(
    metric_id: impl Into<String>,
    workspace_id: WorkspaceId,
    value: MetricValue,
    tags: Vec<String>,
) -> MetricEvent {
    let mut event = create_metric_event(metric_id, workspace_id, value);
    event.tags = tags;
    event
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::storage::{ArtifactFilters, CleanupStats, PredictionFeedback, PredictionRecord, StorageBackend, WorkspaceStats};
    use crate::outcome::{Outcome, OutcomeFilters, OutcomeId};
    use crate::artifact::Artifact;
    use crate::types::SystemEvent;
    use uuid::Uuid;
    use serde_json::Value as JsonValue;
    
    // Mock storage backend for testing
    struct MockStorage {
        metrics: Arc<RwLock<Vec<MetricEvent>>>,
    }
    
    impl MockStorage {
        fn new() -> Self {
            Self {
                metrics: Arc::new(RwLock::new(Vec::new())),
            }
        }
    }
    
    #[async_trait]
    impl StorageBackend for MockStorage {
        async fn store_metrics(&self, events: Vec<MetricEvent>) -> Result<(), CoreError> {
            let mut metrics = self.metrics.write().await;
            metrics.extend(events);
            Ok(())
        }
        
        async fn query_metrics(&self, query: &MetricQuery) -> Result<Vec<MetricEvent>, CoreError> {
            let metrics = self.metrics.read().await;
            let mut result: Vec<MetricEvent> = metrics.clone();
            
            // Apply filters
            if let Some(workspace_id) = &query.workspace_id {
                result.retain(|m| &m.workspace_id == workspace_id);
            }
            
            if let Some(time_range) = &query.time_range {
                result.retain(|m| m.timestamp >= time_range.start && m.timestamp <= time_range.end);
            }
            
            if let Some(limit) = query.limit {
                result.truncate(limit);
            }
            
            Ok(result)
        }

        async fn delete_old_metrics(&self, before: DateTime<Utc>) -> Result<u64, CoreError> {
            let mut metrics = self.metrics.write().await;
            let original_len = metrics.len();
            metrics.retain(|m| m.timestamp >= before);
            Ok((original_len - metrics.len()) as u64)
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

        async fn store_prediction_record(&self, _record: PredictionRecord) -> Result<(), CoreError> {
            Ok(())
        }

        async fn query_predictions(&self, _since: DateTime<Utc>, _limit: usize) -> Result<Vec<PredictionRecord>, CoreError> {
            Ok(Vec::new())
        }

        async fn store_prediction_feedback(&self, _feedback: PredictionFeedback) -> Result<(), CoreError> {
            Ok(())
        }

        async fn get_prediction_feedback(&self, _prediction_id: Uuid) -> Result<Vec<PredictionFeedback>, CoreError> {
            Ok(Vec::new())
        }

        async fn get_prediction_record(&self, _id: Uuid) -> Result<Option<PredictionRecord>, CoreError> {
            Ok(None)
        }

    }
    
    #[tokio::test]
    async fn test_analytics_engine_creation() {
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_metric_recording() {
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await.unwrap();
        
        let event = create_metric_event(
            "test_metric",
            WorkspaceId::new(),
            MetricValue::Float(42.0),
        );
        
        let result = engine.record_metric(event).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let mut config = AnalyticsConfig::default();
        config.rate_limit = 2; // Very low limit for testing
        
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, config).await.unwrap();
        
        // Should succeed for first few events
        for i in 0..2 {
            let event = create_metric_event(
                "rate_test",
                WorkspaceId::new(),
                MetricValue::Float(i as f64),
            );
            
            let result = engine.record_metric(event).await;
            assert!(result.is_ok());
        }
        
        // Should fail on exceeding rate limit
        let event = create_metric_event(
            "rate_test",
            WorkspaceId::new(),
            MetricValue::Float(99.0),
        );
        
        let result = engine.record_metric(event).await;
        assert!(matches!(result, Err(AnalyticsError::RateLimitExceeded(_))));
    }
    
    #[tokio::test]
    async fn test_aggregators() {
        let time_range = TimeRange {
            start: Utc::now() - chrono::Duration::hours(1),
            end: Utc::now(),
        };
        
        let mut sum_agg = SumAggregator::new(time_range);
        
        for i in 1..=10 {
            let event = create_metric_event(
                "sum_test",
                WorkspaceId::new(),
                MetricValue::Float(i as f64),
            );
            
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
        let storage = Arc::new(MockStorage::new());
        
        let engine = AnalyticsEngineBuilder::new()
            .with_storage(storage)
            .with_buffer_size(5000)
            .with_flush_interval(Duration::from_secs(60))
            .with_anomaly_detection(true)
            .with_retention_period(Duration::from_secs(30 * 24 * 3600))
            .build()
            .await;
        
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert_eq!(engine.config.buffer_size, 5000);
        assert_eq!(engine.config.flush_interval, Duration::from_secs(60));
        assert!(engine.config.enable_anomaly_detection);
    }
    
    #[tokio::test]
    async fn test_time_series_management() {
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await.unwrap();
        
        // Record some time series data
        for i in 0..10 {
            let event = create_metric_event(
                "time_series_test",
                WorkspaceId::new(),
                MetricValue::Float(i as f64),
            );
            engine.record_metric(event).await.unwrap();
        }
        
        // Query time series
        let time_range = TimeRange {
            start: Utc::now() - chrono::Duration::hours(1),
            end: Utc::now() + chrono::Duration::hours(1),
        };
        
        let data = engine.get_time_series("time_series_test", time_range).await.unwrap();
        assert!(!data.is_empty());
    }
    
    #[tokio::test]
    async fn test_dashboard_metrics() {
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await.unwrap();
        
        let workspace_id = WorkspaceId::new();
        
        // Record some events
        for _ in 0..5 {
            let event = create_metric_event(
                "dashboard_test",
                workspace_id.clone(),
                MetricValue::Float(rand::random::<f64>() * 100.0),
            );
            engine.record_metric(event).await.unwrap();
        }
        
        let dashboard = engine.get_dashboard_metrics(workspace_id).await.unwrap();
        assert!(dashboard.total_events > 0);
    }
    
    #[tokio::test]
    async fn test_export_formats() {
        let storage = Arc::new(MockStorage::new());
        let engine = AnalyticsEngine::new(storage, AnalyticsConfig::default()).await.unwrap();
        
        let workspace_id = WorkspaceId::new();
        
        // Record some events
        for i in 0..3 {
            let event = create_metric_event(
                format!("export_test_{}", i),
                workspace_id.clone(),
                MetricValue::Float(i as f64),
            );
            engine.record_metric(event).await.unwrap();
        }
        
        // Force flush to ensure data is stored
        engine.force_flush().await.unwrap();
        
        let time_range = TimeRange {
            start: Utc::now() - chrono::Duration::hours(1),
            end: Utc::now() + chrono::Duration::hours(1),
        };
        
        // Test JSON export
        let json_export = engine.export_analytics(
            workspace_id.clone(),
            ExportFormat::Json,
            time_range.clone(),
        ).await.unwrap();
        assert!(!json_export.is_empty());
        
        // Test CSV export
        let csv_export = engine.export_analytics(
            workspace_id,
            ExportFormat::Csv,
            time_range,
        ).await.unwrap();
        assert!(!csv_export.is_empty());
    }
}
            

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use chrono::{DateTime, Utc, Timelike};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{RwLock};
use tracing::{debug, error, info, instrument, warn};

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
    
    #[error("CSV export error: {0}")]
    CsvExportError(String),
    
    #[error("Parquet export error: {0}")]
    ParquetExportError(String),
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
    
    /// User activity tracker
    user_tracker: Arc<RwLock<UserActivityTracker>>,
    
    /// Outcome completion tracker
    outcome_tracker: Arc<RwLock<OutcomeCompletionTracker>>,
    
    /// Response time tracker
    response_tracker: Arc<RwLock<ResponseTimeTracker>>,
    
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

/// User activity tracking
struct UserActivityTracker {
    /// Active users per workspace
    active_users: HashMap<WorkspaceId, HashMap<UserId, DateTime<Utc>>>,
}

impl UserActivityTracker {
    fn new() -> Self {
        Self {
            active_users: HashMap::new(),
        }
    }
    
    fn record_activity(&mut self, workspace_id: WorkspaceId, user_id: UserId, timestamp: DateTime<Utc>) {
        self.active_users
            .entry(workspace_id)
            .or_insert_with(HashMap::new)
            .insert(user_id, timestamp);
    }
    
    fn get_active_count(&self, workspace_id: WorkspaceId, since: DateTime<Utc>) -> u64 {
        self.active_users
            .get(&workspace_id)
            .map(|users| {
                users.values()
                    .filter(|&&last_seen| last_seen >= since)
                    .count() as u64
            })
            .unwrap_or(0)
    }
    
    fn cleanup_inactive(&mut self, cutoff: DateTime<Utc>) {
        for workspace_users in self.active_users.values_mut() {
            workspace_users.retain(|_, &mut last_seen| last_seen >= cutoff);
        }
    }
}

/// Outcome completion tracking
struct OutcomeCompletionTracker {
    /// Completed outcomes per workspace
    completions: HashMap<WorkspaceId, CompletionStats>,
}

#[derive(Debug, Clone, Default)]
struct CompletionStats {
    total_created: u64,
    total_completed: u64,
    completion_times: Vec<Duration>,
    last_updated: DateTime<Utc>,
}

impl OutcomeCompletionTracker {
    fn new() -> Self {
        Self {
            completions: HashMap::new(),
        }
    }
    
    fn record_creation(&mut self, workspace_id: WorkspaceId) {
        let stats = self.completions.entry(workspace_id).or_default();
        stats.total_created += 1;
        stats.last_updated = Utc::now();
    }
    
    fn record_completion(&mut self, workspace_id: WorkspaceId, duration: Duration) {
        let stats = self.completions.entry(workspace_id).or_default();
        stats.total_completed += 1;
        stats.completion_times.push(duration);
        stats.last_updated = Utc::now();
        
        // Keep only last 1000 completion times
        if stats.completion_times.len() > 1000 {
            stats.completion_times.drain(0..100);
        }
    }
    
    fn get_completion_rate(&self, workspace_id: WorkspaceId) -> f64 {
        self.completions
            .get(&workspace_id)
            .map(|stats| {
                if stats.total_created == 0 {
                    0.0
                } else {
                    stats.total_completed as f64 / stats.total_created as f64
                }
            })
            .unwrap_or(0.0)
    }
}

/// Response time tracking
struct ResponseTimeTracker {
    /// Response times per endpoint/metric
    response_times: HashMap<String, VecDeque<Duration>>,
    /// Maximum samples to keep
    max_samples: usize,
}

impl ResponseTimeTracker {
    fn new() -> Self {
        Self {
            response_times: HashMap::new(),
            max_samples: 1000,
        }
    }
    
    fn record_response(&mut self, metric_id: String, duration: Duration) {
        let times = self.response_times.entry(metric_id).or_insert_with(|| {
            VecDeque::with_capacity(self.max_samples)
        });
        
        times.push_back(duration);
        if times.len() > self.max_samples {
            times.pop_front();
        }
    }
    
    fn get_average(&self, metric_id: &str) -> Option<Duration> {
        self.response_times.get(metric_id).and_then(|times| {
            if times.is_empty() {
                None
            } else {
                let total: Duration = times.iter().sum();
                Some(total / times.len() as u32)
            }
        })
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
    buckets: Arc<RwLock<BTreeMap<String, TimeBucket>>>,
    
    /// Bucket duration
    bucket_duration: Duration,
}

/// Time bucket for organizing time-series data
#[derive(Debug, Clone)]
struct TimeBucket {
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    data_points: Vec<DataPoint>,
    compressed: bool,
    compressed_data: Option<Vec<u8>>,
}

impl TimeBucket {
    fn new(start_time: DateTime<Utc>, duration: Duration) -> Self {
        let end_time = start_time + chrono::Duration::from_std(duration).unwrap();
        Self {
            start_time,
            end_time,
            data_points: Vec::new(),
            compressed: false,
            compressed_data: None,
        }
    }
    
    fn add_point(&mut self, point: DataPoint) {
        if !self.compressed {
            self.data_points.push(point);
        }
    }
    
    fn compress(&mut self) -> Result<(), AnalyticsError> {
        if self.compressed || self.data_points.is_empty() {
            return Ok(());
        }
        
        // Simple compression using serde_json and zstd
        let serialized = serde_json::to_vec(&self.data_points)
        .map_err(|e| AnalyticsError::SerializationError(e))?;
        
        let compressed = zstd::encode_all(&serialized[..], 3)
            .map_err(|e| AnalyticsError::SerializationError(
                serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
            ))?;
        
        self.compressed_data = Some(compressed);
        self.data_points.clear();
        self.compressed = true;
        
        Ok(())
    }
    
    fn decompress(&mut self) -> Result<(), AnalyticsError> {
        if !self.compressed {
            return Ok(());
        }
        
        if let Some(compressed_data) = &self.compressed_data {
            let decompressed = zstd::decode_all(&compressed_data[..])
                .map_err(|e| AnalyticsError::SerializationError(
                    serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
                ))?;
            
            self.data_points = serde_json::from_slice(&decompressed)
                .map_err(|e| AnalyticsError::SerializationError(e))?;
            
            self.compressed_data = None;
            self.compressed = false;
        }
        
        Ok(())
    }
    
    fn is_expired(&self, retention_period: Duration) -> bool {
        let age = Utc::now().signed_duration_since(self.end_time);
        age > chrono::Duration::from_std(retention_period).unwrap()
    }
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
        
        // Update running statistics using Welford's algorithm
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
        
        let z_score = (value - self.mean).abs() / self.std_dev.max(0.001);
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
            buckets: Arc::new(RwLock::new(BTreeMap::new())),
            bucket_duration: Duration::from_secs(3600), // 1 hour buckets
        });
        
        let anomaly_detector = Arc::new(AnomalyDetector {
            models: Arc::new(RwLock::new(HashMap::new())),
            anomalies: Arc::new(RwLock::new(Vec::new())),
            sensitivity: 3.0, // 3 standard deviations
        });
        
        let rate_limiter = Arc::new(RateLimiter {
            limits: Arc::new(RwLock::new(HashMap::new())),
        });
        
        let user_tracker = Arc::new(RwLock::new(UserActivityTracker::new()));
        let outcome_tracker = Arc::new(RwLock::new(OutcomeCompletionTracker::new()));
        let response_tracker = Arc::new(RwLock::new(ResponseTimeTracker::new()));
        
        let engine = Self {
            storage,
            buffer,
            aggregators,
            time_series,
            anomaly_detector,
            rate_limiter,
            user_tracker,
            outcome_tracker,
            response_tracker,
            config,
        };
        
        // Start background tasks
        engine.start_background_tasks().await;
        
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
        
        // Track user activity
        if let Some(user_id) = event.user_id.clone() {
            let mut tracker = self.user_tracker.write().await;
            tracker.record_activity(event.workspace_id.clone(), user_id, event.timestamp);
        }
        
        // Track outcome metrics
        if event.metric_id.starts_with("outcome.") {
            self.track_outcome_metrics(&event).await?;
        }
        
        // Track response times
        if event.metric_id.starts_with("response_time.") {
            if let MetricValue::Duration(duration) = event.value {
                let mut tracker = self.response_tracker.write().await;
                tracker.record_response(event.metric_id.clone(), duration);
            }
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
    
    async fn track_outcome_metrics(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        let mut tracker = self.outcome_tracker.write().await;
        
        match event.metric_id.as_str() {
            "outcome.created" => {
                tracker.record_creation(event.workspace_id.clone());
            }
            "outcome.completed" => {
                if let MetricValue::Duration(duration) = event.value {
                    tracker.record_completion(event.workspace_id.clone(), duration);
                }
            }
            _ => {}
        }
        
        Ok(())
    }
    
    async fn get_active_users(
        &self,
        workspace_id: WorkspaceId,
        time_range: &TimeRange,
    ) -> AnalyticsResult<u64> {
        let tracker = self.user_tracker.read().await;
        Ok(tracker.get_active_count(workspace_id, time_range.start))
    }
    
    async fn get_outcome_completion_rate(
        &self,
        workspace_id: WorkspaceId,
        _time_range: &TimeRange,
    ) -> AnalyticsResult<f64> {
        let tracker = self.outcome_tracker.read().await;
        Ok(tracker.get_completion_rate(workspace_id))
    }
    
    async fn get_avg_response_time(
        &self,
        _workspace_id: WorkspaceId,
        _time_range: &TimeRange,
    ) -> AnalyticsResult<Duration> {
        let tracker = self.response_tracker.read().await;
        Ok(tracker.get_average("response_time.api").unwrap_or(Duration::from_millis(0)))
    }
    
    async fn get_trending_metrics(
        &self,
        workspace_id: WorkspaceId,
        limit: usize,
    ) -> AnalyticsResult<Vec<TrendingMetric>> {
        // Query recent metrics
        let now = Utc::now();
        let yesterday = now - chrono::Duration::days(1);
        let last_week = now - chrono::Duration::days(7);
        
        let current_query = MetricQuery {
            workspace_id: Some(workspace_id.clone()),
            time_range: Some(TimeRange {
                start: yesterday,
                end: now,
            }),
            ..Default::default()
        };
        
        let previous_query = MetricQuery {
            workspace_id: Some(workspace_id),
            time_range: Some(TimeRange {
                start: last_week,
                end: yesterday,
            }),
            ..Default::default()
        };
        
        let current_metrics = self.query_metrics(current_query).await?;
        let previous_metrics = self.query_metrics(previous_query).await?;
        
        // Calculate trending metrics
        let mut metric_stats: HashMap<String, (f64, f64)> = HashMap::new();
        
        for metric in current_metrics.metrics {
            if let MetricValue::Float(value) = metric.value {
                let entry = metric_stats.entry(metric.metric_id).or_insert((0.0, 0.0));
                entry.0 += value;
            }
        }
        
        for metric in previous_metrics.metrics {
            if let MetricValue::Float(value) = metric.value {
                let entry = metric_stats.entry(metric.metric_id).or_insert((0.0, 0.0));
                entry.1 += value;
            }
        }
        
        // Calculate trends
        let mut trending: Vec<TrendingMetric> = metric_stats
            .into_iter()
            .map(|(metric_id, (current, previous))| {
                let change_percentage = if previous > 0.0 {
                    ((current - previous) / previous) * 100.0
                } else if current > 0.0 {
                    100.0
                } else {
                    0.0
                };
                
                let trend = if change_percentage > 5.0 {
                    TrendDirection::Up
                } else if change_percentage < -5.0 {
                    TrendDirection::Down
                } else {
                    TrendDirection::Stable
                };
                
                TrendingMetric {
                    metric_id: metric_id.clone(),
                    name: metric_id,
                    current_value: current,
                    previous_value: previous,
                    change_percentage,
                    trend,
                }
            })
            .collect();
        
        // Sort by absolute change percentage
        trending.sort_by(|a, b| {
            b.change_percentage.abs()
                .partial_cmp(&a.change_percentage.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        trending.truncate(limit);
        Ok(trending)
    }
    
    async fn export_as_csv(&self, result: QueryResult) -> AnalyticsResult<Vec<u8>> {
        let mut wtr = Writer::from_writer(vec![]);
        
        // Write headers
        wtr.write_record(&[
            "timestamp",
            "metric_id",
            "workspace_id",
            "user_id",
            "value",
            "tags",
            "outcome_id",
        ]).map_err(|e| AnalyticsError::CsvExportError(e.to_string()))?;
        
        // Write data
        for metric in result.metrics {
            let value_str = match metric.value {
                MetricValue::Float(f) => f.to_string(),
                MetricValue::Integer(i) => i.to_string(),
                MetricValue::Duration(d) => format!("{}ms", d.as_millis()),
                MetricValue::String(s) => s,
                MetricValue::Boolean(b) => b.to_string(),
                MetricValue::Timestamp(t) => t.to_rfc3339(),
                MetricValue::Json(j) => j.to_string(),
                MetricValue::Array(a) => serde_json::to_string(&a).unwrap_or_default(),
                MetricValue::Map(m) => serde_json::to_string(&m).unwrap_or_default(),
            };
            
            wtr.write_record(&[
                metric.timestamp.to_rfc3339(),
                metric.metric_id,
                metric.workspace_id.to_string(),
                metric.user_id.map(|u| u.to_string()).unwrap_or_default(),
                value_str,
                metric.tags.join(","),
                metric.outcome_id.map(|o| o.to_string()).unwrap_or_default(),
            ]).map_err(|e| AnalyticsError::CsvExportError(e.to_string()))?;
        }
        
        wtr.into_inner()
            .map_err(|e| AnalyticsError::CsvExportError(e.to_string()))
    }
    
    async fn export_as_parquet(&self, result: QueryResult) -> AnalyticsResult<Vec<u8>> {
        // Parquet export would require arrow-rs or similar
        // For now, return JSON as fallback
        warn!("Parquet export not yet implemented, falling back to JSON");
        serde_json::to_vec_pretty(&result)
            .map_err(|e| AnalyticsError::ParquetExportError(e.to_string()))
    }
    
    async fn start_background_tasks(&self) {
        // Start flush task
        self.start_flush_task().await;
        
        // Start cleanup task
        self.start_cleanup_task().await;
        
        // Start compression task
        self.start_compression_task().await;
    }
    
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

    pub async fn get_time_series(
        &self,
        metric_id: &str,
        time_range: TimeRange,
    ) -> AnalyticsResult<Vec<DataPoint>> {
        let buckets = self.time_series.buckets.read().await;
        let mut data_points = Vec::new();
        
        for (bucket_id, bucket) in buckets.iter() {
            if bucket_id.starts_with(metric_id) {
                // Check if bucket overlaps with time range
                if bucket.start_time <= time_range.end && bucket.end_time >= time_range.start {
                    let mut bucket_clone = bucket.clone();
                    if bucket_clone.compressed {
                        bucket_clone.decompress()?;
                    }
                    
                    for point in bucket_clone.data_points {
                        if point.timestamp >= time_range.start && point.timestamp <= time_range.end {
                            data_points.push(point);
                        }
                    }
                }
            }
        }
        
        data_points.sort_by_key(|p| p.timestamp);
        Ok(data_points)
    }
    
    /// Get dashboard metrics summary
    pub async fn get_dashboard_metrics(
        &self,
        workspace_id: WorkspaceId,
    ) -> AnalyticsResult<DashboardMetrics> {
        let time_range = TimeRange {
            start: Utc::now() - chrono::Duration::days(7),
            end: Utc::now(),
        };
        
        // Get total events count
        let query = MetricQuery {
            workspace_id: Some(workspace_id.clone()),
            time_range: Some(time_range.clone()),
            ..Default::default()
        };
        
        let result = self.query_metrics(query).await?;
        let total_events = result.total_count as u64;
        
        // Get active users
        let active_users = self.get_active_users(workspace_id.clone(), &time_range).await?;
        
        // Get outcome completion rate
        let outcome_completion_rate = self.get_outcome_completion_rate(
            workspace_id.clone(),
            &time_range
        ).await?;
        
        // Get average response time
        let average_response_time = self.get_avg_response_time(
            workspace_id.clone(),
            &time_range
        ).await?;
        
        // Get anomalies
        let anomalies = {
            let detector_anomalies = self.anomaly_detector.anomalies.read().await;
            detector_anomalies.iter()
                .filter(|a| a.timestamp >= time_range.start && a.timestamp <= time_range.end)
                .cloned()
                .collect()
        };
        
        // Get trending metrics
        let trending_metrics = self.get_trending_metrics(workspace_id.clone(), 5).await?;
        
        Ok(DashboardMetrics {
            workspace_id,
            time_range,
            total_events,
            active_users,
            outcome_completion_rate,
            average_response_time,
            anomalies,
            trending_metrics,
            generated_at: Utc::now(),
        })
    }
    
    /// Export analytics data in specified format
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
                    .map_err(AnalyticsError::SerializationError)
            }
            ExportFormat::Csv => self.export_as_csv(result).await,
            ExportFormat::Parquet => self.export_as_parquet(result).await,
        }
    }
    
    /// Check rate limit for a metric
    async fn check_rate_limit(&self, metric_id: &str) -> AnalyticsResult<bool> {
        let mut limits = self.rate_limiter.limits.write().await;
        let now = Instant::now();
        
        let limit = limits.entry(metric_id.to_string()).or_insert(RateLimit {
            max_per_second: self.config.rate_limit,
            window_start: now,
            count: 0,
        });
        
        // Reset window if needed
        if now.duration_since(limit.window_start) >= Duration::from_secs(1) {
            limit.window_start = now;
            limit.count = 0;
        }
        
        if limit.count >= limit.max_per_second {
            Ok(false)
        } else {
            limit.count += 1;
            Ok(true)
        }
    }
    
    /// Determine if event should be sampled
    fn should_sample(&self) -> bool {
        if self.config.sampling_rate >= 1.0 {
            false
        } else {
            rand::random::<f64>() > self.config.sampling_rate
        }
    }
    
    /// Flush the metric buffer to storage
    pub async fn flush_buffer(&self) -> AnalyticsResult<()> {
        let events = {
            let mut buffer = self.buffer.write().await;
            buffer.flush()
        };
        
        if !events.is_empty() {
            info!("Flushing {} events to storage", events.len());
            self.storage.store_metrics(events).await
                .map_err(AnalyticsError::StorageError)?;
        }
        
        Ok(())
    }
    
    /// Force flush for testing/shutdown
    pub async fn force_flush(&self) -> AnalyticsResult<()> {
        self.flush_buffer().await
    }
    
    /// Update aggregators with new event
    async fn update_aggregators(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        let mut aggregators = self.aggregators.write().await;
        
        for aggregator in aggregators.values_mut() {
            aggregator.aggregate(event).await?;
        }
        
        Ok(())
    }
    
    /// Detect anomalies in metric events
    async fn detect_anomaly(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        if let MetricValue::Float(value) = event.value {
            let mut models = self.anomaly_detector.models.write().await;
            
            let model = models.entry(event.metric_id.clone())
                .or_insert_with(StatisticalModel::new);
            
            // Check for anomaly before updating model
            if model.is_anomaly(value, self.anomaly_detector.sensitivity) {
                let anomaly = Anomaly {
                    metric_id: event.metric_id.clone(),
                    timestamp: event.timestamp,
                    value,
                    expected_range: (
                        model.mean - (model.std_dev * self.anomaly_detector.sensitivity),
                        model.mean + (model.std_dev * self.anomaly_detector.sensitivity),
                    ),
                    severity: self.calculate_anomaly_severity(value, model),
                    metadata: event.metadata.clone(),
                };
                
                let mut anomalies = self.anomaly_detector.anomalies.write().await;
                anomalies.push(anomaly);
                
                // Keep only last 1000 anomalies
                if anomalies.len() > 1000 {
                    anomalies.drain(0..100);
                }
            }
            
            // Update model with new value
            model.update(value);
        }
        
        Ok(())
    }
    
    /// Calculate anomaly severity
    fn calculate_anomaly_severity(&self, value: f64, model: &StatisticalModel) -> AnomalySeverity {
        let z_score = (value - model.mean).abs() / model.std_dev.max(0.001);
        
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
    
    /// Update time series data
    async fn update_time_series(&self, event: &MetricEvent) -> AnalyticsResult<()> {
        let bucket_key = format!(
            "{}_{}",
            event.metric_id,
            event.timestamp.format("%Y%m%d_%H")
        );
        
        let mut buckets = self.time_series.buckets.write().await;
        
        let bucket = buckets.entry(bucket_key).or_insert_with(|| {
            let hour_start = event.timestamp
                .date_naive()
                .and_hms_opt(event.timestamp.hour(), 0, 0)
                .unwrap()
                .and_utc();
            TimeBucket::new(hour_start, self.time_series.bucket_duration)
        });
        
        if let MetricValue::Float(value) = event.value {
            bucket.add_point(DataPoint {
                timestamp: event.timestamp,
                value,
                tags: event.tags.clone(),
            });
        } else if let MetricValue::Integer(value) = event.value {
            bucket.add_point(DataPoint {
                timestamp: event.timestamp,
                value: value as f64,
                tags: event.tags.clone(),
            });
        }
        
        Ok(())
    }
    
    /// Query metrics from storage
    pub async fn query_metrics(&self, query: MetricQuery) -> AnalyticsResult<QueryResult> {
        let start = Instant::now();
        let metrics = self.storage.query_metrics(&query).await
            .map_err(AnalyticsError::StorageError)?;
        
        let total_count = metrics.len();
        let execution_time = start.elapsed();
        
        Ok(QueryResult {
            metrics,
            query,
            execution_time,
            total_count,
        })
    }
    
    /// Start cleanup background task
    async fn start_cleanup_task(&self) {
        let storage = Arc::clone(&self.storage);
        let retention_period = self.config.retention_period;
        let time_series = Arc::clone(&self.time_series);
        let user_tracker = Arc::clone(&self.user_tracker);
        
        tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(3600)); // Every hour
            
            loop {
                cleanup_interval.tick().await;
                
                // Cleanup old metrics
                let cutoff = Utc::now() - chrono::Duration::from_std(retention_period).unwrap();
                if let Err(e) = storage.delete_old_metrics(cutoff).await {
                    error!("Failed to cleanup old metrics: {}", e);
                }
                
                // Cleanup expired time series buckets
                let mut buckets = time_series.buckets.write().await;
                buckets.retain(|_, bucket| !bucket.is_expired(retention_period));
                
                // Cleanup inactive users
                let mut tracker = user_tracker.write().await;
                tracker.cleanup_inactive(cutoff);
            }
        });
    }
    
    /// Start compression background task
    async fn start_compression_task(&self) {
        if !self.config.enable_compression {
            return;
        }
        
        let time_series = Arc::clone(&self.time_series);
        
        tokio::spawn(async move {
            let mut compression_interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
            
            loop {
                compression_interval.tick().await;
                
                let mut buckets = time_series.buckets.write().await;
                let one_hour_ago = Utc::now() - chrono::Duration::hours(1);
                
                for bucket in buckets.values_mut() {
                    // Compress buckets older than 1 hour
                    if bucket.end_time < one_hour_ago && !bucket.compressed {
                        if let Err(e) = bucket.compress() {
                            error!("Failed to compress bucket: {}", e);
                        }
                    }
                }
            }
        });
    }
    
    /// Static flush buffer method for background task
    async fn flush_buffer_static(
        buffer: Arc<RwLock<MetricBuffer>>,
        storage: Arc<dyn StorageBackend>,
    ) -> AnalyticsResult<()> {
        let events = {
            let mut buffer = buffer.write().await;
            buffer.flush()
        };
        
        if !events.is_empty() {
            info!("Background flush: {} events", events.len());
            storage.store_metrics(events).await
                .map_err(AnalyticsError::StorageError)?;
        }
        
        Ok(())
    }
    
    /// Get engine statistics
    pub async fn get_statistics(&self) -> AnalyticsResult<EngineStatistics> {
        let buffer = self.buffer.read().await;
        let aggregators = self.aggregators.read().await;
        let buckets = self.time_series.buckets.read().await;
        let anomalies = self.anomaly_detector.anomalies.read().await;
        let models = self.anomaly_detector.models.read().await;
        
        Ok(EngineStatistics {
            buffered_events: buffer.events.len(),
            active_aggregators: aggregators.len(),
            time_series_buckets: buckets.len(),
            detected_anomalies: anomalies.len(),
            trained_models: models.len(),
            compression_enabled: self.config.enable_compression,
            anomaly_detection_enabled: self.config.enable_anomaly_detection,
        })
    }
}