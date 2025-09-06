//! # Interstice Core
//! 
//! Core domain models and business logic for the Interstice WorkOS platform.
//! This module provides the foundational engine for processing work artifacts
//! across multiple platforms with ML-powered outcome prediction.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

// Re-export all public modules
pub mod analytics;
pub mod artifact;
pub mod error;
pub mod graph;
pub mod outcome;
pub mod storage;
pub mod traits;
pub mod types;

// Re-export core types for convenience
pub use artifact::{
    Artifact, ArtifactError, ArtifactExtractor, ArtifactProcessor, ArtifactQuery, ArtifactStats,
    ArtifactType, ProcessingResult,
};
pub use error::{CoreError, Result};
pub use outcome::{Outcome, OutcomeMapper, OutcomeType};
pub use storage::{ProgressPoint, StorageBackend, WorkspaceStats};
pub use traits::{MLPredictor, OutcomePrediction};
pub use types::{Platform, UserId, WorkspaceId};

use crate::{analytics::{MetricEvent, MetricQuery}, outcome::{OutcomeFilters, OutcomeId}, storage::{ArtifactFilters, CleanupStats}, types::SystemEvent};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Configuration for the Interstice engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum number of artifacts to process in parallel
    pub max_parallel_artifacts: usize,
    
    /// Timeout for processing a single artifact
    pub processing_timeout: Duration,
    
    /// Enable caching of processing results
    pub enable_cache: bool,
    
    /// Cache TTL in seconds
    pub cache_ttl: u64,
    
    /// Maximum artifact content size in bytes
    pub max_content_size: usize,
    
    /// Enable ML predictions
    pub enable_ml: bool,
    
    /// Batch size for ML predictions
    pub ml_batch_size: usize,
    
    /// Enable outcome mapping
    pub enable_outcome_mapping: bool,
    
    /// Telemetry configuration
    pub telemetry: TelemetryConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_parallel_artifacts: 100,
            processing_timeout: Duration::from_secs(30),
            enable_cache: true,
            cache_ttl: 3600, // 1 hour
            max_content_size: 10_485_760, // 10MB
            enable_ml: true,
            ml_batch_size: 32,
            enable_outcome_mapping: true,
            telemetry: TelemetryConfig::default(),
        }
    }
}

impl EngineConfig {
    /// Create a production configuration
    pub fn production() -> Self {
        Self {
            max_parallel_artifacts: 500,
            processing_timeout: Duration::from_secs(60),
            enable_cache: true,
            cache_ttl: 7200, // 2 hours
            max_content_size: 20_971_520, // 20MB
            enable_ml: true,
            ml_batch_size: 64,
            enable_outcome_mapping: true,
            telemetry: TelemetryConfig::production(),
        }
    }
    
    /// Create a development configuration
    pub fn development() -> Self {
        Self {
            max_parallel_artifacts: 50,
            processing_timeout: Duration::from_secs(120),
            enable_cache: false,
            cache_ttl: 300, // 5 minutes
            max_content_size: 5_242_880, // 5MB
            enable_ml: true,
            ml_batch_size: 16,
            enable_outcome_mapping: true,
            telemetry: TelemetryConfig::development(),
        }
    }
}

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Sampling rate for traces (0.0 to 1.0)
    pub trace_sampling_rate: f64,
    
    /// Export metrics interval in seconds
    pub metrics_export_interval: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            trace_sampling_rate: 0.1,
            metrics_export_interval: 60,
        }
    }
}

impl TelemetryConfig {
    pub fn production() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            trace_sampling_rate: 0.01, // 1% sampling in production
            metrics_export_interval: 30,
        }
    }
    
    pub fn development() -> Self {
        Self {
            enable_metrics: false,
            enable_tracing: true,
            trace_sampling_rate: 1.0, // 100% sampling in development
            metrics_export_interval: 10,
        }
    }
}

/// The main engine that processes artifacts from any platform
pub struct IntersticeEngine {
    /// Configuration
    config: EngineConfig,
    
    /// Artifact extractor
    extractor: Arc<ArtifactExtractor>,
    
    /// Artifact processor with ML capabilities
    processor: Arc<ArtifactProcessor>,
    
    /// Outcome mapper
    mapper: Arc<OutcomeMapper>,
    
    /// Storage backend
    storage: Option<Arc<dyn StorageBackend>>,
    
    /// ML predictor
    ml_predictor: Option<Arc<dyn MLPredictor>>,
    
    /// Metrics collector
    metrics: Arc<Metrics>,
}

impl IntersticeEngine {
    /// Create a new engine with default configuration
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }
    
    /// Create a new engine with custom configuration
    pub fn with_config(config: EngineConfig) -> Self {
        let extractor = Arc::new(ArtifactExtractor::new());
        let processor = Arc::new(ArtifactProcessor::new(None));
        let mapper = Arc::new(OutcomeMapper::new(None, Arc::new(NullStorage)));
        let metrics = Arc::new(Metrics::new());
        
        Self {
            config,
            extractor,
            processor,
            mapper,
            storage: None,
            ml_predictor: None,
            metrics,
        }
    }
    
    /// Set the ML predictor
    pub fn with_ml_predictor(mut self, predictor: Arc<dyn MLPredictor>) -> Self {
        // Update processor with ML predictor
        self.processor = Arc::new(ArtifactProcessor::new(Some(predictor.clone())));
        
        // Update mapper with ML predictor
        let storage = self.storage.clone().unwrap_or_else(|| Arc::new(NullStorage));
        self.mapper = Arc::new(OutcomeMapper::new(Some(predictor.clone()), storage));
        
        self.ml_predictor = Some(predictor);
        self
    }
    
    /// Set the storage backend
    pub fn with_storage(mut self, storage: Arc<dyn StorageBackend>) -> Self {
        // Update mapper with new storage
        if let Some(predictor) = &self.ml_predictor {
            self.mapper = Arc::new(OutcomeMapper::new(Some(predictor.clone()), storage.clone()));
        } else {
            self.mapper = Arc::new(OutcomeMapper::new(None, storage.clone()));
        }
        
        self.storage = Some(storage);
        self
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: EngineConfig) {
        self.config = config;
    }
    
    /// Process raw content and extract artifacts
    #[instrument(skip(self, content), fields(platform = %platform))]
    pub async fn process(
        &self,
        content: String,
        platform: Platform,
    ) -> Result<ProcessedData> {
        let start = std::time::Instant::now();
        self.metrics.increment_processing_attempts();
        
        // Validate content size
        if content.len() > self.config.max_content_size {
            self.metrics.increment_processing_errors();
            return Err(CoreError::Validation(format!(
                "Content size {} exceeds maximum {}",
                content.len(),
                self.config.max_content_size
            )));
        }
        
        // Extract artifacts with timeout
        let artifacts = tokio::time::timeout(
            self.config.processing_timeout,
            self.extractor.extract(&content, platform),
        )
        .await
        .map_err(|_| CoreError::Timeout(self.config.processing_timeout))?
        .map_err(|e| CoreError::Internal(e.to_string()))?;
        
        debug!("Extracted {} artifacts from {} content", artifacts.len(), platform);
        
        // Process artifacts in parallel with concurrency limit
        let mut processing_results = Vec::new();
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_parallel_artifacts));
        
        let futures: Vec<_> = artifacts
            .iter()
            .map(|artifact| {
                let processor = self.processor.clone();
                let artifact = artifact.clone();
                let permit = semaphore.clone().acquire_owned();
                
                async move {
                    let _permit = permit.await.ok()?;
                    processor.process(&artifact).await.ok()
                }
            })
            .collect();
        
        let results = futures::future::join_all(futures).await;
        for result in results.into_iter().flatten() {
            processing_results.push(result);
        }
        
        // Generate outcome predictions if ML is enabled
        let predictions = if self.config.enable_ml && !artifacts.is_empty() {
            if let Some(predictor) = &self.ml_predictor {
                match predictor.predict_outcomes(&artifacts).await {
                    Ok(preds) => preds,
                    Err(e) => {
                        warn!("ML prediction failed: {}", e);
                        self.metrics.increment_ml_failures();
                        Vec::new()
                    }
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        
        // Map to outcomes if enabled
        let outcome_predictions = if self.config.enable_outcome_mapping && !artifacts.is_empty() {
            match self.mapper.predict(&artifacts).await {
                Ok(predictions) => predictions,
                Err(e) => {
                    warn!("Outcome mapping failed: {}", e);
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        
        let elapsed = start.elapsed();
        self.metrics.record_processing_time(elapsed);
        self.metrics.increment_processing_success();
        
        info!(
            "Processed {} artifacts with {} predictions in {:?}",
            artifacts.len(),
            predictions.len(),
            elapsed
        );
        
        Ok(ProcessedData {
            artifacts,
            predictions,
            outcomes: Vec::new(), // Empty for now since we only have predictions
            processing_results,
            platform,
            metadata: ProcessingMetadata {
                duration: elapsed,
                timestamp: chrono::Utc::now(),
                engine_version: VERSION.to_string(),
            },
        })
    }
    
    /// Extract artifacts from content
    #[instrument(skip(self, content), fields(platform = %platform))]
    pub async fn extract_artifacts(
        &self,
        content: &str,
        platform: Platform,
    ) -> Result<Vec<Artifact>> {
        self.extractor
            .extract(content, platform)
            .await
            .map_err(|e| CoreError::Internal(e.to_string()))
    }
    
    /// Store processed data
    #[instrument(skip(self, processed), fields(workspace_id = %workspace_id))]
    pub async fn store_processed_data(
        &self,
        processed: &ProcessedData,
        workspace_id: WorkspaceId,
    ) -> Result<()> {
        let storage = self.storage.as_ref().ok_or_else(|| {
            CoreError::Configuration("No storage backend configured".to_string())
        })?;
        
        // Store artifacts and link outcomes
        for artifact in &processed.artifacts {
            let artifact_id = storage.store_artifact(artifact.clone()).await?;
            
            // Store predictions
            for prediction in &processed.predictions {
                storage
                    .link_artifact_outcome(artifact_id, OutcomeId::from_uuid(prediction.outcome_id), prediction.confidence as f64, None)
                    .await?;
            }
            
            // Store mapped outcomes
            for outcome in &processed.outcomes {
                storage.store_outcome(outcome.clone()).await?;
            }
        }
        
        info!("Stored {} artifacts for workspace {}", processed.artifacts.len(), workspace_id);
        Ok(())
    }
    
    /// Get workspace statistics
    #[instrument(skip(self), fields(workspace_id = %workspace_id))]
    pub async fn get_workspace_stats(&self, workspace_id: WorkspaceId) -> Result<WorkspaceStats> {
        let storage = self.storage.as_ref().ok_or_else(|| {
            CoreError::Configuration("No storage backend configured".to_string())
        })?;
        
        storage.get_workspace_stats(workspace_id).await
    }
    
    /// Query artifacts with filters
    #[instrument(skip(self, query))]
    pub async fn query_artifacts(&self, workspace_id: WorkspaceId, query: ArtifactQuery) -> Result<Vec<Artifact>> {
        let storage = self.storage.as_ref().ok_or_else(|| {
            CoreError::Configuration("No storage backend configured".to_string())
        })?;
        
        storage.query_artifacts(workspace_id, None).await
    }
    
    /// Get artifact statistics
    #[instrument(skip(self), fields(workspace_id = %workspace_id))]
    pub async fn get_artifact_stats(&self, workspace_id: WorkspaceId) -> Result<ArtifactStats> {
        let artifacts = self
            .query_artifacts(
                workspace_id,
                ArtifactQuery::builder()
                    .workspace(workspace_id)
                    .limit(1000)
                    .build(),
            )
            .await?;
        
        Ok(ArtifactStats::calculate(&artifacts))
    }
    
    /// Health check
    pub async fn health_check(&self) -> HealthStatus {
        let mut status = HealthStatus {
            healthy: true,
            version: VERSION.to_string(),
            components: Vec::new(),
        };
        
        // Check storage
        if let Some(storage) = &self.storage {
            let storage_health = if storage.health_check().await.is_ok() {
                ComponentHealth {
                    name: "storage".to_string(),
                    healthy: true,
                    message: "Storage backend is operational".to_string(),
                }
            } else {
                status.healthy = false;
                ComponentHealth {
                    name: "storage".to_string(),
                    healthy: false,
                    message: "Storage backend is not responding".to_string(),
                }
            };
            status.components.push(storage_health);
        }
        
        // Check ML predictor
        if let Some(_predictor) = &self.ml_predictor {
            status.components.push(ComponentHealth {
                name: "ml_predictor".to_string(),
                healthy: true,
                message: "ML predictor is configured".to_string(),
            });
        }
        
        // Add metrics
        status.components.push(ComponentHealth {
            name: "metrics".to_string(),
            healthy: true,
            message: format!(
                "Processed: {}, Success: {}, Errors: {}",
                self.metrics.processing_attempts.load(std::sync::atomic::Ordering::Relaxed),
                self.metrics.processing_success.load(std::sync::atomic::Ordering::Relaxed),
                self.metrics.processing_errors.load(std::sync::atomic::Ordering::Relaxed),
            ),
        });
        
        status
    }
    
    /// Get engine metrics
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }
}

/// Processed data from the engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    /// Extracted artifacts
    pub artifacts: Vec<Artifact>,
    
    /// ML predictions
    pub predictions: Vec<OutcomePrediction>,
    
    /// Mapped outcomes
    pub outcomes: Vec<Outcome>,
    
    /// Processing results for each artifact
    pub processing_results: Vec<ProcessingResult>,
    
    /// Source platform
    pub platform: Platform,
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Processing duration
    pub duration: Duration,
    
    /// Processing timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Engine version
    pub engine_version: String,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub version: String,
    pub components: Vec<ComponentHealth>,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub healthy: bool,
    pub message: String,
}

/// Engine metrics
pub struct Metrics {
    processing_attempts: std::sync::atomic::AtomicU64,
    processing_success: std::sync::atomic::AtomicU64,
    processing_errors: std::sync::atomic::AtomicU64,
    ml_failures: std::sync::atomic::AtomicU64,
    total_processing_time: std::sync::Arc<parking_lot::RwLock<Duration>>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            processing_attempts: std::sync::atomic::AtomicU64::new(0),
            processing_success: std::sync::atomic::AtomicU64::new(0),
            processing_errors: std::sync::atomic::AtomicU64::new(0),
            ml_failures: std::sync::atomic::AtomicU64::new(0),
            total_processing_time: Arc::new(parking_lot::RwLock::new(Duration::from_secs(0))),
        }
    }
    
    fn increment_processing_attempts(&self) {
        self.processing_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_processing_success(&self) {
        self.processing_success.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_processing_errors(&self) {
        self.processing_errors.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_ml_failures(&self) {
        self.ml_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_processing_time(&self, duration: Duration) {
        let mut total = self.total_processing_time.write();
        *total += duration;
    }
    
    pub fn processing_attempts(&self) -> u64 {
        self.processing_attempts.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn processing_success(&self) -> u64 {
        self.processing_success.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn processing_errors(&self) -> u64 {
        self.processing_errors.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn ml_failures(&self) -> u64 {
        self.ml_failures.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn average_processing_time(&self) -> Duration {
        let attempts = self.processing_attempts();
        if attempts == 0 {
            Duration::from_secs(0)
        } else {
            let total = self.total_processing_time.read();
            *total / attempts as u32
        }
    }
}

/// Null storage implementation for when no storage is configured
struct NullStorage;

#[async_trait]
impl StorageBackend for NullStorage {
    async fn store_metrics(&self, _events: Vec<MetricEvent>) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn query_metrics(&self, _query: &MetricQuery) -> AnyhowResult<Vec<MetricEvent>, CoreError> {
        Ok(Vec::new())
    }
    
    async fn delete_old_metrics(&self, _before: chrono::DateTime<chrono::Utc>) -> AnyhowResult<u64, CoreError> {
        Ok(0)
    }
    
    async fn store_outcome(&self, _outcome: Outcome) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn get_outcome(&self, _id: OutcomeId) -> AnyhowResult<Option<Outcome>, CoreError> {
        Ok(None)
    }
    
    async fn update_outcome(&self, _outcome: Outcome) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn delete_outcome(&self, _id: OutcomeId) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn query_outcomes(&self, _workspace_id: WorkspaceId, _filters: Option<OutcomeFilters>) -> AnyhowResult<Vec<Outcome>, CoreError> {
        Ok(Vec::new())
    }
    
    async fn store_artifact(&self, _artifact: Artifact) -> AnyhowResult<Uuid, CoreError> {
        Ok(Uuid::new_v4())
    }
    
    async fn get_artifact(&self, _id: Uuid) -> AnyhowResult<Option<Artifact>, CoreError> {
        Ok(None)
    }
    
    async fn query_artifacts(&self, _workspace_id: WorkspaceId, _filters: Option<ArtifactFilters>) -> AnyhowResult<Vec<Artifact>, CoreError> {
        Ok(Vec::new())
    }
    
    async fn delete_artifact(&self, _id: Uuid) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn link_artifact_outcome(
        &self,
        _artifact_id: Uuid,
        _outcome_id: OutcomeId,
        _confidence: f64,
        _metadata: Option<serde_json::Value>,
    ) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn store_event(&self, _event: SystemEvent) -> AnyhowResult<(), CoreError> {
        Ok(())
    }
    
    async fn get_workspace_stats(&self, _workspace_id: WorkspaceId) -> AnyhowResult<WorkspaceStats, CoreError> {
        Ok(WorkspaceStats {
            total_artifacts: 0,
            total_outcomes: 0,
            completed_outcomes: 0,
            total_metrics: 0,
            recent_activity: 0,
            mapped_work_percentage: 0.0,
            created_at: chrono::Utc::now(),
            workspace_id: WorkspaceId::new(),
        })
    }
    
    async fn health_check(&self) -> AnyhowResult<bool, CoreError> {
        Ok(true)
    }
    
    async fn cleanup_expired_data(&self) -> AnyhowResult<CleanupStats, CoreError> {
        Ok(CleanupStats {
            artifacts_deleted: 0,
            tokens_deleted: 0,
            metrics_deleted: 0,
            events_deleted: 0,
            timestamp: chrono::Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_creation() {
        let engine = IntersticeEngine::new();
        assert!(engine.storage.is_none());
        assert!(engine.ml_predictor.is_none());
    }
    
    #[test]
    fn test_config_profiles() {
        let prod = EngineConfig::production();
        assert_eq!(prod.max_parallel_artifacts, 500);
        assert!(prod.enable_cache);
        
        let dev = EngineConfig::development();
        assert_eq!(dev.max_parallel_artifacts, 50);
        assert!(!dev.enable_cache);
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let engine = IntersticeEngine::new();
        let health = engine.health_check().await;
        assert!(health.healthy);
        assert_eq!(health.version, VERSION);
    }
    
    #[test]
    fn test_metrics() {
        let metrics = Metrics::new();
        metrics.increment_processing_attempts();
        metrics.increment_processing_success();
        assert_eq!(metrics.processing_attempts(), 1);
        assert_eq!(metrics.processing_success(), 1);
        assert_eq!(metrics.processing_errors(), 0);
    }
}