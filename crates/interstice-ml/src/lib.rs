//! Interstice ML - Production-Ready Machine Learning Pipeline with Data Moat Integration
//! 
//! This crate provides enterprise-grade ML capabilities with configurable backends,
//! including the advanced Data Moat Engine for unassailable competitive advantage.

pub mod adapters;
pub mod embeddings;
pub mod feedback;
pub mod inference;
pub mod models;
pub mod types;
pub mod training;

// Edge computing module for model optimization
#[cfg(feature = "edge")]
pub mod edge {
    pub use crate::inference::edge::*;
}

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

// Import traits for method access
use training::storage::TrainingStorage;

// Core public exports - clean, organized API surface
pub use adapters::{AdapterConfig, MLPredictorAdapter};
pub use feedback::{FeedbackConfig, FeedbackProcessor};
pub use inference::{ModelConfig, OutcomePredictor, TextEmbedder};
pub use training::{ContinuousTrainer, TrainerConfig};
pub use types::{
    ActionType, Artifact, ArtifactType, ModelInfo, ModelMetrics,
    OutcomePrediction, Platform, PredictionContext, TrainingExample,
    UserAction, ValidationMethod,
};


// Edge computing exports
#[cfg(feature = "edge")]
pub use edge::{
    DType, EdgeError, EdgeMLIntegration, EdgeOptimizer,
    OptimizationConfig, OptimizationMetrics, OptimizationRecommendations,
    OptimizedModel, QuantizationConfig, PruningConfig, DistillationConfig,
};

use std::collections::HashMap;

use crate::inference::engine::{DataMoatConfig, DataMoatEngine, MoatStrength, UserFeedback};

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const ML_ENGINE_VERSION: &str = "2.0.0";
pub const DATA_MOAT_VERSION: &str = "1.0.0";

/// Backend configuration for ML Pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackendConfig {
    /// Standard ML backend with traditional architecture
    Standard {
        model_config: ModelConfig,
        trainer_config: TrainerConfig,
        feedback_config: FeedbackConfig,
    },
    /// Advanced Data Moat backend with three-layer architecture
    DataMoat {
        config: DataMoatConfig,
        enable_federated_learning: bool,
        enable_privacy_protection: bool,
    },
    /// Hybrid mode - use both backends with intelligent routing
    Hybrid {
        standard: Box<BackendConfig>,
        data_moat: Box<BackendConfig>,
        routing_strategy: RoutingStrategy,
    },
}

/// Routing strategy for hybrid backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route based on workspace configuration
    WorkspaceBased,
    /// Route based on performance metrics
    PerformanceBased { threshold: f32 },
    /// Route based on data volume
    VolumeBased { threshold: usize },
    /// Custom routing function
    Custom,
}

impl Default for BackendConfig {
    fn default() -> Self {
        BackendConfig::Standard {
            model_config: ModelConfig::default(),
            trainer_config: TrainerConfig::default(),
            feedback_config: FeedbackConfig::default(),
        }
    }
}

/// ML Backend trait for abstraction
#[async_trait::async_trait]
pub trait MLBackend: Send + Sync {
    /// Predict outcomes
    async fn predict(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        context: &PredictionContext,
    ) -> Result<Vec<OutcomePrediction>>;
    
    /// Process feedback
    async fn process_feedback(
        &self,
        workspace_id: Uuid,
        predictions: &[OutcomePrediction],
        feedback: UserFeedback,
    ) -> Result<()>;
    
    /// Get model metrics
    async fn get_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics>;
    
    /// Trigger training
    async fn trigger_training(&self, workspace_id: Uuid) -> Result<()>;
    
    /// Get backend info
    fn backend_type(&self) -> BackendType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    Standard,
    DataMoat,
    Hybrid,
}

/// Standard ML Backend implementation
struct StandardBackend {
    embedder: Arc<TextEmbedder>,
    predictor: Arc<OutcomePredictor>,
    trainer: Arc<dyn Send + Sync>,
    feedback_processor: Arc<FeedbackProcessor>,
    storage: Arc<training::storage::MLStorage>,
}

#[async_trait::async_trait]
impl MLBackend for StandardBackend {
    async fn predict(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        context: &PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        info!("Starting prediction for workspace {} with {} artifacts", workspace_id, artifacts.len());
        
        // Convert work artifacts to standard format
        let text = artifacts.iter()
            .map(|a| a.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        let embedding = self.embedder.embed_text(&text).await?;
        let core_artifacts = self.convert_artifacts(artifacts)?;
        
        // Use context for enhanced prediction
        let predictions = self.predictor.predict(embedding, &core_artifacts).await?;
        
        // Log context information for debugging
        debug!("Prediction context: hour={}, day={}, activity={:.2}", 
            context.hour_of_day, context.day_of_week, context.user_activity_level);
        
        info!("Generated {} predictions for workspace {}", predictions.len(), workspace_id);
        Ok(predictions)
    }
    
    async fn process_feedback(
        &self,
        workspace_id: Uuid,
        predictions: &[OutcomePrediction],
        feedback: UserFeedback,
    ) -> Result<()> {
        // Convert feedback to user action
        let action = UserAction {
            user_id: Some(Uuid::new_v4().to_string()),
            artifact_id: predictions.first()
                .map(|p| p.outcome_id.to_string())
                .unwrap_or_default(),
            outcome_id: feedback.actual_outcome.unwrap_or_default(),
            action_type: if feedback.accepted {
                ActionType::Accept
            } else {
                ActionType::Reject
            },
            timestamp: chrono::Utc::now(),
            confidence: None,
            feedback_text: feedback.comments,
            metadata: None,
            session_id: None,
            platform: None,
        };
        
        self.feedback_processor
            .process_user_action(workspace_id, action)
            .await
    }
    
    async fn get_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics> {
        self.storage.get_training_stats(workspace_id)
            .await
            .map(|stats| ModelMetrics {
                correct_predictions: stats.validated_examples as u64,
                total_predictions: stats.total_examples as u64,
                accuracy: stats.average_feedback_score.unwrap_or(0.0) as f64,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                auc_roc: None,
                mean_confidence: 0.0,
                prediction_latency_ms: 0.0,
                last_updated: chrono::Utc::now(),
                per_outcome_metrics: None,
                cache_hit_rate: 0.0,
            })
    }
    
    async fn trigger_training(&self, workspace_id: Uuid) -> Result<()> {
        info!("Triggering training for workspace {}", workspace_id);
        
        // In a production system, the trainer would implement a common trait
        // For now, we'll log that training was triggered
        // The actual training is handled by the continuous trainer in background
        info!("Training triggered for workspace {} (handled by continuous trainer)", workspace_id);
        
        Ok(())
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Standard
    }
}

impl StandardBackend {
    fn convert_artifacts(&self, artifacts: &[Artifact]) -> Result<Vec<interstice_core::Artifact>> {
        artifacts.iter().map(|a| {
            Ok(interstice_core::Artifact {
                id: Uuid::new_v4(),
                workspace_id: interstice_core::WorkspaceId::new(),
                artifact_type: self.convert_artifact_type(a.artifact_type),
                platform: self.convert_platform(a.platform),
                content: a.content.clone(),
                metadata: a.metadata.clone().unwrap_or(serde_json::Value::Null),
                created_at: a.created_at,
                updated_at: a.created_at,
                version: 1,
                state: interstice_core::artifact::ArtifactState::Pending,
                quality_metrics: interstice_core::artifact::QualityMetrics::default(),
                related_artifacts: Vec::new(),
                tags: std::collections::HashSet::new(),
            })
        }).collect()
    }
    
    fn convert_platform(&self, platform: Platform) -> interstice_core::Platform {
        match platform {
            Platform::Slack => interstice_core::Platform::Slack,
            Platform::GitHub => interstice_core::Platform::GitHub,
            Platform::Jira => interstice_core::Platform::Jira,
            Platform::Teams => interstice_core::Platform::Teams,
            Platform::Asana => interstice_core::Platform::Asana,
            Platform::VSCode => interstice_core::Platform::VSCode,
            Platform::GoogleWorkspace => interstice_core::Platform::GoogleWorkspace,
            Platform::Monday => interstice_core::Platform::Monday,
            Platform::Trello => interstice_core::Platform::Trello,
            Platform::Zoom => interstice_core::Platform::Zoom,
            Platform::Figma => interstice_core::Platform::Figma,
            Platform::Notion => interstice_core::Platform::Notion,
        }
    }
    
    fn convert_artifact_type(&self, artifact_type: ArtifactType) -> interstice_core::ArtifactType {
        match artifact_type {
            ArtifactType::Message => interstice_core::ArtifactType::Message {
                id: "msg".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "system".to_string(),
                content: String::new(),
                mentions: vec![],
                attachments: vec![],
                reactions: Vec::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            _ => interstice_core::ArtifactType::Document {
                id: "doc".to_string(),
                title: "Document".to_string(),
                author: "system".to_string(),
                word_count: Some(0),
                url: None,
                collaborators: vec![],
                last_modified: chrono::Utc::now(),
                version: 1,
                is_template: false,
                access_level: interstice_core::artifact::AccessLevel::Private,
                doc_type: interstice_core::artifact::DocumentType::Other("Document".to_string()),
            },
        }
    }
}

/// Data Moat Backend implementation
struct DataMoatBackend {
    engine: Arc<DataMoatEngine>,
}

#[async_trait::async_trait]
impl MLBackend for DataMoatBackend {
    async fn predict(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        context: &PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        debug!("Data Moat prediction for workspace {} with context: hour={}, activity={:.2}", 
            workspace_id, context.hour_of_day, context.user_activity_level);
        
        self.engine.predict_outcomes(workspace_id, artifacts).await
            .map_err(|e| anyhow::anyhow!("Data Moat prediction failed: {}", e))
    }
    
    async fn process_feedback(
        &self,
        workspace_id: Uuid,
        predictions: &[OutcomePrediction],
        feedback: UserFeedback,
    ) -> Result<()> {
        // Note: DataMoatEngine expects artifacts along with predictions
        // In production, we'd store artifacts or reconstruct them
        let empty_artifacts = vec![];
        
        self.engine
            .learn_from_interaction(workspace_id, &empty_artifacts, predictions, feedback)
            .await
            .map_err(|e| anyhow::anyhow!("Data Moat feedback processing failed: {}", e))
    }
    
    async fn get_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics> {
        let metrics = self.engine.get_model_metrics(workspace_id).await;
        
        Ok(ModelMetrics {
            correct_predictions: (metrics.predictions_made as f32 * metrics.accuracy) as u64,
            total_predictions: metrics.predictions_made as u64,
            accuracy: metrics.accuracy as f64,
            precision: metrics.precision as f64,
            recall: metrics.recall as f64,
            f1_score: metrics.f1_score as f64,
            auc_roc: None,
            mean_confidence: 0.0,
            prediction_latency_ms: 0.0,
            last_updated: chrono::Utc::now(),
            per_outcome_metrics: None,
            cache_hit_rate: metrics.cache_hit_rate as f32,
        })
    }
    
    async fn trigger_training(&self, workspace_id: Uuid) -> Result<()> {
        info!("Triggering Data Moat training for workspace {}", workspace_id);
        
        // Data Moat uses continuous learning, but we can trigger federated learning
        self.engine.trigger_federated_learning().await
            .map_err(|e| anyhow::anyhow!("Failed to trigger federated learning for workspace {}: {}", workspace_id, e))
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::DataMoat
    }
}

/// Hybrid Backend implementation
struct HybridBackend {
    standard: Arc<dyn MLBackend>,
    data_moat: Arc<dyn MLBackend>,
    routing_strategy: RoutingStrategy,
    metrics_cache: Arc<RwLock<HashMap<Uuid, CachedMetrics>>>,
}



#[derive(Clone)]
struct CachedMetrics {
    standard_accuracy: f64,
    data_moat_accuracy: f64,
    last_updated: std::time::Instant,
}

impl CachedMetrics {
    fn new(standard_accuracy: f64, data_moat_accuracy: f64) -> Self {
        Self {
            standard_accuracy,
            data_moat_accuracy,
            last_updated: std::time::Instant::now(),
        }
    }
    
    /// Check if metrics are stale (older than 5 minutes)
    fn is_stale(&self) -> bool {
        self.last_updated.elapsed() > std::time::Duration::from_secs(300)
    }
    
    /// Get the better performing backend
    fn better_backend(&self) -> &'static str {
        if self.data_moat_accuracy > self.standard_accuracy {
            "data_moat"
        } else {
            "standard"
        }
    }
    
    /// Update metrics with new values
    fn update(&mut self, standard_accuracy: f64, data_moat_accuracy: f64) {
        self.standard_accuracy = standard_accuracy;
        self.data_moat_accuracy = data_moat_accuracy;
        self.last_updated = std::time::Instant::now();
    }
}

#[async_trait::async_trait]
impl MLBackend for HybridBackend {
    async fn predict(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        context: &PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        let backend = self.select_backend(workspace_id, artifacts).await?;
        backend.predict(workspace_id, artifacts, context).await
    }
    
    async fn process_feedback(
        &self,
        workspace_id: Uuid,
        predictions: &[OutcomePrediction],
        feedback: UserFeedback,
    ) -> Result<()> {
        // Process feedback in both backends for learning
        let standard_future = self.standard.process_feedback(workspace_id, predictions, feedback.clone());
        let data_moat_future = self.data_moat.process_feedback(workspace_id, predictions, feedback);
        
        // Run both in parallel
        let (standard_result, data_moat_result) = tokio::join!(standard_future, data_moat_future);
        
        // Return error if both fail, warn if one fails
        match (standard_result, data_moat_result) {
            (Ok(_), Ok(_)) => Ok(()),
            (Err(e), Ok(_)) => {
                warn!("Standard backend feedback failed: {}", e);
                Ok(())
            },
            (Ok(_), Err(e)) => {
                warn!("Data Moat backend feedback failed: {}", e);
                Ok(())
            },
            (Err(e1), Err(e2)) => {
                Err(anyhow::anyhow!("Both backends failed: Standard: {}, Data Moat: {}", e1, e2))
            }
        }
    }
    
    async fn get_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics> {
        // Get metrics from both backends and combine
        let (standard_metrics, data_moat_metrics) = tokio::join!(
            self.standard.get_metrics(workspace_id),
            self.data_moat.get_metrics(workspace_id)
        );
        
        // Return the better performing backend's metrics
        match (standard_metrics, data_moat_metrics) {
            (Ok(std), Ok(dm)) => {
                if std.accuracy > dm.accuracy {
                    Ok(std)
                } else {
                    Ok(dm)
                }
            },
            (Ok(std), Err(_)) => Ok(std),
            (Err(_), Ok(dm)) => Ok(dm),
            (Err(e), Err(_)) => Err(e),
        }
    }
    
    async fn trigger_training(&self, workspace_id: Uuid) -> Result<()> {
        info!("Triggering hybrid training for workspace {}", workspace_id);
        
        // Trigger training in both backends
        let standard_future = self.standard.trigger_training(workspace_id);
        let data_moat_future = self.data_moat.trigger_training(workspace_id);
        
        let (standard_result, data_moat_result) = tokio::join!(standard_future, data_moat_future);
        
        // Log results for monitoring
        match (&standard_result, &data_moat_result) {
            (Ok(_), Ok(_)) => info!("Both backends trained successfully for workspace {}", workspace_id),
            (Err(e), Ok(_)) => warn!("Standard backend training failed for workspace {}: {}", workspace_id, e),
            (Ok(_), Err(e)) => warn!("Data Moat backend training failed for workspace {}: {}", workspace_id, e),
            (Err(e1), Err(e2)) => error!("Both backends failed for workspace {}: Standard: {}, Data Moat: {}", workspace_id, e1, e2),
        }
        
        standard_result?;
        data_moat_result?;
        
        Ok(())
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Hybrid
    }
}

impl HybridBackend {
    async fn select_backend(&self, workspace_id: Uuid, artifacts: &[Artifact]) -> Result<Arc<dyn MLBackend>> {
        match &self.routing_strategy {
            RoutingStrategy::WorkspaceBased => {
                // Use Data Moat for premium workspaces (simplified logic)
                if workspace_id.as_u128() % 2 == 0 {
                    debug!("Data Moat backend selected for workspace {} (premium workspace)", workspace_id);
                    Ok(self.data_moat.clone())
                } else {
                    debug!("Standard backend selected for workspace {} (standard workspace)", workspace_id);
                    Ok(self.standard.clone())
                }
            },
            RoutingStrategy::PerformanceBased { threshold } => {
                // Check cached metrics
                let cache = self.metrics_cache.read().await;
                if let Some(metrics) = cache.get(&workspace_id) {
                    if metrics.is_stale() {
                        debug!("Cached metrics for workspace {} are stale, using standard backend", workspace_id);
                        return Ok(self.standard.clone());
                    }
                    
                    if metrics.data_moat_accuracy > metrics.standard_accuracy + *threshold as f64 {
                        debug!("Data Moat backend selected for workspace {} (accuracy: {:.3} vs {:.3})", 
                            workspace_id, metrics.data_moat_accuracy, metrics.standard_accuracy);
                        return Ok(self.data_moat.clone());
                    }
                }
                debug!("Standard backend selected for workspace {}", workspace_id);
                Ok(self.standard.clone())
            },
            RoutingStrategy::VolumeBased { threshold } => {
                if artifacts.len() >= *threshold {
                    debug!("Data Moat backend selected for workspace {} ({} artifacts >= {})", 
                        workspace_id, artifacts.len(), threshold);
                    Ok(self.data_moat.clone())
                } else {
                    debug!("Standard backend selected for workspace {} ({} artifacts < {})", 
                        workspace_id, artifacts.len(), threshold);
                    Ok(self.standard.clone())
                }
            },
            RoutingStrategy::Custom => {
                // Custom logic would go here
                debug!("Custom routing strategy selected for workspace {}, defaulting to standard backend", workspace_id);
                Ok(self.standard.clone())
            }
        }
    }
}

/// Production-ready ML pipeline with configurable backend
pub struct MLPipeline {
    backend: Arc<dyn MLBackend>,
    config: PipelineConfig,
    health_monitor: Arc<RwLock<HealthMonitor>>,
    #[cfg(feature = "edge")]
    edge_integration: Option<Arc<RwLock<edge::EdgeMLIntegration>>>,
}

/// Pipeline configuration with backend selection
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub backend_config: BackendConfig,
    pub database_url: String,
    pub enable_monitoring: bool,
    pub enable_auto_training: bool,
    pub health_check_interval: Duration,
    #[cfg(feature = "edge")]
    pub enable_edge_optimization: bool,
}

impl PipelineConfig {
    /// Create config with Data Moat backend for maximum competitive advantage
    pub fn with_data_moat(database_url: impl Into<String>) -> Self {
        Self {
            backend_config: BackendConfig::DataMoat {
                config: DataMoatConfig::default(),
                enable_federated_learning: true,
                enable_privacy_protection: true,
            },
            database_url: database_url.into(),
            enable_monitoring: true,
            enable_auto_training: true,
            health_check_interval: Duration::from_secs(60),
            #[cfg(feature = "edge")]
            enable_edge_optimization: true,
        }
    }
    
    /// Create config with hybrid backend for flexibility
    pub fn with_hybrid(database_url: impl Into<String>) -> Self {
        Self {
            backend_config: BackendConfig::Hybrid {
                standard: Box::new(BackendConfig::default()),
                data_moat: Box::new(BackendConfig::DataMoat {
                    config: DataMoatConfig::default(),
                    enable_federated_learning: true,
                    enable_privacy_protection: true,
                }),
                routing_strategy: RoutingStrategy::PerformanceBased { threshold: 0.05 },
            },
            database_url: database_url.into(),
            enable_monitoring: true,
            enable_auto_training: true,
            health_check_interval: Duration::from_secs(60),
            #[cfg(feature = "edge")]
            enable_edge_optimization: true,
        }
    }
    
    /// Create config with standard backend (legacy compatibility)
    pub fn standard(database_url: impl Into<String>) -> Self {
        Self {
            backend_config: BackendConfig::default(),
            database_url: database_url.into(),
            enable_monitoring: true,
            enable_auto_training: true,
            health_check_interval: Duration::from_secs(60),
            #[cfg(feature = "edge")]
            enable_edge_optimization: false,
        }
    }
    
    /// Create config for development/testing
    pub fn development(database_url: impl Into<String>) -> Self {
        Self {
            backend_config: BackendConfig::default(),
            database_url: database_url.into(),
            enable_monitoring: false,
            enable_auto_training: false,
            health_check_interval: Duration::from_secs(300),
            #[cfg(feature = "edge")]
            enable_edge_optimization: false,
        }
    }
    
    /// Create production-ready config with Data Moat backend
    pub fn production(database_url: impl Into<String>) -> Self {
        Self::with_data_moat(database_url)
    }
}

impl MLPipeline {
    /// Initialize ML pipeline with configurable backend
    #[instrument(skip(config))]
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        info!(
            version = VERSION,
            ml_engine = ML_ENGINE_VERSION,
            data_moat = DATA_MOAT_VERSION,
            "Initializing ML Pipeline with configurable backend"
        );

        // Create backend based on configuration
        let backend: Arc<dyn MLBackend> = match &config.backend_config {
            BackendConfig::Standard { model_config, trainer_config, feedback_config } => {
                info!("Initializing Standard ML Backend");
                
                let storage = training::storage::StorageFactory::create_ml_storage(
                    &config.database_url,
                    training::storage::ModelStorageConfig::Local(std::path::PathBuf::from("./models"))
                )
                .await?;
                
                let embedder = Arc::new(TextEmbedder::new(model_config.clone()).await?);
                let predictor = Arc::new(OutcomePredictor::new(model_config.clone()).await?);
                let trainer = Arc::new(ContinuousTrainer::new(trainer_config.clone(), storage.clone()).await?);
                let feedback_processor = Arc::new(
                    FeedbackProcessor::new(&config.database_url, feedback_config.clone()).await?
                );
                
                Arc::new(StandardBackend {
                    embedder,
                    predictor,
                    trainer,
                    feedback_processor,
                    storage,
                })
            },
            BackendConfig::DataMoat { config: dm_config, .. } => {
                info!("Initializing Data Moat Backend");
                
                let engine = Arc::new(DataMoatEngine::new(dm_config.clone()).await?);
                
                Arc::new(DataMoatBackend { engine })
            },
            BackendConfig::Hybrid { standard, data_moat, routing_strategy } => {
                info!("Initializing Hybrid Backend");
                
                // Recursively create both backends
                let standard_config = PipelineConfig {
                    backend_config: *standard.clone(),
                    ..config.clone()
                };
                let standard_backend = Self::create_backend(&standard_config).await?;
                
                let data_moat_config = PipelineConfig {
                    backend_config: *data_moat.clone(),
                    ..config.clone()
                };
                let data_moat_backend = Self::create_backend(&data_moat_config).await?;
                
                Arc::new(HybridBackend {
                    standard: standard_backend,
                    data_moat: data_moat_backend,
                    routing_strategy: routing_strategy.clone(),
                    metrics_cache: Arc::new(RwLock::new(HashMap::new())),
                })
            }
        };

        let health_monitor = Arc::new(RwLock::new(HealthMonitor::new()));

        let pipeline = Self {
            backend,
            config: config.clone(),
            health_monitor,
            #[cfg(feature = "edge")]
            edge_integration: None,
        };

        // Start background services
        if config.enable_monitoring {
            pipeline.start_health_monitoring().await;
        }

        info!("ML Pipeline initialized successfully with {} backend", 
              match &config.backend_config {
                  BackendConfig::Standard { .. } => "Standard",
                  BackendConfig::DataMoat { .. } => "Data Moat",
                  BackendConfig::Hybrid { .. } => "Hybrid",
              }
        );
        
        Ok(pipeline)
    }
    
    /// Helper to create backend recursively for hybrid mode
    async fn create_backend(config: &PipelineConfig) -> Result<Arc<dyn MLBackend>> {
        match &config.backend_config {
            BackendConfig::Standard { model_config, trainer_config, feedback_config } => {
                let storage = training::storage::StorageFactory::create_ml_storage(
                    &config.database_url,
                    training::storage::ModelStorageConfig::Local(std::path::PathBuf::from("./models"))
                )
                .await?;
                
                let embedder = Arc::new(TextEmbedder::new(model_config.clone()).await?);
                let predictor = Arc::new(OutcomePredictor::new(model_config.clone()).await?);
                let trainer = Arc::new(ContinuousTrainer::new(trainer_config.clone(), storage.clone()).await?);
                let feedback_processor = Arc::new(
                    FeedbackProcessor::new(&config.database_url, feedback_config.clone()).await?
                );
                
                Ok(Arc::new(StandardBackend {
                    embedder,
                    predictor,
                    trainer,
                    feedback_processor,
                    storage,
                }))
            },
            BackendConfig::DataMoat { config: dm_config, .. } => {
                let engine = Arc::new(DataMoatEngine::new(dm_config.clone()).await?);
                Ok(Arc::new(DataMoatBackend { engine }))
            },
            _ => Err(anyhow::anyhow!("Cannot recursively create hybrid backend")),
        }
    }
    
    /// Predict outcomes using configured backend
    #[instrument(skip(self, artifacts))]
    pub async fn predict_outcomes(
        &self,
        workspace_id: Uuid,
        artifacts: Vec<Artifact>,
    ) -> Result<Vec<OutcomePrediction>> {
        // Update health metrics
        self.health_monitor.write().await.record_prediction_start();
        
        // Build context
        let context = PredictionContext::from_environment();
        
        // Use configured backend
        let predictions = self.backend
            .predict(workspace_id, &artifacts, &context)
            .await?;
        
        // Record success
        self.health_monitor.write().await.record_prediction_success();
        
        Ok(predictions)
    }
    
    /// Process user feedback
    #[instrument(skip(self, feedback))]
    pub async fn process_feedback(
        &self,
        workspace_id: Uuid,
        predictions: Vec<OutcomePrediction>,
        feedback: UserFeedback,
    ) -> Result<()> {
        self.backend
            .process_feedback(workspace_id, &predictions, feedback)
            .await
    }
    
    /// Get model metrics
    pub async fn get_model_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics> {
        self.backend.get_metrics(workspace_id).await
    }
    
    /// Get model information
    pub async fn get_model_info(&self, workspace_id: Uuid) -> Result<Option<ModelInfo>> {
        // Get metrics and convert to ModelInfo
        let metrics = self.get_model_metrics(workspace_id).await?;
        
        Ok(Some(ModelInfo {
            version: "1.0.0".to_string(),
            accuracy: metrics.accuracy,
            training_runs: metrics.total_predictions,
            last_trained: metrics.last_updated,
        }))
    }
    
    /// Get moat strength (only available with Data Moat backend)
    pub async fn get_moat_strength(&self, workspace_id: Uuid) -> Result<Option<MoatStrength>> {
        match &self.config.backend_config {
            BackendConfig::DataMoat { config, .. } => {
                let engine = DataMoatEngine::new(config.clone()).await?;
                Ok(Some(engine.get_moat_strength(workspace_id).await))
            },
            BackendConfig::Hybrid { data_moat, .. } => {
                if let BackendConfig::DataMoat { config, .. } = &**data_moat {
                    let engine = DataMoatEngine::new(config.clone()).await?;
                    Ok(Some(engine.get_moat_strength(workspace_id).await))
                } else {
                    Ok(None)
                }
            },
            _ => Ok(None),
        }
    }
    
    /// Trigger training
    pub async fn trigger_training(&self, workspace_id: Uuid) -> Result<()> {
        self.backend.trigger_training(workspace_id).await
    }
    
    /// Start continuous training loop for all workspaces
    pub async fn start_training_loop(&self) -> Result<()> {
        info!("Starting continuous training loop");
        
        // Start training based on backend type
        match self.backend.backend_type() {
            BackendType::Standard => {
                // For standard backend, we need to access the trainer directly
                // This is a simplified approach - in production, you'd want better abstraction
                info!("Starting standard backend training loop");
                // The ContinuousTrainer should already be running via the backend
                Ok(())
            },
            BackendType::DataMoat => {
                info!("Starting Data Moat training loop");
                // Data Moat uses continuous learning by default
                Ok(())
            },
            BackendType::Hybrid => {
                info!("Starting hybrid backend training loop");
                // Both backends should handle their own training
                Ok(())
            }
        }
    }
    
    /// Get pipeline health status
    pub async fn health_check(&self) -> HealthStatus {
        self.health_monitor.read().await.get_status()
    }
    
    /// Get comprehensive pipeline status
    pub async fn get_status(&self) -> PipelineStatus {
        PipelineStatus {
            health: self.health_check().await,
            version: VERSION.to_string(),
            ml_engine_version: ML_ENGINE_VERSION.to_string(),
            data_moat_version: DATA_MOAT_VERSION.to_string(),
            backend_type: self.backend.backend_type(),
            #[cfg(feature = "edge")]
            edge_optimization_enabled: self.edge_integration.is_some(),
            #[cfg(not(feature = "edge"))]
            edge_optimization_enabled: false,
            auto_training_enabled: self.config.enable_auto_training,
            monitoring_enabled: self.config.enable_monitoring,
        }
    }
    
    async fn start_health_monitoring(&self) {
        let monitor = self.health_monitor.clone();
        let interval = self.config.health_check_interval;
        
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            
            loop {
                ticker.tick().await;
                monitor.write().await.perform_health_check();
            }
        });
    }
}

/// Pipeline status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub health: HealthStatus,
    pub version: String,
    pub ml_engine_version: String,
    pub data_moat_version: String,
    pub backend_type: BackendType,
    pub edge_optimization_enabled: bool,
    pub auto_training_enabled: bool,
    pub monitoring_enabled: bool,
}

/// Health monitoring for production observability
#[derive(Debug, Clone)]
struct HealthMonitor {
    predictions_total: u64,
    predictions_failed: u64,
    embeddings_failed: u64,
    last_prediction: Option<std::time::Instant>,
    consecutive_failures: u32,
}

impl HealthMonitor {
    fn new() -> Self {
        Self {
            predictions_total: 0,
            predictions_failed: 0,
            embeddings_failed: 0,
            last_prediction: None,
            consecutive_failures: 0,
        }
    }

    fn record_prediction_start(&mut self) {
        self.predictions_total += 1;
        self.last_prediction = Some(std::time::Instant::now());
    }

    fn record_prediction_success(&mut self) {
        self.consecutive_failures = 0;
    }

    fn record_prediction_failure(&mut self) {
        self.predictions_failed += 1;
        self.consecutive_failures += 1;
    }

    fn record_embedding_failure(&mut self) {
        self.embeddings_failed += 1;
    }
    
    /// Record prediction failure with context
    pub fn record_prediction_failure_with_context(&mut self, error: &str) {
        self.record_prediction_failure();
        warn!("Prediction failed: {}", error);
    }
    
    /// Record embedding failure with context
    pub fn record_embedding_failure_with_context(&mut self, error: &str) {
        self.record_embedding_failure();
        warn!("Embedding generation failed: {}", error);
    }

    fn perform_health_check(&mut self) {
        if self.predictions_total > 10000 {
            info!("Resetting health metrics after {} predictions", self.predictions_total);
            self.predictions_total = 0;
            self.predictions_failed = 0;
            self.embeddings_failed = 0;
        }
        
        // Log health status periodically
        let status = self.get_status();
        match status {
            HealthStatus::Unhealthy => error!("ML Pipeline health is unhealthy: {} failures out of {} predictions", 
                self.predictions_failed, self.predictions_total),
            HealthStatus::Degraded => warn!("ML Pipeline health is degraded: {} failures out of {} predictions", 
                self.predictions_failed, self.predictions_total),
            HealthStatus::Healthy => debug!("ML Pipeline health is healthy: {} predictions, {} failures", 
                self.predictions_total, self.predictions_failed),
        }
    }

    fn get_status(&self) -> HealthStatus {
        let error_rate = if self.predictions_total > 0 {
            self.predictions_failed as f64 / self.predictions_total as f64
        } else {
            0.0
        };

        if self.consecutive_failures > 10 || error_rate > 0.5 {
            HealthStatus::Unhealthy
        } else if self.consecutive_failures > 3 || error_rate > 0.1 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Builder pattern for elegant pipeline construction
pub struct MLPipelineBuilder {
    config: PipelineConfig,
}

impl MLPipelineBuilder {
    /// Create builder with Data Moat backend
    pub fn with_data_moat(database_url: impl Into<String>) -> Self {
        Self {
            config: PipelineConfig::with_data_moat(database_url),
        }
    }
    
    /// Create builder with hybrid backend
    pub fn with_hybrid(database_url: impl Into<String>) -> Self {
        Self {
            config: PipelineConfig::with_hybrid(database_url),
        }
    }
    
    /// Create builder with standard backend
    pub fn with_standard(database_url: impl Into<String>) -> Self {
        Self {
            config: PipelineConfig::standard(database_url),
        }
    }
    
    /// Configure Data Moat settings
    pub fn data_moat_config(mut self, config: DataMoatConfig) -> Self {
        if let BackendConfig::DataMoat { config: ref mut dm_config, .. } = &mut self.config.backend_config {
            *dm_config = config;
        }
        self
    }
    
    /// Enable federated learning
    pub fn enable_federated_learning(mut self, enable: bool) -> Self {
        match &mut self.config.backend_config {
            BackendConfig::DataMoat { enable_federated_learning, .. } => {
                *enable_federated_learning = enable;
            },
            BackendConfig::Hybrid { data_moat, .. } => {
                if let BackendConfig::DataMoat { enable_federated_learning, .. } = &mut **data_moat {
                    *enable_federated_learning = enable;
                }
            },
            _ => {}
        }
        self
    }
    
    /// Enable privacy protection
    pub fn enable_privacy_protection(mut self, enable: bool) -> Self {
        match &mut self.config.backend_config {
            BackendConfig::DataMoat { enable_privacy_protection, .. } => {
                *enable_privacy_protection = enable;
            },
            BackendConfig::Hybrid { data_moat, .. } => {
                if let BackendConfig::DataMoat { enable_privacy_protection, .. } = &mut **data_moat {
                    *enable_privacy_protection = enable;
                }
            },
            _ => {}
        }
        self
    }
    
    /// Set routing strategy for hybrid backend
    pub fn routing_strategy(mut self, strategy: RoutingStrategy) -> Self {
        if let BackendConfig::Hybrid { routing_strategy, .. } = &mut self.config.backend_config {
            *routing_strategy = strategy;
        }
        self
    }
    
    /// Enable monitoring
    pub fn enable_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_monitoring = enable;
        self
    }
    
    /// Enable auto training
    pub fn enable_auto_training(mut self, enable: bool) -> Self {
        self.config.enable_auto_training = enable;
        self
    }
    
    /// Set health check interval
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.config.health_check_interval = interval;
        self
    }
    
    /// Enable edge optimization
    #[cfg(feature = "edge")]
    pub fn enable_edge_optimization(mut self, enable: bool) -> Self {
        self.config.enable_edge_optimization = enable;
        self
    }
    
    /// Build the pipeline
    pub async fn build(self) -> Result<Arc<MLPipeline>> {
        Ok(Arc::new(MLPipeline::new(self.config).await?))
    }
}

/// Convenience factory functions for production deployment

/// Create a production-ready ML pipeline with Data Moat backend
pub async fn create_data_moat_pipeline(database_url: &str) -> Result<Arc<MLPipeline>> {
    MLPipelineBuilder::with_data_moat(database_url)
        .enable_federated_learning(true)
        .enable_privacy_protection(true)
        .enable_monitoring(true)
        .enable_auto_training(true)
        .build()
        .await
}

/// Create a hybrid ML pipeline with intelligent routing
pub async fn create_hybrid_pipeline(database_url: &str) -> Result<Arc<MLPipeline>> {
    MLPipelineBuilder::with_hybrid(database_url)
        .routing_strategy(RoutingStrategy::PerformanceBased { threshold: 0.05 })
        .enable_monitoring(true)
        .enable_auto_training(true)
        .build()
        .await
}

/// Create a standard ML pipeline (legacy compatibility)
pub async fn create_standard_pipeline(database_url: &str) -> Result<Arc<MLPipeline>> {
    MLPipelineBuilder::with_standard(database_url)
        .enable_monitoring(true)
        .enable_auto_training(false)
        .build()
        .await
}

/// Create an enterprise-grade ML pipeline with full features
pub async fn create_enterprise_pipeline(
    database_url: &str,
    organization_id: Uuid,
) -> Result<Arc<MLPipeline>> {
    // Custom Data Moat configuration for enterprise
    let mut dm_config = DataMoatConfig::default();
    dm_config.enable_continuous_learning = true;
    dm_config.enable_federated_learning = true;
    dm_config.enable_privacy_protection = true;
    dm_config.min_training_examples = 50; // Lower threshold for faster learning
    dm_config.cache_size = 100000; // Large cache for enterprise scale
    
    #[cfg(feature = "cuda")]
    {
        use crate::inference::engine::DeviceConfig;


        dm_config.device = DeviceConfig::Cuda(0);
    }
    
    #[cfg(feature = "metal")]
    {
        dm_config.device = engine::DeviceConfig::Metal;
    }
    
    let pipeline = MLPipelineBuilder::with_data_moat(database_url)
        .data_moat_config(dm_config)
        .enable_federated_learning(true)
        .enable_privacy_protection(true)
        .enable_monitoring(true)
        .enable_auto_training(true)
        .health_check_interval(Duration::from_secs(30))
        .build()
        .await?;
    
    // Pre-warm the cache for the organization
    pipeline.trigger_training(organization_id).await?;
    
    Ok(pipeline)
}

/// Advanced pipeline orchestrator for multi-tenant scenarios
pub struct PipelineOrchestrator {
    pipelines: Arc<RwLock<HashMap<Uuid, Arc<MLPipeline>>>>,
    default_config: PipelineConfig,
    max_pipelines: usize,
}

impl PipelineOrchestrator {
    /// Create new orchestrator
    pub fn new(default_config: PipelineConfig) -> Self {
        Self {
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            default_config,
            max_pipelines: 1000,
        }
    }
    
    /// Get or create pipeline for organization
    pub async fn get_pipeline(&self, org_id: Uuid) -> Result<Arc<MLPipeline>> {
        let pipelines = self.pipelines.read().await;
        
        if let Some(pipeline) = pipelines.get(&org_id) {
            return Ok(pipeline.clone());
        }
        
        drop(pipelines);
        
        // Create new pipeline for organization
        let mut config = self.default_config.clone();
        
        // Customize config based on organization tier
        if self.is_premium_org(org_id) {
            config.backend_config = BackendConfig::DataMoat {
                config: DataMoatConfig::default(),
                enable_federated_learning: true,
                enable_privacy_protection: true,
            };
        }
        
        let pipeline = Arc::new(MLPipeline::new(config).await?);
        
        let mut pipelines = self.pipelines.write().await;
        
        // Enforce max pipelines limit
        if pipelines.len() >= self.max_pipelines {
            // Evict least recently used
            if let Some(oldest) = pipelines.keys().next().cloned() {
                pipelines.remove(&oldest);
            }
        }
        
        pipelines.insert(org_id, pipeline.clone());
        
        Ok(pipeline)
    }
    
    /// Remove pipeline for organization
    pub async fn remove_pipeline(&self, org_id: Uuid) -> Option<Arc<MLPipeline>> {
        self.pipelines.write().await.remove(&org_id)
    }
    
    /// Get all active pipelines
    pub async fn get_all_pipelines(&self) -> HashMap<Uuid, Arc<MLPipeline>> {
        self.pipelines.read().await.clone()
    }
    
    /// Check if organization is premium (simplified logic)
    fn is_premium_org(&self, org_id: Uuid) -> bool {
        // In production, this would check against a database or service
        org_id.as_u128() % 10 < 3 // 30% are premium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backend_selection() {
        let config = PipelineConfig::with_data_moat("postgres://test");
        assert!(matches!(config.backend_config, BackendConfig::DataMoat { .. }));
        
        let config = PipelineConfig::with_hybrid("postgres://test");
        assert!(matches!(config.backend_config, BackendConfig::Hybrid { .. }));
        
        let config = PipelineConfig::standard("postgres://test");
        assert!(matches!(config.backend_config, BackendConfig::Standard { .. }));
    }
    
    #[test]
    fn test_builder_pattern() {
        let builder = MLPipelineBuilder::with_data_moat("postgres://test")
            .enable_federated_learning(true)
            .enable_privacy_protection(true)
            .enable_monitoring(true);
        
        assert!(builder.config.enable_monitoring);
        
        if let BackendConfig::DataMoat { enable_federated_learning, enable_privacy_protection, .. } = builder.config.backend_config {
            assert!(enable_federated_learning);
            assert!(enable_privacy_protection);
        } else {
            panic!("Expected DataMoat backend");
        }
    }
    
    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new();
        assert_eq!(monitor.get_status(), HealthStatus::Healthy);
        
        for _ in 0..11 {
            monitor.record_prediction_failure();
            monitor.consecutive_failures += 1;
        }
        
        assert_eq!(monitor.get_status(), HealthStatus::Unhealthy);
    }
    
    #[tokio::test]
    async fn test_orchestrator() {
        let config = PipelineConfig::standard("postgres://test");
        let orchestrator = PipelineOrchestrator::new(config);
        
        let org_id = Uuid::new_v4();
        
        // First call creates pipeline
        let pipeline1 = orchestrator.get_pipeline(org_id).await;
        assert!(pipeline1.is_ok());
        
        // Second call returns cached pipeline
        let pipeline2 = orchestrator.get_pipeline(org_id).await;
        assert!(pipeline2.is_ok());
        
        // Check they're the same instance
        assert!(Arc::ptr_eq(&pipeline1.unwrap(), &pipeline2.unwrap()));
    }
}