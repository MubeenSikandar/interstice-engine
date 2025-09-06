//! Interstice ML - Production-Ready Machine Learning Pipeline
//! 
//! This crate provides enterprise-grade ML capabilities for outcome prediction,
//! continuous learning, and intelligent decision support.

pub mod adapters;
pub mod embeddings;
pub mod feedback;
pub mod inference;
pub mod models;
pub mod types;
pub mod training;

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::RwLock;
use tracing::{info, instrument, warn};
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

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const ML_ENGINE_VERSION: &str = "2.0.0";

/// Production-ready ML pipeline with comprehensive monitoring and resilience
pub struct MLPipeline {
    embedder: Arc<TextEmbedder>,
    predictor: Arc<OutcomePredictor>,
    trainer: Arc<ContinuousTrainer>,
    feedback_processor: Arc<FeedbackProcessor>,
    storage: Arc<training::storage::MLStorage>,
    config: PipelineConfig,
    health_monitor: Arc<RwLock<HealthMonitor>>,
}

/// Pipeline configuration with sensible defaults
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub database_url: String,
    pub model_config: ModelConfig,
    pub trainer_config: TrainerConfig,
    pub feedback_config: FeedbackConfig,
    pub enable_monitoring: bool,
    pub enable_auto_training: bool,
    pub health_check_interval: Duration,
}

impl PipelineConfig {
    /// Create config with production defaults
    pub fn production(database_url: impl Into<String>) -> Self {
        Self {
            database_url: database_url.into(),
            model_config: ModelConfig::default(),
            trainer_config: TrainerConfig::default(),
            feedback_config: FeedbackConfig::default(),
            enable_monitoring: true,
            enable_auto_training: true,
            health_check_interval: Duration::from_secs(60),
        }
    }

    /// Create config for development/testing
    pub fn development(database_url: impl Into<String>) -> Self {
        let mut config = Self::production(database_url);
        config.enable_monitoring = false;
        config.enable_auto_training = false;
        config.health_check_interval = Duration::from_secs(300);
        config
    }
}

impl MLPipeline {
    /// Initialize ML pipeline with full production setup
    #[instrument(skip(config))]
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        info!(
            version = VERSION,
            ml_engine = ML_ENGINE_VERSION,
            "Initializing ML Pipeline"
        );

        // Initialize storage layer
        let storage = training::storage::StorageFactory::create_ml_storage(
            &config.database_url,
            training::storage::ModelStorageConfig::Local(std::path::PathBuf::from("./models"))
        )
        .await
        .context("Failed to initialize ML storage")?;

        // Initialize components with proper error handling
        let embedder = Arc::new(
            TextEmbedder::new(config.model_config.clone())
                .await
                .context("Failed to initialize text embedder")?
        );

        let predictor = Arc::new(
            OutcomePredictor::new(config.model_config.clone())
                .await
                .context("Failed to initialize outcome predictor")?
        );

        let trainer = Arc::new(
            ContinuousTrainer::new(config.trainer_config.clone(), storage.clone())
                .await
                .context("Failed to initialize continuous trainer")?
        );

        let feedback_processor = Arc::new(
            FeedbackProcessor::new(&config.database_url, config.feedback_config.clone())
                .await
                .context("Failed to initialize feedback processor")?
        );

        let health_monitor = Arc::new(RwLock::new(HealthMonitor::new()));

        let pipeline = Self {
            embedder,
            predictor,
            trainer,
            feedback_processor,
            storage,
            config: config.clone(),
            health_monitor,
        };

        // Start background services if enabled
        if config.enable_auto_training {
            pipeline.start_training_loop().await?;
        }

        if config.enable_monitoring {
            pipeline.start_health_monitoring().await;
        }

        info!("ML Pipeline initialized successfully");
        Ok(pipeline)
    }

    /// Predict outcomes with comprehensive error handling
    #[instrument(skip(self, artifacts, text))]
    pub async fn predict_outcomes(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        text: &str,
    ) -> Result<Vec<OutcomePrediction>> {
        // Update health metrics
        self.health_monitor.write().await.record_prediction_start();

        // Validate inputs
        if text.is_empty() {
            return Err(anyhow::anyhow!("Input text cannot be empty"));
        }

        if text.len() > 1_000_000 {
            return Err(anyhow::anyhow!("Input text exceeds maximum size (1MB)"));
        }

        // Generate embeddings with error recovery
        let embedding = match self.embedder.embed_text(text).await {
            Ok(emb) => emb,
            Err(e) => {
                warn!("Failed to generate embedding: {}", e);
                self.health_monitor.write().await.record_embedding_failure();
                // Use fallback embedding strategy
                self.generate_fallback_embedding(text)?
            }
        };

        // Convert to core artifacts for predictor
        let core_artifacts = self.convert_to_core_artifacts(artifacts)?;

        // Perform prediction with timeout
        let predictions = tokio::time::timeout(
            Duration::from_secs(30),
            self.predictor.predict(embedding, &core_artifacts)
        )
        .await
        .context("Prediction timeout")?
        .context("Prediction failed")?;

        // Record successful prediction
        self.health_monitor.write().await.record_prediction_success();

        // Learn from the interaction asynchronously
        let storage = self.storage.clone();
        let text = text.to_string();
        tokio::spawn(async move {
            if let Err(e) = Self::learn_vocabulary(storage, workspace_id, &text).await {
                warn!("Failed to learn vocabulary: {}", e);
            }
        });

        Ok(predictions)
    }

    /// Process user feedback with validation
    #[instrument(skip(self, action))]
    pub async fn process_feedback(
        &self,
        workspace_id: Uuid,
        action: UserAction,
    ) -> Result<()> {
        // Validate action
        if action.artifact_id.is_empty() || action.outcome_id.is_empty() {
            return Err(anyhow::anyhow!("Invalid action: missing required IDs"));
        }

        // Check if we should trigger training before moving action
        let should_trigger_training = action.action_type == ActionType::Reject || action.action_type == ActionType::Correct;

        self.feedback_processor
            .process_user_action(workspace_id, action)
            .await
            .context("Failed to process user feedback")?;

        // Trigger training if feedback indicates poor performance
        if should_trigger_training {
            self.maybe_trigger_training(workspace_id).await;
        }

        Ok(())
    }

    /// Get model performance metrics
    #[instrument(skip(self))]
    pub async fn get_model_metrics(&self, workspace_id: Uuid) -> Result<ModelMetrics> {
        // Get training stats from storage
        let stats = self.storage.get_training_stats(workspace_id).await
            .context("Failed to retrieve training statistics")?;
        
        // Convert to ModelMetrics
        Ok(ModelMetrics {
            correct_predictions: stats.validated_examples as u64,
            total_predictions: stats.total_examples as u64,
            accuracy: stats.average_feedback_score.unwrap_or(0.0) as f64,
            precision: 0.0, // Would need to be calculated from confusion matrix
            recall: 0.0,    // Would need to be calculated from confusion matrix
            f1_score: 0.0,  // Would need to be calculated from precision/recall
            auc_roc: None,  // Would need to be calculated from ROC curve
            mean_confidence: 0.0, // Would need to be tracked separately
            prediction_latency_ms: 0.0, // Would need to be tracked separately
            last_updated: chrono::Utc::now(),
            per_outcome_metrics: None, // Would need to be calculated per outcome
        })
    }

    /// Get model information
    #[instrument(skip(self))]
    pub async fn get_model_info(&self, workspace_id: Uuid) -> Result<Option<ModelInfo>> {
        self.trainer.get_model_info(workspace_id).await
    }

    /// Trigger manual training for a workspace
    #[instrument(skip(self))]
    pub async fn train_workspace(&self, workspace_id: Uuid) -> Result<()> {
        info!("Manually triggering training for workspace {}", workspace_id);
        self.trainer.train_workspace_now(workspace_id).await
    }

    /// Get pipeline health status
    pub async fn health_check(&self) -> HealthStatus {
        self.health_monitor.read().await.get_status()
    }

    /// Graceful shutdown
    pub async fn shutdown(self: Arc<Self>) -> Result<()> {
        info!("Initiating ML Pipeline shutdown");
        
        // Stop training loop
        if self.config.enable_auto_training {
            self.trainer.clone().shutdown().await?;
        }

        // Flush pending feedback
        // Note: FeedbackProcessor handles this internally
        
        info!("ML Pipeline shutdown complete");
        Ok(())
    }

    // Private helper methods

    pub async fn start_training_loop(&self) -> Result<()> {
        let trainer = self.trainer.clone();
        tokio::spawn(async move {
            if let Err(e) = trainer.start().await {
                warn!("Training loop failed to start: {}", e);
            }
        });
        Ok(())
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

    async fn maybe_trigger_training(&self, workspace_id: Uuid) {
        let monitor = self.health_monitor.read().await;
        if monitor.should_trigger_training() {
            drop(monitor); // Release lock before triggering
            
            let trainer = self.trainer.clone();
            tokio::spawn(async move {
                if let Err(e) = trainer.train_workspace_now(workspace_id).await {
                    warn!("Auto-triggered training failed: {}", e);
                }
            });
        }
    }

    fn generate_fallback_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Simple deterministic fallback embedding
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut embedding = vec![0.0; self.config.model_config.embedding_dim];
        let mut hasher = DefaultHasher::new();
        
        for (i, chunk) in text.as_bytes().chunks(32).enumerate() {
            chunk.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            
            if i < embedding.len() {
                embedding[i] = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
            }
        }
        
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        Ok(embedding)
    }

    fn convert_to_core_artifacts(&self, artifacts: &[Artifact]) -> Result<Vec<interstice_core::Artifact>> {
        artifacts
            .iter()
            .map(|a| {
                Ok(interstice_core::Artifact {
                    id: Uuid::parse_str(&a.id)?,
                    workspace_id: interstice_core::WorkspaceId::new(),
                    artifact_type: self.convert_artifact_type(&a.artifact_type),
                    platform: self.convert_platform(&a.platform),
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
            })
            .collect()
    }

    fn convert_platform(&self, platform: &Platform) -> interstice_core::Platform {
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
            Platform::Notion => interstice_core::Platform::Notion
        }
    }

    fn convert_artifact_type(&self, artifact_type: &ArtifactType) -> interstice_core::ArtifactType {
        match artifact_type {
            ArtifactType::PullRequest => interstice_core::ArtifactType::PullRequest {
                number: 0,
                title: "Unknown PR".to_string(),
                state: interstice_core::artifact::PullRequestState::Open,
                files_changed: 0,
                additions: 0,
                deletions: 0,
                merged: false,
                draft: false,
                base_branch: "main".to_string(),
                head_branch: "feature".to_string(),
                author: "unknown".to_string(),
                reviewers: vec![],
                labels: vec![],
                merge_conflict: false,
                ci_status: None,
            },
            ArtifactType::Issue => interstice_core::ArtifactType::Issue {
                id: "unknown".to_string(),
                title: "Unknown Issue".to_string(),
                state: interstice_core::artifact::IssueState::Open,
                priority: interstice_core::artifact::Priority::Medium,
                assignees: vec![],
                labels: vec![],
                story_points: None,
                sprint: None,
                epic: None,
                blocked: false,
                blockers: vec![],
                time_estimate: None,
                time_spent: None,
            },
            ArtifactType::Commit => interstice_core::ArtifactType::Commit {
                sha: "unknown".to_string(),
                message: "Unknown commit".to_string(),
                author: "unknown".to_string(),
                committer: "unknown".to_string(),
                files_changed: 0,
                additions: 0,
                deletions: 0,
                branch: "main".to_string(),
                is_merge: false,
                signed: false,
                verified: false,
            },
            ArtifactType::Document => interstice_core::ArtifactType::Document {
                id: "unknown".to_string(),
                title: "Unknown Document".to_string(),
                doc_type: interstice_core::artifact::DocumentType::Other("Unknown".to_string()),
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: chrono::Utc::now(),
                version: 1,
                is_template: false,
                access_level: interstice_core::artifact::AccessLevel::Internal,
            },
            ArtifactType::Message => interstice_core::ArtifactType::Message {
                id: "unknown".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "unknown".to_string(),
                content: "Unknown message".to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            _ => interstice_core::ArtifactType::Message {
                id: "unknown".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "unknown".to_string(),
                content: artifact_type.to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
        }
    }

    async fn learn_vocabulary(
        storage: Arc<training::storage::MLStorage>,
        workspace_id: Uuid,
        text: &str,
    ) -> Result<()> {
        // Extract key terms and store for future training
        let terms = Self::extract_key_terms(text);
        
        if !terms.is_empty() {
            // Store vocabulary terms in training examples table
            // This is a simplified approach - in production you'd have a dedicated vocabulary table
            for term in terms {
                let example = training::storage::TrainingExample {
                    id: Uuid::new_v4(),
                    workspace_id,
                    artifact_id: None,
                    input_text: term,
                    suggested_outcome_id: None,
                    actual_outcome_id: None,
                    user_feedback: None,
                    feedback_score: None,
                    context: Some(serde_json::json!({"type": "vocabulary"})),
                    created_at: chrono::Utc::now(),
                    is_validated: false,
                    validation_method: None,
                    input_embedding: None,
                };
                
                if let Err(e) = storage.store_training_example(workspace_id, None, &example).await {
                    warn!("Failed to store vocabulary term: {}", e);
                }
            }
        }
        
        Ok(())
    }

    fn extract_key_terms(text: &str) -> Vec<String> {
        // Simple term extraction - in production, use NLP library
        text.split_whitespace()
            .filter(|word| word.len() > 4 && word.chars().all(|c| c.is_alphanumeric()))
            .take(20)
            .map(|s| s.to_lowercase())
            .collect()
    }
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

    fn should_trigger_training(&self) -> bool {
        self.consecutive_failures > 5 ||
        (self.predictions_total > 100 && 
         self.predictions_failed as f64 / self.predictions_total as f64 > 0.1)
    }

    fn perform_health_check(&mut self) {
        // Reset counters periodically
        if self.predictions_total > 10000 {
            self.predictions_total = 0;
            self.predictions_failed = 0;
            self.embeddings_failed = 0;
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Convenience factory functions for quick setup

/// Create a production-ready ML predictor adapter
pub async fn create_ml_predictor() -> Result<MLPredictorAdapter> {
    MLPredictorAdapter::with_defaults().await
}

/// Create ML predictor with custom configuration
pub async fn create_ml_predictor_with_config(config: AdapterConfig) -> Result<MLPredictorAdapter> {
    MLPredictorAdapter::new(config).await
}

/// Create a complete ML pipeline
pub async fn create_ml_pipeline(database_url: &str) -> Result<MLPipeline> {
    let config = PipelineConfig::production(database_url);
    MLPipeline::new(config).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        // This would require a test database
        // let pipeline = create_ml_pipeline("postgres://test").await;
        // assert!(pipeline.is_ok());
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
}