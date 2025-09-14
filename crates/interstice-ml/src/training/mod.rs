// interstice-ml/src/training/mod.rs
pub mod model_storage;
pub mod monitoring;
pub mod storage;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, Transaction};
use tokio::sync::{RwLock, Semaphore};
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

// Import from the correct location
use crate::models::OrgModel;
use crate::training::storage::{
    MLStorage, TrainingStorage,
    TrainingExampleFilters
};

// Re-export monitoring types
pub use monitoring::{
    MetricsCollector, AlertManager, AlertNotifier, Alert, AlertSeverity,
    HealthStatus, ComponentHealth, MetricsSnapshot
};

// Configuration
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    /// Training cycle configuration
    pub training: TrainingConfig,
    
    /// Model persistence configuration  
    pub persistence: PersistenceConfig,
    
    /// Observability configuration
    pub observability: ObservabilityConfig,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            training: TrainingConfig::default(),
            persistence: PersistenceConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Interval between training cycles
    #[serde(with = "humantime_serde")]
    pub cycle_interval: Duration,
    
    /// Minimum examples required to start training
    pub min_examples: usize,
    
    /// Maximum examples to use for training
    pub max_examples: usize,
    
    /// Window for considering examples as "new"
    #[serde(with = "humantime_serde")]
    pub new_data_window: Duration,
    
    /// Maximum concurrent workspace trainings
    pub max_concurrent_workspaces: usize,
    
    /// Timeout for individual workspace training
    #[serde(with = "humantime_serde")]
    pub workspace_timeout: Duration,
    
    /// Minimum accuracy improvement to save model
    pub min_accuracy_improvement: f64,
    
    /// Enable automatic model rollback on regression
    pub enable_rollback: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            cycle_interval: Duration::from_secs(3600),
            min_examples: 100,
            max_examples: 10000,
            new_data_window: Duration::from_secs(86400),
            max_concurrent_workspaces: 4,
            workspace_timeout: Duration::from_secs(300),
            min_accuracy_improvement: 0.01,
            enable_rollback: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub enable_versioning: bool,
    pub max_versions: usize,
    pub compression_enabled: bool,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            enable_versioning: true,
            max_versions: 5,
            compression_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub metrics_port: u16,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            metrics_port: 9090,
        }
    }
}

// Core Types
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub loss: f64,
    pub training_duration: Duration,
    pub examples_used: usize,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct WorkspaceTrainingState {
    pub workspace_id: Uuid,
    pub last_training: Option<DateTime<Utc>>,
    pub current_accuracy: f64,
    pub training_count: u64,
    pub failed_attempts: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Insufficient training data: {available}/{required}")]
    InsufficientData { available: usize, required: usize },
    
    #[error("Training timeout for workspace {0}")]
    Timeout(Uuid),
    
    #[error("Model regression detected: accuracy {current} < {previous}")]
    ModelRegression { current: f64, previous: f64 },
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub workspace_id: Uuid,
    pub metrics: TrainingMetrics,
    pub created_at: DateTime<Utc>,
    pub size_bytes: u64,
}

// Training Pipeline
// -----------------------------------------------------------------------------

pub struct ContinuousTrainer {
    config: TrainerConfig,
    storage: Arc<MLStorage>,
    model_registry: Arc<RwLock<HashMap<Uuid, Arc<RwLock<OrgModel>>>>>,
    workspace_states: Arc<RwLock<HashMap<Uuid, WorkspaceTrainingState>>>,
    training_semaphore: Arc<Semaphore>,
    metrics_collector: Arc<MetricsCollector>,
}


impl ContinuousTrainer {
    #[instrument(skip(config, storage))]
    pub async fn new(config: TrainerConfig, storage: Arc<MLStorage>) -> Result<Self> {
        let metrics_collector = Arc::new(
            MetricsCollector::new()
                .context("Failed to create metrics collector")?
        );
        
        Ok(Self {
            training_semaphore: Arc::new(Semaphore::new(config.training.max_concurrent_workspaces)),
            model_registry: Arc::new(RwLock::new(HashMap::new())),
            workspace_states: Arc::new(RwLock::new(HashMap::new())),
            metrics_collector,
            storage,
            config,
        })
    }
    
    #[instrument(skip(self))]
    pub async fn start(self: Arc<Self>) -> Result<()> {
        // Initialize workspace states from database
        self.initialize_workspace_states().await?;
        
        // Start metrics server if enabled
        if self.config.observability.enable_metrics {
            let metrics = self.metrics_collector.clone();
            let port = self.config.observability.metrics_port;
            tokio::spawn(async move {
                if let Err(e) = metrics.start_metrics_server(port).await {
                    error!("Failed to start metrics server: {}", e);
                }
            });
        }
        
        // Start the training loop
        // Note: Training loop is currently disabled
        // TODO: Implement proper training loop activation
        
        info!("Continuous trainer started successfully");
        Ok(())
    }
    
    async fn initialize_workspace_states(&self) -> Result<()> {
        let pool = self.storage.pool();
        
        let workspaces = sqlx::query!(
            r#"
            SELECT DISTINCT 
                w.id as workspace_id,
                COUNT(te.id) as example_count,
                MAX(te.created_at) as last_example
            FROM workspaces w
            LEFT JOIN training_examples te ON w.id = te.workspace_id
            WHERE w.ml_enabled = true
            GROUP BY w.id
            "#
        )
        .fetch_all(pool)
        .await
        .context("Failed to fetch workspace states")?;
        
        let mut states = self.workspace_states.write().await;
        for record in workspaces {
            states.insert(
                record.workspace_id,
                WorkspaceTrainingState {
                    workspace_id: record.workspace_id,
                    last_training: None,
                    current_accuracy: 0.0,
                    training_count: 0,
                    failed_attempts: 0,
                }
            );
        }
        
        info!("Initialized {} workspace states", states.len());
        Ok(())
    }
    
    
    
    
    #[instrument(skip(self, _permit))]
    async fn train_workspace(
        &self,
        workspace_id: Uuid,
        _permit: tokio::sync::SemaphorePermit<'_>,
    ) -> Result<()> {
        let start = Instant::now();
        
        // Record training start
        self.metrics_collector.record_training_start(workspace_id);
        
        // Begin transaction for atomic operations
        let pool = self.storage.pool();
        let mut tx = pool.begin().await?;
        
        // 1. Fetch training examples using storage trait
        let filters = TrainingExampleFilters {
            validated_only: false,
            has_feedback: true,
            min_feedback_score: Some(0.7),
            created_after: Some(Utc::now() - chrono::Duration::days(30)),
            limit: Some(self.config.training.max_examples as i64),
            ..Default::default()
        };
        
        let examples = self.storage
            .get_training_examples(workspace_id, filters)
            .await?;
        
        if examples.len() < self.config.training.min_examples {
            self.metrics_collector.record_training_failure(
                workspace_id, 
                "insufficient_data"
            );
            return Err(TrainingError::InsufficientData {
                available: examples.len(),
                required: self.config.training.min_examples,
            }.into());
        }
        
        // 2. Get or create model
        let model = self.get_or_create_model(workspace_id).await?;
        
        // 3. Perform training (convert TrainingExample to format expected by model)
        let model_examples = examples.iter().map(|e| {
            crate::types::TrainingExample {
                id: e.id,
                input_text: e.input_text.clone(),
                suggested_outcome_id: e.suggested_outcome_id,
                actual_outcome_id: e.actual_outcome_id,
                user_feedback: e.user_feedback.clone(),
                feedback_score: e.feedback_score,
                context: e.context.as_ref().and_then(|v| serde_json::from_value(v.clone()).ok()),
                created_at: e.created_at,
                is_validated: e.is_validated,
                validation_method: e.validation_method.as_ref().map(|s| crate::types::ValidationMethod::from(s.clone())),
                input_embedding: e.input_embedding.clone(),
            }
        }).collect::<Vec<_>>();
        
        let _training_result = {
            let mut model_guard = model.write().await;
            model_guard.fine_tune(&model_examples).await
                .context("Model fine-tuning failed")?
        };
        
        // 4. Evaluate model
        let metrics = {
            let model_guard = model.read().await;
            self.evaluate_model(&*model_guard, &model_examples).await?
        };
        
        // 5. Handle model persistence based on performance
        let saved = self.handle_model_persistence(
            workspace_id,
            &model,
            &metrics,
            &mut tx
        ).await?;
        
        // 6. Update workspace state
        self.update_workspace_state(workspace_id, &metrics, saved).await?;
        
        // 7. Record training completion
        self.record_training_completion(&mut tx, workspace_id, &metrics).await?;
        
        tx.commit().await?;
        
        // Record metrics
        let duration = start.elapsed();
        self.metrics_collector.record_training_complete(
            workspace_id,
            duration,
            examples.len(),
            metrics.accuracy,
            metrics.loss,
            "latest", // Should get actual version from model storage
        );
        
        Ok(())
    }
    
    async fn get_or_create_model(&self, workspace_id: Uuid) -> Result<Arc<RwLock<OrgModel>>> {
        let mut registry = self.model_registry.write().await;
        
        if let Some(model) = registry.get(&workspace_id) {
            return Ok(Arc::clone(model));
        }
        
        // Try loading from storage first
        let model = if let Some(stored_model) = self.storage.models().load(workspace_id).await? {
            info!("Loaded existing model for workspace {}", workspace_id);
            stored_model
        } else {
            info!("Creating new model for workspace {}", workspace_id);
            OrgModel::new(workspace_id).await
                .context("Failed to create new model")?
        };
        
        let model = Arc::new(RwLock::new(model));
        registry.insert(workspace_id, Arc::clone(&model));
        
        Ok(model)
    }
    
    async fn evaluate_model(
        &self,
        model: &OrgModel,
        examples: &[crate::types::TrainingExample],
    ) -> Result<TrainingMetrics> {
        // Note: Evaluation split removed as eval_examples was unused
        
        let eval_result = model.evaluate().await
            .context("Model evaluation failed")?;
        
        Ok(TrainingMetrics {
            accuracy: eval_result.accuracy,
            precision: eval_result.precision,
            recall: eval_result.recall,
            f1_score: eval_result.f1_score,
            loss: 0.0, // Loss not available in ModelMetrics
            training_duration: Duration::from_secs(0), // Will be set later
            examples_used: examples.len(),
            timestamp: Utc::now(),
        })
    }
    
    async fn handle_model_persistence(
        &self,
        workspace_id: Uuid,
        model: &Arc<RwLock<OrgModel>>,
        metrics: &TrainingMetrics,
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<bool> {
        // Get current best accuracy
        let current_best = sqlx::query!(
            r#"
            SELECT best_accuracy 
            FROM workspace_models 
            WHERE workspace_id = $1
            "#,
            workspace_id
        )
        .fetch_optional(&mut **tx)
        .await?
        .map(|r| r.best_accuracy as f64)
        .unwrap_or(0.0);
        
        let improvement = metrics.accuracy - current_best;
        
        if improvement >= self.config.training.min_accuracy_improvement {
            info!(
                "Model improved by {:.2}% for workspace {}",
                improvement * 100.0,
                workspace_id
            );
            
            // Save model to storage
            let model_guard = model.read().await;
            self.storage.models().save(workspace_id, &*model_guard, metrics).await?;
            
            // Update database
            sqlx::query!(
                r#"
                INSERT INTO workspace_models (workspace_id, best_accuracy, last_updated)
                VALUES ($1, $2, NOW())
                ON CONFLICT (workspace_id) 
                DO UPDATE SET 
                    best_accuracy = EXCLUDED.best_accuracy,
                    last_updated = EXCLUDED.last_updated
                "#,
                workspace_id,
                metrics.accuracy as f32
            )
            .execute(&mut **tx)
            .await?;
            
            Ok(true)
        } else if improvement < -0.05 && self.config.training.enable_rollback {
            // Model regression detected
            warn!(
                "Model regression detected for workspace {}: {:.2}% decrease",
                workspace_id,
                -improvement * 100.0
            );
            
            self.metrics_collector.record_model_rollback(workspace_id, "regression");
            
            // Rollback to previous version if available
            let versions = self.storage.models().list_versions(workspace_id).await?;
            if versions.len() > 1 {
                self.storage.models().rollback(workspace_id, &versions[1].version).await?;
                warn!("Rolled back to previous model version");
            }
            
            Ok(false)
        } else {
            info!(
                "Model accuracy insufficient improvement ({:.2}%) for workspace {}",
                improvement * 100.0,
                workspace_id
            );
            Ok(false)
        }
    }
    
    async fn update_workspace_state(
        &self,
        workspace_id: Uuid,
        metrics: &TrainingMetrics,
        saved: bool,
    ) -> Result<()> {
        let mut states = self.workspace_states.write().await;
        
        let state = states.entry(workspace_id).or_insert_with(|| WorkspaceTrainingState {
            workspace_id,
            last_training: None,
            current_accuracy: 0.0,
            training_count: 0,
            failed_attempts: 0,
        });
        
        state.last_training = Some(Utc::now());
        if saved {
            state.current_accuracy = metrics.accuracy;
        }
        state.training_count += 1;
        state.failed_attempts = 0; // Reset on success
        
        Ok(())
    }
    
    async fn record_training_completion(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        workspace_id: Uuid,
        metrics: &TrainingMetrics,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO training_history (
                id,
                workspace_id,
                accuracy,
                precision,
                recall,
                f1_score,
                loss,
                examples_used,
                duration_ms,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            "#,
            Uuid::new_v4(),
            workspace_id,
            metrics.accuracy as f32,
            metrics.precision as f32,
            metrics.recall as f32,
            metrics.f1_score as f32,
            metrics.loss as f32,
            metrics.examples_used as i32,
            metrics.training_duration.as_millis() as i64,
            metrics.timestamp
        )
        .execute(&mut **tx)
        .await?;
        
        Ok(())
    }
    
    
    /// Trigger manual training for a workspace
    pub async fn train_workspace_now(&self, workspace_id: Uuid) -> Result<()> {
        let permit = self.training_semaphore.acquire().await?;
        self.train_workspace(workspace_id, permit).await?;
        Ok(())
    }
    
    /// Get model information for a workspace
    pub async fn get_model_info(&self, workspace_id: Uuid) -> Result<Option<crate::types::ModelInfo>> {
        let pool = self.storage.pool();
        
        let info = sqlx::query!(
            r#"
            SELECT 
                wm.best_accuracy,
                wm.last_updated,
                wm.total_training_runs as "total_training_runs?",
                wm.current_version
            FROM workspace_models wm
            WHERE wm.workspace_id = $1
            "#,
            workspace_id
        )
        .fetch_optional(pool)
        .await?;
        
        Ok(info.map(|r| crate::types::ModelInfo {
            version: r.current_version.unwrap_or_else(|| "unknown".to_string()),
            accuracy: r.best_accuracy as f64,
            last_trained: r.last_updated.unwrap_or_else(|| Utc::now()),
            training_runs: r.total_training_runs.unwrap_or(0) as u64,
        }))
    }
    
    // Graceful shutdown
    pub async fn shutdown(self: Arc<Self>) -> Result<()> {
        info!("Shutting down continuous trainer");
        
        // Save all models currently in memory
        let registry = self.model_registry.read().await;
        for (workspace_id, model) in registry.iter() {
            if let Err(e) = self.save_current_model(*workspace_id, model).await {
                error!("Failed to save model for workspace {} during shutdown: {}", workspace_id, e);
            }
        }
        
        info!("Continuous trainer shutdown complete");
        Ok(())
    }
    
    async fn save_current_model(
        &self,
        workspace_id: Uuid,
        model: &Arc<RwLock<OrgModel>>,
    ) -> Result<()> {
        let model_guard = model.read().await;
        let metrics = TrainingMetrics {
            accuracy: model_guard.best_accuracy as f64,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            loss: 0.0,
            training_duration: Duration::from_secs(0),
            examples_used: 0,
            timestamp: Utc::now(),
        };
        
        self.storage.models().save(workspace_id, &*model_guard, &metrics).await
    }
}