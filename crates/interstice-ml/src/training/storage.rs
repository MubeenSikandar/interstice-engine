// interstice-ml/src/training/storage.rs
// This module extends core storage with ML-specific functionality

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, QueryBuilder, Postgres};
use uuid::Uuid;

use interstice_core::{outcome::OutcomeId, storage::{PostgresStorage, StorageBackend as CoreStorage}, WorkspaceId};

pub use crate::training::model_storage::{
    ModelStorage, S3ModelStorage, LocalModelStorage, HybridModelStorage,
    CompressionStrategy, EncryptionConfig, S3Config,
};

// ML-Specific Storage Extension
// -----------------------------------------------------------------------------

/// Extended storage that combines core storage with ML-specific storage
pub struct MLStorage {
    /// Core storage for artifacts/outcomes
    core: Arc<PostgresStorage>,
    
    /// Model binary storage
    models: Arc<dyn ModelStorage>,
    
    /// Shared database pool for ML-specific tables
    pool: PgPool,
}

impl MLStorage {
    pub fn new(
        core: Arc<PostgresStorage>,
        models: Arc<dyn ModelStorage>,
        pool: PgPool,
    ) -> Self {
        Self { core, models, pool }
    }
    
    /// Get reference to core storage
    pub fn core(&self) -> &Arc<PostgresStorage> {
        &self.core
    }
    
    /// Get reference to model storage
    pub fn models(&self) -> &Arc<dyn ModelStorage> {
        &self.models
    }
    
    /// Get reference to database pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

// Training Data Storage
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: Uuid,
    pub workspace_id: Uuid,
    pub artifact_id: Option<Uuid>,
    pub input_text: String,
    pub suggested_outcome_id: Option<Uuid>,
    pub actual_outcome_id: Option<Uuid>,
    pub user_feedback: Option<String>,
    pub feedback_score: Option<f32>,
    pub context: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub is_validated: bool,
    pub validation_method: Option<String>,
    pub input_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainingExampleFilters {
    pub min_feedback_score: Option<f32>,
    pub validated_only: bool,
    pub has_feedback: bool,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct TrainingFeedback {
    pub user_feedback: Option<String>,
    pub feedback_score: f32,
    pub actual_outcome_id: Option<Uuid>,
    pub is_validated: bool,
    pub validator_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainingStats {
    pub total_examples: i64,
    pub validated_examples: i64,
    pub examples_with_feedback: i64,
    pub average_feedback_score: Option<f32>,
    pub examples_last_7_days: i64,
    pub examples_last_30_days: i64,
}

#[async_trait]
pub trait TrainingStorage: Send + Sync {
    async fn store_training_example(
        &self,
        workspace_id: Uuid,
        artifact_id: Option<Uuid>,
        example: &TrainingExample,
    ) -> Result<Uuid>;
    
    async fn get_training_examples(
        &self,
        workspace_id: Uuid,
        filters: TrainingExampleFilters,
    ) -> Result<Vec<TrainingExample>>;
    
    async fn update_training_feedback(
        &self,
        example_id: Uuid,
        feedback: TrainingFeedback,
    ) -> Result<()>;
    
    async fn get_training_stats(
        &self,
        workspace_id: Uuid,
    ) -> Result<TrainingStats>;
}

#[async_trait]
impl TrainingStorage for MLStorage {
    async fn store_training_example(
        &self,
        workspace_id: Uuid,
        artifact_id: Option<Uuid>,
        example: &TrainingExample,
    ) -> Result<Uuid> {
        let example_id = example.id;
        
        // Convert embedding to PostgreSQL array format if present
        let embedding_array: Option<Vec<f32>> = example.input_embedding.clone();
        
        sqlx::query!(
            r#"
            INSERT INTO training_examples (
                id, workspace_id, artifact_id, input_text, 
                suggested_outcome_id, actual_outcome_id,
                user_feedback, feedback_score, context,
                is_validated, validation_method, input_embedding,
                created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            "#,
            example_id,
            workspace_id,
            artifact_id,
            example.input_text,
            example.suggested_outcome_id,
            example.actual_outcome_id,
            example.user_feedback,
            example.feedback_score,
            example.context,
            example.is_validated,
            Some(serde_json::to_value(example.validation_method.clone())?),
            Some(serde_json::to_value(embedding_array.as_deref())?),
            example.created_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to store training example")?;
        
        Ok(example_id)
    }
    
    async fn get_training_examples(
        &self,
        workspace_id: Uuid,
        filters: TrainingExampleFilters,
    ) -> Result<Vec<TrainingExample>> {
        // Build dynamic query
        let mut query_builder: QueryBuilder<Postgres> = QueryBuilder::new(
            r#"
            SELECT 
                id, workspace_id, artifact_id, input_text,
                suggested_outcome_id, actual_outcome_id,
                user_feedback, feedback_score, context,
                is_validated, validation_method, input_embedding,
                created_at
            FROM training_examples
            WHERE workspace_id = "#
        );
        
        query_builder.push_bind(workspace_id);
        
        // Apply filters
        if filters.validated_only {
            query_builder.push(" AND is_validated = true");
        }
        
        if filters.has_feedback {
            query_builder.push(" AND user_feedback IS NOT NULL");
        }
        
        if let Some(min_score) = filters.min_feedback_score {
            query_builder.push(" AND feedback_score >= ");
            query_builder.push_bind(min_score);
        }
        
        if let Some(after) = filters.created_after {
            query_builder.push(" AND created_at >= ");
            query_builder.push_bind(after);
        }
        
        if let Some(before) = filters.created_before {
            query_builder.push(" AND created_at <= ");
            query_builder.push_bind(before);
        }
        
        query_builder.push(" ORDER BY created_at DESC");
        
        if let Some(limit) = filters.limit {
            query_builder.push(" LIMIT ");
            query_builder.push_bind(limit);
        }
        
        if let Some(offset) = filters.offset {
            query_builder.push(" OFFSET ");
            query_builder.push_bind(offset);
        }
        
        let query = query_builder.build_query_as::<TrainingExampleRow>();
        let rows = query
            .fetch_all(&self.pool)
            .await
            .context("Failed to fetch training examples")?;
        
        Ok(rows.into_iter().map(Into::into).collect())
    }
    
    async fn update_training_feedback(
        &self,
        example_id: Uuid,
        feedback: TrainingFeedback,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE training_examples
            SET 
                user_feedback = COALESCE($2, user_feedback),
                feedback_score = $3,
                actual_outcome_id = COALESCE($4, actual_outcome_id),
                is_validated = $5,
                validator_id = COALESCE($6, validator_id),
                validated_at = CASE WHEN $5 THEN NOW() ELSE validated_at END,
                updated_at = NOW()
            WHERE id = $1
            "#,
            example_id,
            feedback.user_feedback,
            feedback.feedback_score,
            feedback.actual_outcome_id,
            feedback.is_validated,
            feedback.validator_id
        )
        .execute(&self.pool)
        .await
        .context("Failed to update training feedback")?;
        
        Ok(())
    }
    
    async fn get_training_stats(
        &self,
        workspace_id: Uuid,
    ) -> Result<TrainingStats> {
        let stats = sqlx::query!(
            r#"
            SELECT 
                COUNT(*) as "total_examples!",
                COUNT(*) FILTER (WHERE is_validated = true) as "validated_examples!",
                COUNT(*) FILTER (WHERE user_feedback IS NOT NULL) as "examples_with_feedback!",
                AVG(feedback_score) FILTER (WHERE feedback_score IS NOT NULL) as avg_feedback_score,
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '7 days') as "examples_last_7_days!",
                COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as "examples_last_30_days!"
            FROM training_examples
            WHERE workspace_id = $1
            "#,
            workspace_id
        )
        .fetch_one(&self.pool)
        .await
        .context("Failed to fetch training statistics")?;
        
        Ok(TrainingStats {
            total_examples: stats.total_examples,
            validated_examples: stats.validated_examples,
            examples_with_feedback: stats.examples_with_feedback,
            average_feedback_score: stats.avg_feedback_score.map(|v| v as f32),
            examples_last_7_days: stats.examples_last_7_days,
            examples_last_30_days: stats.examples_last_30_days,
        })
    }
}

// Inference Storage
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub id: Uuid,
    pub workspace_id: Uuid,
    pub artifact_id: Option<Uuid>,
    pub model_version: String,
    pub input_text: String,
    pub predicted_outcome_id: Uuid,
    pub confidence: f32,
    pub latency_ms: i64,
    pub created_at: DateTime<Utc>,
}

#[async_trait]
pub trait InferenceStorage: Send + Sync {
    async fn store_inference(
        &self,
        workspace_id: Uuid,
        inference: &InferenceResult,
    ) -> Result<Uuid>;
    
    async fn get_recent_inferences(
        &self,
        workspace_id: Uuid,
        limit: i64,
    ) -> Result<Vec<InferenceResult>>;
    
    async fn cache_prediction(
        &self,
        workspace_id: Uuid,
        input_hash: &str,
        prediction: &serde_json::Value,
        model_version: &str,
        ttl_seconds: i64,
    ) -> Result<()>;
    
    async fn get_cached_prediction(
        &self,
        workspace_id: Uuid,
        input_hash: &str,
        model_version: &str,
    ) -> Result<Option<serde_json::Value>>;
    
    async fn invalidate_cache(
        &self,
        workspace_id: Uuid,
        model_version: Option<&str>,
    ) -> Result<u64>;
}

#[async_trait]
impl InferenceStorage for MLStorage {
    async fn store_inference(
        &self,
        workspace_id: Uuid,
        inference: &InferenceResult,
    ) -> Result<Uuid> {
        let inference_id = inference.id;
        
        sqlx::query!(
            r#"
            INSERT INTO inference_history (
                id, workspace_id, artifact_id, model_version,
                input_text, predicted_outcome_id, confidence,
                latency_ms, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
            inference_id,
            workspace_id,
            inference.artifact_id,
            inference.model_version,
            inference.input_text,
            inference.predicted_outcome_id,
            inference.confidence as f64,
            inference.latency_ms,
            inference.created_at
        )
        .execute(&self.pool)
        .await
        .context("Failed to store inference")?;
        
        Ok(inference_id)
    }
    
    async fn get_recent_inferences(
        &self,
        workspace_id: Uuid,
        limit: i64,
    ) -> Result<Vec<InferenceResult>> {
        let rows = sqlx::query!(
            r#"
            SELECT 
                id, workspace_id, artifact_id, model_version,
                input_text, predicted_outcome_id, confidence,
                latency_ms, created_at
            FROM inference_history
            WHERE workspace_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
            workspace_id,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to fetch recent inferences")?;
        
        let results = rows.into_iter().map(|row| InferenceResult {
            id: row.id,
            workspace_id: row.workspace_id,
            artifact_id: row.artifact_id,
            model_version: row.model_version,
            input_text: row.input_text,
            predicted_outcome_id: row.predicted_outcome_id.unwrap(),
            confidence: row.confidence as f32,
            latency_ms: row.latency_ms,
            created_at: row.created_at.unwrap(),
        }).collect();
        
        Ok(results)
    }
    
    async fn cache_prediction(
        &self,
        workspace_id: Uuid,
        input_hash: &str,
        prediction: &serde_json::Value,
        model_version: &str,
        ttl_seconds: i64,
    ) -> Result<()> {
        let expires_at = Utc::now() + chrono::Duration::seconds(ttl_seconds);
        
        sqlx::query!(
            r#"
            INSERT INTO prediction_cache (
                id, workspace_id, input_hash, prediction, 
                model_version, confidence, expires_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (workspace_id, input_hash, model_version)
            DO UPDATE SET 
                prediction = EXCLUDED.prediction,
                expires_at = EXCLUDED.expires_at,
                created_at = NOW()
            "#,
            Uuid::new_v4(),
            workspace_id,
            input_hash,
            prediction,
            model_version,
            0.0f32 as f64, // Default confidence, should be extracted from prediction
            expires_at,
            Utc::now()
        )
        .execute(&self.pool)
        .await
        .context("Failed to cache prediction")?;
        
        Ok(())
    }
    
    async fn get_cached_prediction(
        &self,
        workspace_id: Uuid,
        input_hash: &str,
        model_version: &str,
    ) -> Result<Option<serde_json::Value>> {
        let row = sqlx::query!(
            r#"
            SELECT prediction
            FROM prediction_cache
            WHERE workspace_id = $1 
                AND input_hash = $2 
                AND model_version = $3
                AND expires_at > NOW()
            "#,
            workspace_id,
            input_hash,
            model_version
        )
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch cached prediction")?;
        
        Ok(row.map(|r| r.prediction))
    }
    
    async fn invalidate_cache(
        &self,
        workspace_id: Uuid,
        model_version: Option<&str>,
    ) -> Result<u64> {
        let result = if let Some(version) = model_version {
            sqlx::query!(
                r#"
                DELETE FROM prediction_cache
                WHERE workspace_id = $1 AND model_version = $2
                "#,
                workspace_id,
                version
            )
            .execute(&self.pool)
            .await?
        } else {
            sqlx::query!(
                r#"
                DELETE FROM prediction_cache
                WHERE workspace_id = $1
                "#,
                workspace_id
            )
            .execute(&self.pool)
            .await?
        };
        
        Ok(result.rows_affected())
    }
}

// Bridge to Core Storage
// -----------------------------------------------------------------------------

/// Bridge service that creates training examples from core artifacts
pub struct TrainingDataBridge {
    storage: Arc<MLStorage>,
}

impl TrainingDataBridge {
    pub fn new(storage: Arc<MLStorage>) -> Self {
        Self { storage }
    }
    
    /// Convert artifact to training example
    pub async fn create_training_example_from_artifact(
        &self,
        workspace_id: Uuid,
        artifact_id: Uuid,
    ) -> Result<Uuid> {
        // Fetch artifact from core storage
        let workspace_id = WorkspaceId::from_uuid(workspace_id);
        let artifacts = self.storage.core()
            .query_artifacts(workspace_id, None)
            .await?;
        
        let artifact = artifacts.into_iter()
            .find(|a| a.id == artifact_id)
            .context("Artifact not found")?;
        
        // Create training example
        let example = TrainingExample {
            id: Uuid::new_v4(),
            workspace_id: workspace_id.as_uuid().clone(),
            artifact_id: Some(artifact_id),
            input_text: artifact.content.clone(),
            suggested_outcome_id: None,
            actual_outcome_id: None,
            user_feedback: None,
            feedback_score: None,
            context: Some(serde_json::json!({
                "platform": artifact.platform.to_string(),
                "artifact_type": format!("{:?}", artifact.artifact_type),
                "metadata": artifact.metadata,
                "original_timestamp": artifact.created_at,
            })),
            created_at: Utc::now(),
            is_validated: false,
            validation_method: None,
            input_embedding: None,
        };
        
        self.storage.store_training_example(
            workspace_id.as_uuid().clone(),
            Some(artifact_id),
            &example,
        ).await
    }
    
    /// Link training feedback to core outcomes
    pub async fn link_training_to_outcome(
        &self,
        training_example_id: Uuid,
        outcome_id: Uuid,
        confidence: f32,
    ) -> Result<()> {
        // Get the training example
        let example = sqlx::query!(
            r#"
            SELECT artifact_id
            FROM training_examples
            WHERE id = $1
            "#,
            training_example_id
        )
        .fetch_one(&self.storage.pool)
        .await
        .context("Training example not found")?;
        
        // If there's an associated artifact, link it to the outcome in core storage
        if let Some(artifact_id) = example.artifact_id {
            self.storage.core()
                .link_artifact_outcome(artifact_id, OutcomeId::from_uuid(outcome_id), confidence as f64, None)
                .await?;
        }
        
        // Update the training example with the outcome
        let feedback = TrainingFeedback {
            user_feedback: Some("ML-predicted outcome".to_string()),
            feedback_score: confidence,
            actual_outcome_id: Some(outcome_id),
            is_validated: false,
            validator_id: None,
        };
        
        self.storage.update_training_feedback(training_example_id, feedback).await
    }
    
    /// Batch create training examples from multiple artifacts
    pub async fn batch_create_from_artifacts(
        &self,
        workspace_id: Uuid,
        artifact_ids: Vec<Uuid>,
    ) -> Result<Vec<Uuid>> {
        let mut example_ids = Vec::new();
        
        for artifact_id in artifact_ids {
            match self.create_training_example_from_artifact(workspace_id, artifact_id).await {
                Ok(id) => example_ids.push(id),
                Err(e) => {
                    tracing::warn!(
                        artifact_id = %artifact_id,
                        error = %e,
                        "Failed to create training example from artifact"
                    );
                }
            }
        }
        
        Ok(example_ids)
    }
}

// Helper Types
// -----------------------------------------------------------------------------

#[derive(Debug, sqlx::FromRow)]
struct TrainingExampleRow {
    id: Uuid,
    workspace_id: Uuid,
    artifact_id: Option<Uuid>,
    input_text: String,
    suggested_outcome_id: Option<Uuid>,
    actual_outcome_id: Option<Uuid>,
    user_feedback: Option<String>,
    feedback_score: Option<f32>,
    context: Option<serde_json::Value>,
    created_at: DateTime<Utc>,
    is_validated: bool,
    validation_method: Option<String>,
    input_embedding: Option<Vec<f32>>,
}

impl From<TrainingExampleRow> for TrainingExample {
    fn from(row: TrainingExampleRow) -> Self {
        TrainingExample {
            id: row.id,
            workspace_id: row.workspace_id,
            artifact_id: row.artifact_id,
            input_text: row.input_text,
            suggested_outcome_id: row.suggested_outcome_id,
            actual_outcome_id: row.actual_outcome_id,
            user_feedback: row.user_feedback,
            feedback_score: row.feedback_score,
            context: row.context,
            created_at: row.created_at,
            is_validated: row.is_validated,
            validation_method: row.validation_method,
            input_embedding: row.input_embedding,
        }
    }
}

// Storage Factory
// -----------------------------------------------------------------------------

pub struct StorageFactory;

impl StorageFactory {
    /// Create ML storage with all integrations
    pub async fn create_ml_storage(
        database_url: &str,
        model_storage_config: ModelStorageConfig,
    ) -> Result<Arc<MLStorage>> {
        // Create shared database pool with optimized settings
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(32)
            .min_connections(5)
            .connect(database_url)
            .await
            .context("Failed to connect to database")?;
        
        // Run migrations if needed
        sqlx::migrate!("../../migrations")
            .run(&pool)
            .await
            .context("Failed to run migrations")?;
        
        // Create core storage
        let config = interstice_core::storage::StorageConfig {
            database_url: database_url.to_string(),
            ..Default::default()
        };
        let core_storage = Arc::new(PostgresStorage::new(config).await?);
        
        // Create model storage based on config
        let model_storage: Arc<dyn ModelStorage> = match model_storage_config {
            ModelStorageConfig::S3(config) => {
                Arc::new(
                    S3ModelStorage::new(
                        config.bucket,
                        config.prefix,
                        config.encryption,
                        config.compression,
                    ).await
                    .context("Failed to create S3 model storage")?
                )
            }
            ModelStorageConfig::Local(path) => {
                Arc::new(
                    LocalModelStorage::new(
                        path,
                        CompressionStrategy::Gzip { level: 6 },
                    ).await
                    .context("Failed to create local model storage")?
                )
            }
            ModelStorageConfig::Hybrid { local_path, s3_config, cache_ttl } => {
                Arc::new(
                    HybridModelStorage::new(
                        local_path,
                        s3_config,
                        cache_ttl,
                    ).await
                    .context("Failed to create hybrid model storage")?
                )
            }
        };
        
        Ok(Arc::new(MLStorage::new(
            core_storage,
            model_storage,
            pool,
        )))
    }
}

#[derive(Debug, Clone)]
pub enum ModelStorageConfig {
    S3(S3Config),
    Local(std::path::PathBuf),
    Hybrid {
        local_path: std::path::PathBuf,
        s3_config: S3Config,
        cache_ttl: std::time::Duration,
    },
}

// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    async fn create_test_storage() -> Arc<MLStorage> {
        let temp_dir = TempDir::new().unwrap();
        let config = ModelStorageConfig::Local(temp_dir.path().to_path_buf());
        
        StorageFactory::create_ml_storage(
            "postgres://test:test@localhost/test_db",
            config,
        ).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_training_example_lifecycle() {
        let storage = create_test_storage().await;
        
        // Create example
        let example = TrainingExample {
            id: Uuid::new_v4(),
            workspace_id: Uuid::new_v4(),
            artifact_id: None,
            input_text: "Test input".to_string(),
            suggested_outcome_id: None,
            actual_outcome_id: None,
            user_feedback: None,
            feedback_score: None,
            context: None,
            created_at: Utc::now(),
            is_validated: false,
            validation_method: None,
            input_embedding: None,
        };
        
        // Store
        let id = storage.store_training_example(
            example.workspace_id,
            None,
            &example,
        ).await.unwrap();
        
        // Retrieve
        let examples = storage.get_training_examples(
            example.workspace_id,
            TrainingExampleFilters::default(),
        ).await.unwrap();
        
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].id, id);
    }
    
    #[tokio::test]
    async fn test_prediction_caching() {
        let storage = create_test_storage().await;
        
        let workspace_id = Uuid::new_v4();
        let input_hash = "test_hash";
        let prediction = serde_json::json!({"outcome": "test"});
        let model_version = "v1.0.0";
        
        // Cache prediction
        storage.cache_prediction(
            workspace_id,
            input_hash,
            &prediction,
            model_version,
            3600,
        ).await.unwrap();
        
        // Retrieve cached prediction
        let cached = storage.get_cached_prediction(
            workspace_id,
            input_hash,
            model_version,
        ).await.unwrap();
        
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), prediction);
    }
}