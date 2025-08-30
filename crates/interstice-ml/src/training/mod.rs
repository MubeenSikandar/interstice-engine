use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use uuid::Uuid;
use sqlx::PgPool;
use anyhow::Result;
use tracing::{info, error};

use crate::models::OrgModel;
use crate::types::{TrainingExample, ModelMetrics};

pub struct ContinuousTrainer {
    db: PgPool,
    model_registry: Arc<RwLock<HashMap<Uuid, OrgModel>>>,
}

impl ContinuousTrainer {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        
        Ok(Self {
            db: pool,
            model_registry: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn connect_lazy(database_url: &str) -> Result<Self> {
        // For lazy connection, create a dummy pool that will connect when first used
        let pool = PgPool::connect_lazy(database_url)
            .map_err(|e| anyhow::anyhow!("Failed to create lazy connection: {}", e))?;
        
        Ok(Self {
            db: pool,
            model_registry: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn start_training_loop(self: Arc<Self>) -> Result<()> {
        tokio::spawn(async move {
            loop {
                // Run every hour
                tokio::time::sleep(Duration::from_secs(3600)).await;
                
                match self.run_training_cycle().await {
                    Ok(_) => info!("Training cycle completed successfully"),
                    Err(e) => error!("Training cycle failed: {}", e),
                }
            }
        });
        
        Ok(())
    }

    async fn run_training_cycle(&self) -> Result<()> {
        // Get workspaces with new training data
        let workspaces = self.get_workspaces_needing_training().await?;
        
        for workspace_id in workspaces {
            if let Err(e) = self.train_workspace_model(workspace_id).await {
                error!("Failed to train workspace {}: {}", workspace_id, e);
            }
        }
        
        Ok(())
    }
    
    async fn get_workspaces_needing_training(&self) -> Result<Vec<Uuid>> {
        let records = sqlx::query!(
            r#"
            SELECT DISTINCT workspace_id 
            FROM training_examples 
            WHERE created_at > NOW() - INTERVAL '24 hours'
            AND user_feedback IS NOT NULL
            "#
        )
        .fetch_all(&self.db)
        .await?;
        
        Ok(records.into_iter().filter_map(|r| r.workspace_id).collect())
    }
    
    async fn train_workspace_model(&self, workspace_id: Uuid) -> Result<()> {
        // 1. Get training examples
        let examples = self.get_training_examples(workspace_id, 1000).await?;
        
        if examples.len() < 100 {
            return Ok(()); // Need minimum data
        }
        
        // 2. Get or create org model
        let mut models = self.model_registry.write().await;
        let model = models.entry(workspace_id)
            .or_insert_with(|| OrgModel::new(workspace_id));
        
        // 3. Fine-tune with LoRA
        model.fine_tune(&examples).await?;
        
        // 4. Evaluate performance
        let metrics = model.evaluate().await?;
        
        // 5. Save if improved
        if metrics.accuracy > model.best_accuracy {
            self.save_model(workspace_id, model).await?;
            model.best_accuracy = metrics.accuracy;
        }
        
        Ok(())
    }

    async fn get_training_examples(&self, workspace_id: Uuid, limit: i64) -> Result<Vec<TrainingExample>> {
        let records = sqlx::query!(
            r#"
            SELECT id, input_text, suggested_outcome_id, actual_outcome_id, user_feedback
            FROM training_examples 
            WHERE workspace_id = $1 
            AND user_feedback IS NOT NULL
            ORDER BY created_at DESC 
            LIMIT $2
            "#,
            workspace_id,
            limit
        )
        .fetch_all(&self.db)
        .await?;
        
        let examples = records.into_iter().map(|r| TrainingExample {
            id: r.id,
            input_text: r.input_text,
            suggested_outcome_id: r.suggested_outcome_id,
            actual_outcome_id: r.actual_outcome_id,
            user_feedback: r.user_feedback,
        }).collect();
        
        Ok(examples)
    }

    async fn save_model(&self, workspace_id: Uuid, model: &OrgModel) -> Result<()> {
        // In a real implementation, this would save the model weights
        // For now, we'll just log that we would save it
        info!("Would save improved model for workspace {}", workspace_id);
        Ok(())
    }
}