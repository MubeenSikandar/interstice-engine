pub mod embeddings;
pub mod training;
pub mod inference;
pub mod feedback;
pub mod types;
pub mod models;

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use anyhow::Result;

use crate::embeddings::Embedder;
use crate::training::ContinuousTrainer;
use crate::inference::OutcomePredictor;
use crate::feedback::FeedbackProcessor;
use crate::types::OutcomePrediction;
use interstice_core::Artifact;

pub struct MLPipeline {
    embedder: Arc<Embedder>,
    predictor: Arc<OutcomePredictor>,
    trainer: Arc<ContinuousTrainer>,
    feedback_loop: Arc<FeedbackProcessor>,
}

impl MLPipeline {
    pub async fn new(database_url: &str) -> Result<Self> {
        Ok(Self {
            embedder: Arc::new(Embedder::new().await?),
            predictor: Arc::new(OutcomePredictor::new().await?),
            trainer: Arc::new(ContinuousTrainer::new(database_url).await?),
            feedback_loop: Arc::new(FeedbackProcessor::new_with_db(database_url).await?),
        })
    }

    pub fn connect_lazy(database_url: &str) -> Result<Self> {
        // For lazy connection, create a dummy pipeline that will connect when first used
        // This is a temporary solution until we implement proper lazy loading
        Ok(Self {
            embedder: Arc::new(Embedder::connect_lazy().unwrap()),
            predictor: Arc::new(OutcomePredictor::connect_lazy().unwrap()),
            trainer: Arc::new(ContinuousTrainer::connect_lazy(database_url).unwrap()),
            feedback_loop: Arc::new(FeedbackProcessor::new()),
        })
    }
    
    pub async fn predict_outcomes(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        text: &str,
    ) -> Result<Vec<OutcomePrediction>> {
        // 1. Generate embeddings
        let embedding = self.embedder.embed_text(text).await?;
        
        // 2. Get org-specific predictions
        let predictions = self.predictor.predict(
            workspace_id,
            embedding,
            artifacts
        ).await?;
        
        // 3. Learn vocabulary
        self.learn_org_vocabulary(workspace_id, text).await?;
        
        Ok(predictions)
    }

    async fn learn_org_vocabulary(&self, workspace_id: Uuid, text: &str) -> Result<()> {
        // In a real implementation, this would:
        // 1. Extract key terms from text
        // 2. Generate embeddings for terms
        // 3. Store in org_vocabulary table
        
        // For now, just log that we would learn vocabulary
        tracing::debug!("Would learn vocabulary for workspace {} from text: {}", workspace_id, text);
        Ok(())
    }

    pub async fn start_training_loop(&self) -> Result<()> {
        let trainer = Arc::clone(&self.trainer);
        trainer.start_training_loop().await
    }

    pub async fn process_feedback(
        &self,
        workspace_id: Uuid,
        action: crate::types::UserAction,
    ) -> Result<()> {
        self.feedback_loop.process_user_action(workspace_id, action).await
    }

    pub async fn get_model_performance(&self, workspace_id: Uuid) -> Result<Option<crate::types::ModelMetrics>> {
        self.predictor.get_model_performance(workspace_id).await
    }
}