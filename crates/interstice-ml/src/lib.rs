//interstice-ml/src/lib.rs
pub mod embeddings;
pub mod training;
pub mod inference;
pub mod feedback;
pub mod types;
pub mod models;
pub mod adapters;

use std::sync::Arc;
use uuid::Uuid;
use anyhow::Result;

// Export the adapter so other crates can use it
pub use adapters::MLPredictorAdapter;

// Export other public types
pub use inference::{OutcomeEngine, EngineConfig};
use crate::inference::{TextEmbedder, OutcomePredictor};
use crate::training::ContinuousTrainer;
use crate::feedback::FeedbackProcessor;
pub use types::{OutcomePrediction, ModelMetrics, Artifact};

pub struct MLPipeline {
    embedder: Arc<TextEmbedder>,
    predictor: Arc<OutcomePredictor>,
    trainer: Arc<ContinuousTrainer>,
    feedback_loop: Arc<FeedbackProcessor>,
}

impl MLPipeline {
    pub async fn new(database_url: &str) -> Result<Self> {
        Ok(Self {
            embedder: Arc::new(TextEmbedder::connect_lazy()?),
            predictor: Arc::new(OutcomePredictor::connect_lazy()?),
            trainer: Arc::new(ContinuousTrainer::new(database_url).await?),
            feedback_loop: Arc::new(FeedbackProcessor::new_with_db(database_url).await?),
        })
    }

    pub fn connect_lazy(database_url: &str) -> Result<Self> {
        Ok(Self {
            embedder: Arc::new(TextEmbedder::connect_lazy().unwrap()),
            predictor: Arc::new(OutcomePredictor::connect_lazy().unwrap()),
            trainer: Arc::new(ContinuousTrainer::connect_lazy(database_url).unwrap()),
            feedback_loop: Arc::new(FeedbackProcessor::new()),
        })
    }
    
    pub async fn predict_outcomes(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact], // Using ML's Artifact type
        text: &str,
    ) -> Result<Vec<OutcomePrediction>> {
        // Generate embeddings from text
        let embedding = self.embedder.embed_text(text).await?;
        
        // Now predictor.predict() expects ML's Artifact type
        let predictions = self.predictor.predict_ml(
            embedding,
            artifacts  // No conversion needed - same type
        ).await?;
        
        // Learn vocabulary from the text
        self.learn_org_vocabulary(workspace_id, text).await?;
        
        Ok(predictions)
    }

    // Alternative method for when you have interstice_core artifacts
    pub async fn predict_outcomes_from_core(
        &self,
        workspace_id: Uuid,
        core_artifacts: &[interstice_core::Artifact],
        text: &str,
    ) -> Result<Vec<OutcomePrediction>> {
        // Convert core artifacts to ML artifacts
        let ml_artifacts: Vec<Artifact> = core_artifacts.iter()
            .map(|a| Artifact {
                id: a.id.to_string(),
                version: 1,
                content: a.raw_text.clone(),
                platform: convert_core_platform(&a.platform),
                artifact_type: convert_core_artifact_type(&a.artifact_type),
                metadata: None,
            })
            .collect();
        
        // Call the main predict method
        self.predict_outcomes(workspace_id, &ml_artifacts, text).await
    }

    async fn learn_org_vocabulary(&self, workspace_id: Uuid, text: &str) -> Result<()> {
        // Extract key terms from text for vocabulary learning
        tracing::debug!("Learning vocabulary for workspace {} from text: {}", workspace_id, text);
        
        // In a production implementation, this would:
        // 1. Tokenize and extract key terms
        // 2. Generate embeddings for new terms
        // 3. Store in org_vocabulary table for fine-tuning
        
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

    pub async fn get_model_performance(&self, _workspace_id: Uuid) -> Result<Option<crate::types::ModelMetrics>> {
        self.predictor.get_model_performance().await
    }
}

// Helper functions for type conversion
fn convert_core_platform(platform: &interstice_core::Platform) -> crate::types::Platform {
    use crate::types::Platform;
    match platform {
        interstice_core::Platform::Slack => Platform::Slack,
        interstice_core::Platform::GitHub => Platform::GitHub,
        interstice_core::Platform::Jira => Platform::Jira,
        interstice_core::Platform::Teams => Platform::Teams,
        interstice_core::Platform::Asana => Platform::Asana,
        interstice_core::Platform::VSCode => Platform::VSCode,
        interstice_core::Platform::GoogleWorkspace => Platform::GoogleWorkspace,
        interstice_core::Platform::Monday => Platform::Monday,
        interstice_core::Platform::Trello => Platform::Trello,
        interstice_core::Platform::Zoom => Platform::Zoom,
        interstice_core::Platform::Figma => Platform::Figma,
        interstice_core::Platform::Notion => Platform::Notion,
    }
}

fn convert_core_artifact_type(artifact_type: &interstice_core::ArtifactType) -> crate::types::ArtifactType {
    use crate::types::ArtifactType;
    match artifact_type {
        interstice_core::ArtifactType::PullRequest { .. } => ArtifactType::PullRequest,
        interstice_core::ArtifactType::Issue { .. } => ArtifactType::Issue,
        interstice_core::ArtifactType::Commit { .. } => ArtifactType::Commit,
        interstice_core::ArtifactType::Document { .. } => ArtifactType::Document,
        interstice_core::ArtifactType::Message { .. } => ArtifactType::Message,
    }
}

// Convenience function to create an ML predictor for interstice-core
pub async fn create_ml_predictor() -> Result<MLPredictorAdapter> {
    MLPredictorAdapter::new().await
}