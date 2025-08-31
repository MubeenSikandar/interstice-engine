use crate::{Artifact, Result};
use crate::traits::OutcomePrediction;
use crate::storage::Storage;
use std::sync::Arc;
use uuid::Uuid;
use tracing::info;

/// Evidence graph for tracking work-outcome relationships
pub struct EvidenceGraph {
    storage: Arc<dyn Storage>,
}

impl EvidenceGraph {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    /// Build relationships between artifacts and outcomes
    pub async fn build_relationships(
        &self,
        workspace_id: Uuid,
        artifacts: &[Artifact],
        predictions: &[OutcomePrediction],
    ) -> Result<()> {
        info!("Building relationships for {} artifacts and {} predictions", 
              artifacts.len(), predictions.len());

        for artifact in artifacts {
            // Store the artifact
            let artifact_id = self.storage.store_artifact(artifact, workspace_id).await?;
            
            // Link to outcomes based on predictions
            for prediction in predictions {
                self.storage.link_artifact_outcome(
                    artifact_id,
                    prediction.outcome_id,
                    prediction.confidence,
                ).await?;
            }
        }

        info!("Successfully built {} artifact-outcome relationships", 
              artifacts.len() * predictions.len());
        Ok(())
    }
}