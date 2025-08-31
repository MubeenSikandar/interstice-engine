use crate::{Artifact, Outcome, OutcomePrediction, Result, Error};
use crate::storage::{Storage, WorkspaceStats, ProgressPoint};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;
use chrono::{DateTime, Utc, Duration};
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
        info!("Building relationships for {} artifacts and {} predictions", artifacts.len(), predictions.len());

        for artifact in artifacts {
            // Store the artifact
            let artifact_id = self.storage.store_artifact(artifact, workspace_id).await?;
            
            // Link to outcomes based on predictions
            for prediction in predictions {
                // Use the outcome_id directly since it's already a Uuid
                self.storage.link_artifact_outcome(
                    artifact_id,
                    prediction.outcome_id,
                    prediction.confidence,
                ).await?;
            }
        }

        info!("Successfully built {} artifact-outcome relationships", artifacts.len() * predictions.len());
        Ok(())
    }
}
