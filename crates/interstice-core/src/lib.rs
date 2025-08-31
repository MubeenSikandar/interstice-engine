//! Core domain models and business logic for Interstice

pub mod analytics;
pub mod artifact;
pub mod error;
pub mod graph;
pub mod outcome;
pub mod types;
pub mod storage;
pub use storage::{Storage, DatabaseStorage, WorkspaceStats, ProgressPoint};

// Re-export main types
pub use artifact::{Artifact, ArtifactExtractor, ArtifactType};
pub use error::{Error, Result};
pub use outcome::{Outcome, OutcomeMapper, OutcomePrediction};
pub use types::{Platform, UserId, WorkspaceId};

use std::sync::Arc;
use uuid::Uuid;

/// The main engine that processes artifacts from any platform
pub struct IntersticeEngine {
    extractor: Arc<ArtifactExtractor>,
    mapper: Arc<OutcomeMapper>,
    storage: Option<Arc<dyn Storage>>,
}

impl IntersticeEngine {
    pub fn new() -> Self {
        Self {
            extractor: Arc::new(ArtifactExtractor::new()),
            mapper: Arc::new(OutcomeMapper::new()),
            storage: None,
        }
    }

    pub fn with_storage(mut self, storage: Arc<dyn Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    pub async fn process(&self, content: String, platform: Platform) -> Result<ProcessedArtifact> {
        // Extract artifacts from content
        let artifacts = self.extractor.extract(&content, platform).await?;

        // Map to potential outcomes
        let predictions = self.mapper.predict(&artifacts).await?;

        Ok(ProcessedArtifact {
            artifacts,
            predictions,
            platform,
        })
    }

    /// Extract artifacts from text content
    pub async fn extract_artifacts(&self, content: &str, platform: Platform) -> Result<Vec<Artifact>> {
        self.extractor.extract(content, platform).await
    }

    /// Store artifacts and outcomes in the database
    pub async fn store_processed_data(
        &self,
        processed: &ProcessedArtifact,
        workspace_id: Uuid,
    ) -> Result<()> {
        if let Some(storage) = &self.storage {
            // Store artifacts
            for artifact in &processed.artifacts {
                let artifact_id = storage.store_artifact(artifact, workspace_id).await?;
                
                // Link to outcomes if we have predictions
                for prediction in &processed.predictions {
                    storage.link_artifact_outcome(artifact_id, prediction.outcome_id, prediction.confidence).await?;
                }
            }
        }
        Ok(())
    }

    /// Get workspace statistics
    pub async fn get_workspace_stats(&self, workspace_id: Uuid) -> Result<WorkspaceStats> {
        if let Some(storage) = &self.storage {
            storage.get_workspace_stats(workspace_id).await
                .map_err(|e| Error::Other(anyhow::anyhow!("Failed to get workspace stats: {}", e)))
        } else {
            Err(Error::Other(anyhow::anyhow!("No storage configured")))
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessedArtifact {
    pub artifacts: Vec<Artifact>,
    pub predictions: Vec<OutcomePrediction>,
    pub platform: Platform,
}
