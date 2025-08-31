//! Core domain models and business logic for Interstice
//interstice-core/src/lib.rs
pub mod analytics;
pub mod artifact;
pub mod error;
pub mod graph;
pub mod outcome;
pub mod types;
pub mod storage;
pub mod traits; // Add this module

pub use storage::{Storage, DatabaseStorage, WorkspaceStats, ProgressPoint};

// Re-export main types
pub use artifact::{Artifact, ArtifactExtractor, ArtifactType};
pub use error::{Error, Result};
pub use outcome::{Outcome, OutcomeMapper};
pub use traits::{MLPredictor, OutcomePrediction}; // Export from traits, not outcome
pub use types::{Platform, UserId, WorkspaceId};
// Remove the interstice_ml import - it causes circular dependency

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
            mapper: Arc::new(OutcomeMapper::new(None)), // Pass None for ML predictor
            storage: None,
        }
    }
    
    pub fn with_ml_predictor(mut self, predictor: Arc<dyn MLPredictor>) -> Self {
        self.mapper = Arc::new(OutcomeMapper::new(Some(predictor)));
        self
    }

    pub fn with_storage(mut self, storage: Arc<dyn Storage>) -> Self {
        self.storage = Some(storage);
        self
    }

    // Rest of the implementation stays the same...
    pub async fn process(&self, content: String, platform: Platform) -> Result<ProcessedArtifact> {
        let artifacts = self.extractor.extract(&content, platform).await?;
        let predictions = self.mapper.predict(&artifacts).await?;

        Ok(ProcessedArtifact {
            artifacts,
            predictions,
            platform,
        })
    }

    pub async fn extract_artifacts(&self, content: &str, platform: Platform) -> Result<Vec<Artifact>> {
        self.extractor.extract(content, platform).await
    }

    pub async fn store_processed_data(
        &self,
        processed: &ProcessedArtifact,
        workspace_id: Uuid,
    ) -> Result<()> {
        if let Some(storage) = &self.storage {
            for artifact in &processed.artifacts {
                let artifact_id = storage.store_artifact(artifact, workspace_id).await?;
                
                for prediction in &processed.predictions {
                    storage.link_artifact_outcome(artifact_id, prediction.outcome_id, prediction.confidence).await?;
                }
            }
        }
        Ok(())
    }

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