//! Core domain models and business logic for Interstice

pub mod analytics;
pub mod artifact;
pub mod error;
pub mod graph;
pub mod outcome;
pub mod types;
pub mod storage;
pub use storage::Storage;

// Re-export main types
pub use artifact::{Artifact, ArtifactExtractor, ArtifactType};
pub use error::{Error, Result};
pub use outcome::{Outcome, OutcomeMapper, OutcomePrediction};
pub use types::{Platform, UserId, WorkspaceId};

use std::sync::Arc;

/// The main engine that processes artifacts from any platform
pub struct IntersticeEngine {
    extractor: Arc<ArtifactExtractor>,
    mapper: Arc<OutcomeMapper>,
    // We'll add more components as we build them
}

impl IntersticeEngine {
    pub fn new() -> Self {
        Self {
            extractor: Arc::new(ArtifactExtractor::new()),
            mapper: Arc::new(OutcomeMapper::new()),
        }
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
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcessedArtifact {
    pub artifacts: Vec<Artifact>,
    pub predictions: Vec<OutcomePrediction>,
    pub platform: Platform,
}
