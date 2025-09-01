//interstice-core/src/traits.rs
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

#[async_trait]
pub trait MLPredictor: Send + Sync {
    async fn predict_outcomes(
        &self,
        artifacts: &[crate::Artifact],
    ) -> anyhow::Result<Vec<OutcomePrediction>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePrediction {
    pub outcome_id: Uuid,
    pub outcome_name: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
}

// Platform conversion trait
pub trait PlatformConverter {
    fn to_ml_platform(&self) -> i32;
}

// ArtifactType conversion trait  
pub trait ArtifactTypeConverter {
    fn to_ml_type(&self) -> i32;
}