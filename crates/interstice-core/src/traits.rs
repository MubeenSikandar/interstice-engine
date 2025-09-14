//interstice-core/src/traits.rs
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

use crate::types::Priority;

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
    pub suggested_targets: Vec<String>,
    pub estimated_impact: f64,
    pub recommended_priority: Priority,
    pub alternative_outcomes: Vec<AlternativeOutcome>,
    pub contributing_factors: Vec<ContributingFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub factor_id: String,
    pub name: String,
    pub weight: f32,
    pub description: Option<String>,
    pub category: FactorCategory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorCategory {
    Historical,
    Contextual,
    Environmental,
    Behavioral,
    Technical,
    External,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeOutcome {
    pub outcome_id: Uuid,
    pub outcome_name: String,
    pub probability: f32,
    pub relative_likelihood: f32,
    pub key_differences: Vec<String>,
}

// Platform conversion trait
pub trait PlatformConverter {
    fn to_ml_platform(&self) -> i32;
}

// ArtifactType conversion trait  
pub trait ArtifactTypeConverter {
    fn to_ml_type(&self) -> i32;
}