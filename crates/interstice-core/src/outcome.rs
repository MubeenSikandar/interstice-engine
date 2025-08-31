use std::sync::Arc;
use crate::{Artifact, Result};
use crate::traits::{MLPredictor, OutcomePrediction};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub target_value: Option<f64>,
    pub current_value: Option<f64>,
    pub parent_id: Option<Uuid>,
}

pub struct OutcomeMapper {
    ml_predictor: Option<Arc<dyn MLPredictor>>,
}

impl OutcomeMapper {
    pub fn new(ml_predictor: Option<Arc<dyn MLPredictor>>) -> Self {
        Self { ml_predictor }
    }

    pub async fn predict(&self, artifacts: &[Artifact]) -> Result<Vec<OutcomePrediction>> {
        if let Some(predictor) = &self.ml_predictor {
            predictor.predict_outcomes(artifacts).await
                .map_err(|e| crate::Error::Other(e))
        } else {
            self.fallback_predict(artifacts)
        }
    }
    
    fn fallback_predict(&self, artifacts: &[Artifact]) -> Result<Vec<OutcomePrediction>> {
        let mut predictions = Vec::new();
        
        for artifact in artifacts {
            match &artifact.artifact_type {
                crate::ArtifactType::PullRequest { .. } => {
                    predictions.push(OutcomePrediction {
                        outcome_id: Uuid::new_v4(),
                        outcome_name: "Code Quality Improvement".to_string(),
                        confidence: 0.75,
                        reasoning: Some("PR indicates code changes".to_string()),
                    });
                }
                crate::ArtifactType::Issue { .. } => {
                    predictions.push(OutcomePrediction {
                        outcome_id: Uuid::new_v4(),
                        outcome_name: "Bug Resolution".to_string(),
                        confidence: 0.65,
                        reasoning: Some("Issue tracking indicates problem solving".to_string()),
                    });
                }
                _ => {}
            }
        }
        
        Ok(predictions)
    }
}