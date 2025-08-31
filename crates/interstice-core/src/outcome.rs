use std::sync::Arc;
use crate::{Artifact, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use interstice_ml::{OutcomeEngine, EngineConfig};

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
    ml_engine: Option<Arc<OutcomeEngine>>,
}

impl OutcomeMapper {
    pub async fn new() -> Result<Self> {
        // Create config with default paths
        let config = EngineConfig {
            embedding_model_path: "models/embeddings.onnx".to_string(),
            predictor_model_path: "models/predictor.onnx".to_string(),
            n_outcomes: 10,
        };
        
        // Try to load the engine, but don't fail if models aren't available yet
        let ml_engine = match OutcomeEngine::new(config).await {
            Ok(engine) => Some(Arc::new(engine)),
            Err(e) => {
                tracing::warn!("ML engine not available: {}", e);
                None
            }
        };
        
        Ok(Self { ml_engine })
    }

    pub async fn predict(&self, artifacts: &[Artifact]) -> Result<Vec<interstice_ml::types::OutcomePrediction>> {
        if let Some(engine) = &self.ml_engine {
            // Convert artifacts to ML format
            let ml_artifacts: Vec<interstice_ml::inference::Artifact> = artifacts.iter()
                .map(|a| self.convert_artifact(a))
                .collect();
            
            let context = self.create_prediction_context();
            
            // Use ML predictions
            engine.predict(ml_artifacts, context).await
        } else {
            // Fallback to rule-based predictions
            self.fallback_predict(artifacts)
        }
    }
    
    fn convert_artifact(&self, artifact: &Artifact) -> interstice_ml::inference::Artifact {
        // Convert from core Artifact to ML Artifact
        interstice_ml::inference::Artifact {
            id: artifact.id.to_string(),
            version: 1,
            content: artifact.raw_text.clone(),
            platform: self.map_platform(&artifact.platform),
            artifact_type: self.map_artifact_type(&artifact.artifact_type),
        }
    }
    
    fn create_prediction_context(&self) -> interstice_ml::inference::PredictionContext {
        let now = chrono::Local::now();
        interstice_ml::inference::PredictionContext {
            hour_of_day: now.hour(),
            day_of_week: now.weekday().num_days_from_monday(),
            days_until_deadline: 7.0, // Default, should come from project data
            user_activity_level: 0.7,
            user_expertise_score: 0.8,
            team_size: 5,
        }
    }
    
    fn fallback_predict(&self, artifacts: &[Artifact]) -> Result<Vec<interstice_ml::types::OutcomePrediction>> {
        // Your existing mock predictions
        let mut predictions = Vec::new();
        // ... existing code ...
        Ok(predictions)
    }
}