use async_trait::async_trait;
use interstice_core::traits::{MLPredictor, OutcomePrediction as CorePrediction};
use interstice_core::{Artifact as CoreArtifact, Platform as CorePlatform, ArtifactType as CoreArtifactType};
use crate::inference::{OutcomeEngine, EngineConfig};
use crate::types::{Artifact, Platform, ArtifactType, PredictionContext, OutcomePrediction};
use chrono::{Timelike, Datelike};

pub struct MLPredictorAdapter {
    engine: OutcomeEngine,
}

impl MLPredictorAdapter {
    pub async fn new() -> anyhow::Result<Self> {
        let config = EngineConfig {
            embedding_model_path: "models/embeddings.onnx".to_string(),
            predictor_model_path: "models/predictor.onnx".to_string(),
            n_outcomes: 10,
        };
        
        let engine = OutcomeEngine::new(config).await?;
        Ok(Self { engine })
    }
}

#[async_trait]
impl MLPredictor for MLPredictorAdapter {
    async fn predict_outcomes(
        &self,
        artifacts: &[CoreArtifact],
    ) -> anyhow::Result<Vec<CorePrediction>> {
        // Convert core artifacts to ML artifacts
        let ml_artifacts: Vec<Artifact> = artifacts.iter()
            .map(|a| convert_artifact(a))
            .collect();
        
        let context = create_prediction_context();
        
        // Get ML predictions
        let ml_predictions = self.engine.predict(ml_artifacts, context).await?;
        
        // Convert back to core predictions
        Ok(ml_predictions.into_iter().map(|p| CorePrediction {
            outcome_id: uuid::Uuid::parse_str(&p.outcome_id).unwrap_or_else(|_| uuid::Uuid::new_v4()),
            outcome_name: p.outcome_name,
            confidence: p.confidence,
            reasoning: p.reasoning,
        }).collect())
    }
}

// ADD THIS MISSING FUNCTION
fn convert_artifact(artifact: &CoreArtifact) -> Artifact {
    Artifact {
        id: artifact.id.to_string(),
        version: 1,
        content: artifact.raw_text.clone(),
        platform: convert_platform(&artifact.platform),
        artifact_type: convert_artifact_type(&artifact.artifact_type),
        metadata: None,
    }
}

fn convert_platform(platform: &CorePlatform) -> Platform {
    match platform {
        CorePlatform::Slack => Platform::Slack,
        CorePlatform::GitHub => Platform::GitHub,
        CorePlatform::Jira => Platform::Jira,
        CorePlatform::Teams => Platform::Teams,
        CorePlatform::Asana => Platform::Asana,
        CorePlatform::VSCode => Platform::VSCode,
        CorePlatform::GoogleWorkspace => Platform::GoogleWorkspace,
        CorePlatform::Monday => Platform::Monday,
        CorePlatform::Trello => Platform::Trello,
        CorePlatform::Zoom => Platform::Zoom,
        CorePlatform::Figma => Platform::Figma,
        CorePlatform::Notion => Platform::Notion,
    }
}

fn convert_artifact_type(artifact_type: &CoreArtifactType) -> ArtifactType {
    match artifact_type {
        CoreArtifactType::PullRequest { .. } => ArtifactType::PullRequest,
        CoreArtifactType::Issue { .. } => ArtifactType::Issue,
        CoreArtifactType::Commit { .. } => ArtifactType::Commit,
        CoreArtifactType::Document { .. } => ArtifactType::Document,
        CoreArtifactType::Message { .. } => ArtifactType::Message,
    }
}

fn create_prediction_context() -> PredictionContext {
    let now = chrono::Local::now();
    PredictionContext {
        hour_of_day: now.hour(),
        day_of_week: now.weekday().num_days_from_monday(),
        days_until_deadline: 7.0,
        user_activity_level: 0.7,
        user_expertise_score: 0.8,
        team_size: 5,
    }
}