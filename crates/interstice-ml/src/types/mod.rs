use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: Uuid,
    pub input_text: String,
    pub suggested_outcome_id: Option<Uuid>,
    pub actual_outcome_id: Option<Uuid>,
    pub user_feedback: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub total_predictions: u32,
    pub correct_predictions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePrediction {
    pub outcome_id: Uuid,
    pub outcome_name: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    pub action_type: String,
    pub artifact_id: Uuid,
    pub outcome_id: Option<Uuid>,
    pub feedback: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrgVocabulary {
    pub workspace_id: Uuid,
    pub term: String,
    pub term_type: String,
    pub frequency: i32,
    pub embedding: Option<Vec<f32>>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
}
