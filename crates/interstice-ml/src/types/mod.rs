use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Define ML-specific versions of types to avoid circular dependency

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub id: String,
    pub version: u32,
    pub content: String,
    pub platform: Platform,
    pub artifact_type: ArtifactType,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Platform {
    Slack = 0,
    GitHub = 1,
    Jira = 2,
    Teams = 3,
    Asana = 4,
    VSCode = 5,
    GoogleWorkspace = 6,
    Monday = 7,
    Trello = 8,
    Zoom = 9,
    Figma = 10,
    Notion = 11,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ArtifactType {
    PullRequest = 0,
    Issue = 1,
    Commit = 2,
    Document = 3,
    Message = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContext {
    pub hour_of_day: u32,
    pub day_of_week: u32,
    pub days_until_deadline: f32,
    pub user_activity_level: f32,
    pub user_expertise_score: f32,
    pub team_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePrediction {
    pub outcome_id: String,
    pub outcome_name: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub correct_predictions: u64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub total_predictions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Accept,
    Reject,
    Modify,
    Defer,
    Correct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    pub action_type: ActionType,
    pub artifact_id: String,
    pub outcome_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: Uuid,
    pub input_text: String,
    pub suggested_outcome_id: Option<Uuid>,
    pub actual_outcome_id: Option<Uuid>,
    pub user_feedback: Option<String>,
}

