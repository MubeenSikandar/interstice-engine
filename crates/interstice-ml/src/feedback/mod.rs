use uuid::Uuid;
use anyhow::Result;
use sqlx::PgPool;
use tracing::{info, debug};

use crate::types::{UserAction, TrainingExample};

pub struct FeedbackProcessor {
    db: PgPool,
}

impl FeedbackProcessor {
    pub fn new() -> Self {
        Self {
            db: PgPool::connect_lazy("postgresql://dummy").unwrap(),
        }
    }

    pub async fn new_with_db(database_url: &str) -> Result<Self> {
        let pool = PgPool::connect(database_url).await?;
        
        Ok(Self { db: pool })
    }

    pub async fn process_user_action(
        &self,
        workspace_id: Uuid,
        action: UserAction,
    ) -> Result<()> {
        debug!("Processing user action: {:?} for workspace {}", action.action_type, workspace_id);
        
        match action.action_type.as_str() {
            "accepted" => {
                self.record_feedback(
                    workspace_id,
                    action.artifact_id,
                    action.outcome_id,
                    "accepted",
                    1.0
                ).await?;
            }
            
            "rejected" => {
                if let Some(outcome_id) = action.outcome_id {
                    self.record_feedback(
                        workspace_id,
                        action.artifact_id,
                        Some(outcome_id),
                        "rejected",
                        -1.0
                    ).await?;
                }
            }
            
            "corrected" => {
                if let Some(outcome_id) = action.outcome_id {
                    self.record_feedback(
                        workspace_id,
                        action.artifact_id,
                        Some(outcome_id),
                        "corrected",
                        1.0
                    ).await?;
                }
            }
            
            _ => {
                info!("Unknown action type: {}", action.action_type);
            }
        }
        
        Ok(())
    }

    async fn record_feedback(
        &self,
        workspace_id: Uuid,
        artifact_id: Uuid,
        outcome_id: Option<Uuid>,
        feedback_type: &str,
        score: f32,
    ) -> Result<()> {
        // Update training example with feedback
        sqlx::query!(
            r#"
            UPDATE training_examples 
            SET user_feedback = $1, feedback_timestamp = NOW()
            WHERE workspace_id = $2 AND artifact_id = $3
            "#,
            feedback_type,
            workspace_id,
            artifact_id
        )
        .execute(&self.db)
        .await?;

        // Update model performance metrics
        self.update_performance_metrics(workspace_id, feedback_type, score as f64).await?;
        
        info!("Recorded {} feedback for artifact {} in workspace {}", 
              feedback_type, artifact_id, workspace_id);
        
        Ok(())
    }

    async fn update_performance_metrics(
        &self,
        workspace_id: Uuid,
        feedback_type: &str,
        score: f64,
    ) -> Result<()> {
        let today = chrono::Utc::now().date_naive();
        
        // Try to update existing record
        let result = sqlx::query!(
            r#"
            UPDATE model_performance 
            SET 
                predictions_made = predictions_made + 1,
                predictions_accepted = predictions_accepted + CASE WHEN $1 = 'accepted' THEN 1 ELSE 0 END,
                predictions_rejected = predictions_rejected + CASE WHEN $1 = 'rejected' THEN 1 ELSE 0 END,
                avg_confidence = (avg_confidence * predictions_made + $2) / (predictions_made + 1)
            WHERE workspace_id = $3 AND date = $4
            "#,
            feedback_type,
            score,
            workspace_id,
            today
        )
        .execute(&self.db)
        .await?;

        // If no rows were updated, create new record
        if result.rows_affected() == 0 {
            sqlx::query!(
                r#"
                INSERT INTO model_performance 
                (workspace_id, date, predictions_made, predictions_accepted, predictions_rejected, avg_confidence)
                VALUES ($1, $2, 1, $3, $4, $5)
                "#,
                workspace_id,
                today,
                if feedback_type == "accepted" { 1 } else { 0 },
                if feedback_type == "rejected" { 1 } else { 0 },
                score as f64
            )
            .execute(&self.db)
            .await?;
        }
        
        Ok(())
    }

    pub async fn get_feedback_summary(&self, workspace_id: Uuid) -> Result<Vec<TrainingExample>> {
        let records = sqlx::query!(
            r#"
            SELECT id, input_text, suggested_outcome_id, actual_outcome_id, user_feedback
            FROM training_examples 
            WHERE workspace_id = $1 
            AND user_feedback IS NOT NULL
            ORDER BY feedback_timestamp DESC 
            LIMIT 100
            "#,
            workspace_id
        )
        .fetch_all(&self.db)
        .await?;
        
        let examples = records.into_iter().map(|r| TrainingExample {
            id: r.id,
            input_text: r.input_text,
            suggested_outcome_id: r.suggested_outcome_id,
            actual_outcome_id: r.actual_outcome_id,
            user_feedback: r.user_feedback,
        }).collect();
        
        Ok(examples)
    }
}