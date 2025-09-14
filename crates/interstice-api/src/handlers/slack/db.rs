// src/handlers/slack/db.rs
use chrono::Utc;
use interstice_core::{artifact::{ArtifactState, QualityMetrics}, Artifact, ArtifactType, Platform, WorkspaceId};
use interstice_ml::OutcomePrediction;
use sqlx::PgPool;
use uuid::Uuid;

use crate::handlers::slack::{WorkspacePatterns, WorkspaceStatistics};

pub async fn fetch_workspace_patterns(
    team_id: &str,
    db: &PgPool,
) -> Result<WorkspacePatterns, Box<dyn std::error::Error>> {
    let peak_hour = sqlx::query_scalar!(
        r#"
        SELECT EXTRACT(HOUR FROM created_at)::INT as hour
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '30 days'
        GROUP BY hour
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .map(|h| format!("{}:00", h.unwrap_or(0)))
    .unwrap_or_else(|| "N/A".to_string());
    
    let most_active_channel = sqlx::query_scalar!(
        r#"
        SELECT channel_id
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND channel_id IS NOT NULL
        GROUP BY channel_id
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .flatten()
    .unwrap_or_else(|| "N/A".to_string());
    
    let avg_daily = sqlx::query_scalar!(
        r#"
        SELECT AVG(daily_count)::FLOAT
        FROM (
            SELECT DATE(created_at) as day, COUNT(*) as daily_count
            FROM artifacts
            WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
              AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY day
        ) as daily_counts
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .flatten()
    .unwrap_or(0.0) as f64;
    
    let common_type = sqlx::query_scalar!(
        r#"
        SELECT artifact_type
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        GROUP BY artifact_type
        ORDER BY COUNT(*) DESC
        LIMIT 1
        "#,
        team_id
    )
    .fetch_optional(db)
    .await?
    .unwrap_or_else(|| "N/A".to_string());
    
    Ok(WorkspacePatterns {
        peak_hour,
        most_active_channel,
        avg_daily_artifacts: avg_daily,
        common_artifact_type: common_type,
    })
}

pub async fn fetch_recent_artifacts(
    team_id: &str,
    db: &PgPool,
    limit: i32,
) -> Result<Vec<Artifact>, Box<dyn std::error::Error>> {
    let records = sqlx::query!(
        r#"
        SELECT id, artifact_type, content, channel_id, metadata, created_at
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        ORDER BY created_at DESC
        LIMIT $2
        "#,
        team_id,
        limit as i64
    )
    .fetch_all(db)
    .await?;
    
    let artifacts = records.into_iter().map(|r| Artifact {
        id: r.id,
        workspace_id: WorkspaceId::from_uuid(Uuid::new_v4()),
        artifact_type: ArtifactType::Message {
            id: r.id.to_string(),
            channel: r.channel_id.unwrap_or_default(),
            thread_id: None,
            author: "unknown".to_string(),
            content: r.content.clone(),
            mentions: vec![],
            attachments: vec![],
            reactions: Vec::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        },
        platform: Platform::Slack,
        content: r.content,
        metadata: r.metadata.unwrap_or_else(|| serde_json::json!({})),
        created_at: r.created_at.unwrap_or_else(|| Utc::now()),
        updated_at: r.created_at.unwrap_or_else(|| Utc::now()),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: Default::default(),
    }).collect();
    
    Ok(artifacts)
}

pub async fn fetch_channel_artifacts(
    team_id: &str,
    channel_id: &str,
    db: &PgPool,
    limit: i32,
) -> Result<Vec<Artifact>, Box<dyn std::error::Error>> {
    let records = sqlx::query!(
        r#"
        SELECT id, artifact_type, content, metadata, created_at
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1) AND channel_id = $2
        ORDER BY created_at DESC
        LIMIT $3
        "#,
        team_id,
        channel_id,
        limit as i64
    )
    .fetch_all(db)
    .await?;
    
    let artifacts = records.into_iter().map(|r| Artifact {
        id: r.id,
        workspace_id: WorkspaceId::from_uuid(Uuid::new_v4()),
        artifact_type: ArtifactType::Message {
            id: r.id.to_string(),
            channel: channel_id.to_string(),
            thread_id: None,
            author: "unknown".to_string(),
            content: r.content.clone(),
            mentions: vec![],
            attachments: vec![],
            reactions: Vec::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        },
        platform: Platform::Slack,
        content: r.content,
        metadata: r.metadata.unwrap_or_else(|| serde_json::json!({})),
        created_at: r.created_at.unwrap_or_else(|| Utc::now()),
        updated_at: r.created_at.unwrap_or_else(|| Utc::now()),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: Default::default(),
    }).collect();
    
    Ok(artifacts)
}

pub async fn fetch_workspace_statistics(
    team_id: &str,
    db: &PgPool,
) -> Result<WorkspaceStatistics, Box<dyn std::error::Error>> {
    let weekly_artifacts = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    let weekly_commands = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM slack_command_usage
        WHERE team_id = $1
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    let active_users = sqlx::query_scalar!(
        r#"
        SELECT COUNT(DISTINCT user_id)
        FROM slack_command_usage
        WHERE team_id = $1
          AND created_at > NOW() - INTERVAL '7 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    let monthly_artifacts = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM artifacts
        WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
          AND created_at > NOW() - INTERVAL '30 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    let monthly_predictions = sqlx::query_scalar!(
        r#"
        SELECT COUNT(*)
        FROM inference_history
        WHERE artifact_id IN (
            SELECT id FROM artifacts WHERE workspace_id = (SELECT id FROM workspaces WHERE slack_team_id = $1)
        )
        AND created_at > NOW() - INTERVAL '30 days'
        "#,
        team_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(0);
    
    Ok(WorkspaceStatistics {
        weekly_artifacts,
        weekly_commands,
        active_users,
        monthly_artifacts,
        monthly_predictions,
        success_rate: 0.85,
    })
}

pub async fn store_prediction(
    prediction: &OutcomePrediction,
    team_id: &str,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    sqlx::query!(
        r#"
        INSERT INTO inference_history (
            id, workspace_id, artifact_id,
            confidence, prediction_data, created_at
        )
        VALUES ($1, 
                (SELECT id FROM workspaces WHERE slack_team_id = $2 LIMIT 1),
                NULL, $3, $4, NOW())
        "#,
        Uuid::new_v4(),
        team_id,
        prediction.confidence as f32,
        serde_json::to_value(prediction)?
    )
    .execute(db)
    .await?;
    
    Ok(())
}

pub fn format_predictions_response(predictions: &[OutcomePrediction]) -> serde_json::Value {
    let mut blocks = vec![
        serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*🔮 Outcome Predictions*"
            }
        })
    ];
    
    for (i, pred) in predictions.iter().enumerate().take(5) {
        let confidence_bar = "█".repeat((pred.confidence * 10.0) as usize);
        let empty_bar = "░".repeat(10 - (pred.confidence * 10.0) as usize);
        
        blocks.push(serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": format!(
                    "*{}. {}*\n{}{} _{:.1}% confidence_",
                    i + 1,
                    pred.outcome_name,
                    confidence_bar,
                    empty_bar,
                    pred.confidence * 100.0
                )
            }
        }));
    }
    
    serde_json::json!({
        "response_type": "in_channel",
        "blocks": blocks
    })
}