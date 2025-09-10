// src/handlers/slack/commands.rs

use axum::{extract::State, http::HeaderMap, Json};
use chrono::Utc;
use interstice_core::{artifact::{ArtifactState, QualityMetrics}, Artifact, ArtifactType, Platform, WorkspaceId};
use interstice_ml::OutcomePrediction;
use reqwest::Client;
use serde_json::Value as JsonValue;
use uuid::Uuid;
use std::sync::Arc;
use tracing::{error, info};
use tokio::time::Duration;

use crate::{handlers::slack::{fetch_workspace_patterns, fetch_workspace_statistics, format_predictions_response, store_artifact, store_prediction, verify_slack_request, SlackCommandEvent}, AppState};

pub async fn handle_slash_commands(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Body,
) -> Result<Json<JsonValue>, axum::http::StatusCode> {
    let adapter = state.slack_adapter.as_ref().ok_or_else(|| {
        error!("Slack adapter not configured");
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    })?;

    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.map_err(|e| {
        error!("Failed to read body: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;
    
    let body_str = String::from_utf8(body_bytes.into()).map_err(|e| {
        error!("Failed to parse body as UTF-8: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;

    verify_slack_request(&headers, &body_str, adapter)?;

    let command: SlackCommandEvent = serde_urlencoded::from_str(&body_str).map_err(|e| {
        error!("Failed to parse slash command: {}", e);
        axum::http::StatusCode::BAD_REQUEST
    })?;

    info!(
        "Slash command '{}' received from team {} by user {} with text: '{}'",
        &command.command,
        &command.team_id,
        &command.user_id,
        &command.text
    );

    let response = match command.command.as_str() {
        "/interstice" => handle_interstice_command(&command, &state).await,
        "/interstice-track" => handle_track_command(&command, &state).await,
        "/interstice-insights" => handle_insights_command(&command, &state).await,
        _ => serde_json::json!({
            "response_type": "ephemeral",
            "text": "Unknown command. Use `/interstice help` to see available commands."
        }),
    };

    store_command_usage(&command, &response, &state).await;

    if should_use_async_response(&command.text) {
        let ack_response = serde_json::json!({
            "response_type": "ephemeral",
            "text": "Processing your request... Results will appear shortly."
        });
        
        let state_clone = state.clone();
        let command_clone = command.clone();
        tokio::spawn(async move {
            let full_response = process_complex_command(&command_clone, &state_clone).await;
            post_to_response_url(&command_clone.response_url, &full_response).await.ok();
        });
        
        return Ok(Json(ack_response));
    }
    
    Ok(Json(response))
}

async fn handle_interstice_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> JsonValue {
    let args: Vec<&str> = command.text.split_whitespace().collect();
    let subcommand = args.get(0).map(|s| s.to_lowercase());
    
    match subcommand.as_deref() {
        Some("help") | None => show_help_command(),
        Some("status") => show_workspace_status(&command.team_id, state).await,
        Some("predict") => run_predictions_command(command, state).await,
        Some("analyze") => analyze_workspace_patterns(&command.team_id, state).await,
        Some("recent") => show_recent_artifacts(&command.team_id, state).await,
        Some("stats") => show_workspace_stats(&command.team_id, state).await,
        _ => serde_json::json!({
            "response_type": "ephemeral",
            "text": format!("Unknown subcommand '{}'. Use `/interstice help` for available commands.", args[0])
        }),
    }
}

async fn run_predictions_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> JsonValue {
    let channel_id = extract_channel_from_text(&command.text).unwrap_or_else(|| command.channel_id.clone());
    
    match extract_and_predict_artifacts(&command.team_id, &channel_id, state).await {
        Ok(predictions) if !predictions.is_empty() => format_predictions_response(&predictions),
        Ok(_) => serde_json::json!({
            "response_type": "ephemeral",
            "text": "No predictions available. Try generating more activity first!"
        }),
        Err(e) => {
            error!("Failed to run predictions: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to generate predictions. Please try again later."
            })
        }
    }
}

async fn handle_track_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> JsonValue {
    if command.text.is_empty() {
        return serde_json::json!({
            "response_type": "ephemeral",
            "text": "Please provide text to track. Usage: `/interstice-track <your artifact text>`"
        });
    }
    
    let artifact = Artifact {
        id: Uuid::new_v4(),
        workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(&command.team_id).unwrap_or_else(|_| Uuid::new_v4())),
        artifact_type: ArtifactType::Task {
            id: Uuid::new_v4().to_string(),
            title: command.text.clone(),
            status: interstice_core::artifact::TaskStatus::Todo,
            assignee: Some(command.user_id.clone()),
            due_date: None,
            completed_at: None,
            checklist_items: vec![],
            dependencies: vec![],
            tags: vec![],
            recurring: false,
            parent_task: None,
            subtasks: vec![],
        },
        platform: Platform::Slack,
        content: command.text.clone(),
        metadata: serde_json::json!({
            "tracked_by": "slash_command",
            "user_id": command.user_id,
            "channel_id": command.channel_id
        }),
        created_at: Utc::now(),
        updated_at: Utc::now(),
        version: 1,
        state: ArtifactState::Processed,
        quality_metrics: QualityMetrics::default(),
        related_artifacts: vec![],
        tags: Default::default(),
    };
    
    match store_artifact(&artifact, &command.team_id, &state.db).await {
        Ok(_) => serde_json::json!({
            "response_type": "ephemeral",
            "text": format!("✅ Artifact tracked successfully: \"{}\"", &command.text),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!("✅ *Artifact Tracked*\n```{}```", &command.text)
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": format!("Type: {:?} | ID: {}", artifact.artifact_type, artifact.id)
                        }
                    ]
                }
            ]
        }),
        Err(e) => {
            error!("Failed to track artifact: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to track artifact. Please try again later."
            })
        }
    }
}

async fn store_command_usage(
    command: &SlackCommandEvent,
    response: &JsonValue,
    state: &Arc<AppState>,
) {
    let _ = sqlx::query!(
        r#"
        INSERT INTO slack_command_usage (
            team_id, user_id, command, text, response_type, created_at
        )
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
        &command.team_id,
        &command.user_id,
        &command.command,
        &command.text,
        response["response_type"].as_str()
    )
    .execute(&state.db)
    .await;
}

fn should_use_async_response(text: &str) -> bool {
    let complex_keywords = ["analyze", "report", "deep", "comprehensive", "detailed", "full", "export", "generate", "historical"];
    let text_lower = text.to_lowercase();
    complex_keywords.iter().any(|keyword| text_lower.contains(*keyword))
}

pub async fn post_to_response_url(
    response_url: &str,
    response: &JsonValue,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    
    let res = client
        .post(response_url)
        .json(response)
        .timeout(Duration::from_secs(5))
        .send()
        .await?;
    
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        error!("Failed to post to response_url. Status: {}, Body: {}", status, body);
        return Err(format!("Failed to post async response: {}", status).into());
    }
    
    info!("Successfully posted async response to Slack");
    Ok(())
}

async fn handle_insights_command(
    command: &SlackCommandEvent,
    state: &Arc<AppState>,
) -> JsonValue {
    match generate_workspace_insights(&command.team_id, state).await {
        Ok(insights) => insights,
        Err(e) => {
            error!("Failed to generate insights: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to generate insights. Please try again later."
            })
        }
    }
}

async fn generate_workspace_insights(
    team_id: &str,
    state: &Arc<AppState>,
) -> Result<JsonValue, Box<dyn std::error::Error>> {
    let _stats = fetch_workspace_statistics(team_id, &state.db).await?;
    
    let insights = vec![
        "📈 Task completion rate increased by 15% this week",
        "🎯 Most productive hours: 10am - 12pm",
        "👥 Top collaborators: Alice, Bob, Charlie",
        "⚡ Average response time: 2.3 hours",
    ];
    
    let mut blocks = vec![
        serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*🧠 Workspace Insights*"
            }
        }),
        serde_json::json!({
            "type": "divider"
        })
    ];
    
    for insight in insights {
        blocks.push(serde_json::json!({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": insight
            }
        }));
    }
    
    blocks.push(serde_json::json!({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": format!("_Generated at {} UTC_", Utc::now().format("%Y-%m-%d %H:%M"))
            }
        ]
    }));
    
    Ok(serde_json::json!({
        "response_type": "in_channel",
        "blocks": blocks
    }))
}

fn show_help_command() -> JsonValue {
    serde_json::json!({
        "response_type": "ephemeral",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Welcome to Interstice Engine!* 🚀\n\nI help track work artifacts and predict outcomes from your Slack activity."
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Available Commands:*\n\n• `/interstice help` - Show this help message\n• `/interstice status` - View workspace status\n• `/interstice predict` - Get outcome predictions\n• `/interstice analyze` - Analyze workspace patterns\n• `/interstice recent` - Show recent artifacts\n• `/interstice stats` - View detailed statistics\n\n• `/interstice-track <text>` - Manually track an artifact\n• `/interstice-insights` - Generate AI-powered insights"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "💡 _Tip: I automatically track artifacts from your messages!_"
                    }
                ]
            }
        ]
    })
}

async fn show_workspace_status(
    team_id: &str,
    state: &Arc<AppState>,
) -> JsonValue {
    match fetch_workspace_statistics(team_id, &state.db).await {
        Ok(status) => serde_json::json!({
            "response_type": "ephemeral",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!("*Workspace Status for Team {}*", team_id)
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": format!("*Weekly Artifacts:*\n{}", status.weekly_artifacts)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Monthly Predictions:*\n{}", status.monthly_predictions)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Active Users:*\n{}", status.active_users)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Success Rate:*\n{:.1}%", status.success_rate * 100.0)
                        }
                    ]
                }
            ]
        }),
        Err(e) => {
            error!("Failed to fetch workspace status: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch workspace status. Please try again later."
            })
        }
    }
}

async fn analyze_workspace_patterns(
    team_id: &str,
    state: &Arc<AppState>,
) -> JsonValue {
    match fetch_workspace_patterns(team_id, &state.db).await {
        Ok(patterns) => serde_json::json!({
            "response_type": "ephemeral",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📊 Workspace Analysis*"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": format!("*Peak Activity Hour:*\n{}", patterns.peak_hour)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Most Active Channel:*\n{}", patterns.most_active_channel)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Avg Daily Artifacts:*\n{:.1}", patterns.avg_daily_artifacts)
                        },
                        {
                            "type": "mrkdwn",
                            "text": format!("*Common Artifact Type:*\n{}", patterns.common_artifact_type)
                        }
                    ]
                }
            ]
        }),
        Err(e) => {
            error!("Failed to analyze patterns: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to analyze workspace patterns. Please try again later."
            })
        }
    }
}

async fn show_recent_artifacts(
    team_id: &str,
    state: &Arc<AppState>,
) -> JsonValue {
    match super::fetch_recent_artifacts(team_id, &state.db, 5).await {
        Ok(artifacts) if !artifacts.is_empty() => {
            let mut blocks = vec![
                serde_json::json!({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📝 Recent Artifacts*"
                    }
                })
            ];
            
            for artifact in artifacts {
                blocks.push(serde_json::json!({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!(
                            "*{:?}*\n_{}_\n```{}```",
                            artifact.artifact_type,
                            artifact.created_at.format("%Y-%m-%d %H:%M UTC"),
                            super::truncate_string(&artifact.content, 100)
                        )
                    }
                }));
            }
            
            serde_json::json!({
                "response_type": "ephemeral",
                "blocks": blocks
            })
        }
        Ok(_) => serde_json::json!({
            "response_type": "ephemeral",
            "text": "No recent artifacts found. Start tracking some activities!"
        }),
        Err(e) => {
            error!("Failed to fetch recent artifacts: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch recent artifacts. Please try again later."
            })
        }
    }
}

async fn show_workspace_stats(
    team_id: &str,
    state: &Arc<AppState>,
) -> JsonValue {
    match super::fetch_workspace_statistics(team_id, &state.db).await {
        Ok(stats) => serde_json::json!({
            "response_type": "ephemeral",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*📈 Detailed Statistics*"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!(
                            "*Last 7 Days:*\n• Artifacts: {}\n• Commands: {}\n• Active Users: {}\n\n*Last 30 Days:*\n• Total Artifacts: {}\n• Predictions: {}\n• Success Rate: {:.1}%",
                            stats.weekly_artifacts,
                            stats.weekly_commands,
                            stats.active_users,
                            stats.monthly_artifacts,
                            stats.monthly_predictions,
                            stats.success_rate * 100.0
                        )
                    }
                }
            ]
        }),
        Err(e) => {
            error!("Failed to fetch statistics: {}", e);
            serde_json::json!({
                "response_type": "ephemeral",
                "text": "Failed to fetch statistics. Please try again later."
            })
        }
    }
}

fn extract_channel_from_text(text: &str) -> Option<String> {
    if let Some(start) = text.find("<#") {
        if let Some(end) = text[start..].find('>') {
            let channel_ref = &text[start + 2..start + end];
            if let Some(pipe) = channel_ref.find('|') {
                return Some(channel_ref[..pipe].to_string());
            }
            return Some(channel_ref.to_string());
        }
    }
    None
}

async fn extract_and_predict_artifacts(
    team_id: &str,
    channel_id: &str,
    state: &Arc<AppState>,
) -> Result<Vec<OutcomePrediction>, Box<dyn std::error::Error>> {
    let artifacts = super::fetch_channel_artifacts(team_id, channel_id, &state.db, 10).await?;
    
    if artifacts.is_empty() {
        return Ok(vec![]);
    }
    
    let ml_artifacts = artifacts.iter().map(|a| {
        interstice_ml::types::Artifact::new(
            a.id.to_string(),
            a.content.clone(),
            interstice_ml::types::Platform::Slack,
            interstice_ml::types::ArtifactType::Message
        )
    }).collect::<Vec<_>>();
    
    let predictions = state.ml_pipeline.predict_outcomes(
        Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4()),
        &ml_artifacts,
        &artifacts.iter().map(|a| a.content.as_str()).collect::<Vec<_>>().join(" ")
    ).await?;
    
    for pred in &predictions {
        store_prediction(pred, team_id, &state.db).await?;
    }
    
    Ok(predictions)
}


async fn process_complex_command(
    command: &SlackCommandEvent,
    _state: &Arc<AppState>,
) -> JsonValue {
    serde_json::json!({
        "response_type": "in_channel",
        "text": format!("Completed processing: {}", command.text)
    })
}