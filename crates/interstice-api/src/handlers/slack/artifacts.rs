// src/handlers/slack/artifacts.rs

use chrono::Utc;
use futures::{stream, StreamExt};
use interstice_core::{artifact::{AccessLevel, ArtifactState, DesignType, DocumentType, IssueState, Priority, QualityMetrics}, Artifact, ArtifactType, Platform, WorkspaceId};
use interstice_ml::{types::{ArtifactType as MlArtifactType, Platform as MlPlatform}, OutcomePrediction, MLPipeline};
use serde_json::Value as JsonValue;
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::sleep;
use tracing::{error, info, warn};
use uuid::Uuid;
use anyhow::{anyhow, Result as AnyhowResult};

use super::{SlackPushEvent};

#[allow(dead_code)]
pub async fn extract_artifacts_from_event(
    event: &SlackPushEvent,
    team_id: &str,
) -> AnyhowResult<Vec<Artifact>> {
    let mut artifacts = Vec::with_capacity(4);
    
    if let Some(event_data) = &event.event {
        let event_type = event_data.get("event_type").and_then(|v| v.as_str()).unwrap_or("unknown");
        
        match event_type {
            "message" => {
                if let Some(text) = event_data.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        let channel_id = event_data.get("channel").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let user_id = event_data.get("user").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let ts = event_data.get("ts").and_then(|v| v.as_str()).unwrap_or("");
                        
                        let artifact_type = determine_artifact_type(text);
                        
                        let artifact = Artifact {
                            id: Uuid::new_v4(),
                            workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())),
                            artifact_type,
                            platform: Platform::Slack,
                            content: text.to_string(),
                            metadata: serde_json::json!({
                                "channel_id": channel_id,
                                "user_id": user_id,
                                "message_ts": ts,
                                "event_type": "message",
                            }),
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            version: 1,
                            state: ArtifactState::Pending,
                            quality_metrics: QualityMetrics::default(),
                            related_artifacts: vec![],
                            tags: Default::default(),
                        };
                        
                        artifacts.push(artifact);
                    }
                }
                
                if let Some(attachments) = event_data.get("attachments").and_then(|a| a.as_array()) {
                    for attachment in attachments {
                        if let Some(artifact) = process_attachment(attachment, team_id) {
                            artifacts.push(artifact);
                        }
                    }
                }
            }
            "file_shared" => {
                if let Some(file) = event_data.get("file") {
                    if let Some(artifact) = process_file_share(file, team_id) {
                        artifacts.push(artifact);
                    }
                }
            }
            "reaction_added" => {
                if let Some(reaction) = event_data.get("reaction").and_then(|v| v.as_str()) {
                    if let Some(item) = event_data.get("item") {
                        let channel = item.get("channel").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let ts = item.get("ts").and_then(|v| v.as_str()).unwrap_or("");
                        let user = event_data.get("user").and_then(|v| v.as_str()).unwrap_or("unknown");

                        let artifact = Artifact {
                            id: Uuid::new_v4(),
                            workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())),
                            artifact_type: ArtifactType::Message {
                                id: Uuid::new_v4().to_string(),
                                channel: "unknown".to_string(),
                                thread_id: None,
                                author: "unknown".to_string(),
                                content: format!("Reaction added: {}", reaction),
                                mentions: vec![],
                                attachments: vec![],
                                reactions: HashMap::new(),
                                sentiment: interstice_core::artifact::Sentiment::Neutral,
                                intent: interstice_core::artifact::MessageIntent::Discussion,
                                is_edited: false,
                                reply_count: 0,
                            },
                            platform: Platform::Slack,
                            content: format!("Reaction added: {}", reaction),
                            metadata: serde_json::json!({
                                "reaction": reaction,
                                "channel": channel,
                                "message_ts": ts,
                                "user": user,
                                "event_type": "reaction_added",
                            }),
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                            version: 1,
                            state: ArtifactState::Pending,
                            quality_metrics: QualityMetrics::default(),
                            related_artifacts: vec![],
                            tags: Default::default(),
                        };

                        artifacts.push(artifact);
                    }
                }
            }
            _ => info!("Unhandled event type: {}", event_type),
        }
    }
    
    Ok(artifacts)
}

fn determine_artifact_type(text: &str) -> ArtifactType {
    if text.contains("PR #") || text.contains("pull request") {
        ArtifactType::PullRequest {
            title: text.to_string(),
            state: interstice_core::artifact::PullRequestState::Open,
            author: "unknown".to_string(),
            reviewers: vec![],
            labels: vec![],
            merge_conflict: false,
            ci_status: None,
            base_branch: "main".to_string(),
            head_branch: "feature".to_string(),
            files_changed: 0,
            additions: 0,
            deletions: 0,
            merged: false,
            draft: false,
            number: 0,
        }
    } else if text.contains("#") && text.contains("issue") {
        ArtifactType::Issue {
            id: Uuid::new_v4().to_string(),
            title: text.to_string(),
            state: IssueState::Open,
            priority: Priority::Medium,
            assignees: vec![],
            labels: vec![],
            story_points: None,
            sprint: None,
            epic: None,
            blocked: false,
            blockers: vec![],
            time_estimate: None,
            time_spent: None,
        }
    } else if text.contains("task") || text.contains("TODO") {
        ArtifactType::Task {
            id: Uuid::new_v4().to_string(),
            title: text.to_string(),
            status: interstice_core::artifact::TaskStatus::Todo,
            assignee: None,
            due_date: None,
            completed_at: None,
            checklist_items: vec![],
            dependencies: vec![],
            tags: vec![],
            recurring: false,
            parent_task: None,
            subtasks: vec![],
        }
    } else {
        ArtifactType::Message {
            id: Uuid::new_v4().to_string(),
            channel: "unknown".to_string(),
            thread_id: None,
            author: "unknown".to_string(),
            content: text.to_string(),
            mentions: vec![],
            attachments: vec![],
            reactions: HashMap::new(),
            sentiment: interstice_core::artifact::Sentiment::Neutral,
            intent: interstice_core::artifact::MessageIntent::Discussion,
            is_edited: false,
            reply_count: 0,
        }
    }
}

fn process_attachment(attachment: &JsonValue, team_id: &str) -> Option<Artifact> {
    attachment.get("title").and_then(|v| v.as_str()).filter(|title| !title.is_empty()).map(|title| {
        let artifact_type = determine_artifact_type(title);
        Artifact {
            id: Uuid::new_v4(),
            workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())),
            artifact_type,
            platform: Platform::Slack,
            content: title.to_string(),
            metadata: serde_json::json!({
                "attachment": attachment.clone(),
                "event_type": "attachment",
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
            state: ArtifactState::Pending,
            quality_metrics: QualityMetrics::default(),
            related_artifacts: vec![],
            tags: Default::default(),
        }
    })
}

fn process_file_share(file: &JsonValue, team_id: &str) -> Option<Artifact> {
    file.get("name").and_then(|v| v.as_str()).filter(|name| !name.is_empty()).map(|name| {
        let artifact_type = if name.ends_with(".md") || name.ends_with(".txt") {
            ArtifactType::Document {
                id: Uuid::new_v4().to_string(),
                title: name.to_string(),
                doc_type: DocumentType::Wiki,
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: Utc::now(),
                version: 1,
                is_template: false,
                access_level: AccessLevel::Internal,
            }
        } else if name.ends_with(".png") || name.ends_with(".jpg") {
            ArtifactType::Design {
                id: Uuid::new_v4().to_string(),
                title: name.to_string(),
                design_type: DesignType::Figma,
                version: None,
                collaborators: vec![],
                components: 0,
                screens: 0,
                last_modified: Utc::now(),
                design_system: None,
                accessibility_score: None,
            }
        } else {
            ArtifactType::Document {
                id: Uuid::new_v4().to_string(),
                title: name.to_string(),
                doc_type: DocumentType::Wiki,
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: Utc::now(),
                version: 1,
                is_template: false,
                access_level: AccessLevel::Internal,
            }
        };
        Artifact {
            id: Uuid::new_v4(),
            workspace_id: WorkspaceId::from_uuid(Uuid::parse_str(team_id).unwrap_or_else(|_| Uuid::new_v4())),
            artifact_type,
            platform: Platform::Slack,
            content: name.to_string(),
            metadata: serde_json::json!({
                "file": file.clone(),
                "event_type": "file_shared",
            }),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            version: 1,
            state: ArtifactState::Pending,
            quality_metrics: QualityMetrics::default(),
            related_artifacts: vec![],
            tags: Default::default(),
        }
    })
}


pub async fn run_ml_on_artifacts(
    artifacts: &[Artifact],
    workspace_id: Uuid,
    ml_pipeline: &Arc<MLPipeline>,
) -> AnyhowResult<Vec<OutcomePrediction>> {
    let ml_artifacts: Vec<interstice_ml::types::Artifact> = artifacts.iter().map(|a| {
        interstice_ml::types::Artifact::new(
            a.id.to_string(),
            a.content.clone(),
            MlPlatform::Slack,
            MlArtifactType::Message
        )
    }).collect();
    
    let context = artifacts.iter().map(|a| a.content.as_str()).collect::<Vec<_>>().join(" ");
    
    ml_pipeline.predict_outcomes(workspace_id, &ml_artifacts, &context).await.map_err(Into::into)
}


pub async fn update_workspace_analytics(
    team_id: String,
    artifacts: Vec<Artifact>,
    predictions: Vec<OutcomePrediction>,
    db: PgPool,
) -> AnyhowResult<()> {
    sqlx::query!(
        r#"
        INSERT INTO workspace_analytics (
            workspace_id, date, artifact_count, prediction_count,
            message_count, task_count, document_count
        )
        VALUES (
            (SELECT id FROM workspaces WHERE slack_team_id = $1),
            CURRENT_DATE,
            $2,
            $3,
            $4,
            $5,
            $6
        )
        ON CONFLICT (workspace_id, date) DO UPDATE SET
            artifact_count = workspace_analytics.artifact_count + EXCLUDED.artifact_count,
            prediction_count = workspace_analytics.prediction_count + EXCLUDED.prediction_count,
            message_count = workspace_analytics.message_count + EXCLUDED.message_count,
            task_count = workspace_analytics.task_count + EXCLUDED.task_count,
            document_count = workspace_analytics.document_count + EXCLUDED.document_count,
            updated_at = NOW()
        "#,
        &team_id,
        artifacts.len() as i32,
        predictions.len() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Message { .. })).count() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Task { .. })).count() as i32,
        artifacts.iter().filter(|a| matches!(a.artifact_type, ArtifactType::Document { .. })).count() as i32
    )
    .execute(&db)
    .await?;
    
    Ok(())
}

pub async fn store_artifact(
    artifact: &Artifact,
    team_id: &str,
    db: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace_id = sqlx::query_scalar!(
        "SELECT id FROM workspaces WHERE slack_team_id = $1",
        team_id
    )
    .fetch_optional(db)
    .await?
    .ok_or("Workspace not found")?;

    let channel_id = artifact.metadata.get("channel_id").and_then(|v| v.as_str()).map(str::to_string);

    sqlx::query!(
        r#"
        INSERT INTO artifacts (
            id, workspace_id, artifact_type, content, 
            channel_id, metadata, platform, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
        artifact.id,
        workspace_id,
        format!("{:?}", artifact.artifact_type),
        &artifact.content,
        channel_id,
        &artifact.metadata,
        artifact.platform.to_string()
    )
    .execute(db)
    .await?;
    
    Ok(())
}

pub async fn store_artifacts_batch(
    artifacts: Vec<Artifact>,
    workspace_id: Uuid,
    db: PgPool,
) -> AnyhowResult<()> {
    if artifacts.is_empty() {
        return Ok(());
    }

    let total_artifacts = artifacts.len();
    info!(
        workspace_id = %workspace_id,
        artifact_count = total_artifacts,
        "Starting batch artifact storage"
    );

    // Configuration for batch processing
    const CHUNK_SIZE: usize = 100; // Optimal chunk size for PostgreSQL
    const MAX_CONCURRENCY: usize = 4; // Parallel database connections
    const MAX_RETRIES: u32 = 3;
    const BASE_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(100);

    // Validate workspace exists first (fail fast)
    validate_workspace_exists(workspace_id, &db).await?;

    // Process artifacts in chunks for better performance and memory usage
    let chunks: Vec<Vec<Artifact>> = artifacts
        .chunks(CHUNK_SIZE)
        .map(|chunk| chunk.to_vec())
        .collect();

    let total_chunks = chunks.len();
    let mut processed_count = 0;
    let mut failed_artifacts = Vec::new();

    // Process chunks with controlled concurrency
    let results = stream::iter(chunks.into_iter().enumerate())
        .map(|(chunk_index, chunk)| {
            let db = db.clone();
            async move {
                let chunk_start = chunk_index * CHUNK_SIZE;
                let chunk_end = std::cmp::min(chunk_start + chunk.len(), total_artifacts);
                
                info!(
                    workspace_id = %workspace_id,
                    chunk = format!("{}/{}", chunk_index + 1, total_chunks),
                    artifacts = format!("{}-{}", chunk_start, chunk_end),
                    "Processing artifact chunk"
                );

                store_chunk_with_retry(
                    chunk,
                    workspace_id,
                    &db,
                    MAX_RETRIES,
                    BASE_RETRY_DELAY,
                ).await
            }
        })
        .buffer_unordered(MAX_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    // Aggregate results and handle partial failures
    for (chunk_index, result) in results.into_iter().enumerate() {
        match result {
            Ok(count) => {
                processed_count += count;
            }
            Err(ChunkError { failed_indices, error }) => {
                let chunk_start = chunk_index * CHUNK_SIZE;
                for idx in failed_indices {
                    let artifact_index = chunk_start + idx;
                    if artifact_index < artifacts.len() {
                        failed_artifacts.push((artifact_index, error.to_string()));
                    }
                }
            }
        }
    }

    // Handle partial success scenarios
    if !failed_artifacts.is_empty() {
        let failure_rate = (failed_artifacts.len() as f64 / total_artifacts as f64) * 100.0;
        
        warn!(
            workspace_id = %workspace_id,
            total_artifacts = total_artifacts,
            successful = processed_count,
            failed = failed_artifacts.len(),
            failure_rate = format!("{:.2}%", failure_rate),
            "Partial batch storage failure"
        );

        // If more than 50% failed, consider it a critical failure
        if failure_rate > 50.0 {
            return Err(anyhow!("Critical batch storage failure: {}/{} artifacts failed",
                failed_artifacts.len(),
                total_artifacts
            ));
        }

        // Otherwise, try individual fallback for failed items
        let recovered = attempt_individual_recovery(
            &failed_artifacts,
            &artifacts,
            workspace_id,
            &db,
        ).await;

        if recovered > 0 {
            info!(
                workspace_id = %workspace_id,
                recovered = recovered,
                "Recovered failed artifacts through individual storage"
            );
            processed_count += recovered;
        }
    }

    // Final success metrics
    info!(
        workspace_id = %workspace_id,
        total_artifacts = total_artifacts,
        successfully_stored = processed_count,
        duration_ms = 0, // Would need timing logic here
        "Batch artifact storage completed"
    );

    // Update batch processing metrics
    update_batch_metrics(workspace_id, total_artifacts, processed_count, &db).await?;

    Ok(())
}

/// Store a chunk of artifacts with retry logic
async fn store_chunk_with_retry(
    chunk: Vec<Artifact>,
    workspace_id: Uuid,
    db: &PgPool,
    max_retries: u32,
    base_delay: std::time::Duration,
) -> Result<usize, ChunkError> {
    let mut retry_count = 0;
    let mut delay = base_delay;

    loop {
        match store_chunk_transactional(chunk.clone(), workspace_id, db).await {
            Ok(count) => return Ok(count),
            Err(e) if retry_count < max_retries => {
                retry_count += 1;
                warn!(
                    workspace_id = %workspace_id,
                    retry = format!("{}/{}", retry_count, max_retries),
                    delay_ms = delay.as_millis(),
                    error = %e,
                    "Retrying chunk storage after error"
                );
                
                sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
            Err(e) => {
                error!(
                    workspace_id = %workspace_id,
                    error = %e,
                    "Chunk storage failed after all retries"
                );
                
                // Attempt to identify which specific artifacts failed
                let failed_indices = identify_failed_artifacts(&chunk, workspace_id, db).await;
                
                return Err(ChunkError {
                    failed_indices,
                    error: anyhow!("Chunk storage failed: {}", e),
                });
            }
        }
    }
}

/// Store a chunk of artifacts within a database transaction
async fn store_chunk_transactional(
    chunk: Vec<Artifact>,
    workspace_id: Uuid,
    db: &PgPool,
) -> AnyhowResult<usize> {
    let mut tx = db.begin().await?;
    let mut stored_count = 0;

    for artifact in chunk {
        let channel_id = artifact.metadata
            .get("channel_id")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        
        let message_id = artifact.metadata
            .get("message_ts")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // Serialize artifact type with proper error handling
        let artifact_type_str = serde_json::to_string(&artifact.artifact_type)
            .unwrap_or_else(|_| format!("{:?}", artifact.artifact_type));

        // Use UPSERT pattern for idempotency
        let result = sqlx::query!(
            r#"
            INSERT INTO artifacts (
                id, workspace_id, artifact_type, content,
                channel_id, message_id, metadata, platform,
                created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata
            RETURNING id
            "#,
            artifact.id,
            workspace_id,
            artifact_type_str,
            &artifact.content,
            channel_id,
            message_id,
            &artifact.metadata,
            artifact.platform.to_string(),
            artifact.created_at,
        )
        .fetch_optional(&mut *tx)
        .await?;

        if result.is_some() {
            stored_count += 1;
        }

        // Store related artifacts if any
        if !artifact.related_artifacts.is_empty() {
            store_artifact_relations(
                artifact.id,
                &artifact.related_artifacts,
                &mut tx,
            ).await?;
        }

        // Store tags if any
        if !artifact.tags.is_empty() {
            store_artifact_tags(
                artifact.id,
                &artifact.tags,
                &mut tx,
            ).await?;
        }
    }

    tx.commit().await?;
    Ok(stored_count)
}

/// Store artifact relationships
async fn store_artifact_relations(
    artifact_id: Uuid,
    related_ids: &[Uuid],
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> AnyhowResult<()> {
    for related_id in related_ids {
        sqlx::query!(
            r#"
            INSERT INTO artifact_relations (
                artifact_id, related_artifact_id, created_at
            )
            VALUES ($1, $2, NOW())
            ON CONFLICT (artifact_id, related_artifact_id) DO NOTHING
            "#,
            artifact_id,
            related_id
        )
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Store artifact tags
async fn store_artifact_tags(
    artifact_id: Uuid,
    tags: &std::collections::HashSet<String>,
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
) -> AnyhowResult<()> {
    for tag in tags {
        sqlx::query!(
            r#"
            INSERT INTO artifact_tags (
                artifact_id, tag, created_at
            )
            VALUES ($1, $2, NOW())
            ON CONFLICT (artifact_id, tag) DO NOTHING
            "#,
            artifact_id,
            tag
        )
        .execute(&mut **tx)
        .await?;
    }
    Ok(())
}

/// Attempt to recover failed artifacts through individual storage
async fn attempt_individual_recovery(
    failed_artifacts: &[(usize, String)],
    all_artifacts: &[Artifact],
    workspace_id: Uuid,
    db: &PgPool,
) -> usize {
    let mut recovered = 0;

    for (index, _error) in failed_artifacts {
        if let Some(artifact) = all_artifacts.get(*index) {
            // Try individual storage with dedicated connection
            if let Ok(_) = store_single_artifact_safe(artifact, workspace_id, db).await {
                recovered += 1;
            }
        }
    }

    recovered
}

/// Store a single artifact with maximum safety
async fn store_single_artifact_safe(
    artifact: &Artifact,
    workspace_id: Uuid,
    db: &PgPool,
) -> AnyhowResult<()> {
    let channel_id = artifact.metadata
        .get("channel_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    
    let message_id = artifact.metadata
        .get("message_ts")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    sqlx::query!(
        r#"
        INSERT INTO artifacts (
            id, workspace_id, artifact_type, content,
            channel_id, message_id, metadata, platform, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        ON CONFLICT (id) DO NOTHING
        "#,
        artifact.id,
        workspace_id,
        format!("{:?}", artifact.artifact_type),
        &artifact.content,
        channel_id,
        message_id,
        &artifact.metadata,
        artifact.platform.to_string()
    )
    .execute(db)
    .await?;
    
    Ok(())
}

/// Identify which specific artifacts in a chunk failed
async fn identify_failed_artifacts(
    chunk: &[Artifact],
    workspace_id: Uuid,
    db: &PgPool,
) -> Vec<usize> {
    let mut failed_indices = Vec::new();

    for (idx, artifact) in chunk.iter().enumerate() {
        // Check if artifact can be stored individually
        if store_single_artifact_safe(artifact, workspace_id, db).await.is_err() {
            failed_indices.push(idx);
        }
    }

    failed_indices
}

/// Validate that the workspace exists
async fn validate_workspace_exists(
    workspace_id: Uuid,
    db: &PgPool,
) -> AnyhowResult<()> {
    let exists = sqlx::query_scalar!(
        "SELECT EXISTS(SELECT 1 FROM workspaces WHERE id = $1)",
        workspace_id
    )
    .fetch_one(db)
    .await?
    .unwrap_or(false);

    if !exists {
        return Err(anyhow!("Workspace {} does not exist", workspace_id));
    }

    Ok(())
}

/// Update batch processing metrics for monitoring
async fn update_batch_metrics(
    workspace_id: Uuid,
    total: usize,
    successful: usize,
    db: &PgPool,
) -> AnyhowResult<()> {
    let _ = sqlx::query!(
        r#"
        INSERT INTO batch_processing_metrics (
            workspace_id, operation_type, total_items,
            successful_items, failed_items, created_at
        )
        VALUES ($1, 'artifact_storage', $2, $3, $4, NOW())
        "#,
        workspace_id,
        total as i32,
        successful as i32,
        (total - successful) as i32
    )
    .execute(db)
    .await;

    Ok(())
}

/// Custom error type for chunk processing
#[derive(Debug)]
struct ChunkError {
    failed_indices: Vec<usize>,
    error: anyhow::Error,
}

/// Similarly robust implementation for storing predictions in batch
pub async fn store_predictions_batch(
    predictions: Vec<OutcomePrediction>,
    artifact_ids: Vec<Uuid>,
    workspace_id: Uuid,
    db: PgPool,
) -> AnyhowResult<()> {
    if predictions.is_empty() {
        return Ok(());
    }

    const CHUNK_SIZE: usize = 50;
    const MAX_CONCURRENCY: usize = 3;

    info!(
        workspace_id = %workspace_id,
        prediction_count = predictions.len(),
        "Starting batch prediction storage"
    );

    let chunks: Vec<(Vec<OutcomePrediction>, Vec<Uuid>)> = predictions
        .chunks(CHUNK_SIZE)
        .zip(artifact_ids.chunks(CHUNK_SIZE))
        .map(|(pred_chunk, id_chunk)| (pred_chunk.to_vec(), id_chunk.to_vec()))
        .collect();

    let results = stream::iter(chunks)
        .map(|(pred_chunk, id_chunk)| {
            let db = db.clone();
            async move {
                store_prediction_chunk(pred_chunk, id_chunk, workspace_id, &db).await
            }
        })
        .buffer_unordered(MAX_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    let mut total_stored = 0;
    for result in results {
        match result {
            Ok(count) => total_stored += count,
            Err(e) => warn!("Failed to store prediction chunk: {}", e),
        }
    }

    info!(
        workspace_id = %workspace_id,
        stored = total_stored,
        total = predictions.len(),
        "Batch prediction storage completed"
    );

    Ok(())
}

/// Store a chunk of predictions
async fn store_prediction_chunk(
    pred_chunk: Vec<OutcomePrediction>,
    id_chunk: Vec<Uuid>,
    workspace_id: Uuid,
    db: &PgPool,
) -> AnyhowResult<usize> {
    let mut tx = db.begin().await?;
    let mut stored_count = 0;

    for (prediction, artifact_id) in pred_chunk.iter().zip(id_chunk.iter()) {

        sqlx::query!(
            r#"
            INSERT INTO inference_history (
                id, workspace_id, artifact_id,
                confidence, prediction_data, created_at
            )
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (id) DO NOTHING
            "#,
            Uuid::new_v4(),
            workspace_id,
            artifact_id,
            prediction.confidence as f32,
            serde_json::to_value(prediction)?
        )
        .execute(&mut *tx)
        .await?;

        stored_count += 1;
    }

    tx.commit().await?;
    Ok(stored_count)
}