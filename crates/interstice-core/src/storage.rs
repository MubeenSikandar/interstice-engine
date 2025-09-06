//! # Storage Module
//! 
//! Comprehensive storage abstraction layer for the INTERSTICE-ENGINE WorkOS.
//! Provides unified interface for persistent storage with multiple backend implementations.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sqlx::{postgres::PgPoolOptions, PgPool, Postgres, Row};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, error, instrument, warn};
use uuid::Uuid;

use crate::analytics::{MetricEvent, MetricQuery};
use crate::artifact::Artifact;
use crate::error::CoreError;
use crate::outcome::{Outcome, OutcomeFilters, OutcomeId};
use crate::types::{
    Platform, UserId, WorkspaceId, SystemEvent
};

/// Storage-specific error types
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Query failed: {0}")]
    QueryFailed(String),
    
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),
    
    #[error("Entity not found: {entity_type} with id {entity_id}")]
    NotFound {
        entity_type: String,
        entity_id: String,
    },
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("Migration error: {0}")]
    MigrationError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
}

impl From<StorageError> for CoreError {
    fn from(err: StorageError) -> Self {
        CoreError::Storage(crate::error::StorageError::QueryFailed(err.to_string()))
    }
}

/// Main storage backend trait
#[async_trait]
pub trait StorageBackend: Send + Sync {
    // Metrics operations
    async fn store_metrics(&self, events: Vec<MetricEvent>) -> Result<(), CoreError>;
    async fn query_metrics(&self, query: &MetricQuery) -> Result<Vec<MetricEvent>, CoreError>;
    async fn delete_old_metrics(&self, before: DateTime<Utc>) -> Result<u64, CoreError>;
    
    // Outcome operations
    async fn store_outcome(&self, outcome: Outcome) -> Result<(), CoreError>;
    async fn get_outcome(&self, id: OutcomeId) -> Result<Option<Outcome>, CoreError>;
    async fn update_outcome(&self, outcome: Outcome) -> Result<(), CoreError>;
    async fn delete_outcome(&self, id: OutcomeId) -> Result<(), CoreError>;
    async fn query_outcomes(
        &self,
        workspace_id: WorkspaceId,
        filters: Option<OutcomeFilters>,
    ) -> Result<Vec<Outcome>, CoreError>;
    
    // Artifact operations
    async fn store_artifact(&self, artifact: Artifact) -> Result<Uuid, CoreError>;
    async fn get_artifact(&self, id: Uuid) -> Result<Option<Artifact>, CoreError>;
    async fn query_artifacts(
        &self,
        workspace_id: WorkspaceId,
        filters: Option<ArtifactFilters>,
    ) -> Result<Vec<Artifact>, CoreError>;
    async fn delete_artifact(&self, id: Uuid) -> Result<(), CoreError>;
    
    // Linking operations
    async fn link_artifact_outcome(
        &self,
        artifact_id: Uuid,
        outcome_id: OutcomeId,
        confidence: f64,
        metadata: Option<JsonValue>,
    ) -> Result<(), CoreError>;
    
    // System operations
    async fn store_event(&self, event: SystemEvent) -> Result<(), CoreError>;
    async fn get_workspace_stats(&self, workspace_id: WorkspaceId) -> Result<WorkspaceStats, CoreError>;
    async fn health_check(&self) -> Result<bool, CoreError>;
    async fn cleanup_expired_data(&self) -> Result<CleanupStats, CoreError>;
}

/// PostgreSQL storage implementation
pub struct PostgresStorage {
    pool: PgPool,
    cache: Arc<RwLock<StorageCache>>,
    config: StorageConfig,
}

impl PostgresStorage {
    /// Create new PostgreSQL storage instance
    pub async fn new(config: StorageConfig) -> Result<Self, StorageError> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(std::time::Duration::from_secs(config.connection_timeout_seconds))
            .connect(&config.database_url)
            .await
            .map_err(|e| StorageError::ConnectionFailed(e.to_string()))?;
        
        // Run migrations
        if config.auto_migrate {
            sqlx::migrate!("../../migrations")
                .run(&pool)
                .await
                .map_err(|e| StorageError::MigrationError(e.to_string()))?;
        }
        
        Ok(Self {
            pool,
            cache: Arc::new(RwLock::new(StorageCache::new(config.cache_size))),
            config,
        })
    }
    
    /// Execute query with retry logic
    async fn execute_with_retry<F>(&self, operation: F) -> Result<sqlx::postgres::PgQueryResult, StorageError>
    where
        F: Fn() -> sqlx::query::Query<'static, Postgres, sqlx::postgres::PgArguments>,
    {
        let mut attempts = 0;
        let max_attempts = self.config.retry_attempts;
        
        loop {
            match operation().execute(&self.pool).await {
                Ok(result) => return Ok(result),
                Err(e) if attempts < max_attempts && Self::is_retryable_error(&e) => {
                    attempts += 1;
                    let delay = std::time::Duration::from_millis(
                        self.config.retry_delay_ms * (2_u64.pow(attempts as u32))
                    );
                    warn!("Retrying operation after {:?}, attempt {}/{}", delay, attempts, max_attempts);
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(StorageError::DatabaseError(e)),
            }
        }
    }
    
    fn is_retryable_error(error: &sqlx::Error) -> bool {
        matches!(
            error,
            sqlx::Error::Io(_) |
            sqlx::Error::PoolTimedOut |
            sqlx::Error::PoolClosed
        )
    }
}

#[async_trait]
impl StorageBackend for PostgresStorage {
    #[instrument(skip(self, events))]
    async fn store_metrics(&self, events: Vec<MetricEvent>) -> Result<(), CoreError> {
        if events.is_empty() {
            return Ok(());
        }
        
        let event_count = events.len();
        let mut tx = self.pool.begin().await
            .map_err(|e| StorageError::TransactionFailed(e.to_string()))?;
        
        for event in events {
            let value_json = serde_json::to_value(&event.value)?;
            let tags_json = serde_json::to_value(&event.tags)?;
            
            sqlx::query!(
                r#"
                INSERT INTO metrics (
                    id, metric_id, workspace_id, user_id, value, timestamp,
                    metadata, outcome_id, tags, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                )
                "#,
                Uuid::new_v4(),
                event.metric_id,
                event.workspace_id.as_uuid(),
                event.user_id.as_ref().map(|u| u.as_str()),
                value_json,
                event.timestamp,
                serde_json::to_value(&event.metadata)?,
                event.outcome_id.map(|o| o.0),
                tags_json,
                Utc::now()
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        }
        
        tx.commit().await
            .map_err(|e| StorageError::TransactionFailed(e.to_string()))?;
        
        debug!("Stored {} metric events", event_count);
        Ok(())
    }
    
    #[instrument(skip(self))]
    async fn query_metrics(&self, query: &MetricQuery) -> Result<Vec<MetricEvent>, CoreError> {
        let mut sql = String::from(
            "SELECT id, metric_id, workspace_id, user_id, value, timestamp, metadata, outcome_id, tags 
             FROM metrics WHERE 1=1"
        );
        
        let mut params: Vec<String> = Vec::new();
        let mut param_count = 1;
        
        // Build dynamic query
        if let Some(workspace_id) = &query.workspace_id {
            sql.push_str(&format!(" AND workspace_id = ${}", param_count));
            params.push(workspace_id.to_string());
            param_count += 1;
        }
        
        if let Some(user_id) = &query.user_id {
            sql.push_str(&format!(" AND user_id = ${}", param_count));
            params.push(user_id.to_string());
            param_count += 1;
        }
        
        if let Some(time_range) = &query.time_range {
            sql.push_str(&format!(" AND timestamp >= ${}", param_count));
            params.push(time_range.start.to_rfc3339());
            param_count += 1;
            
            sql.push_str(&format!(" AND timestamp <= ${}", param_count));
            params.push(time_range.end.to_rfc3339());
            
        }
        
        if let Some(limit) = query.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }
        
        sql.push_str(" ORDER BY timestamp DESC");
        
        // Execute query
        let rows = sqlx::query(&sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        let mut events = Vec::new();
        for row in rows {
            events.push(MetricEvent {
                metric_id: row.get("metric_id"),
                workspace_id: WorkspaceId::from_uuid(row.get("workspace_id")),
                user_id: row.get::<Option<String>, _>("user_id").map(UserId::from),
                value: serde_json::from_value(row.get("value"))?,
                timestamp: row.get("timestamp"),
                metadata: serde_json::from_value(row.get("metadata")).unwrap_or_default(),
                outcome_id: row.get::<Option<Uuid>, _>("outcome_id").map(OutcomeId::from_uuid),
                tags: serde_json::from_value(row.get("tags")).unwrap_or_default(),
            });
        }
        
        Ok(events)
    }
    
    async fn delete_old_metrics(&self, before: DateTime<Utc>) -> Result<u64, CoreError> {
        let result = sqlx::query!(
            "DELETE FROM metrics WHERE timestamp < $1",
            before
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        Ok(result.rows_affected())
    }
    
    #[instrument(skip(self, outcome))]
    async fn store_outcome(&self, outcome: Outcome) -> Result<(), CoreError> {
        let tags_json = serde_json::to_value(&outcome.tags)?;
        let platforms_json = serde_json::to_value(&outcome.platforms)?;
        let children_json = serde_json::to_value(&outcome.children)?;
        let dependencies_json = serde_json::to_value(&outcome.dependencies)?;
        let assignees_json = serde_json::to_value(&outcome.assignees)?;
        let artifacts_json = serde_json::to_value(&outcome.artifacts)?;
        
        let state_json = serde_json::to_value(&outcome.state)?;
        let outcome_type_json = serde_json::to_value(&outcome.outcome_type)?;
        let priority_json = serde_json::to_value(&outcome.priority)?;
        let risk_level_json = serde_json::to_value(&outcome.risk_level)?;
        let automation_level_json = serde_json::to_value(&outcome.automation_level)?;

        sqlx::query(
            r#"
            INSERT INTO outcomes (
                id, workspace_id, name, description, state, outcome_type,
                priority, progress, parent_id, children, dependencies,
                assignees, owner_id, artifacts, tags, platforms,
                metadata, created_at, updated_at, due_date, completed_at,
                estimated_hours, actual_hours, value_score, risk_level,
                automation_level
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                state = EXCLUDED.state,
                outcome_type = EXCLUDED.outcome_type,
                priority = EXCLUDED.priority,
                progress = EXCLUDED.progress,
                parent_id = EXCLUDED.parent_id,
                children = EXCLUDED.children,
                dependencies = EXCLUDED.dependencies,
                assignees = EXCLUDED.assignees,
                artifacts = EXCLUDED.artifacts,
                tags = EXCLUDED.tags,
                platforms = EXCLUDED.platforms,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at,
                due_date = EXCLUDED.due_date,
                completed_at = EXCLUDED.completed_at,
                estimated_hours = EXCLUDED.estimated_hours,
                actual_hours = EXCLUDED.actual_hours,
                value_score = EXCLUDED.value_score,
                risk_level = EXCLUDED.risk_level,
                automation_level = EXCLUDED.automation_level
            "#
        )
        .bind(outcome.id.0)
        .bind(outcome.workspace_id.as_uuid())
        .bind(&outcome.name)
        .bind(&outcome.description)
        .bind(&state_json)
        .bind(&outcome_type_json)
        .bind(&priority_json)
        .bind(outcome.progress)
        .bind(outcome.parent_id.map(|id| id.0))
        .bind(&children_json)
        .bind(&dependencies_json)
        .bind(&assignees_json)
        .bind(outcome.owner_id.as_str())
        .bind(&artifacts_json)
        .bind(&tags_json)
        .bind(&platforms_json)
        .bind(&serde_json::to_value(&outcome.metadata)?)
        .bind(outcome.created_at)
        .bind(outcome.updated_at)
        .bind(outcome.due_date)
        .bind(outcome.completed_at)
        .bind(outcome.estimated_hours)
        .bind(outcome.actual_hours)
        .bind(outcome.value_score)
        .bind(&risk_level_json)
        .bind(&automation_level_json)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        // Invalidate cache
        self.cache.write().await.invalidate_outcome(outcome.id);
        
        Ok(())
    }
    
    #[instrument(skip(self))]
    async fn get_outcome(&self, id: OutcomeId) -> Result<Option<Outcome>, CoreError> {
        // Check cache first
        if let Some(outcome) = self.cache.read().await.get_outcome(id) {
            return Ok(Some(outcome));
        }
        
        let row = sqlx::query(
            r#"
            SELECT * FROM outcomes WHERE id = $1
            "#
        )
        .bind(id.0)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        if let Some(row) = row {
            // Build targets from separate table
            let targets_rows = sqlx::query!(
                "SELECT * FROM outcome_targets WHERE outcome_id = $1",
                id.0
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
            
            let mut targets = Vec::new();
            for target_row in targets_rows {
                targets.push(serde_json::from_value(target_row.data)?);
            }
            
            let outcome = Outcome {
                id,
                workspace_id: WorkspaceId::from_uuid(row.get("workspace_id")),
                name: row.get("name"),
                description: row.get("description"),
                state: serde_json::from_value(row.get("state"))?,
                outcome_type: serde_json::from_value(row.get("outcome_type"))?,
                priority: serde_json::from_value(row.get("priority"))?,
                targets,
                progress: row.get("progress"),
                parent_id: row.get::<Option<Uuid>, _>("parent_id").map(OutcomeId::from_uuid),
                children: serde_json::from_value(row.get("children"))?,
                dependencies: serde_json::from_value(row.get("dependencies"))?,
                assignees: serde_json::from_value(row.get("assignees"))?,
                owner_id: UserId::from(row.get::<String, _>("owner_id")),
                artifacts: serde_json::from_value(row.get("artifacts"))?,
                tags: serde_json::from_value(row.get("tags"))?,
                platforms: serde_json::from_value(row.get("platforms"))?,
                metadata: serde_json::from_value(row.get("metadata")).unwrap_or_default(),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                due_date: row.get("due_date"),
                completed_at: row.get("completed_at"),
                estimated_hours: row.get("estimated_hours"),
                actual_hours: row.get("actual_hours"),
                value_score: row.get("value_score"),
                risk_level: serde_json::from_value(row.get("risk_level"))?,
                automation_level: serde_json::from_value(row.get("automation_level"))?,
            };
            
            // Update cache
            self.cache.write().await.store_outcome(outcome.clone());
            
            Ok(Some(outcome))
        } else {
            Ok(None)
        }
    }
    
    async fn update_outcome(&self, outcome: Outcome) -> Result<(), CoreError> {
        self.store_outcome(outcome).await
    }
    
    async fn delete_outcome(&self, id: OutcomeId) -> Result<(), CoreError> {
        sqlx::query!(
            "DELETE FROM outcomes WHERE id = $1",
            id.0
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        self.cache.write().await.invalidate_outcome(id);
        
        Ok(())
    }
    
    async fn query_outcomes(
        &self,
        workspace_id: WorkspaceId,
        filters: Option<OutcomeFilters>,
    ) -> Result<Vec<Outcome>, CoreError> {
        let mut query = String::from(
            "SELECT * FROM outcomes WHERE workspace_id = $1"
        );
        
        // Apply filters if provided
        if let Some(filters) = filters {
            if let Some(states) = filters.states {
                let states_json = serde_json::to_value(states)?;
                query.push_str(&format!(" AND state = ANY('{}')", states_json));
            }
            
            if let Some(priorities) = filters.priorities {
                let priorities_json = serde_json::to_value(priorities)?;
                query.push_str(&format!(" AND priority = ANY('{}')", priorities_json));
            }
            
            if let Some(parent_id) = filters.parent_id {
                query.push_str(&format!(" AND parent_id = '{}'", parent_id.0));
            }
            
            if let Some(min_progress) = filters.min_progress {
                query.push_str(&format!(" AND progress >= {}", min_progress));
            }
            
            if let Some(max_progress) = filters.max_progress {
                query.push_str(&format!(" AND progress <= {}", max_progress));
            }
        }
        
        query.push_str(" ORDER BY created_at DESC");
        
        let rows = sqlx::query(&query)
            .bind(workspace_id.as_uuid())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        let mut outcomes = Vec::new();
        for row in rows {
            let outcome_id = OutcomeId::from_uuid(row.get("id"));
        
            // Fetch targets from separate table
            let targets_rows = sqlx::query!(
                "SELECT data FROM outcome_targets WHERE outcome_id = $1",
                row.get::<Uuid, _>("id")
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
            let mut targets = Vec::new();
            for target_row in targets_rows {
                targets.push(serde_json::from_value(target_row.data)?);
            }
        
            let outcome = Outcome {
                id: outcome_id,
                workspace_id: WorkspaceId::from_uuid(row.get("workspace_id")),
                name: row.get("name"),
                description: row.get("description"),
                state: serde_json::from_value(row.get("state"))?,
                outcome_type: serde_json::from_value(row.get("outcome_type"))?,
                priority: serde_json::from_value(row.get("priority"))?,
                targets,
                progress: row.get("progress"),
                parent_id: row.get::<Option<Uuid>, _>("parent_id").map(OutcomeId::from_uuid),
                children: serde_json::from_value(row.get("children"))?,
                dependencies: serde_json::from_value(row.get("dependencies"))?,
                assignees: serde_json::from_value(row.get("assignees"))?,
                owner_id: UserId::from(row.get::<String, _>("owner_id")),
                artifacts: serde_json::from_value(row.get("artifacts"))?,
                tags: serde_json::from_value(row.get("tags"))?,
                platforms: serde_json::from_value(row.get("platforms"))?,
                metadata: serde_json::from_value(row.get("metadata")).unwrap_or_default(),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                due_date: row.get("due_date"),
                completed_at: row.get("completed_at"),
                estimated_hours: row.get("estimated_hours"),
                actual_hours: row.get("actual_hours"),
                value_score: row.get("value_score"),
                risk_level: serde_json::from_value(row.get("risk_level"))?,
                automation_level: serde_json::from_value(row.get("automation_level"))?,
            };
        
            outcomes.push(outcome);
        }
        
        Ok(outcomes)
    }
    
    #[instrument(skip(self, artifact))]
    async fn store_artifact(&self, artifact: Artifact) -> Result<Uuid, CoreError> {
        let metadata_json = serde_json::to_value(&artifact.metadata)?;
        
        sqlx::query!(
            r#"
            INSERT INTO artifacts (
                id, workspace_id, artifact_type, platform, content,
                metadata, created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
            "#,
            artifact.id,
            artifact.workspace_id.as_uuid(),
            &serde_json::to_value(&artifact.artifact_type)?.to_string(),
            artifact.platform.to_string(),
            artifact.content,
            metadata_json,
            artifact.created_at,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        Ok(artifact.id)
    }
    
    async fn get_artifact(&self, id: Uuid) -> Result<Option<Artifact>, CoreError> {
        let row = sqlx::query!(
            "SELECT * FROM artifacts WHERE id = $1",
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        if let Some(row) = row {
            Ok(Some(Artifact {
                id: row.id,
                workspace_id: WorkspaceId::from_uuid(row.workspace_id.unwrap()),
                artifact_type: serde_json::from_str(&row.artifact_type)?,
                platform: row.platform.parse().unwrap_or(Platform::Slack),
                content: row.content,
                metadata: row.metadata.unwrap_or_default(),
                created_at: row.created_at.unwrap(),
                updated_at: row.created_at.unwrap(),
                version: 1,
                state: crate::artifact::ArtifactState::Pending,
                quality_metrics: crate::artifact::QualityMetrics::default(),
                related_artifacts: Vec::new(),
                tags: std::collections::HashSet::new(),
            }))
        } else {
            Ok(None)
        }
    }
    
    async fn query_artifacts(
        &self,
        workspace_id: WorkspaceId,
        filters: Option<ArtifactFilters>,
    ) -> Result<Vec<Artifact>, CoreError> {
        let mut query = String::from(
            "SELECT * FROM artifacts WHERE workspace_id = $1"
        );
        
        if let Some(filters) = filters {
            if let Some(platforms) = filters.platforms {
                let platforms_str: Vec<String> = platforms.iter()
                    .map(|p| p.to_string())
                    .collect();
                query.push_str(&format!(" AND platform = ANY(ARRAY[{}])", 
                    platforms_str.iter().map(|p| format!("'{}'", p)).collect::<Vec<_>>().join(",")
                ));
            }
            
            if let Some(created_after) = filters.created_after {
                query.push_str(&format!(" AND created_at >= '{}'", created_after.to_rfc3339()));
            }
            
            if let Some(created_before) = filters.created_before {
                query.push_str(&format!(" AND created_at <= '{}'", created_before.to_rfc3339()));
            }
        }
        
        query.push_str(" ORDER BY created_at DESC");
        
        let rows = sqlx::query(&query)
            .bind(workspace_id.as_uuid())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        let mut artifacts = Vec::new();
        for row in rows {
            artifacts.push(Artifact {
                id: row.get("id"),
                workspace_id: WorkspaceId::from_uuid(row.get("workspace_id")),
                artifact_type: serde_json::from_value(row.get("artifact_type"))?,
                platform: row.get::<String, _>("platform").parse().unwrap_or(Platform::Slack),
                content: row.get("content"),
                metadata: row.get("metadata"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                version: 1,
                state: crate::artifact::ArtifactState::Pending,
                quality_metrics: crate::artifact::QualityMetrics::default(),
                related_artifacts: Vec::new(),
                tags: std::collections::HashSet::new(),
            });
        }
        
        Ok(artifacts)
    }
    
    async fn delete_artifact(&self, id: Uuid) -> Result<(), CoreError> {
        sqlx::query!(
            "DELETE FROM artifacts WHERE id = $1",
            id
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        Ok(())
    }
    
    async fn link_artifact_outcome(
        &self,
        artifact_id: Uuid,
        outcome_id: OutcomeId,
        confidence: f64,
        metadata: Option<JsonValue>,
    ) -> Result<(), CoreError> {
        sqlx::query!(
            r#"
            INSERT INTO artifact_outcomes (
                artifact_id, outcome_id, confidence, metadata, created_at
            ) VALUES (
                $1, $2, $3, $4, $5
            )
            ON CONFLICT (artifact_id, outcome_id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            "#,
            artifact_id,
            outcome_id.0,
            confidence as f64,
            metadata,
            Utc::now()
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        Ok(())
    }
    
    async fn store_event(&self, event: SystemEvent) -> Result<(), CoreError> {
        sqlx::query!(
            r#"
            INSERT INTO system_events (
                id, event_type, workspace_id, user_id, timestamp,
                metadata, correlation_id, platform
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            "#,
            event.id,
            serde_json::to_value(&event.event_type)?,
            event.workspace_id.map(|w| w.as_uuid().clone()),
            event.user_id.as_ref().map(|u| u.as_str()),
            event.timestamp,
            serde_json::to_value(&event.metadata)?,
            event.correlation_id,
            event.platform.map(|p| p.to_string())
        )
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?;
        
        Ok(())
    }
    
    async fn get_workspace_stats(&self, workspace_id: WorkspaceId) -> Result<WorkspaceStats, CoreError> {
        let total_artifacts = sqlx::query!(
            "SELECT COUNT(*) as count FROM artifacts WHERE workspace_id = $1",
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let total_outcomes = sqlx::query!(
            "SELECT COUNT(*) as count FROM outcomes WHERE workspace_id = $1",
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let completed_outcomes = sqlx::query!(
            "SELECT COUNT(*) as count FROM outcomes WHERE workspace_id = $1 AND state = 'completed'",
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let total_metrics = sqlx::query!(
            "SELECT COUNT(*) as count FROM metrics WHERE workspace_id = $1",
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let recent_activity = sqlx::query!(
            r#"
            SELECT COUNT(*) as count FROM metrics 
            WHERE workspace_id = $1 AND timestamp >= NOW() - INTERVAL '7 days'
            "#,
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let linked_artifacts = sqlx::query!(
            r#"
            SELECT COUNT(DISTINCT artifact_id) as count 
            FROM artifact_outcomes ao
            JOIN artifacts a ON ao.artifact_id = a.id
            WHERE a.workspace_id = $1
            "#,
            workspace_id.as_uuid()
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .count
        .unwrap_or(0);
        
        let mapped_work_percentage = if total_artifacts > 0 {
            (linked_artifacts as f64 / total_artifacts as f64) * 100.0
        } else {
            0.0
        };
        
        Ok(WorkspaceStats {
            workspace_id,
            total_artifacts: total_artifacts as u64,
            total_outcomes: total_outcomes as u64,
            completed_outcomes: completed_outcomes as u64,
            total_metrics: total_metrics as u64,
            recent_activity: recent_activity as u64,
            mapped_work_percentage,
            created_at: Utc::now(),
        })
    }
    
    async fn health_check(&self) -> Result<bool, CoreError> {
        match sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
        {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    async fn cleanup_expired_data(&self) -> Result<CleanupStats, CoreError> {
        let mut tx = self.pool.begin().await
            .map_err(|e| StorageError::TransactionFailed(e.to_string()))?;
        
        // Clean old metrics based on retention period
        let metrics_deleted = sqlx::query!(
            "DELETE FROM metrics WHERE timestamp < NOW() - INTERVAL '90 days'"
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .rows_affected();
        
        // Clean orphaned artifacts
        let artifacts_deleted = sqlx::query!(
            r#"
            DELETE FROM artifacts 
            WHERE id NOT IN (
                SELECT DISTINCT artifact_id FROM artifact_outcomes
            ) AND created_at < NOW() - INTERVAL '30 days'
            "#
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .rows_affected();
        
        // Clean expired sessions/tokens
        let tokens_deleted = sqlx::query!(
            "DELETE FROM auth_tokens WHERE expires_at < NOW()"
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .rows_affected();
        
        // Clean old system events
        let events_deleted = sqlx::query!(
            "DELETE FROM system_events WHERE timestamp < NOW() - INTERVAL '60 days'"
        )
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::QueryFailed(e.to_string()))?
        .rows_affected();
        
        tx.commit().await
            .map_err(|e| StorageError::TransactionFailed(e.to_string()))?;
        
        Ok(CleanupStats {
            metrics_deleted,
            artifacts_deleted,
            tokens_deleted,
            events_deleted,
            timestamp: Utc::now(),
        })
    }
}

/// In-memory cache for frequently accessed data
struct StorageCache {
    outcomes: HashMap<OutcomeId, (Outcome, DateTime<Utc>)>,
    artifacts: HashMap<Uuid, (Artifact, DateTime<Utc>)>,
    max_size: usize,
    ttl: std::time::Duration,
}

impl StorageCache {
    fn new(max_size: usize) -> Self {
        Self {
            outcomes: HashMap::new(),
            artifacts: HashMap::new(),
            max_size,
            ttl: std::time::Duration::from_secs(300), // 5 minutes
        }
    }
    
    fn get_outcome(&self, id: OutcomeId) -> Option<Outcome> {
        if let Some((outcome, timestamp)) = self.outcomes.get(&id) {
            if Utc::now().signed_duration_since(*timestamp).to_std().unwrap_or_default() < self.ttl {
                return Some(outcome.clone());
            }
        }
        None
    }
    
    fn store_outcome(&mut self, outcome: Outcome) {
        // Evict old entries if cache is full
        if self.outcomes.len() >= self.max_size {
            let oldest = self.outcomes
                .iter()
                .min_by_key(|(_, (_, ts))| ts)
                .map(|(id, _)| *id);
            
            if let Some(id) = oldest {
                self.outcomes.remove(&id);
            }
        }
        
        self.outcomes.insert(outcome.id, (outcome, Utc::now()));
    }
    
    fn invalidate_outcome(&mut self, id: OutcomeId) {
        self.outcomes.remove(&id);
    }
}

/// Storage configuration
#[derive(Debug, Clone, Deserialize)]
pub struct StorageConfig {
    pub database_url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_seconds: u64,
    pub auto_migrate: bool,
    pub cache_size: usize,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_url: String::from("postgresql://localhost/interstice"),
            max_connections: 10,
            min_connections: 2,
            connection_timeout_seconds: 30,
            auto_migrate: true,
            cache_size: 1000,
            retry_attempts: 3,
            retry_delay_ms: 100,
        }
    }
}

/// Artifact query filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactFilters {
    pub platforms: Option<Vec<Platform>>,
    pub artifact_types: Option<Vec<String>>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub has_outcome: Option<bool>,
    pub tags: Option<Vec<String>>,
}

/// Workspace statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceStats {
    pub workspace_id: WorkspaceId,
    pub total_artifacts: u64,
    pub total_outcomes: u64,
    pub completed_outcomes: u64,
    pub total_metrics: u64,
    pub recent_activity: u64,
    pub mapped_work_percentage: f64,
    pub created_at: DateTime<Utc>,
}

/// Cleanup statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupStats {
    pub metrics_deleted: u64,
    pub artifacts_deleted: u64,
    pub tokens_deleted: u64,
    pub events_deleted: u64,
    pub timestamp: DateTime<Utc>,
}

/// Progress tracking point for outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressPoint {
    pub date: NaiveDate,
    pub artifact_count: i64,
    pub outcome_progress: f64,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::Outcome;
    
    #[tokio::test]
    async fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.cache_size, 1000);
        assert!(config.auto_migrate);
    }
    
    #[test]
    fn test_workspace_stats() {
        let stats = WorkspaceStats {
            workspace_id: WorkspaceId::new(),
            total_artifacts: 100,
            total_outcomes: 20,
            completed_outcomes: 15,
            total_metrics: 1000,
            recent_activity: 50,
            mapped_work_percentage: 75.0,
            created_at: Utc::now(),
        };
        
        assert_eq!(stats.total_artifacts, 100);
        assert_eq!(stats.mapped_work_percentage, 75.0);
    }
    
    #[test]
    fn test_cache_operations() {
        let mut cache = StorageCache::new(2);
        
        let outcome1 = Outcome::new(
            WorkspaceId::new(),
            "Test 1".to_string(),
            UserId::from("user1"),
        );
        let outcome2 = Outcome::new(
            WorkspaceId::new(),
            "Test 2".to_string(),
            UserId::from("user2"),
        );
        let outcome3 = Outcome::new(
            WorkspaceId::new(),
            "Test 3".to_string(),
            UserId::from("user3"),
        );
        
        cache.store_outcome(outcome1.clone());
        cache.store_outcome(outcome2.clone());
        
        assert!(cache.get_outcome(outcome1.id).is_some());
        assert!(cache.get_outcome(outcome2.id).is_some());
        
        // Adding third should evict the oldest
        cache.store_outcome(outcome3.clone());
        assert_eq!(cache.outcomes.len(), 2);
    }
    
    #[test]
    fn test_artifact_filters() {
        let filters = ArtifactFilters {
            platforms: Some(vec![Platform::GitHub, Platform::Slack]),
            created_after: Some(Utc::now() - chrono::Duration::days(7)),
            created_before: Some(Utc::now()),
            ..Default::default()
        };
        
        assert_eq!(filters.platforms.unwrap().len(), 2);
    }
}