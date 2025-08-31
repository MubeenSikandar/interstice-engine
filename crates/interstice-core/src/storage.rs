//interstice-core/src/storage.rs
use crate::{Artifact, Outcome, Platform, Result, Error};
use sqlx::PgPool;
use uuid::Uuid;
use std::str::FromStr;

/// Database-backed storage implementation
pub struct DatabaseStorage {
    pool: PgPool,
}

#[async_trait::async_trait]
impl Storage for DatabaseStorage {
    async fn store_artifact(&self, artifact: &Artifact, workspace_id: Uuid) -> Result<Uuid> {
        self.store_artifact(artifact, workspace_id).await
    }
    
    async fn store_outcome(&self, outcome: &Outcome, workspace_id: Uuid) -> Result<Uuid> {
        self.store_outcome(outcome, workspace_id).await
    }
    
    async fn link_artifact_outcome(&self, artifact_id: Uuid, outcome_id: Uuid, confidence: f32) -> Result<()> {
        self.link_artifact_outcome(artifact_id, outcome_id, confidence).await
    }
    
    async fn get_artifacts(&self, workspace_id: Uuid, limit: Option<i64>) -> Result<Vec<Artifact>> {
        self.get_artifacts(workspace_id, limit).await
    }
    
    async fn get_outcomes(&self, workspace_id: Uuid) -> Result<Vec<Outcome>> {
        self.get_outcomes(workspace_id).await
    }
    
    async fn get_workspace_stats(&self, workspace_id: Uuid) -> Result<WorkspaceStats> {
        self.get_workspace_stats(workspace_id).await
    }
    
    async fn get_outcome_progress(&self, outcome_id: Uuid, days: i32) -> Result<Vec<ProgressPoint>> {
        self.get_outcome_progress(outcome_id, days).await
    }
}

impl DatabaseStorage {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Store a new artifact in the database
    pub async fn store_artifact(&self, artifact: &Artifact, workspace_id: Uuid) -> Result<Uuid> {
        let artifact_id = Uuid::new_v4();
        
        let artifact_type = match &artifact.artifact_type {
            crate::ArtifactType::PullRequest { number, repo } => {
                format!("pull_request:{}:{}", number, repo.as_deref().unwrap_or("unknown"))
            }
            crate::ArtifactType::Issue { id, project } => {
                format!("issue:{}:{}", id, project.as_deref().unwrap_or("unknown"))
            }
            crate::ArtifactType::Commit { sha } => {
                format!("commit:{}", sha)
            }
            crate::ArtifactType::Document { title, url } => {
                format!("document:{}:{}", title, url.as_deref().unwrap_or("unknown"))
            }
            crate::ArtifactType::Message { content } => {
                format!("message:{}", &content[..content.len().min(50)])
            }
        };

        sqlx::query!(
            r#"
            INSERT INTO artifacts (id, workspace_id, platform, artifact_type, content, raw_text, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            artifact_id,
            workspace_id,
            artifact.platform.to_string(),
            artifact_type,
            artifact.raw_text,
            artifact.raw_text,
            artifact.metadata,
            artifact.timestamp
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(format!("Failed to store artifact: {}", e)))?;

        Ok(artifact_id)
    }

    /// Store a new outcome in the database
    pub async fn store_outcome(&self, outcome: &Outcome, workspace_id: Uuid) -> Result<Uuid> {
        let outcome_id = outcome.id;
        let now = chrono::Utc::now();

        sqlx::query!(
            r#"
            INSERT INTO outcomes (id, workspace_id, name, description, target_value, current_value, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                target_value = EXCLUDED.target_value,
                current_value = EXCLUDED.current_value,
                updated_at = NOW()
            "#,
            outcome_id,
            workspace_id,
            outcome.name,
            outcome.description,
            outcome.target_value.map(|v| sqlx::types::BigDecimal::from_str(&v.to_string()).unwrap_or_default()),
            outcome.current_value.map(|v| sqlx::types::BigDecimal::from_str(&v.to_string()).unwrap_or_default()),
            now,
            now
        )
        .execute(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to store outcome: {}", e))?;

        Ok(outcome_id)
    }

    /// Link an artifact to an outcome with confidence score
    pub async fn link_artifact_outcome(
        &self,
        artifact_id: Uuid,
        outcome_id: Uuid,
        confidence: f32,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO artifact_outcomes (artifact_id, outcome_id, confidence, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (artifact_id, outcome_id) DO UPDATE SET
                confidence = EXCLUDED.confidence,
                created_at = NOW()
            "#,
            artifact_id,
            outcome_id,
            confidence as f64
        )
        .execute(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to link artifact to outcome: {}", e))?;

        Ok(())
    }

    /// Get all artifacts for a workspace
    pub async fn get_artifacts(&self, workspace_id: Uuid, limit: Option<i64>) -> Result<Vec<Artifact>> {
        let limit = limit.unwrap_or(100);
        
        let rows = sqlx::query!(
            r#"
            SELECT id, platform, artifact_type, content, raw_text, metadata, created_at
            FROM artifacts
            WHERE workspace_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#,
            workspace_id,
            limit
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to fetch artifacts: {}", e))?;

        let mut artifacts = Vec::new();
        for row in rows {
            let artifact_type = self.parse_artifact_type(&row.artifact_type)?;
            
            let artifact = Artifact {
                id: row.id.to_string(),
                artifact_type,
                platform: row.platform.parse().unwrap_or(Platform::Slack),
                raw_text: row.raw_text.unwrap_or_default(),
                metadata: row.metadata.unwrap_or_default(),
                timestamp: row.created_at.unwrap_or_default(),
            };
            artifacts.push(artifact);
        }

        Ok(artifacts)
    }

    /// Get all outcomes for a workspace
    pub async fn get_outcomes(&self, workspace_id: Uuid) -> Result<Vec<Outcome>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, name as "name!", description, target_value, current_value, created_at, updated_at
            FROM outcomes
            WHERE workspace_id = $1
            ORDER BY name
            "#,
            workspace_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to fetch outcomes: {}", e))?;

        let mut outcomes = Vec::new();
        for row in rows {
            let outcome = Outcome {
                id: row.id,
                name: row.name,
                description: row.description,
                target_value: row.target_value.map(|v| v.to_string().parse::<f64>().unwrap_or(0.0)),
                current_value: row.current_value.map(|v| v.to_string().parse::<f64>().unwrap_or(0.0)),
                parent_id: None, // TODO: Implement parent-child relationships
            };
            outcomes.push(outcome);
        }

        Ok(outcomes)
    }

    /// Get workspace statistics
    pub async fn get_workspace_stats(&self, workspace_id: Uuid) -> Result<WorkspaceStats> {
        let artifacts_count = sqlx::query!(
            r#"
            SELECT COUNT(*) as count
            FROM artifacts
            WHERE workspace_id = $1
            "#,
            workspace_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to count artifacts: {}", e))?
        .count
        .unwrap_or(0);

        let outcomes_count = sqlx::query!(
            r#"
            SELECT COUNT(*) as count
            FROM outcomes
            WHERE workspace_id = $1
            "#,
            workspace_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to count outcomes: {}", e))?
        .count
        .unwrap_or(0);

        let mapped_work_percentage = if artifacts_count > 0 {
            let mapped_count = sqlx::query!(
                r#"
                SELECT COUNT(DISTINCT artifact_id) as count
                FROM artifact_outcomes ao
                JOIN artifacts a ON ao.artifact_id = a.id
                WHERE a.workspace_id = $1
                "#,
                workspace_id
            )
            .fetch_one(&self.pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to count mapped artifacts: {}", e))?
            .count
            .unwrap_or(0);
            
            (mapped_count as f64 / artifacts_count as f64) * 100.0
        } else {
            0.0
        };

        let recent_artifacts = sqlx::query!(
            r#"
            SELECT COUNT(*) as count
            FROM artifacts
            WHERE workspace_id = $1 AND created_at >= NOW() - INTERVAL '7 days'
            "#,
            workspace_id
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to count recent artifacts: {}", e))?
        .count
        .unwrap_or(0);

        Ok(WorkspaceStats {
            total_artifacts: artifacts_count,
            total_outcomes: outcomes_count,
            mapped_work_percentage,
            recent_artifacts,
        })
    }

    /// Get outcome progress over time
    pub async fn get_outcome_progress(&self, outcome_id: Uuid, days: i32) -> Result<Vec<ProgressPoint>> {
        let rows = sqlx::query!(
            r#"
            SELECT 
                DATE(a.created_at) as date,
                COUNT(DISTINCT a.id) as artifact_count,
                AVG(ao.confidence) as avg_confidence
            FROM artifacts a
            JOIN artifact_outcomes ao ON a.id = ao.artifact_id
            WHERE ao.outcome_id = $1 
                AND a.created_at >= NOW() - INTERVAL '1 day' * $2
            GROUP BY DATE(a.created_at)
            ORDER BY date
            "#,
            outcome_id,
            days as f64
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to fetch outcome progress: {}", e))?;

        let mut progress = Vec::new();
        for row in rows {
                    if let (Some(date), Some(artifact_count), Some(avg_confidence)) = 
            (row.date, row.artifact_count, row.avg_confidence) {
            progress.push(ProgressPoint {
                date,
                artifact_count: artifact_count as i64,
                avg_confidence: avg_confidence.to_string().parse::<f32>().unwrap_or(0.0),
            });
        }
        }

        Ok(progress)
    }

    /// Helper method to parse artifact type from database string
    fn parse_artifact_type(&self, type_str: &str) -> Result<crate::ArtifactType> {
        let parts: Vec<&str> = type_str.split(':').collect();
        
        match parts.as_slice() {
            ["pull_request", number, repo] => {
                let number = number.parse()
                    .map_err(|_| Error::Other(anyhow::anyhow!("Invalid PR number")))?;
                Ok(crate::ArtifactType::PullRequest {
                    number,
                    repo: if *repo == "unknown" { None } else { Some(repo.to_string()) }
                })
            }
            ["issue", id, project] => {
                Ok(crate::ArtifactType::Issue {
                    id: id.to_string(),
                    project: if *project == "unknown" { None } else { Some(project.to_string()) }
                })
            }
            ["commit", sha] => {
                Ok(crate::ArtifactType::Commit {
                    sha: sha.to_string()
                })
            }
            ["document", title, url] => {
                Ok(crate::ArtifactType::Document {
                    title: title.to_string(),
                    url: if *url == "unknown" { None } else { Some(url.to_string()) }
                })
            }
            ["message", content] => {
                Ok(crate::ArtifactType::Message {
                    content: content.to_string()
                })
            }
            _ => Err(Error::Other(anyhow::anyhow!("Unknown artifact type: {}", type_str)))
        }
    }
}

/// Trait defining storage operations for artifacts and outcomes
#[async_trait::async_trait]
pub trait Storage: Send + Sync {
    /// Store a new artifact in the database
    async fn store_artifact(&self, artifact: &Artifact, workspace_id: Uuid) -> Result<Uuid>;
    
    /// Store a new outcome in the database
    async fn store_outcome(&self, outcome: &Outcome, workspace_id: Uuid) -> Result<Uuid>;
    
    /// Link an artifact to an outcome with confidence score
    async fn link_artifact_outcome(&self, artifact_id: Uuid, outcome_id: Uuid, confidence: f32) -> Result<()>;
    
    /// Get all artifacts for a workspace
    async fn get_artifacts(&self, workspace_id: Uuid, limit: Option<i64>) -> Result<Vec<Artifact>>;
    
    /// Get all outcomes for a workspace
    async fn get_outcomes(&self, workspace_id: Uuid) -> Result<Vec<Outcome>>;
    
    /// Get workspace statistics
    async fn get_workspace_stats(&self, workspace_id: Uuid) -> Result<WorkspaceStats>;
    
    /// Get outcome progress over time
    async fn get_outcome_progress(&self, outcome_id: Uuid, days: i32) -> Result<Vec<ProgressPoint>>;
}



/// Workspace statistics
#[derive(Debug, Clone)]
pub struct WorkspaceStats {
    pub total_artifacts: i64,
    pub total_outcomes: i64,
    pub mapped_work_percentage: f64,
    pub recent_artifacts: i64,
}

/// Progress tracking point
#[derive(Debug, Clone)]
pub struct ProgressPoint {
    pub date: chrono::NaiveDate,
    pub artifact_count: i64,
    pub avg_confidence: f32,
}