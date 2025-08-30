use sqlx::{PgPool, postgres::PgPoolOptions};
use uuid::Uuid;
use serde_json;
use crate::{Artifact, Result, Error, Platform};

#[derive(Clone)]
pub struct Storage {
    pool: PgPool,
}

impl Storage {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await
            .map_err(|e| Error::DatabaseError(format!("Failed to connect: {}", e)))?;
        
        Ok(Self { pool })
    }

    pub fn connect_lazy(database_url: &str) -> Result<Self> {
        // For lazy connection, we'll create a pool that connects when first used
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect_lazy(database_url)
            .map_err(|e| Error::DatabaseError(format!("Failed to create lazy connection: {}", e)))?;
        
        Ok(Self { pool })
    }

    pub async fn get_or_create_workspace(&self, slack_team_id: &str, name: &str) -> Result<Uuid> {
        // First try to get existing workspace
        let existing = sqlx::query!(
            "SELECT id FROM workspaces WHERE slack_team_id = $1",
            slack_team_id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(e.to_string()))?;
        
        if let Some(record) = existing {
            return Ok(record.id);
        }
        
        // Create new workspace
        let id = Uuid::new_v4();
        sqlx::query!(
            "INSERT INTO workspaces (id, slack_team_id, name) VALUES ($1, $2, $3)",
            id,
            slack_team_id,
            name
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(e.to_string()))?;
        
        Ok(id)
    }

    pub async fn save_artifact(&self, artifact: &Artifact, workspace_id: Uuid) -> Result<Uuid> {
        let id = Uuid::new_v4();
        let artifact_type = serde_json::to_string(&artifact.artifact_type)
            .map_err(|e| Error::Other(e.into()))?;
        
        sqlx::query!(
            r#"
            INSERT INTO artifacts (id, workspace_id, platform, artifact_type, content, raw_text, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
            id,
            workspace_id,
            artifact.platform.to_string(),
            artifact_type,
            serde_json::to_string(&artifact).unwrap_or_default(),
            artifact.raw_text,
            artifact.metadata
        )
        .execute(&self.pool)
        .await
        .map_err(|e| Error::DatabaseError(e.to_string()))?;
        
        Ok(id)
    }

    pub async fn get_recent_artifacts(&self, workspace_id: Uuid, limit: i64) -> Result<Vec<serde_json::Value>> {
        let records = sqlx::query!(
            r#"
            SELECT id, platform, artifact_type, raw_text, created_at
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
        .map_err(|e| Error::DatabaseError(e.to_string()))?;
        
        let artifacts: Vec<serde_json::Value> = records
            .into_iter()
            .map(|r| serde_json::json!({
                "id": r.id,
                "platform": r.platform,
                "type": r.artifact_type,
                "text": r.raw_text,
                "created_at": r.created_at
            }))
            .collect();
        
        Ok(artifacts)
    }
}