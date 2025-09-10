//interstice-api/src/routes/api.rs
use axum::{
    extract::{ws, Path, Query, State}, http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}, Json, Router
};
use interstice_ml::Platform as MLPlatform;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::str::FromStr;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::info;
use interstice_core::{
    analytics::{ExportFormat, MetricQuery}, outcome::{AutomationLevel, OutcomeState, RiskLevel}, types::{MetricValue, Platform, Priority, TimeRange, WorkspaceId}, Outcome, OutcomeType, UserId
};
use crate::{AppState};

#[derive(Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    page: u32,
    #[serde(default = "default_limit")]
    limit: u32,
}

fn default_page() -> u32 { 1 }
fn default_limit() -> u32 { 20 }

#[derive(Deserialize)]
struct ArtifactQueryParams {
    #[serde(flatten)]
    pagination: PaginationParams,
    workspace_id: Option<Uuid>,
    platform: Option<String>,
}

#[derive(Deserialize)]
struct OutcomeQueryParams {
    #[serde(flatten)]
    _pagination: PaginationParams,
    workspace_id: Option<WorkspaceId>,
}
#[derive(sqlx::FromRow)]
struct ArtifactRow {
    id: Uuid,
    workspace_id: Uuid,
    content: String,
    platform: String,
    artifact_type: Option<String>,
    created_at: Option<DateTime<Utc>>,
    metadata: Option<serde_json::Value>,
}

// ============================================================================
// Workspace Routes
// ============================================================================

pub fn workspace_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(list_workspaces).post(create_workspace))
        .route("/:id", get(get_workspace).put(update_workspace).delete(delete_workspace))
}

// ============================================================================
// Artifact Routes
// ============================================================================

pub fn artifact_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(list_artifacts).post(create_artifact))
        .route("/:id", get(get_artifact))
}

// ============================================================================
// Outcome Routes
// ============================================================================

pub fn outcome_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(list_outcomes).post(create_outcome))
        .route("/:id", get(get_outcome))
        .route("/:id/predict", post(predict_outcomes))
}

// ============================================================================
// Analytics Routes
// ============================================================================

pub fn analytics_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/dashboard", get(get_dashboard))
        .route("/metrics", get(get_metrics))
        .route("/metrics/query", post(query_metrics))
        .route("/metrics/export", post(export_metrics))
        .route("/health", get(analytics_health))
        .route("/workspaces/:id/stats", get(get_workspace_analytics))
        .route("/workspaces/:id/insights", get(get_workspace_insights))
        .route("/ws/:workspace_id", get(analytics_websocket))
}

// Workspace handlers
async fn list_workspaces(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<WorkspaceResponse>>, StatusCode> {
    let offset = ((params.page - 1) * params.limit) as i64;
    let limit = params.limit as i64;
    
    let workspaces = sqlx::query!(
        r#"
        SELECT id, name, created_at, updated_at, description,
               (SELECT COUNT(*) FROM artifacts WHERE workspace_id = workspaces.id) as artifact_count,
               (SELECT COUNT(*) FROM outcomes WHERE workspace_id = workspaces.id) as outcome_count
        FROM workspaces
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        "#,
        limit,
        offset
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list workspaces: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let response: Vec<WorkspaceResponse> = workspaces
        .into_iter()
        .map(|w| WorkspaceResponse {
            id: w.id,
            name: w.name,
            description: w.description,
            created_at: Some(w.created_at),
            updated_at: Some(w.updated_at),
            artifact_count: Some(w.artifact_count.unwrap_or(0) as u64),
            outcome_count: Some(w.outcome_count.unwrap_or(0) as u64),
        })
        .collect();
    
    Ok(Json(response))
}

async fn create_workspace(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateWorkspaceRequest>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    // Validate input
    if payload.name.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    
    let workspace_id = Uuid::new_v4();
    
    let workspace = sqlx::query!(
        r#"
        INSERT INTO workspaces (id, name, description, created_at, updated_at)
        VALUES ($1, $2, $3, NOW(), NOW())
        RETURNING id, name, description, created_at, updated_at
        "#,
        workspace_id,
        payload.name.trim(),
        payload.description.as_deref()
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create workspace: {}", e);
        if e.to_string().contains("duplicate") {
            StatusCode::CONFLICT
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    })?;
    
    // Track workspace creation in analytics
    if let Some(analytics) = &state.analytics {
        let event = interstice_core::analytics::create_tagged_metric(
            "workspace.created",
            WorkspaceId::from_uuid(workspace_id),
            MetricValue::Integer(1),
            vec!["action:create".to_string()],
        );
        let _ = analytics.record_metric(event).await;
    }
    
    tracing::info!("Created workspace {} with name: {}", workspace_id, payload.name);
    
    Ok(Json(WorkspaceResponse {
        id: workspace.id,
        name: workspace.name,
        description: workspace.description,
        created_at: Some(workspace.created_at),
        updated_at: Some(workspace.updated_at),
        artifact_count: Some(0),
        outcome_count: Some(0),
    }))
}

async fn get_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    let workspace = sqlx::query!(
        r#"
        SELECT w.id, w.name, w.description, w.created_at, w.updated_at,
               (SELECT COUNT(*) FROM artifacts WHERE workspace_id = w.id) as artifact_count,
               (SELECT COUNT(*) FROM outcomes WHERE workspace_id = w.id) as outcome_count
        FROM workspaces w
        WHERE w.id = $1
        "#,
        id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get workspace {}: {}", id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    match workspace {
        Some(w) => Ok(Json(WorkspaceResponse {
            id: w.id,
            name: w.name,
            description: w.description,
            created_at: Some(w.created_at),
            updated_at: Some(w.updated_at),
            artifact_count: Some(w.artifact_count.unwrap_or(0) as u64),
            outcome_count: Some(w.outcome_count.unwrap_or(0) as u64),
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn update_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpdateWorkspaceRequest>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    // Build dynamic update query
    let mut updates = vec![];
    let mut params: Vec<String> = vec![];
    
    if let Some(name) = &payload.name {
        if !name.trim().is_empty() {
            updates.push("name = $2");
            params.push(name.trim().to_string());
        }
    }
    
    if let Some(desc) = &payload.description {
        updates.push("description = $3");
        params.push(desc.clone());
    }
    
    if updates.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    
    // Update workspace
    let workspace = sqlx::query!(
        r#"
        UPDATE workspaces 
        SET name = COALESCE($2, name),
            description = COALESCE($3, description),
            updated_at = NOW()
        WHERE id = $1
        RETURNING id, name, description, created_at, updated_at
        "#,
        id,
        payload.name.as_deref(),
        payload.description.as_deref()
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to update workspace {}: {}", id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    match workspace {
        Some(w) => {
            tracing::info!("Updated workspace {}", id);
            Ok(Json(WorkspaceResponse {
                id: w.id,
                name: w.name,
                description: w.description,
                created_at: Some(w.created_at),
                updated_at: Some(w.updated_at),
                artifact_count: None,
                outcome_count: None,
            }))
        }
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn delete_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, StatusCode> {
    // Start transaction for cascading delete
    let mut tx = state.db.begin().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Delete related data first
    sqlx::query!("DELETE FROM artifacts WHERE workspace_id = $1", id)
        .execute(&mut *tx)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    sqlx::query!("DELETE FROM outcomes WHERE workspace_id = $1", id)
        .execute(&mut *tx)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    // Delete workspace
    let result = sqlx::query!("DELETE FROM workspaces WHERE id = $1", id)
        .execute(&mut *tx)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }
    
    tx.commit().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    tracing::info!("Deleted workspace {} and all related data", id);
    Ok(StatusCode::NO_CONTENT)
}

// Artifact handlers
async fn list_artifacts(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ArtifactQueryParams>,
) -> Result<Json<Vec<ArtifactResponse>>, StatusCode> {
    let offset = ((params.pagination.page - 1) * params.pagination.limit) as i64;
    let limit = params.pagination.limit as i64;
    
    let mut query = "SELECT * FROM artifacts WHERE 1=1".to_string();
    
    if let Some(workspace_id) = params.workspace_id {
        query.push_str(&format!(" AND workspace_id = '{}'", workspace_id));
    }
    
    if let Some(platform) = params.platform {
        query.push_str(&format!(" AND platform = '{}'", platform));
    }
    
    query.push_str(&format!(" ORDER BY created_at DESC LIMIT {} OFFSET {}", limit, offset));
    
    let artifacts = sqlx::query_as::<_, ArtifactRow>(&query)
        .fetch_all(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Failed to list artifacts: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    let response: Vec<ArtifactResponse> = artifacts
        .into_iter()
        .map(|a| ArtifactResponse {
            id: a.id,
            workspace_id: a.workspace_id,
            content: a.content,
            platform: a.platform,
            artifact_type: a.artifact_type,
            created_at: Some(a.created_at.unwrap_or_else(Utc::now)),
            metadata: a.metadata,
        })
        .collect();
    
    Ok(Json(response))
}

async fn create_artifact(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateArtifactRequest>,
) -> Result<Json<ArtifactResponse>, StatusCode> {
    // Parse platform
    let platform = match payload.platform.to_lowercase().as_str() {
        "slack" => Platform::Slack,
        "github" => Platform::GitHub,
        "jira" => Platform::Jira,
        _ => Platform::Slack,
    };
    
    let artifact_id = Uuid::new_v4();
    let now = Utc::now();
    
    // Store directly in database
    let result = sqlx::query!(
        r#"
        INSERT INTO artifacts (
            id, workspace_id, artifact_type, content, 
            platform, metadata, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        "#,
        artifact_id,
        payload.workspace_id,
        "message", // artifact type
        payload.content,
        platform.to_string(),
        serde_json::json!({
            "source": "api",
            "author": "api_user"
        }),
        now,
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create artifact: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    // Track creation in analytics
    if let Some(analytics) = &state.analytics {
        let event = interstice_core::analytics::create_tagged_metric(
            "artifact.created",
            WorkspaceId::from_uuid(payload.workspace_id),
            MetricValue::Integer(1),
            vec![
                format!("platform:{}", platform),
                "source:api".to_string()
            ],
        );
        let _ = analytics.record_metric(event).await;
    }
    
    tracing::info!("Created artifact {} in workspace {}", artifact_id, payload.workspace_id);
    
    Ok(Json(ArtifactResponse {
        id: result.id,
        workspace_id: payload.workspace_id,
        content: payload.content,
        platform: payload.platform,
        artifact_type: Some("message".to_string()),
        created_at: Some(now),
        metadata: Some(serde_json::json!({
            "source": "api",
            "author": "api_user"
        })),
    }))
}

async fn get_artifact(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ArtifactResponse>, StatusCode> {
    let artifact = sqlx::query_as::<_, ArtifactRow>(
        "SELECT * FROM artifacts WHERE id = $1"
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get artifact {}: {}", id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    match artifact {
        Some(a) => Ok(Json(ArtifactResponse {
            id: a.id,
            workspace_id: a.workspace_id,
            content: a.content,
            platform: a.platform,
            artifact_type: a.artifact_type,
            created_at: a.created_at,
            metadata: a.metadata,
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}


// Outcome handlers
async fn list_outcomes(
    State(state): State<Arc<AppState>>,
    Query(params): Query<OutcomeQueryParams>,
) -> Result<Json<Vec<OutcomeResponse>>, StatusCode> {
    let workspace_id = params.workspace_id
        .unwrap_or_else(WorkspaceId::new);
    
    let outcomes = sqlx::query!(
        "SELECT id, name, description, state::text, created_at FROM outcomes WHERE workspace_id = $1 ORDER BY created_at DESC",
        workspace_id.as_uuid()
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list outcomes: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    let response: Vec<OutcomeResponse> = outcomes
        .into_iter()
        .map(|o| OutcomeResponse {
            id: o.id,
            name: o.name,
            description: o.description,
            status: o.state,
            confidence: None,
            created_at: o.created_at,
        })
        .collect();
    
    Ok(Json(response))
}

async fn create_outcome(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateOutcomeRequest>,
) -> Result<Json<OutcomeResponse>, StatusCode> {
    let workspace_id = payload.workspace_id
        .unwrap_or_else(WorkspaceId::new);
    
    let outcome = Outcome {
        id: interstice_core::outcome::OutcomeId::new(),
        workspace_id,
        name: payload.name.clone(),
        description: payload.description.clone(),
        state: OutcomeState::Draft,
        outcome_type: OutcomeType::Task,
        priority: Priority::Medium,
        targets: vec![],
        progress: 0.0,
        parent_id: None,
        children: vec![],
        dependencies: vec![],
        assignees: vec![],
        owner_id: UserId::new("api_user"),
        artifacts: vec![],
        tags: HashSet::new(),
        platforms: HashSet::new(),
        due_date: None,
        estimated_hours: None,
        actual_hours: None,
        value_score: None,
        risk_level: RiskLevel::Low,
        automation_level: AutomationLevel::Manual,
        completed_at: None,
        metadata: HashMap::new(),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    
    let outcome_id = sqlx::query_scalar::<_, Uuid>(
        r#"
        INSERT INTO outcomes (id, workspace_id, name, description, state, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::outcome_state, $6, $7)
        RETURNING id
        "#
    )
    .bind(outcome.id.0)
    .bind(outcome.workspace_id.0)
    .bind(&outcome.name)
    .bind(&outcome.description)
    .bind(format!("{:?}", outcome.state).to_lowercase())
    .bind(outcome.created_at)
    .bind(outcome.updated_at)
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create outcome: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    info!("Created outcome {} with name: {}", outcome_id, payload.name);
    
    Ok(Json(OutcomeResponse {
        id: outcome_id,
        name: payload.name,
        description: payload.description,
        status: Some("draft".to_string()),
        confidence: Some(0.0),
        created_at: Some(Utc::now()),
    }))
}

async fn get_outcome(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<OutcomeResponse>, StatusCode> {
    let outcome = sqlx::query!(
        "SELECT id, name, description, state::text, created_at FROM outcomes WHERE id = $1",
        id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get outcome {}: {}", id, e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    
    match outcome {
        Some(o) => Ok(Json(OutcomeResponse {
            id: o.id,
            name: o.name,
            description: o.description,
            status: o.state,
            confidence: None,
            created_at: o.created_at,
        })),
        None => Err(StatusCode::NOT_FOUND),
    }
}


async fn predict_outcomes(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, StatusCode> {
    // Fetch artifacts
    let mut artifacts = vec![];
    for artifact_id in &payload.artifacts {
        if let Ok(Some(artifact)) = sqlx::query_as::<_, ArtifactRow>(
            "SELECT * FROM artifacts WHERE id = $1"
        )
        .bind(artifact_id)
        .fetch_optional(&state.db)
        .await
        {
            artifacts.push(artifact);
        }
    }
    
    if artifacts.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    
    // Run ML prediction
    let predictions = state.ml_pipeline
        .predict_outcomes(
            id,
            &artifacts.iter().map(|a| {
                interstice_ml::types::Artifact::new(
                    a.id.to_string(),
                    a.content.clone(),
                    MLPlatform::from_str(&a.platform.to_string())
                        .unwrap_or(MLPlatform::Slack),
                    interstice_ml::types::ArtifactType::Message,
                )
            }).collect::<Vec<_>>(),
            &artifacts.iter().map(|a| a.content.as_str()).collect::<Vec<_>>().join(" "),
        )
        .await
        .map_err(|e| {
            tracing::error!("ML prediction failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    let response = PredictResponse {
        predictions: predictions.into_iter().map(|p| PredictionResult {
            outcome_id: Uuid::new_v4(),
            confidence: p.confidence as f32,
            reasoning: p.reasoning,
        }).collect(),
    };
    
    Ok(Json(response))
}

// Analytics handlers
async fn get_dashboard(
    State(state): State<Arc<AppState>>,
    Query(params): Query<DashboardParams>,
) -> Result<Json<DashboardResponse>, StatusCode> {
    let workspace_id = params.workspace_id.unwrap_or_else(WorkspaceId::new);
    
    if let Some(analytics) = &state.analytics {
        // Use the full analytics engine capabilities
        match analytics.get_dashboard_metrics(workspace_id).await {
            Ok(metrics) => {
                Ok(Json(DashboardResponse {
                    total_artifacts: metrics.total_events as u64,
                    total_outcomes: 0, // You can get this from engine if needed
                    recent_activity: vec![], // Convert metrics.trending_metrics if needed
                    workspace_stats: Some(WorkspaceStatsResponse {
                        completed_outcomes: (metrics.outcome_completion_rate * 100.0) as u64,
                        mapped_work_percentage: metrics.outcome_completion_rate,
                        recent_activity_count: metrics.active_users,
                    }),
                }))
            }
            Err(e) => {
                tracing::error!("Dashboard error: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    } else {
        // Fallback to engine stats if analytics disabled
        let stats = state.engine().get_workspace_stats(workspace_id).await
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        
        Ok(Json(DashboardResponse {
            total_artifacts: stats.total_artifacts,
            total_outcomes: stats.total_outcomes,
            recent_activity: vec![],
            workspace_stats: Some(WorkspaceStatsResponse {
                completed_outcomes: stats.completed_outcomes,
                mapped_work_percentage: stats.mapped_work_percentage,
                recent_activity_count: stats.recent_activity,
            }),
        }))
    }
}


async fn get_metrics(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MetricsParams>,
) -> Result<Json<MetricsResponse>, StatusCode> {
    if let Some(_analytics) = &state.analytics {
        let workspace_id = params.workspace_id.unwrap_or_else(|| WorkspaceId::new());
        
        // Get ML model metrics if available with timeout
        let model_metrics = match state.ml_pipeline.get_model_metrics(*workspace_id.as_uuid()).await {
            Ok(metrics) => Some(MLModelMetrics {
                accuracy: metrics.accuracy,
                precision: metrics.precision,
                recall: metrics.recall,
                f1_score: metrics.f1_score,
                total_predictions: metrics.total_predictions,
                correct_predictions: metrics.correct_predictions,
                last_updated: metrics.last_updated,
            }),
            Err(_) => None,
        };
        
        Ok(Json(MetricsResponse {
            model_metrics,
            system_metrics: SystemMetrics {
                avg_response_time_ms: 0.0,
                requests_per_minute: 0.0,
                error_rate: 0.0,
            },
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// New analytics handlers
async fn query_metrics(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<QueryMetricsRequest>,
) -> Result<Json<QueryMetricsResponse>, StatusCode> {
    let analytics = state.analytics.as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    
    let query = MetricQuery {
        workspace_id: Some(payload.workspace_id),
        user_id: None,
        metric_ids: payload.metric_name.map(|n| vec![n]),
        time_range: match (payload.start_time, payload.end_time) {
            (Some(start), Some(end)) => Some(TimeRange { start, end }),
            _ => None,
        },
        tags: payload.tags.map(|h| h.into_keys().collect()),
        aggregation: payload.aggregation,
        group_by: payload.group_by,
        outcome_ids: None,
        sort_by: None,
        limit: payload.limit,
    };
    
    let result = analytics.query_metrics(query).await
        .map_err(|e| {
            tracing::error!("Query failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
    
    let results = result.metrics.into_iter()
        .map(|m| MetricResult {
            metric_name: m.metric_id,
            value: m.value,
            timestamp: m.timestamp,
            tags: m.metadata.into_iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                .collect(),
        })
        .collect();
    
    Ok(Json(QueryMetricsResponse { results }))
}

async fn analytics_websocket(
    ws: ws::WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(workspace_id): Path<Uuid>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_analytics_ws(socket, state, workspace_id))
}

async fn handle_analytics_ws(
    mut socket: ws::WebSocket,
    state: Arc<AppState>,
    workspace_id: Uuid,
) {
    let analytics = match &state.analytics {
        Some(a) => a,
        None => return,
    };
    
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
    
    loop {
        tokio::select! {
            _ = interval.tick() => {
                // Send real-time metrics
                if let Ok(metrics) = analytics.get_dashboard_metrics(WorkspaceId::from_uuid(workspace_id)).await {
                    let msg = ws::Message::Text(
                        serde_json::to_string(&metrics).unwrap_or_default().into()
                    );
                    if socket.send(msg).await.is_err() {
                        break;
                    }
                }
            }
            
            Some(msg) = socket.recv() => {
                // Handle client messages if needed
                if msg.is_err() {
                    break;
                }
            }
        }
    }
}

async fn export_metrics(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ExportMetricsRequest>,
) -> Result<Response, StatusCode> {
    let analytics = state.analytics.as_ref()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    
    let time_range = TimeRange {
        start: payload.start_time.unwrap_or_else(|| Utc::now() - chrono::Duration::days(7)),
        end: payload.end_time.unwrap_or_else(|| Utc::now()),
    };
    
    let format = payload.format.unwrap_or(ExportFormat::Csv);
    
    let data = analytics.export_analytics(
        payload.workspace_id, 
        format.clone(), 
        time_range
    ).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let (content_type, ext) = match format {
        ExportFormat::Json => ("application/json", "json"),
        ExportFormat::Csv => ("text/csv", "csv"),
        ExportFormat::Parquet => ("application/octet-stream", "parquet"),
    };
    
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", content_type)
        .header("Content-Disposition", 
            format!("attachment; filename=\"export_{}.{}\"", 
                Utc::now().format("%Y%m%d_%H%M%S"), ext))
        .body(axum::body::Body::from(data))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

async fn analytics_health(
    State(state): State<Arc<AppState>>,
) -> Result<Json<AnalyticsHealthResponse>, StatusCode> {
    if let Some(_analytics) = &state.analytics {
        Ok(Json(AnalyticsHealthResponse {
            status: "healthy".to_string(),
            components: vec![
                ComponentHealth {
                    name: "analytics_engine".to_string(),
                    healthy: true,
                    message: "Analytics engine is running".to_string(),
                },
            ],
        }))
    } else {
        Ok(Json(AnalyticsHealthResponse {
            status: "disabled".to_string(),
            components: vec![],
        }))
    }
}

async fn get_workspace_analytics(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(params): Query<AnalyticsTimeRange>,
) -> Result<Json<WorkspaceAnalyticsResponse>, StatusCode> {
    let _workspace_id = WorkspaceId::from_uuid(id);
    
    if let Some(_analytics) = &state.analytics {
        let end_time = Utc::now();
        let start_time = end_time - params.duration.unwrap_or(chrono::Duration::days(7));
        
        Ok(Json(WorkspaceAnalyticsResponse {
            workspace_id: id,
            time_range: TimeRange {
                start: start_time,
                end: end_time,
            },
            artifact_stats: ArtifactStats {
                total_created: 0,
                by_platform: HashMap::new(),
            },
            ml_stats: MLStats {
                avg_accuracy: 0.0,
                total_predictions: 0,
            },
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn get_workspace_insights(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(_params): Query<InsightsParams>,
) -> Result<Json<WorkspaceInsightsResponse>, StatusCode> {
    if state.analytics.is_some() {
        Ok(Json(WorkspaceInsightsResponse {
            workspace_id: id,
            insights: vec![],
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// Request/Response types
#[derive(Deserialize)]
struct CreateWorkspaceRequest {
    name: String,
    description: Option<String>,
}

#[derive(Serialize)]
struct WorkspaceResponse {
    id: Uuid,
    name: String,
    description: Option<String>,
    created_at: Option<DateTime<Utc>>,
    updated_at: Option<DateTime<Utc>>,
    artifact_count: Option<u64>,
    outcome_count: Option<u64>,
}

#[derive(Deserialize)]
struct UpdateWorkspaceRequest {
    name: Option<String>,
    description: Option<String>,
}

#[derive(Deserialize)]
struct CreateArtifactRequest {
    workspace_id: Uuid,
    content: String,
    platform: String,
}

#[derive(Serialize)]
struct ArtifactResponse {
    id: Uuid,
    workspace_id: Uuid,
    content: String,
    platform: String,
    artifact_type: Option<String>,
    created_at: Option<DateTime<Utc>>,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct CreateOutcomeRequest {
    name: String,
    description: Option<String>,
    workspace_id: Option<WorkspaceId>,
}

#[derive(Serialize)]
struct OutcomeResponse {
    id: Uuid,
    name: String,
    description: Option<String>,
    status: Option<String>,
    confidence: Option<f64>,
    created_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize)]
struct PredictRequest {
    artifacts: Vec<Uuid>,
}

#[derive(Serialize)]
struct PredictResponse {
    predictions: Vec<PredictionResult>,
}

#[derive(Serialize)]
struct PredictionResult {
    outcome_id: Uuid,
    confidence: f32,
    reasoning: Option<String>,
}

#[derive(Deserialize)]
struct DashboardParams {
    workspace_id: Option<WorkspaceId>,
}

#[derive(Serialize)]
struct DashboardResponse {
    total_artifacts: u64,
    total_outcomes: u64,
    recent_activity: Vec<serde_json::Value>,
    workspace_stats: Option<WorkspaceStatsResponse>,
}

#[derive(Serialize)]
struct WorkspaceStatsResponse {
    completed_outcomes: u64,
    mapped_work_percentage: f64,
    recent_activity_count: u64,
}

#[derive(Deserialize)]
struct MetricsParams {
    workspace_id: Option<WorkspaceId>,
}

#[derive(Serialize)]
struct MetricsResponse {
    model_metrics: Option<MLModelMetrics>,
    system_metrics: SystemMetrics,
}

#[derive(Serialize)]
struct MLModelMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1_score: f64,
    total_predictions: u64,
    correct_predictions: u64,
    last_updated: DateTime<Utc>,
}

#[derive(Serialize)]
struct SystemMetrics {
    avg_response_time_ms: f64,
    requests_per_minute: f64,
    error_rate: f64,
}

#[derive(Deserialize)]
struct QueryMetricsRequest {
    workspace_id: WorkspaceId,
    metric_name: Option<String>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    tags: Option<HashMap<String, String>>,
    aggregation: Option<interstice_core::analytics::AggregatorType>,
    group_by: Option<Vec<String>>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct QueryMetricsResponse {
    results: Vec<MetricResult>,
}

#[derive(Serialize)]
struct MetricResult {
    metric_name: String,
    value: MetricValue,
    timestamp: DateTime<Utc>,
    tags: HashMap<String, String>,
}

#[derive(Deserialize)]
struct ExportMetricsRequest {
    workspace_id: WorkspaceId,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
   
    format: Option<ExportFormat>,
}

#[derive(Serialize)]
struct AnalyticsHealthResponse {
    status: String,
    components: Vec<ComponentHealth>,
}

#[derive(Serialize)]
struct ComponentHealth {
    name: String,
    healthy: bool,
    message: String,
}

#[derive(Deserialize)]
struct AnalyticsTimeRange {
    duration: Option<chrono::Duration>,
}

#[derive(Serialize)]
struct WorkspaceAnalyticsResponse {
    workspace_id: Uuid,
    time_range: TimeRange,
    artifact_stats: ArtifactStats,
    ml_stats: MLStats,
}

#[derive(Serialize)]
struct ArtifactStats {
    total_created: u64,
    by_platform: HashMap<String, u64>,
}

#[derive(Serialize)]
struct MLStats {
    avg_accuracy: f64,
    total_predictions: u64,
}

#[derive(Deserialize)]
struct InsightsParams {
    _insight_types: Option<Vec<String>>,
}

#[derive(Serialize)]
struct WorkspaceInsightsResponse {
    workspace_id: Uuid,
    insights: Vec<InsightResult>,
}

#[derive(Serialize)]
struct InsightResult {
    insight_type: String,
    title: String,
    description: String,
    confidence: f64,
    recommendations: Vec<String>,
}