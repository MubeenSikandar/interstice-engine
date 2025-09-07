use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json, Router,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use interstice_core::{
    analytics::{ExportFormat, MetricEvent, MetricQuery},
    types::{MetricValue, TimeRange, WorkspaceId},
};
use crate::AppState;

#[derive(Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    page: u32,
    #[serde(default = "default_limit")]
    limit: u32,
}

fn default_page() -> u32 { 1 }
fn default_limit() -> u32 { 20 }

pub fn api_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Workspace routes
        .route("/workspaces", get(list_workspaces).post(create_workspace))
        .route("/workspaces/:id", get(get_workspace).put(update_workspace).delete(delete_workspace))
        
        // Artifact routes
        .route("/artifacts", get(list_artifacts).post(create_artifact))
        .route("/artifacts/:id", get(get_artifact))
        
        // Outcome routes
        .route("/outcomes", get(list_outcomes).post(create_outcome))
        .route("/outcomes/:id", get(get_outcome))
        .route("/outcomes/:id/predict", post(predict_outcomes))
        
        // Analytics routes
        .route("/analytics/dashboard", get(get_dashboard))
        .route("/analytics/metrics", get(get_metrics))
        .route("/analytics/metrics/query", post(query_metrics))
        .route("/analytics/metrics/export", post(export_metrics))
        .route("/analytics/health", get(analytics_health))
        .route("/analytics/workspaces/:id/stats", get(get_workspace_analytics))
        .route("/analytics/workspaces/:id/insights", get(get_workspace_insights))
}

// Workspace handlers
async fn list_workspaces(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<WorkspaceResponse>>, StatusCode> {
    // TODO: Implement
    Ok(Json(vec![]))
}

async fn create_workspace(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateWorkspaceRequest>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn get_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn update_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpdateWorkspaceRequest>,
) -> Result<Json<WorkspaceResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn delete_workspace(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, StatusCode> {
    // TODO: Implement
    Ok(StatusCode::NO_CONTENT)
}

// Artifact handlers
async fn list_artifacts(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<ArtifactResponse>>, StatusCode> {
    // TODO: Implement
    Ok(Json(vec![]))
}

async fn create_artifact(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateArtifactRequest>,
) -> Result<Json<ArtifactResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn get_artifact(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ArtifactResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

// Outcome handlers
async fn list_outcomes(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> Result<Json<Vec<OutcomeResponse>>, StatusCode> {
    // TODO: Implement
    Ok(Json(vec![]))
}

async fn create_outcome(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateOutcomeRequest>,
) -> Result<Json<OutcomeResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn get_outcome(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<OutcomeResponse>, StatusCode> {
    // TODO: Implement
    Err(StatusCode::NOT_IMPLEMENTED)
}

async fn predict_outcomes(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, StatusCode> {
    // TODO: Implement ML prediction
    Err(StatusCode::NOT_IMPLEMENTED)
}

// Analytics handlers
async fn get_dashboard(
    State(state): State<Arc<AppState>>,
    Query(params): Query<DashboardParams>,
) -> Result<Json<DashboardResponse>, StatusCode> {
    let workspace_id = params.workspace_id.unwrap_or_else(|| WorkspaceId::new());
    
    if let Some(analytics) = &state.analytics {
        // Get basic workspace stats
        let stats = match state.engine().get_workspace_stats(workspace_id).await {
            Ok(stats) => stats,
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };
        
        Ok(Json(DashboardResponse {
            total_artifacts: stats.total_artifacts,
            total_outcomes: stats.total_outcomes,
            recent_activity: vec![], // Simplified for now
            workspace_stats: Some(WorkspaceStatsResponse {
                completed_outcomes: stats.completed_outcomes,
                mapped_work_percentage: stats.mapped_work_percentage,
                recent_activity_count: stats.recent_activity,
            }),
        }))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn get_metrics(
    State(state): State<Arc<AppState>>,
    Query(params): Query<MetricsParams>,
) -> Result<Json<MetricsResponse>, StatusCode> {
    if let Some(_analytics) = &state.analytics {
        let workspace_id = params.workspace_id.unwrap_or_else(|| WorkspaceId::new());
        
        // Get ML model metrics if available
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
    if let Some(analytics) = &state.analytics {
        let query = MetricQuery {
            workspace_id: Some(payload.workspace_id),
            user_id: None,
            metric_ids: None,
           
            time_range: None,
            tags: payload.tags.map(|h| h.into_keys().collect()),
            aggregation: Some(payload.aggregation.unwrap_or(interstice_core::analytics::AggregatorType::Count)),
            group_by: Some(payload.group_by.unwrap_or_default()),
            outcome_ids: None,
            sort_by: None,
            limit: payload.limit,
        };
        
        match analytics.query_metrics(query).await {
            Ok(query_result) => {
                let results = query_result.metrics.into_iter()
                    .map(|m| MetricResult {
                        metric_name: m.metric_id,
                        value: m.value,
                        timestamp: m.timestamp,
                        tags: HashMap::new(),
                    })
                    .collect();
                
                Ok(Json(QueryMetricsResponse { results }))
            }
            Err(e) => {
                tracing::error!("Failed to query metrics: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

async fn export_metrics(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ExportMetricsRequest>,
) -> Result<Json<ExportResponse>, StatusCode> {
    if let Some(analytics) = &state.analytics {
        let query = MetricQuery {
            workspace_id: Some(payload.workspace_id),
            user_id: None,
            metric_ids: None,
            time_range: None,
            tags: payload.tags.map(|h| h.into_keys().collect()),
            aggregation: Some(payload.aggregation.unwrap_or(interstice_core::analytics::AggregatorType::Count)),
            group_by: Some(payload.group_by.unwrap_or_default()),
            outcome_ids: None,
            sort_by: None,
            limit: payload.limit,
        };
        
        let format = payload.format.unwrap_or(ExportFormat::Csv);
        let format_clone = format.clone();
        
        match analytics.export_analytics(payload.workspace_id, format, TimeRange {
            start: payload.start_time.unwrap_or_else(|| Utc::now() - chrono::Duration::days(7)),
            end: payload.end_time.unwrap_or_else(|| Utc::now()),
        }).await {
            Ok(export_data) => {
                Ok(Json(ExportResponse {
                    download_url: format!("/api/v1/analytics/downloads/{}", uuid::Uuid::new_v4()),
                    file_name: format!("analytics_export_{}.csv", Utc::now().format("%Y%m%d_%H%M%S")),
                    format: format_clone,
                    expires_at: Utc::now() + chrono::Duration::hours(24),
                }))
            }
            Err(e) => {
                tracing::error!("Failed to export metrics: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
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
    let workspace_id = WorkspaceId::from_uuid(id);
    
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
}

#[derive(Deserialize)]
struct CreateOutcomeRequest {
    name: String,
    description: Option<String>,
}

#[derive(Serialize)]
struct OutcomeResponse {
    id: Uuid,
    name: String,
    description: Option<String>,
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
    metric_name: Option<String>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
    tags: Option<HashMap<String, String>>,
    aggregation: Option<interstice_core::analytics::AggregatorType>,
    group_by: Option<Vec<String>>,
    limit: Option<usize>,
    format: Option<ExportFormat>,
}

#[derive(Serialize)]
struct ExportResponse {
    download_url: String,
    file_name: String,
    format: ExportFormat,
    expires_at: DateTime<Utc>,
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
    insight_types: Option<Vec<String>>,
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