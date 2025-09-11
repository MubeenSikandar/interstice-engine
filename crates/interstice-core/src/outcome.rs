//! # Outcome Module
//! 
//! Comprehensive outcome management for the INTERSTICE-ENGINE WorkOS.
//! Handles outcome creation, tracking, prediction, and lifecycle management.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::artifact::{Artifact, ArtifactType, AttendeeResponse, CIStatus, DeploymentStrategy, Environment, IssueState, MeetingType, PullRequestState, Severity};
use crate::error::CoreError;
use crate::storage::StorageBackend;
use crate::traits::{MLPredictor, OutcomePrediction};
use crate::types::{
    Context, Platform, Priority, UserId, WorkspaceId,
};

/// Result type for outcome operations
pub type OutcomeResult<T> = Result<T, OutcomeError>;

/// Outcome-specific error types
#[derive(Error, Debug)]
pub enum OutcomeError {
    #[error("Outcome not found: {0}")]
    NotFound(OutcomeId),
    
    #[error("Invalid outcome state transition: {0}")]
    InvalidStateTransition(String),
    
    #[error("Dependency cycle detected")]
    DependencyCycle,
    
    #[error("Parent outcome not found: {0}")]
    ParentNotFound(OutcomeId),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Storage error: {0}")]
    StorageError(#[from] CoreError),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Outcome identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeId(pub Uuid);

impl OutcomeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl Default for OutcomeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for OutcomeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Core outcome structure with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    /// Unique identifier
    pub id: OutcomeId,
    
    /// Workspace context
    pub workspace_id: WorkspaceId,
    
    /// Human-readable name
    pub name: String,
    
    /// Detailed description
    pub description: Option<String>,
    
    /// Current state of the outcome
    pub state: OutcomeState,
    
    /// Type classification
    pub outcome_type: OutcomeType,
    
    /// Priority level
    pub priority: Priority,
    
    /// Target metrics
    pub targets: Vec<OutcomeTarget>,
    
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    
    /// Parent outcome for hierarchical structures
    pub parent_id: Option<OutcomeId>,
    
    /// Child outcomes
    pub children: Vec<OutcomeId>,
    
    /// Dependencies on other outcomes
    pub dependencies: Vec<OutcomeId>,
    
    /// Assigned users
    pub assignees: Vec<UserId>,
    
    /// Owner/creator
    pub owner_id: UserId,
    
    /// Associated artifacts
    pub artifacts: Vec<Uuid>,
    
    /// Tags for categorization
    pub tags: HashSet<String>,
    
    /// Platform associations
    pub platforms: HashSet<Platform>,
    
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Target completion date
    pub due_date: Option<DateTime<Utc>>,
    
    /// Actual completion date
    pub completed_at: Option<DateTime<Utc>>,
    
    /// Estimated effort in hours
    pub estimated_hours: Option<f64>,
    
    /// Actual effort spent in hours
    pub actual_hours: Option<f64>,
    
    /// Business value score
    pub value_score: Option<f64>,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
    
    /// Automation level
    pub automation_level: AutomationLevel,
}

impl Outcome {
    /// Create a new outcome with basic information
    pub fn new(workspace_id: WorkspaceId, name: String, owner_id: UserId) -> Self {
        let now = Utc::now();
        Self {
            id: OutcomeId::new(),
            workspace_id,
            name,
            description: None,
            state: OutcomeState::Draft,
            outcome_type: OutcomeType::Task,
            priority: Priority::Medium,
            targets: Vec::new(),
            progress: 0.0,
            parent_id: None,
            children: Vec::new(),
            dependencies: Vec::new(),
            assignees: vec![owner_id.clone()],
            owner_id,
            artifacts: Vec::new(),
            tags: HashSet::new(),
            platforms: HashSet::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            due_date: None,
            completed_at: None,
            estimated_hours: None,
            actual_hours: None,
            value_score: None,
            risk_level: RiskLevel::Low,
            automation_level: AutomationLevel::Manual,
        }
    }
    
    /// Update progress based on targets
    pub fn update_progress(&mut self) {
        if self.targets.is_empty() {
            return;
        }
        
        let total_progress: f64 = self.targets.iter()
            .map(|t| t.calculate_progress() * t.weight)
            .sum();
        
        let total_weight: f64 = self.targets.iter()
            .map(|t| t.weight)
            .sum();
        
        self.progress = if total_weight > 0.0 {
            (total_progress / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        self.updated_at = Utc::now();
    }
    
    /// Check if outcome is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.state, OutcomeState::Completed | OutcomeState::Archived)
    }
    
    /// Check if outcome is blocked
    pub fn is_blocked(&self) -> bool {
        matches!(self.state, OutcomeState::Blocked)
    }
    
    /// Add a child outcome
    pub fn add_child(&mut self, child_id: OutcomeId) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
            self.updated_at = Utc::now();
        }
    }
    
    /// Add a dependency
    pub fn add_dependency(&mut self, dependency_id: OutcomeId) -> OutcomeResult<()> {
        if dependency_id == self.id {
            return Err(OutcomeError::ValidationError(
                "Cannot add self as dependency".to_string()
            ));
        }
        
        if !self.dependencies.contains(&dependency_id) {
            self.dependencies.push(dependency_id);
            self.updated_at = Utc::now();
        }
        
        Ok(())
    }
    
    /// Transition to a new state
    pub fn transition_state(&mut self, new_state: OutcomeState) -> OutcomeResult<()> {
        if !self.state.can_transition_to(&new_state) {
            return Err(OutcomeError::InvalidStateTransition(
                format!("{:?} -> {:?}", self.state, new_state)
            ));
        }
        
        self.state = new_state;
        self.updated_at = Utc::now();
        
        if matches!(new_state, OutcomeState::Completed) {
            self.completed_at = Some(Utc::now());
            self.progress = 1.0;
        }
        
        Ok(())
    }
}

/// Outcome state lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeState {
    Draft,
    Planning,
    Ready,
    InProgress,
    Review,
    Blocked,
    Completed,
    Cancelled,
    Archived,
}

impl OutcomeState {
    /// Check if transition to another state is valid
    pub fn can_transition_to(&self, target: &OutcomeState) -> bool {
        match (self, target) {
            // Draft can go to planning or cancelled
            (Self::Draft, Self::Planning | Self::Cancelled) => true,
            
            // Planning can go to ready, draft, or cancelled
            (Self::Planning, Self::Ready | Self::Draft | Self::Cancelled) => true,
            
            // Ready can go to in progress, planning, or cancelled
            (Self::Ready, Self::InProgress | Self::Planning | Self::Cancelled) => true,
            
            // In Progress can go to review, blocked, or cancelled
            (Self::InProgress, Self::Review | Self::Blocked | Self::Cancelled) => true,
            
            // Review can go to completed, in progress, or cancelled
            (Self::Review, Self::Completed | Self::InProgress | Self::Cancelled) => true,
            
            // Blocked can go to in progress or cancelled
            (Self::Blocked, Self::InProgress | Self::Cancelled) => true,
            
            // Completed can only go to archived
            (Self::Completed, Self::Archived) => true,
            
            // Cancelled can go to draft (restart) or archived
            (Self::Cancelled, Self::Draft | Self::Archived) => true,
            
            // Archived is terminal
            (Self::Archived, _) => false,
            
            // All other transitions are invalid
            _ => false,
        }
    }
}

/// Outcome type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeType {
    Strategic,
    Tactical,
    Operational,
    Project,
    Epic,
    Story,
    Task,
    Bug,
    Improvement,
    Research,
    Experiment,
}

impl OutcomeType {
    pub fn default_priority(&self) -> Priority {
        match self {
            Self::Strategic => Priority::Critical,
            Self::Bug => Priority::High,
            Self::Tactical | Self::Project | Self::Epic => Priority::High,
            Self::Story | Self::Task | Self::Improvement => Priority::Medium,
            Self::Operational | Self::Research | Self::Experiment => Priority::Low,
        }
    }
}

/// Measurable target for outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeTarget {
    pub id: Uuid,
    pub name: String,
    pub metric_type: MetricType,
    pub target_value: f64,
    pub current_value: f64,
    pub unit: String,
    pub weight: f64,
    pub threshold: Option<f64>,
}

impl OutcomeTarget {
    pub fn new(name: String, metric_type: MetricType, target_value: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            metric_type,
            target_value,
            current_value: 0.0,
            unit: String::new(),
            weight: 1.0,
            threshold: None,
        }
    }
    
    pub fn calculate_progress(&self) -> f64 {
        if self.target_value == 0.0 {
            return 0.0;
        }
        
        match self.metric_type {
            MetricType::Percentage | MetricType::Ratio => {
                (self.current_value / self.target_value).clamp(0.0, 1.0)
            }
            MetricType::Boolean => {
                if self.current_value >= self.target_value { 1.0 } else { 0.0 }
            }
            MetricType::Count | MetricType::Duration | MetricType::Custom => {
                (self.current_value / self.target_value).clamp(0.0, 1.0)
            }
        }
    }
}

/// Metric type for targets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    Count,
    Percentage,
    Duration,
    Boolean,
    Ratio,
    Custom,
}

/// Risk level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Automation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutomationLevel {
    Manual,
    SemiAutomated,
    Automated,
    Autonomous,
}

/// Outcome mapper for ML predictions
pub struct OutcomeMapper {
    ml_predictor: Option<Arc<dyn MLPredictor>>,
    storage: Arc<dyn StorageBackend>,
    cache: Arc<RwLock<HashMap<String, OutcomePrediction>>>,
}

impl OutcomeMapper {
    pub fn new(
        ml_predictor: Option<Arc<dyn MLPredictor>>,
        storage: Arc<dyn StorageBackend>,
    ) -> Self {
        Self {
            ml_predictor,
            storage,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Predict outcomes from artifacts
    #[instrument(skip(self, artifacts))]
    pub async fn predict(&self, artifacts: &[Artifact]) -> OutcomeResult<Vec<OutcomePrediction>> {
        // Check cache first
        let cache_key = self.generate_cache_key(artifacts);
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                debug!("Returning cached prediction");
                return Ok(vec![cached.clone()]);
            }
        }
        
        // Use ML predictor if available
        let predictions = if let Some(predictor) = &self.ml_predictor {
            predictor.predict_outcomes(artifacts).await
                .map_err(|e| OutcomeError::PredictionFailed(e.to_string()))?
        } else {
            self.fallback_predict(artifacts).await?
        };
        
        // Cache results
        if !predictions.is_empty() {
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, predictions[0].clone());
        }
        
        Ok(predictions)
    }
    
    /// Rule-based fallback prediction
    async fn fallback_predict(&self, artifacts: &[Artifact]) -> OutcomeResult<Vec<OutcomePrediction>> {
        let mut prediction_map: HashMap<String, PredictionAccumulator> = HashMap::new();
        
        for artifact in artifacts {
            let predictions_from_artifact = self.analyze_artifact_deep(artifact).await?;
            
            for pred in predictions_from_artifact {
                prediction_map.entry(pred.outcome_name.clone())
                    .and_modify(|acc| acc.merge(pred.clone()))
                    .or_insert_with(|| PredictionAccumulator::from(pred));
            }
        }
        
        // Convert accumulators to final predictions
        let mut predictions: Vec<OutcomePrediction> = prediction_map.into_values()
            .map(|acc| acc.finalize())
            .collect();
        
        // Merge similar predictions to reduce redundancy
        predictions = self.merge_similar_predictions(predictions);
        
        // Sort by confidence descending
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        // Take top predictions if too many
        predictions.truncate(10);
        
        Ok(predictions)
    }

    async fn analyze_artifact_deep(&self, artifact: &Artifact) -> OutcomeResult<Vec<OutcomePrediction>> {
        let mut predictions = Vec::new();
        
        match &artifact.artifact_type {
            ArtifactType::PullRequest { 
                number, title, state, files_changed, additions, deletions, 
                merged, draft, base_branch, head_branch, author, reviewers, 
                labels, merge_conflict, ci_status 
            } => {
                // Size-based predictions
                let change_size = *additions + *deletions;
                let size_category = match change_size {
                    0..=50 => "small",
                    51..=200 => "medium",
                    201..=500 => "large",
                    _ => "xlarge"
                };
                
                // Branch strategy analysis
                let is_hotfix = head_branch.contains("hotfix") || head_branch.contains("fix");
                let is_feature = head_branch.contains("feature") || head_branch.contains("feat");
                let is_release = base_branch.contains("release") || head_branch.contains("release");
                
                // CI/CD health
                let ci_health = match ci_status {
                    Some(CIStatus::Success) => 1.0,
                    Some(CIStatus::Running) => 0.7,
                    Some(CIStatus::Failed) => 0.3,
                    _ => 0.5,
                };
                
                // Review complexity
                let review_complexity = reviewers.len() as f64 * 0.15 + 
                    if *merge_conflict { 0.3 } else { 0.0 };
                
                // Generate contextual predictions
                if is_hotfix && matches!(state, PullRequestState::Merged) {
                    predictions.push(self.create_prediction(
                        "Critical Bug Resolution",
                        0.85 * ci_health,
                        format!("Hotfix PR #{} by {} merged to {} affecting {} files", number, author, base_branch, files_changed),
                        vec!["incident_resolution_time", "defect_escape_rate"],
                        0.9,
                        if labels.iter().any(|l| l.contains("critical")) { Priority::Critical } else { Priority::High }
                    ));
                }
                
                if is_feature && *files_changed > 10 {
                    predictions.push(self.create_prediction(
                        "Feature Development",
                        0.75 + (reviewers.len() as f32 * 0.05).min(0.2),
                        format!("{} change by {} with {} additions across {} files", size_category, author, additions, files_changed),
                        vec!["feature_completion_rate", "velocity_points"],
                        0.7,
                        Priority::Medium
                    ));
                }
                
                if *draft {
                    predictions.push(self.create_prediction(
                        "Work In Progress",
                        0.6,
                        format!("Draft PR '{}' by {} with {} reviewers assigned", title, author, reviewers.len()),
                        vec!["wip_limit", "cycle_time"],
                        0.4,
                        Priority::Low
                    ));
                }
                
                if labels.iter().any(|l| l.contains("refactor") || l.contains("tech-debt")) {
                    predictions.push(self.create_prediction(
                        "Technical Debt Reduction",
                        0.7 + review_complexity.min(0.25) as f32,
                        format!("Refactoring '{}' - {} lines across {} files", title, change_size, files_changed),
                        vec!["code_quality_score", "technical_debt_ratio"],
                        0.6,
                        Priority::Medium
                    ));
                }
                
                // Use merged and is_release for additional context
                if *merged && is_release {
                    predictions.push(self.create_prediction(
                        "Release Integration",
                        0.8,
                        format!("Release PR '{}' by {} successfully merged", title, author),
                        vec!["release_frequency", "deployment_success_rate"],
                        0.75,
                        Priority::High
                    ));
                }
            }
            
            ArtifactType::Issue { 
                id, title, state, priority, assignees, labels, story_points, 
                sprint, epic, blocked, blockers, time_estimate, time_spent 
            } => {
                // Effort analysis
                let effort_variance = if let (Some(estimate), Some(spent)) = (time_estimate, time_spent) {
                    (spent.num_hours() as f64 / estimate.num_hours() as f64).min(2.0)
                } else { 1.0 };
                
                // Team capacity
                let team_load = assignees.len() as f64;
                let is_distributed = assignees.len() > 1;
                
                // Sprint health
                let in_active_sprint = sprint.is_some();
                let has_epic = epic.is_some();
                
                // Blocker analysis
                let blocker_severity = if *blocked { 
                    0.3 + (blockers.len() as f64 * 0.1).min(0.4)
                } else { 0.0 };
                
                // Generate predictions based on state and context
                match state {
                    IssueState::InProgress if *blocked => {
                        predictions.push(self.create_prediction(
                            "Impediment Resolution Required",
                            0.9 - blocker_severity as f32,
                            format!("Issue '{}' blocked by {} dependencies", title, blockers.len()),
                            vec!["blocker_resolution_time", "dependency_health"],
                            0.8,
                            Priority::High
                        ));
                    }
                    IssueState::InProgress if effort_variance > 1.5 => {
                        predictions.push(self.create_prediction(
                            "Scope Creep Management",
                            0.75,
                            format!("Issue '{}' exceeding estimate by {:.0}%", title, (effort_variance - 1.0) * 100.0),
                            vec!["estimation_accuracy", "scope_change_rate"],
                            0.7,
                            Priority::Medium
                        ));
                    }
                    _ => {}
                }
                
                if let Some(points) = story_points {
                    if *points > 8 {
                        predictions.push(self.create_prediction(
                            "Epic Decomposition",
                            0.8,
                            format!("Large story '{}' ({} points) with {} assignees", title, points, team_load),
                            vec!["story_breakdown_ratio", "velocity_predictability"],
                            0.65,
                            Priority::Medium
                        ));
                    }
                }
                
                // Use labels for additional context
                if labels.iter().any(|l| l.contains("bug") || l.contains("defect")) {
                    predictions.push(self.create_prediction(
                        "Bug Resolution",
                        0.85,
                        format!("Bug issue #{} '{}' with labels: {}", id, title, labels.join(", ")),
                        vec!["bug_resolution_time", "defect_escape_rate"],
                        0.8,
                        Priority::High
                    ));
                }
                
                if has_epic && in_active_sprint {
                    predictions.push(self.create_prediction(
                        "Sprint Goal Progress",
                        0.7 + if is_distributed { 0.1 } else { 0.0 },
                        format!("Epic-linked sprint work for issue #{} with {} team members", id, assignees.len()),
                        vec!["sprint_goal_achievement", "epic_progress"],
                        0.6,
                        priority.to_types_priority()
                    ));
                }
            }
            
            ArtifactType::Meeting { 
                duration, attendees, organizer, meeting_type, recurring, 
                recording_available, action_items, decisions, .. 
            } => {
                let meeting_efficiency = (action_items.len() + decisions.len()) as f64 / 
                    duration.num_minutes() as f64 * 60.0;
                
                let attendance_rate = attendees.iter()
                    .filter(|a| a.response == AttendeeResponse::Accepted)
                    .count() as f64 / attendees.len().max(1) as f64;
                
                let is_strategic = matches!(meeting_type, 
                    MeetingType::Planning | MeetingType::Review | MeetingType::AllHands);
                
                if !decisions.is_empty() {
                    predictions.push(self.create_prediction(
                        "Strategic Decision Making",
                        0.85 * attendance_rate as f32,
                        format!("{} decisions made by {} with {:.0}% attendance", decisions.len(), organizer, attendance_rate * 100.0),
                        vec!["decision_velocity", "decision_quality_score"],
                        0.8,
                        if is_strategic { Priority::High } else { Priority::Medium }
                    ));
                }
                
                if action_items.len() > 3 {
                    predictions.push(self.create_prediction(
                        "Action Item Generation",
                        0.7 + meeting_efficiency.min(0.2) as f32,
                        format!("{} action items from {}-minute {} organized by {}", 
                            action_items.len(), duration.num_minutes(), meeting_type.name(), organizer),
                        vec!["action_completion_rate", "follow_through_score"],
                        0.65,
                        Priority::Medium
                    ));
                }
                
                if *recurring && attendance_rate < 0.5 {
                    predictions.push(self.create_prediction(
                        "Meeting Optimization Opportunity",
                        0.6,
                        format!("Recurring meeting organized by {} with low attendance ({:.0}%)", organizer, attendance_rate * 100.0),
                        vec!["meeting_effectiveness", "time_optimization"],
                        0.5,
                        Priority::Low
                    ));
                }
                
                // Use recording_available for knowledge management insights
                if *recording_available {
                    predictions.push(self.create_prediction(
                        "Knowledge Capture",
                        0.75,
                        format!("Meeting organized by {} with recording available for future reference", organizer),
                        vec!["knowledge_retention", "meeting_documentation"],
                        0.6,
                        Priority::Low
                    ));
                }
            }
            
            ArtifactType::Deployment { 
                environment, version, success, duration, rollback, 
                auto_deployed, affected_services, deployment_strategy, health_checks, .. 
            } => {
                let health_score = health_checks.iter()
                    .filter(|h| h.passed)
                    .count() as f64 / health_checks.len().max(1) as f64;
                
                let deployment_risk = match deployment_strategy {
                    DeploymentStrategy::BlueGreen => 0.2,
                    DeploymentStrategy::Canary => 0.3,
                    DeploymentStrategy::RollingUpdate => 0.4,
                    DeploymentStrategy::Recreate => 0.6,
                    DeploymentStrategy::Shadow => 0.1,
                };
                
                if *rollback {
                    predictions.push(self.create_prediction(
                        "Deployment Rollback Recovery",
                        0.95,
                        format!("Rollback of {} to {} affecting {} services", 
                            version, environment.name(), affected_services.len()),
                        vec!["mttr", "deployment_failure_rate"],
                        0.9,
                        Priority::Critical
                    ));
                } else if *success && matches!(environment, Environment::Production) {
                    let deployment_type = if *auto_deployed { "automated" } else { "manual" };
                    predictions.push(self.create_prediction(
                        "Successful Production Release",
                        (0.8 + health_score * 0.2 - deployment_risk) as f32,
                        format!("{} {} deployment via {} strategy", version, deployment_type, deployment_strategy.name()),
                        vec!["deployment_frequency", "change_failure_rate"],
                        0.85,
                        Priority::High
                    ));
                }
                
                if duration.num_minutes() > 30 {
                    let deployment_type = if *auto_deployed { "automated" } else { "manual" };
                    predictions.push(self.create_prediction(
                        "Deployment Performance Optimization",
                        0.65,
                        format!("Long {} deployment duration: {} minutes", deployment_type, duration.num_minutes()),
                        vec!["deployment_duration", "pipeline_efficiency"],
                        0.5,
                        Priority::Low
                    ));
                }
                
                // Use auto_deployed for automation insights
                if *auto_deployed {
                    predictions.push(self.create_enhanced_prediction(
                        artifact,
                        "Automated Deployment Success",
                        0.8,
                        format!("Automated deployment of {} to {} completed successfully", version, environment.name()),
                        Priority::Medium
                    ));
                }
            }
            
            ArtifactType::Alert { 
                severity, resolved, resolution_time, affected_services, 
                root_cause, escalation_level, .. 
            } => {
                let impact_score = affected_services.len() as f64 * 0.2 + 
                    *escalation_level as f64 * 0.1;
                
                if !resolved {
                    predictions.push(self.create_prediction(
                        "Active Incident Management",
                        match severity {
                            Severity::Critical => 0.95,
                            Severity::High => 0.85,
                            Severity::Medium => 0.7,
                            _ => 0.5,
                        },
                        format!("{:?} severity incident affecting {} services", severity, affected_services.len()),
                        vec!["mttr", "incident_rate"],
                        0.95,
                        severity.to_artifact_severity()
                    ));
                } else if let Some(resolution_duration) = resolution_time {
                    let resolution_quality = if root_cause.is_some() { 0.9 } else { 0.7 };
                    
                    predictions.push(self.create_prediction(
                        "Incident Resolution Complete",
                        resolution_quality,
                        format!("Resolved in {} minutes with{} root cause", 
                            resolution_duration.num_minutes(),
                            if root_cause.is_some() { "" } else { "out" }),
                        vec!["incident_resolution_time", "root_cause_identification_rate"],
                        impact_score.min(1.0),
                        Priority::Medium
                    ));
                }
            }
            
            _ => {
                // Handle other artifact types with enhanced analysis
                let outcome_name = format!("{} Processing", artifact.artifact_type.type_name());
                let base_confidence = 0.5 + (if let serde_json::Value::Object(map) = &artifact.metadata { map.len() } else { 0 } as f32 * 0.02).min(0.3);
                
                predictions.push(self.create_enhanced_prediction(
                    artifact,
                    &outcome_name,
                    base_confidence,
                    format!("Standard processing for {} artifact", artifact.artifact_type.type_name()),
                    Priority::Medium
                ));
            }
        }
        
        Ok(predictions)
    }

    fn create_prediction(
        &self,
        name: &str,
        confidence: f32,
        reasoning: String,
        targets: Vec<&str>,
        impact: f64,
        priority: Priority,
    ) -> OutcomePrediction {
        OutcomePrediction {
            outcome_id: OutcomeId::new().0,
            outcome_name: name.to_string(),
            confidence: confidence.min(1.0).max(0.0),
            reasoning: Some(reasoning),
            suggested_targets: targets.iter().map(|s| s.to_string()).collect(),
            estimated_impact: impact,
            recommended_priority: priority,
        }
    }
    
    fn create_enhanced_prediction(
        &self,
        artifact: &Artifact,
        name: &str,
        confidence: f32,
        base_reasoning: String,
        _priority: Priority,
    ) -> OutcomePrediction {
        let enhanced_reasoning = self.generate_reasoning(artifact, name);
        let suggested_targets = self.suggest_targets(&artifact.artifact_type);
        let estimated_impact = self.estimate_impact(&artifact.artifact_type);
        let recommended_priority = self.recommend_priority(&artifact.artifact_type);
        
        OutcomePrediction {
            outcome_id: OutcomeId::new().0,
            outcome_name: name.to_string(),
            confidence: confidence.min(1.0).max(0.0),
            reasoning: Some(format!("{} | {}", base_reasoning, enhanced_reasoning)),
            suggested_targets: suggested_targets.iter().map(|t| t.name.clone()).collect(),
            estimated_impact: estimated_impact,
            recommended_priority: recommended_priority,
        }
    }


    pub async fn store_predictions(
        &self,
        predictions: &[OutcomePrediction],
        artifacts: &[Artifact],
    ) -> OutcomeResult<()> {
        let prediction_record = crate::storage::PredictionRecord {
            id: Uuid::new_v4(),
            predictions: predictions.to_vec(),
            artifact_ids: artifacts.iter().map(|a| a.id).collect(),
            created_at: Utc::now(),
        };
        
        self.storage.store_prediction_record(prediction_record).await
            .map_err(|e| OutcomeError::StorageError(e))
    }
    
    /// Retrieve historical predictions for learning
    pub async fn get_historical_predictions(
        &self,
        since: DateTime<Utc>,
        limit: usize,
    ) -> OutcomeResult<Vec<crate::storage::PredictionRecord>> {
        self.storage.query_predictions(since, limit).await
            .map_err(|e| OutcomeError::StorageError(e))
    }
    
    /// Update prediction confidence based on feedback
    pub async fn update_prediction_confidence(
        &self,
        outcome_id: Uuid,
        actual_outcome: &str,
        was_accurate: bool,
    ) -> OutcomeResult<()> {
        let feedback = crate::storage::PredictionFeedback {
            prediction_id: outcome_id,
            actual_outcome: actual_outcome.to_string(),
            was_accurate,
            feedback_at: Utc::now(),
        };
        
        self.storage.store_prediction_feedback(feedback).await
            .map_err(|e| OutcomeError::StorageError(e))?;
        
        // Update cache if present
        if was_accurate {
            let mut cache = self.cache.write().await;
            for (_, pred) in cache.iter_mut() {
                if pred.outcome_id == outcome_id {
                    pred.confidence = (pred.confidence * 1.1).min(1.0);
                }
            }
        }
        
        Ok(())
    }
    
    fn generate_cache_key(&self, artifacts: &[Artifact]) -> String {
        let mut ids: Vec<String> = artifacts.iter()
            .map(|a| a.id.to_string())
            .collect();
        ids.sort();
        format!("predict:{}", ids.join(":"))
    }
    
    fn generate_reasoning(&self, artifact: &Artifact, outcome_type: &str) -> String {
        format!(
            "{} detected from {} on platform {} with {} confidence indicators",
            outcome_type,
            artifact.artifact_type.type_name(),
            artifact.platform,
            if let serde_json::Value::Object(map) = &artifact.metadata { map.len() } else { 0 }
        )
    }
    
    fn suggest_targets(&self, artifact_type: &ArtifactType) -> Vec<OutcomeTarget> {
        match artifact_type {
            ArtifactType::PullRequest { .. } => vec![
                OutcomeTarget::new("Code Coverage".to_string(), MetricType::Percentage, 80.0),
                OutcomeTarget::new("Review Approval".to_string(), MetricType::Boolean, 1.0),
            ],
            ArtifactType::Issue { .. } => vec![
                OutcomeTarget::new("Resolution Time".to_string(), MetricType::Duration, 48.0),
                OutcomeTarget::new("Customer Satisfaction".to_string(), MetricType::Percentage, 90.0),
            ],
            _ => vec![
                OutcomeTarget::new("Completion".to_string(), MetricType::Boolean, 1.0),
            ],
        }
    }
    
    fn estimate_impact(&self, artifact_type: &ArtifactType) -> f64 {
        match artifact_type {
            ArtifactType::Deployment { environment, .. } => {
                match environment {
                    Environment::Production => 0.9,
                    Environment::Staging => 0.6,
                    _ => 0.3,
                }
            }
            ArtifactType::Alert { severity, .. } => {
                match severity {
                    Severity::Critical => 1.0,
                    Severity::High => 0.8,
                    _ => 0.5,
                }
            }
            _ => 0.5,
        }
    }
    
    fn recommend_priority(&self, artifact_type: &ArtifactType) -> Priority {
        match artifact_type {
            ArtifactType::Alert { severity, .. } => {
                match severity {
                    Severity::Critical => Priority::Critical,
                    Severity::High => Priority::High,
                    _ => Priority::Medium,
                }
            }
            ArtifactType::Issue { id, .. } => {
                if id.contains("CRITICAL") || id.contains("BLOCKER") {
                    Priority::Critical
                } else if id.contains("HIGH") {
                    Priority::High
                } else if id.contains("LOW") {
                    Priority::Low
                } else {
                    Priority::Medium
                }
            }
            _ => Priority::Medium,
        }
    }
    
    fn merge_similar_predictions(&self, predictions: Vec<OutcomePrediction>) -> Vec<OutcomePrediction> {
        let mut merged: HashMap<String, OutcomePrediction> = HashMap::new();
        
        for pred in predictions {
            merged.entry(pred.outcome_name.clone())
                .and_modify(|existing| {
                    existing.confidence = existing.confidence.max(pred.confidence);
                    if pred.confidence > existing.confidence {
                        existing.reasoning = pred.reasoning.clone();
                    }
                })
                .or_insert(pred);
        }
        
        merged.into_values().collect()
    }
}

struct PredictionAccumulator {
    outcome_name: String,
    total_confidence: f32,
    count: usize,
    reasonings: Vec<String>,
    all_targets: HashSet<String>,
    max_impact: f64,
    highest_priority: Priority,
}

impl PredictionAccumulator {
    fn from(pred: OutcomePrediction) -> Self {
        Self {
            outcome_name: pred.outcome_name,
            total_confidence: pred.confidence,
            count: 1,
            reasonings: pred.reasoning.into_iter().collect(),
            all_targets: pred.suggested_targets.into_iter().collect(),
            max_impact: pred.estimated_impact,
            highest_priority: pred.recommended_priority,
        }
    }
    
    fn merge(&mut self, pred: OutcomePrediction) {
        self.total_confidence += pred.confidence;
        self.count += 1;
        if let Some(reasoning) = pred.reasoning {
            self.reasonings.push(reasoning);
        }
        self.all_targets.extend(pred.suggested_targets);
        self.max_impact = self.max_impact.max(pred.estimated_impact);
        if pred.recommended_priority < self.highest_priority {
            self.highest_priority = pred.recommended_priority;
        }
    }
    
    fn finalize(self) -> OutcomePrediction {
        OutcomePrediction {
            outcome_id: OutcomeId::new().0,
            outcome_name: self.outcome_name,
            confidence: (self.total_confidence / self.count as f32).min(1.0),
            reasoning: Some(self.reasonings.join("; ")),
            suggested_targets: self.all_targets.into_iter().collect(),
            estimated_impact: self.max_impact,
            recommended_priority: self.highest_priority,
        }
    }
}

/// Outcome manager for CRUD operations
pub struct OutcomeManager {
    storage: Arc<dyn StorageBackend>,
    mapper: Arc<OutcomeMapper>,
    outcomes: Arc<RwLock<HashMap<OutcomeId, Outcome>>>,
}

impl OutcomeManager {
    pub fn new(
        storage: Arc<dyn StorageBackend>,
        ml_predictor: Option<Arc<dyn MLPredictor>>,
    ) -> Self {
        let mapper = Arc::new(OutcomeMapper::new(ml_predictor, storage.clone()));
        
        Self {
            storage,
            mapper,
            outcomes: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Create a new outcome
    pub async fn create_outcome(&self, outcome: Outcome) -> OutcomeResult<OutcomeId> {
        // Validate outcome
        self.validate_outcome(&outcome).await?;
        
        // Check for dependency cycles
        if !outcome.dependencies.is_empty() {
            self.check_dependency_cycle(&outcome).await?;
        }
        
        let id = outcome.id;
        
        // Store in memory
        self.outcomes.write().await.insert(id, outcome.clone());
        
        // Persist to storage
        self.storage.store_outcome(outcome).await
            .map_err(|e| OutcomeError::StorageError(e))?;
        
        
        
        info!("Outcome created: {}", id);
        
        Ok(id)
    }
    
    /// Get an outcome by ID
    pub async fn get_outcome(&self, id: OutcomeId) -> OutcomeResult<Outcome> {
        // Check cache first
        {
            let outcomes = self.outcomes.read().await;
            if let Some(outcome) = outcomes.get(&id) {
                return Ok(outcome.clone());
            }
        }
        
        // Load from storage
        let outcome = self.storage.get_outcome(id).await
            .map_err(|e| OutcomeError::StorageError(e))?
            .ok_or(OutcomeError::NotFound(id))?;
        
        // Update cache
        self.outcomes.write().await.insert(id, outcome.clone());
        
        Ok(outcome)
    }
    
    /// Update an existing outcome
    pub async fn update_outcome(&self, mut outcome: Outcome) -> OutcomeResult<()> {
        // Validate changes
        self.validate_outcome(&outcome).await?;
        
        outcome.updated_at = Utc::now();
        
        // Update cache
        self.outcomes.write().await.insert(outcome.id, outcome.clone());
        
        // Persist changes
        self.storage.update_outcome(outcome.clone()).await
            .map_err(|e| OutcomeError::StorageError(e))?;
        
        info!("Outcome updated: {}", outcome.id);
        
        Ok(())
    }
    
    /// Delete an outcome
    pub async fn delete_outcome(&self, id: OutcomeId) -> OutcomeResult<()> {
        // Check if outcome exists
        let outcome = self.get_outcome(id).await?;
        
        // Check for dependent outcomes
        if !outcome.children.is_empty() {
            return Err(OutcomeError::ValidationError(
                "Cannot delete outcome with children".to_string()
            ));
        }
        
        // Remove from cache
        self.outcomes.write().await.remove(&id);
        
        // Remove from storage
        self.storage.delete_outcome(id).await
            .map_err(|e| OutcomeError::StorageError(e))?;
        
        info!("Outcome deleted: {}", id);
        
        Ok(())
    }
    
    /// Validate outcome data
    async fn validate_outcome(&self, outcome: &Outcome) -> OutcomeResult<()> {
        // Check required fields
        if outcome.name.is_empty() {
            return Err(OutcomeError::ValidationError(
                "Outcome name cannot be empty".to_string()
            ));
        }
        
        // Validate progress
        if !(0.0..=1.0).contains(&outcome.progress) {
            return Err(OutcomeError::ValidationError(
                "Progress must be between 0 and 1".to_string()
            ));
        }
        
        // Validate parent relationship
        if let Some(parent_id) = outcome.parent_id {
            if parent_id == outcome.id {
                return Err(OutcomeError::ValidationError(
                    "Outcome cannot be its own parent".to_string()
                ));
            }
            
            // Check parent exists
            self.get_outcome(parent_id).await
                .map_err(|_| OutcomeError::ParentNotFound(parent_id))?;
        }
        
        // Validate dates
        if let Some(due_date) = outcome.due_date {
            if due_date < outcome.created_at {
                return Err(OutcomeError::ValidationError(
                    "Due date cannot be before creation date".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    /// Check for dependency cycles
    async fn check_dependency_cycle(&self, outcome: &Outcome) -> OutcomeResult<()> {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        
        self.detect_cycle_dfs(outcome.id, &mut visited, &mut stack).await
    }
    
    /// DFS helper for cycle detection
    async fn detect_cycle_dfs(
        &self,
        outcome_id: OutcomeId,
        visited: &mut HashSet<OutcomeId>,
        stack: &mut HashSet<OutcomeId>,
    ) -> OutcomeResult<()> {
        if stack.contains(&outcome_id) {
            return Err(OutcomeError::DependencyCycle);
        }
        
        if visited.contains(&outcome_id) {
            return Ok(());
        }
        
        visited.insert(outcome_id);
        stack.insert(outcome_id);
        
        if let Ok(outcome) = self.get_outcome(outcome_id).await {
            for dep_id in &outcome.dependencies {
                Box::pin(self.detect_cycle_dfs(*dep_id, visited, stack)).await?;
            }
        }
        
        stack.remove(&outcome_id);
        Ok(())
    }
    
    /// Get outcomes by workspace
    pub async fn get_workspace_outcomes(
        &self,
        workspace_id: WorkspaceId,
        filters: Option<OutcomeFilters>,
    ) -> OutcomeResult<Vec<Outcome>> {
        let outcomes = self.storage.query_outcomes(workspace_id, filters).await
            .map_err(|e| OutcomeError::StorageError(e))?;
        
        Ok(outcomes)
    }
    
    /// Calculate outcome hierarchy
    pub async fn get_outcome_hierarchy(
        &self,
        root_id: OutcomeId,
    ) -> OutcomeResult<OutcomeHierarchy> {
        let root = self.get_outcome(root_id).await?;
        let children = self.build_hierarchy_tree(root_id, 0, 5).await?;
        
        let total_progress = self.calculate_hierarchical_progress(&root, &children);
        let depth = self.calculate_hierarchy_depth(&children);
        
        Ok(OutcomeHierarchy {
            root: root.clone(),
            children,
            total_progress,
            depth,
        })
    }
    
    async fn build_hierarchy_tree(
        &self,
        parent_id: OutcomeId,
        current_depth: usize,
        max_depth: usize,
    ) -> OutcomeResult<Vec<OutcomeNode>> {
        if current_depth >= max_depth {
            return Ok(Vec::new());
        }
        
        let parent = self.get_outcome(parent_id).await?;
        let mut nodes = Vec::new();
        
        for child_id in &parent.children {
            if let Ok(child) = self.get_outcome(*child_id).await {
                let sub_children = Box::pin(self.build_hierarchy_tree(
                    *child_id,
                    current_depth + 1,
                    max_depth
                )).await?;
                
                nodes.push(OutcomeNode {
                    outcome: child,
                    children: sub_children,
                });
            }
        }
        
        Ok(nodes)
    }
    
    fn calculate_hierarchical_progress(
        &self,
        root: &Outcome,
        children: &[OutcomeNode],
    ) -> f64 {
        if children.is_empty() {
            return root.progress;
        }
        
        let child_progress: f64 = children.iter()
            .map(|node| {
                let node_progress = if node.children.is_empty() {
                    node.outcome.progress
                } else {
                    self.calculate_hierarchical_progress(&node.outcome, &node.children)
                };
                node_progress / children.len() as f64
            })
            .sum();
        
        // Weight: 30% parent, 70% children
        root.progress * 0.3 + child_progress * 0.7
    }
    
    fn calculate_hierarchy_depth(&self, nodes: &[OutcomeNode]) -> usize {
        if nodes.is_empty() {
            return 0;
        }
        
        nodes.iter()
            .map(|node| 1 + self.calculate_hierarchy_depth(&node.children))
            .max()
            .unwrap_or(0)
    }
    
    /// Predict outcomes from artifacts
    pub async fn predict_from_artifacts(
        &self,
        artifacts: &[Artifact],
        context: &Context,
    ) -> OutcomeResult<Vec<SuggestedOutcome>> {
        let predictions = self.mapper.predict(artifacts).await?;
        
        let mut suggestions = Vec::new();
        for pred in predictions {
            let outcome = Outcome::new(
                context.workspace_id,
                pred.outcome_name.clone(),
                context.user_id.clone().unwrap_or_else(|| UserId::from("system")),
            );
            
            suggestions.push(SuggestedOutcome {
                outcome,
                prediction: pred,
                artifacts: artifacts.iter().map(|a| a.id).collect(),
                created_at: Utc::now(),
            });
        }
        
        Ok(suggestions)
    }
}

/// Outcome query filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OutcomeFilters {
    pub states: Option<Vec<OutcomeState>>,
    pub types: Option<Vec<OutcomeType>>,
    pub priorities: Option<Vec<Priority>>,
    pub assignees: Option<Vec<UserId>>,
    pub parent_id: Option<OutcomeId>,
    pub tags: Option<Vec<String>>,
    pub platforms: Option<Vec<Platform>>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub due_after: Option<DateTime<Utc>>,
    pub due_before: Option<DateTime<Utc>>,
    pub min_progress: Option<f64>,
    pub max_progress: Option<f64>,
}

/// Outcome hierarchy structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeHierarchy {
    pub root: Outcome,
    pub children: Vec<OutcomeNode>,
    pub total_progress: f64,
    pub depth: usize,
}

/// Node in outcome tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeNode {
    pub outcome: Outcome,
    pub children: Vec<OutcomeNode>,
}

/// Suggested outcome from predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedOutcome {
    pub outcome: Outcome,
    pub prediction: OutcomePrediction,
    pub artifacts: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
}

/// Outcome statistics
#[derive(Debug, Clone, Serialize)]
pub struct OutcomeStatistics {
    pub total_outcomes: usize,
    pub completed_outcomes: usize,
    pub in_progress_outcomes: usize,
    pub blocked_outcomes: usize,
    pub average_completion_time: Duration,
    pub completion_rate: f64,
    pub average_progress: f64,
    pub overdue_count: usize,
    pub by_priority: HashMap<Priority, usize>,
    pub by_type: HashMap<OutcomeType, usize>,
    pub by_state: HashMap<OutcomeState, usize>,
}

impl OutcomeStatistics {
    pub async fn calculate(
        outcomes: &[Outcome],
    ) -> Self {
        let total = outcomes.len();
        let completed = outcomes.iter().filter(|o| o.is_complete()).count();
        let in_progress = outcomes.iter()
            .filter(|o| matches!(o.state, OutcomeState::InProgress))
            .count();
        let blocked = outcomes.iter().filter(|o| o.is_blocked()).count();
        
        let completion_times: Vec<Duration> = outcomes.iter()
            .filter_map(|o| {
                o.completed_at.map(|completed| {
                    completed - o.created_at
                })
            })
            .collect();
        
        let avg_completion_time = if !completion_times.is_empty() {
            let total_seconds: i64 = completion_times.iter()
                .map(|d| d.num_seconds())
                .sum();
            Duration::seconds(total_seconds / completion_times.len() as i64)
        } else {
            Duration::seconds(0)
        };
        
        let now = Utc::now();
        let overdue = outcomes.iter()
            .filter(|o| {
                !o.is_complete() && o.due_date.map(|d| d < now).unwrap_or(false)
            })
            .count();
        
        let avg_progress = if total > 0 {
            outcomes.iter().map(|o| o.progress).sum::<f64>() / total as f64
        } else {
            0.0
        };
        
        let mut by_priority = HashMap::new();
        let mut by_type = HashMap::new();
        let mut by_state = HashMap::new();
        
        for outcome in outcomes {
            *by_priority.entry(outcome.priority).or_insert(0) += 1;
            *by_type.entry(outcome.outcome_type).or_insert(0) += 1;
            *by_state.entry(outcome.state).or_insert(0) += 1;
        }
        
        Self {
            total_outcomes: total,
            completed_outcomes: completed,
            in_progress_outcomes: in_progress,
            blocked_outcomes: blocked,
            average_completion_time: avg_completion_time,
            completion_rate: if total > 0 { completed as f64 / total as f64 } else { 0.0 },
            average_progress: avg_progress,
            overdue_count: overdue,
            by_priority,
            by_type,
            by_state,
        }
    }
}

// Additional storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictionRecord {
    id: Uuid,
    predictions: Vec<OutcomePrediction>,
    artifact_ids: Vec<Uuid>,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PredictionFeedback {
    prediction_id: Uuid,
    actual_outcome: String,
    was_accurate: bool,
    feedback_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_outcome_creation() {
        let workspace_id = WorkspaceId::new();
        let user_id = UserId::from("test_user");
        
        let outcome = Outcome::new(
            workspace_id,
            "Test Outcome".to_string(),
            user_id.clone(),
        );
        
        assert_eq!(outcome.workspace_id, workspace_id);
        assert_eq!(outcome.name, "Test Outcome");
        assert_eq!(outcome.owner_id, user_id);
        assert_eq!(outcome.state, OutcomeState::Draft);
        assert_eq!(outcome.progress, 0.0);
    }
    
    #[test]
    fn test_state_transitions() {
        let mut outcome = Outcome::new(
            WorkspaceId::new(),
            "Test".to_string(),
            UserId::from("user"),
        );
        
        // Valid transitions
        assert!(outcome.transition_state(OutcomeState::Planning).is_ok());
        assert_eq!(outcome.state, OutcomeState::Planning);
        
        assert!(outcome.transition_state(OutcomeState::Ready).is_ok());
        assert_eq!(outcome.state, OutcomeState::Ready);
        
        // Invalid transition
        assert!(outcome.transition_state(OutcomeState::Archived).is_err());
    }
    
    #[test]
    fn test_progress_calculation() {
        let mut target = OutcomeTarget::new(
            "Test Target".to_string(),
            MetricType::Percentage,
            100.0,
        );
        
        target.current_value = 50.0;
        assert_eq!(target.calculate_progress(), 0.5);
        
        target.current_value = 100.0;
        assert_eq!(target.calculate_progress(), 1.0);
        
        target.current_value = 150.0;
        assert_eq!(target.calculate_progress(), 1.0); // Clamped to 1.0
    }
    
    #[test]
    fn test_dependency_management() {
        let mut outcome = Outcome::new(
            WorkspaceId::new(),
            "Test".to_string(),
            UserId::from("user"),
        );
        
        let dep_id = OutcomeId::new();
        assert!(outcome.add_dependency(dep_id).is_ok());
        assert!(outcome.dependencies.contains(&dep_id));
        
        // Cannot add self as dependency
        assert!(outcome.add_dependency(outcome.id).is_err());
    }
    
    #[tokio::test]
    async fn test_outcome_statistics() {
        let mut outcomes = vec![];
        
        for i in 0..10 {
            let mut outcome = Outcome::new(
                WorkspaceId::new(),
                format!("Outcome {}", i),
                UserId::from("user"),
            );
            
            if i < 3 {
                outcome.state = OutcomeState::Completed;
                outcome.progress = 1.0;
            } else if i < 6 {
                outcome.state = OutcomeState::InProgress;
                outcome.progress = 0.5;
            } else {
                outcome.state = OutcomeState::Planning;
                outcome.progress = 0.1;
            }
            
            outcomes.push(outcome);
        }
        
        let stats = OutcomeStatistics::calculate(&outcomes).await;
        
        assert_eq!(stats.total_outcomes, 10);
        assert_eq!(stats.completed_outcomes, 3);
        assert_eq!(stats.in_progress_outcomes, 3);
        assert_eq!(stats.completion_rate, 0.3);
    }
}
