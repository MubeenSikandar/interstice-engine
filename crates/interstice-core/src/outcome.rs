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

use crate::artifact::{Artifact, ArtifactType, Environment, Severity, TaskStatus};
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
        let mut predictions = Vec::new();
        let mut confidence_boost = 0.0;
        
        for artifact in artifacts {
            // Analyze artifact patterns
            let (outcome_type, base_confidence) = match &artifact.artifact_type {
                ArtifactType::PullRequest { number, title, state, files_changed, additions, deletions, merged, draft, base_branch, head_branch, author, reviewers, labels, merge_conflict, ci_status } => {
                    let size_factor = (*number as f64).min(10.0) / 10.0;
                    
                    if *number > 1000 {
                        ("Major Code Refactoring", 0.85 * size_factor)
                    } else if *number > 100 {
                        ("Feature Implementation", 0.75 * size_factor)
                    } else {
                        ("Code Quality Improvement", 0.65)
                    }
                }
                
                ArtifactType::Issue { id, title, state, priority, assignees, labels, story_points, sprint, epic, blocked, blockers, time_estimate, time_spent } => {
                    let priority_factor = if id.contains("CRITICAL") || id.contains("BLOCKER") {
                        0.95
                    } else if id.contains("HIGH") {
                        0.85
                    } else if id.contains("MEDIUM") {
                        0.70
                    } else {
                        0.60
                    };
                    
                    if id.contains("BUG") {
                        ("Bug Resolution", priority_factor)
                    } else if id.contains("FEATURE") {
                        ("Feature Request Implementation", priority_factor * 0.9)
                    } else {
                        ("Task Completion", priority_factor * 0.8)
                    }
                }
                
                ArtifactType::Message { id, channel, thread_id, author, content, mentions, attachments, reactions, sentiment, intent, is_edited, reply_count } => {
                    if content.contains("?") {
                        ("Knowledge Sharing", 0.60)
                    } else if content.contains("decision") || content.contains("decide") {
                        ("Decision Making", 0.80)
                    } else if content.contains("action") || content.contains("todo") {
                        ("Action Item Creation", 0.75)
                    } else {
                        ("Communication", 0.50)
                    }
                }
                
                ArtifactType::Document { id, title, doc_type, url, author, collaborators, word_count, last_modified, version, is_template, access_level } => {
                    if title.to_lowercase().contains("design") {
                        ("Design Documentation", 0.85)
                    } else if title.to_lowercase().contains("requirement") {
                        ("Requirements Definition", 0.90)
                    } else if title.to_lowercase().contains("technical") {
                        ("Technical Documentation", 0.80)
                    } else {
                        ("Documentation", 0.70)
                    }
                }
                
                ArtifactType::Meeting { duration, attendees, .. } => {
                    let attendee_factor = (attendees.len() as f64 / 10.0).min(1.0);
                    let duration_factor = (duration.num_minutes() as f64 / 60.0).min(1.0);
                    
                    ("Team Alignment", 0.65 * attendee_factor * duration_factor)
                }
                
                ArtifactType::Task { status, .. } => {
                    match status {
                        &TaskStatus::Done => ("Task Completion", 0.95),
                        &TaskStatus::InProgress => ("Work Progress", 0.70),
                        _ => ("Task Management", 0.60),
                    }
                }
                
                ArtifactType::Review { approved, changes_requested, .. } => {
                    if *approved {
                        ("Quality Assurance", 0.90)
                    } else if *changes_requested > 0 {
                        ("Code Review Feedback", 0.75)
                    } else {
                        ("Review Process", 0.65)
                    }
                }
                
                ArtifactType::Deployment { success, environment, .. } => {
                    let env_factor = match environment {
                        &Environment::Production => 1.0,
                        &Environment::Staging => 0.8,
                        _ => 0.6,
                    };
                    
                    if *success {
                        ("Successful Deployment", 0.95 * env_factor)
                    } else {
                        ("Deployment Attempt", 0.40 * env_factor)
                    }

                }
                
                ArtifactType::Metric {   .. } => {
                    ("Performance Tracking", 0.70)
                }
                
                ArtifactType::Alert { severity, resolved, .. } => {
                    let severity_factor = match severity {
                        Severity::Critical => 0.95,
                        Severity::High => 0.85,
                        _ => 0.70,
                    };
                    
                    if *resolved {
                        ("Incident Resolution", severity_factor)
                    } else {
                        ("Incident Response", severity_factor * 0.7)
                    }
                }
                
                ArtifactType::Custom { category, .. } => {
                    (category.as_str(), 0.50)
                }
                
                ArtifactType::Commit { .. } => {
                    ("Code Development", 0.70)
                }
                
                ArtifactType::Design { .. } => {
                    ("Design Work", 0.80)
                }
                
                ArtifactType::TestResult { .. } => {
                    ("Quality Assurance", 0.75)
                }
            };
            
            // Boost confidence based on artifact recency
            let age = Utc::now().signed_duration_since(artifact.created_at);
            if age.num_hours() < 24 {
                confidence_boost = 0.1;
            } else if age.num_days() < 7 {
                confidence_boost = 0.05;
            }
            
            predictions.push(OutcomePrediction {
                outcome_id: OutcomeId::new().0,
                outcome_name: outcome_type.to_string(),
                confidence: (base_confidence + confidence_boost).min(1.0) as f32,
                reasoning: Some(self.generate_reasoning(artifact, outcome_type)),
                suggested_targets: self.suggest_targets(&artifact.artifact_type).iter().map(|t| t.name.clone()).collect(),
                estimated_impact: self.estimate_impact(&artifact.artifact_type),
                recommended_priority: self.recommend_priority(&artifact.artifact_type),
            });
        }
        
        // Deduplicate and merge similar predictions
        predictions = self.merge_similar_predictions(predictions);
        
        Ok(predictions)
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
