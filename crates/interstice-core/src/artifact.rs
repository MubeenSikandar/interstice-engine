//! # Artifacts Module
//! 
//! Comprehensive artifact management for the INTERSTICE-ENGINE WorkOS.
//! Production-ready implementation with ML prediction capabilities via trait abstraction.
//interstice-core/src/artifact.rs
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, Value};
use thiserror::Error;
use tracing::{debug, instrument, warn};
use uuid::Uuid;
use regex::Regex;

use crate::traits::{MLPredictor, OutcomePrediction};
use crate::types::{Platform, WorkspaceId};

/// Artifact-specific errors
#[derive(Error, Debug)]
pub enum ArtifactError {
    #[error("Invalid artifact type: {0}")]
    InvalidType(String),
    
    #[error("Metadata validation failed: {0}")]
    MetadataValidation(String),
    
    #[error("Complexity calculation error: {0}")]
    ComplexityError(String),
    
    #[error("Extraction failed: {0}")]
    ExtractionError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Parse error: {0}")]
    ParseError(String),
}

/// Core artifact structure representing any work item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Unique identifier
    pub id: Uuid,
    
    /// Workspace context
    pub workspace_id: WorkspaceId,
    
    /// Type-specific artifact data
    pub artifact_type: ArtifactType,
    
    /// Source platform
    pub platform: Platform,
    
    /// Raw content or description
    pub content: String,
    
    /// Additional metadata
    pub metadata: JsonValue,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Version for optimistic locking
    pub version: u32,
    
    /// Processing state
    pub state: ArtifactState,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    
    /// Related artifact IDs
    pub related_artifacts: Vec<Uuid>,
    
    /// Tags for categorization
    pub tags: HashSet<String>,
}

/// Artifact processing state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ArtifactState {
    Pending,
    Processing,
    Processed,
    Failed(String),
    Archived,
}

/// Quality metrics for artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub completeness: f64,      // 0.0 to 1.0
    pub clarity: f64,           // 0.0 to 1.0
    pub impact: f64,            // 0.0 to 1.0
    pub urgency: f64,           // 0.0 to 1.0
    pub complexity: f64,        // 0.0 to 10.0
    pub confidence: f64,        // 0.0 to 1.0
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            completeness: 0.5,
            clarity: 0.5,
            impact: 0.5,
            urgency: 0.5,
            complexity: 5.0,
            confidence: 0.5,
        }
    }
}

impl Artifact {
    /// Create a new artifact with validation
    #[instrument(skip(content))]
    pub fn new(
        workspace_id: WorkspaceId,
        artifact_type: ArtifactType,
        platform: Platform,
        content: String,
    ) -> Result<Self, ArtifactError> {
        // Validate content length
        if content.is_empty() {
            return Err(ArtifactError::MetadataValidation(
                "Content cannot be empty".to_string()
            ));
        }
        
        if content.len() > 1_000_000 { // 1MB limit
            return Err(ArtifactError::MetadataValidation(
                "Content exceeds maximum size".to_string()
            ));
        }
        
        let now = Utc::now();
        let complexity = artifact_type.complexity_score();
        
        Ok(Self {
            id: Uuid::new_v4(),
            workspace_id,
            artifact_type,
            platform,
            content,
            metadata: JsonValue::Object(serde_json::Map::new()),
            created_at: now,
            updated_at: now,
            version: 1,
            state: ArtifactState::Pending,
            quality_metrics: QualityMetrics {
                complexity,
                ..Default::default()
            },
            related_artifacts: Vec::new(),
            tags: HashSet::new(),
        })
    }
    
    /// Builder pattern for adding metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: JsonValue) -> Self {
        if let JsonValue::Object(ref mut map) = self.metadata {
            map.insert(key.into(), value);
        }
        self
    }
    
    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) -> &mut Self {
        self.tags.insert(tag.into());
        self.updated_at = Utc::now();
        self.version += 1;
        self
    }
    
    /// Add related artifact
    pub fn add_related(&mut self, artifact_id: Uuid) -> &mut Self {
        if !self.related_artifacts.contains(&artifact_id) {
            self.related_artifacts.push(artifact_id);
            self.updated_at = Utc::now();
            self.version += 1;
        }
        self
    }
    
    /// Update content with validation
    pub fn update_content(&mut self, content: String) -> Result<(), ArtifactError> {
        if content.is_empty() {
            return Err(ArtifactError::MetadataValidation(
                "Content cannot be empty".to_string()
            ));
        }
        
        self.content = content;
        self.updated_at = Utc::now();
        self.version += 1;
        Ok(())
    }
    
    /// Calculate relevance score based on age and quality
    pub fn relevance_score(&self) -> f64 {
        let age_days = (Utc::now() - self.created_at).num_days() as f64;
        let age_factor = (-age_days / 30.0).exp(); // Exponential decay over 30 days
        
        let quality_score = (
            self.quality_metrics.completeness +
            self.quality_metrics.clarity +
            self.quality_metrics.impact +
            self.quality_metrics.urgency
        ) / 4.0;
        
        (age_factor * 0.3 + quality_score * 0.7).min(1.0)
    }
    
    /// Check if artifact is stale
    pub fn is_stale(&self, max_age_days: i64) -> bool {
        let age = Utc::now() - self.updated_at;
        age.num_days() > max_age_days
    }
    
    /// Compute signature for deduplication
    pub fn signature(&self) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        hasher.update(self.workspace_id.to_string().as_bytes());
        hasher.update(self.platform.to_string().as_bytes());
        hasher.update(self.artifact_type.type_name().as_bytes());
        hasher.update(self.content.as_bytes());
        
        format!("{:x}", hasher.finalize())
    }
}

/// Artifact type enumeration with platform-specific details
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ArtifactType {
    /// Pull request from version control
    PullRequest {
        number: u32,
        title: String,
        state: PullRequestState,
        files_changed: u32,
        additions: u32,
        deletions: u32,
        merged: bool,
        draft: bool,
        base_branch: String,
        head_branch: String,
        author: String,
        reviewers: Vec<String>,
        labels: Vec<String>,
        merge_conflict: bool,
        ci_status: Option<CIStatus>,
    },
    
    /// Issue from project management
    Issue {
        id: String,
        title: String,
        state: IssueState,
        priority: Priority,
        assignees: Vec<String>,
        labels: Vec<String>,
        story_points: Option<u32>,
        sprint: Option<String>,
        epic: Option<String>,
        blocked: bool,
        blockers: Vec<String>,
        time_estimate: Option<Duration>,
        time_spent: Option<Duration>,
    },
    
    /// Code commit
    Commit {
        sha: String,
        message: String,
        author: String,
        committer: String,
        files_changed: u32,
        additions: u32,
        deletions: u32,
        branch: String,
        is_merge: bool,
        signed: bool,
        verified: bool,
    },
    
    /// Document or wiki page
    Document {
        id: String,
        title: String,
        doc_type: DocumentType,
        url: Option<String>,
        author: String,
        collaborators: Vec<String>,
        word_count: Option<u32>,
        last_modified: DateTime<Utc>,
        version: u32,
        is_template: bool,
        access_level: AccessLevel,
    },
    
    /// Chat message or comment
    Message {
        id: String,
        channel: String,
        thread_id: Option<String>,
        author: String,
        content: String,
        mentions: Vec<String>,
        attachments: Vec<Attachment>,
        reactions: HashMap<String, u32>,
        sentiment: Sentiment,
        intent: MessageIntent,
        is_edited: bool,
        reply_count: u32,
    },
    
    /// Calendar event or meeting
    Meeting {
        id: String,
        title: String,
        duration: Duration,
        attendees: Vec<Attendee>,
        organizer: String,
        meeting_type: MeetingType,
        recurring: bool,
        recording_available: bool,
        notes: Option<String>,
        action_items: Vec<String>,
        decisions: Vec<String>,
        location: Option<String>,
    },
    
    /// Task from task management system
    Task {
        id: String,
        title: String,
        status: TaskStatus,
        assignee: Option<String>,
        due_date: Option<DateTime<Utc>>,
        completed_at: Option<DateTime<Utc>>,
        checklist_items: Vec<ChecklistItem>,
        dependencies: Vec<String>,
        tags: Vec<String>,
        recurring: bool,
        parent_task: Option<String>,
        subtasks: Vec<String>,
    },
    
    /// Code review
    Review {
        id: String,
        pull_request_id: String,
        reviewer: String,
        state: ReviewState,
        approved: bool,
        changes_requested: u32,
        comments: Vec<ReviewComment>,
        files_reviewed: u32,
        review_time: Duration,
        suggested_changes: Vec<String>,
    },
    
    /// Deployment event
    Deployment {
        id: String,
        environment: Environment,
        version: String,
        success: bool,
        duration: Duration,
        rollback: bool,
        auto_deployed: bool,
        triggered_by: String,
        affected_services: Vec<String>,
        deployment_strategy: DeploymentStrategy,
        health_checks: Vec<HealthCheck>,
    },
    
    /// Metric or measurement
    Metric {
        name: String,
        value: f64,
        metric_type: MetricType,
        unit: String,
        tags: HashMap<String, String>,
        anomaly: bool,
        baseline: Option<f64>,
        threshold: Option<f64>,
        trend: Trend,
    },
    
    /// Alert or incident
    Alert {
        id: String,
        title: String,
        severity: Severity,
        source: String,
        resolved: bool,
        resolved_by: Option<String>,
        resolution_time: Option<Duration>,
        affected_services: Vec<String>,
        root_cause: Option<String>,
        runbook_url: Option<String>,
        escalation_level: u32,
    },
    
    /// Design file or mockup
    Design {
        id: String,
        title: String,
        design_type: DesignType,
        version: Option<String>,
        collaborators: Vec<String>,
        components: u32,
        screens: u32,
        last_modified: DateTime<Utc>,
        design_system: Option<String>,
        accessibility_score: Option<f64>,
    },
    
    /// Test result
    TestResult {
        id: String,
        suite: String,
        passed: u32,
        failed: u32,
        skipped: u32,
        duration: Duration,
        coverage: Option<f64>,
        test_type: TestType,
        flaky_tests: Vec<String>,
        performance_metrics: HashMap<String, f64>,
    },
    
    /// Custom artifact type
    Custom {
        category: String,
        attributes: HashMap<String, JsonValue>,
        schema_version: String,
    },
}

impl MeetingType {
   pub fn name(&self) -> &str {
        match self {
            Self::Standup => "standup",
            Self::Planning => "planning",
            Self::Review => "review",
            Self::Retrospective => "retrospective",
            Self::OneOnOne => "1:1",
            Self::AllHands => "all-hands",
            Self::Interview => "interview",
            Self::Training => "training",
            Self::Other(s) => s,
        }
    }
}

impl Environment {
   pub fn name(&self) -> &str {
        match self {
            Self::Development => "development",
            Self::Staging => "staging",
            Self::Production => "production",
            Self::Testing => "testing",
            Self::Preview => "preview",
            Self::Custom(s) => s,
        }
    }
}

impl DeploymentStrategy {
   pub fn name(&self) -> &str {
        match self {
            Self::BlueGreen => "blue-green",
            Self::Canary => "canary",
            Self::RollingUpdate => "rolling-update",
            Self::Recreate => "recreate",
            Self::Shadow => "shadow",
        }
    }
}

impl Severity {
   pub fn to_artifact_severity(&self) -> crate::types::Priority {
        match self {
            Self::Critical => crate::types::Priority::Critical,
            Self::High => crate::types::Priority::High,
            Self::Medium => crate::types::Priority::Medium,
            Self::Low | Self::Info => crate::types::Priority::Low,
        }
    }
}

// Enums for better type safety
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PullRequestState {
    Open,
    Closed,
    Merged,
    Draft,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueState {
    Open,
    InProgress,
    Resolved,
    Closed,
    Reopened,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

impl Priority {
    pub fn to_types_priority(&self) -> crate::types::Priority {
        match self {
            Self::Critical => crate::types::Priority::Critical,
            Self::High => crate::types::Priority::High,
            Self::Medium => crate::types::Priority::Medium,
            Self::Low => crate::types::Priority::Low,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    Wiki,
    Specification,
    Design,
    Readme,
    Tutorial,
    API,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessLevel {
    Public,
    Internal,
    Private,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub name: String,
    pub url: String,
    pub mime_type: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Sentiment {
    Positive,
    Neutral,
    Negative,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageIntent {
    Question,
    Answer,
    Announcement,
    Discussion,
    Command,
    Feedback,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attendee {
    pub name: String,
    pub email: String,
    pub response: AttendeeResponse,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AttendeeResponse {
    Accepted,
    Declined,
    Tentative,
    NoResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeetingType {
    Standup,
    Planning,
    Review,
    Retrospective,
    OneOnOne,
    AllHands,
    Interview,
    Training,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Todo,
    InProgress,
    Blocked,
    InReview,
    Done,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecklistItem {
    pub text: String,
    pub completed: bool,
    pub assignee: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReviewState {
    Pending,
    Approved,
    ChangesRequested,
    Commented,
    Dismissed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewComment {
    pub file: String,
    pub line: Option<u32>,
    pub text: String,
    pub severity: CommentSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommentSeverity {
    Blocker,
    Major,
    Minor,
    Nitpick,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
    Testing,
    Preview,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    BlueGreen,
    Canary,
    RollingUpdate,
    Recreate,
    Shadow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub passed: bool,
    pub response_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Gauge,
    Counter,
    Histogram,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Trend {
    Rising,
    Falling,
    Stable,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Ord, PartialOrd, Eq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignType {
    Figma,
    Sketch,
    AdobeXD,
    InVision,
    Framer,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestType {
    Unit,
    Integration,
    EndToEnd,
    Performance,
    Security,
    Accessibility,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CIStatus {
    Success,
    Failed,
    Running,
    Cancelled,
    Skipped,
}

impl ArtifactType {
    /// Get a human-readable type name
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::PullRequest { .. } => "Pull Request",
            Self::Issue { .. } => "Issue",
            Self::Commit { .. } => "Commit",
            Self::Document { .. } => "Document",
            Self::Message { .. } => "Message",
            Self::Meeting { .. } => "Meeting",
            Self::Task { .. } => "Task",
            Self::Review { .. } => "Code Review",
            Self::Deployment { .. } => "Deployment",
            Self::Metric { .. } => "Metric",
            Self::Alert { .. } => "Alert",
            Self::Design { .. } => "Design",
            Self::TestResult { .. } => "Test Result",
            Self::Custom { .. } => "Custom",
        }
    }
    
    /// Calculate a complexity score for the artifact
    #[instrument(skip(self))]
    pub fn complexity_score(&self) -> f64 {
        let score = match self {
            Self::PullRequest { 
                files_changed, 
                additions, 
                deletions, 
                reviewers,
                merge_conflict,
                .. 
            } => {
                let size_score = (*files_changed as f64).ln().max(1.0);
                let change_score = ((*additions + *deletions) as f64).ln() / 10.0;
                let review_score = reviewers.len() as f64 * 0.5;
                let conflict_penalty = if *merge_conflict { 2.0 } else { 0.0 };
                size_score + change_score + review_score + conflict_penalty
            }
            
            Self::Issue { 
                story_points, 
                assignees, 
                blocked,
                blockers,
                priority,
                .. 
            } => {
                let points_score = story_points.unwrap_or(3) as f64;
                let assignee_score = assignees.len() as f64 * 0.5;
                let blocker_penalty = if *blocked { 2.0 } else { blockers.len() as f64 * 0.5 };
                let priority_factor = match priority {
                    Priority::Critical => 3.0,
                    Priority::High => 2.0,
                    Priority::Medium => 1.0,
                    Priority::Low => 0.5,
                };
                points_score + assignee_score + blocker_penalty + priority_factor
            }
            
            Self::Document { 
                word_count, 
                collaborators,
                version,
                .. 
            } => {
                let size_score = (word_count.unwrap_or(500) as f64 / 500.0).min(5.0);
                let collab_score = collaborators.len() as f64 * 0.3;
                let version_factor = (*version as f64).ln().max(1.0) * 0.2;
                size_score + collab_score + version_factor
            }
            
            Self::Meeting { 
                duration, 
                attendees,
                action_items,
                decisions,
                meeting_type,
                .. 
            } => {
                let duration_score = duration.num_minutes() as f64 / 30.0;
                let attendee_score = attendees.len() as f64 / 5.0;
                let action_score = action_items.len() as f64 * 0.5;
                let decision_score = decisions.len() as f64 * 0.8;
                let type_factor = match meeting_type {
                    MeetingType::AllHands => 2.0,
                    MeetingType::Planning | MeetingType::Review => 1.5,
                    _ => 1.0,
                };
                (duration_score + attendee_score + action_score + decision_score) * type_factor
            }
            
            Self::Task { 
                checklist_items, 
                dependencies,
                subtasks,
                status,
                .. 
            } => {
                let checklist_score = checklist_items.len() as f64 * 0.5;
                let dep_score = dependencies.len() as f64 * 0.8;
                let subtask_score = subtasks.len() as f64 * 0.6;
                let status_factor = match status {
                    TaskStatus::Blocked => 2.0,
                    TaskStatus::InReview => 1.5,
                    _ => 1.0,
                };
                (checklist_score + dep_score + subtask_score) * status_factor
            }
            
            Self::Deployment {
                affected_services,
                rollback,
                environment,
                ..
            } => {
                let service_score = affected_services.len() as f64;
                let rollback_penalty = if *rollback { 3.0 } else { 0.0 };
                let env_factor = match environment {
                    Environment::Production => 3.0,
                    Environment::Staging => 2.0,
                    _ => 1.0,
                };
                (service_score + rollback_penalty) * env_factor
            }
            
            Self::Alert {
                severity,
                affected_services,
                escalation_level,
                ..
            } => {
                let severity_score = match severity {
                    Severity::Critical => 5.0,
                    Severity::High => 4.0,
                    Severity::Medium => 3.0,
                    Severity::Low => 2.0,
                    Severity::Info => 1.0,
                };
                let service_score = affected_services.len() as f64;
                let escalation_score = *escalation_level as f64;
                severity_score + service_score + escalation_score
            }
            
            _ => 5.0, // Default medium complexity
        };
        
        score.min(10.0).max(0.0)
    }
    
    /// Extract key participants from the artifact
    pub fn participants(&self) -> Vec<String> {
        let mut participants = match self {
            Self::PullRequest { author, reviewers, .. } => {
                let mut p = vec![author.clone()];
                p.extend(reviewers.clone());
                p
            }
            Self::Issue { assignees, .. } => assignees.clone(),
            Self::Document { author, collaborators, .. } => {
                let mut p = vec![author.clone()];
                p.extend(collaborators.clone());
                p
            }
            Self::Message { author, mentions, .. } => {
                let mut p = vec![author.clone()];
                p.extend(mentions.clone());
                p
            }
            Self::Meeting { organizer, attendees, .. } => {
                let mut p = vec![organizer.clone()];
                p.extend(attendees.iter().map(|a| a.name.clone()));
                p
            }
            Self::Task { assignee, .. } => {
                assignee.as_ref().map(|a| vec![a.clone()]).unwrap_or_default()
            }
            Self::Review { reviewer, .. } => vec![reviewer.clone()],
            Self::Deployment { triggered_by, .. } => vec![triggered_by.clone()],
            _ => Vec::new(),
        };
        
        // Deduplicate
        participants.sort();
        participants.dedup();
        participants
    }
    
    /// Check if artifact is actionable
    pub fn is_actionable(&self) -> bool {
        match self {
            Self::PullRequest { state, .. } => matches!(state, PullRequestState::Open),
            Self::Issue { state, .. } => !matches!(state, IssueState::Closed | IssueState::Resolved),
            Self::Task { status, .. } => !matches!(status, TaskStatus::Done | TaskStatus::Cancelled),
            Self::Alert { resolved, .. } => !resolved,
            Self::Review { state, .. } => matches!(state, ReviewState::Pending),
            _ => false,
        }
    }
    
    /// Get priority level
    pub fn priority_level(&self) -> Priority {
        match self {
            Self::Issue { priority, .. } => priority.clone(),
            Self::Alert { severity, .. } => match severity {
                Severity::Critical => Priority::Critical,
                Severity::High => Priority::High,
                Severity::Medium => Priority::Medium,
                _ => Priority::Low,
            },
            Self::Task { due_date, .. } => {
                if let Some(due) = due_date {
                    let days_until = (*due - Utc::now()).num_days();
                    if days_until < 1 {
                        Priority::Critical
                    } else if days_until < 3 {
                        Priority::High
                    } else if days_until < 7 {
                        Priority::Medium
                    } else {
                        Priority::Low
                    }
                } else {
                    Priority::Medium
                }
            }
            _ => Priority::Medium,
        }
    }
}

/// Artifact extractor for parsing content from different platforms
pub struct ArtifactExtractor {
    parsers: HashMap<Platform, Box<dyn Parser>>,
}

impl ArtifactExtractor {
    pub fn new() -> Self {
        let mut parsers: HashMap<Platform, Box<dyn Parser>> = HashMap::new();
        
        // Register default parsers for all platforms
        parsers.insert(Platform::Slack, Box::new(SlackParser));
        parsers.insert(Platform::Teams, Box::new(TeamsParser));
        parsers.insert(Platform::Jira, Box::new(JiraParser));
        parsers.insert(Platform::Asana, Box::new(AsanaParser));
        parsers.insert(Platform::Notion, Box::new(NotionParser));
        parsers.insert(Platform::GitHub, Box::new(GitHubParser));
        parsers.insert(Platform::VSCode, Box::new(VSCodeParser));
        parsers.insert(Platform::GoogleWorkspace, Box::new(GoogleWorkspaceParser));
        parsers.insert(Platform::Monday, Box::new(MondayParser));
        parsers.insert(Platform::Trello, Box::new(TrelloParser));
        parsers.insert(Platform::Zoom, Box::new(ZoomParser));
        parsers.insert(Platform::Figma, Box::new(FigmaParser));
        
        Self { parsers }
    }
    
    /// Extract artifacts from raw content
    #[instrument(skip(self, content))]
    pub async fn extract(&self, content: &str, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
        let parser = self.parsers
            .get(&platform)
            .ok_or_else(|| ArtifactError::ExtractionError(format!("No parser for platform: {:?}", platform)))?;
        
        parser.parse(content.to_string(), platform).await
    }
    
    /// Register a custom parser for a platform
    pub fn register_parser(&mut self, platform: Platform, parser: Box<dyn Parser>) {
        self.parsers.insert(platform, parser);
    }
}

/// Parser trait for platform-specific content extraction
#[async_trait]
pub trait Parser: Send + Sync {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError>;
}

// To make it BoxFuture, but since async fn, it's fine, but to match, use Box::pin

// Platform-specific parsers
struct SlackParser;
struct TeamsParser;
struct JiraParser;
struct AsanaParser;
struct NotionParser;
struct GitHubParser;
struct VSCodeParser;
struct GoogleWorkspaceParser;
struct MondayParser;
struct TrelloParser;
struct ZoomParser;
struct FigmaParser;

#[async_trait]
impl Parser for GitHubParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                if value.get("pull_request").is_some() {
                    let pr = &value["pull_request"];
                    
                    let number = pr["number"].as_u64().unwrap_or(0) as u32;
                    let title = pr["title"].as_str().unwrap_or("Untitled Pull Request").to_string();
                    let state_str = pr["state"].as_str().unwrap_or("open");
                    let merged = pr["merged"].as_bool().unwrap_or(false);
                    let state = match state_str {
                        "open" => PullRequestState::Open,
                        "closed" => if merged { PullRequestState::Merged } else { PullRequestState::Closed },
                        _ => PullRequestState::Open,
                    };
                    let files_changed = pr["changed_files"].as_u64().unwrap_or(0) as u32;
                    let additions = pr["additions"].as_u64().unwrap_or(0) as u32;
                    let deletions = pr["deletions"].as_u64().unwrap_or(0) as u32;
                    let draft = pr["draft"].as_bool().unwrap_or(false);
                    let base_branch = pr["base"]["ref"].as_str().unwrap_or("main").to_string();
                    let head_branch = pr["head"]["ref"].as_str().unwrap_or("feature").to_string();
                    let author = pr["user"]["login"].as_str().unwrap_or("unknown").to_string();
                    let reviewers: Vec<String> = pr["requested_reviewers"].as_array()
                        .map(|arr| arr.iter().filter_map(|r| r["login"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let labels: Vec<String> = pr["labels"].as_array()
                        .map(|arr| arr.iter().filter_map(|l| l["name"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let merge_conflict = pr["mergeable"].as_bool().map(|b| !b).unwrap_or(false);
                    let ci_status = None; // Can be extended with check_run events
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::PullRequest {
                            number,
                            title,
                            state,
                            files_changed,
                            additions,
                            deletions,
                            merged,
                            draft,
                            base_branch,
                            head_branch,
                            author,
                            reviewers,
                            labels,
                            merge_conflict,
                            ci_status,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                } else if value.get("issue").is_some() {
                    let issue = &value["issue"];
                    
                    let id = issue["number"].as_str().unwrap_or("0").to_string();
                    let title = issue["title"].as_str().unwrap_or("Untitled Issue").to_string();
                    let state_str = issue["state"].as_str().unwrap_or("open");
                    let state = match state_str {
                        "open" => IssueState::Open,
                        "closed" => IssueState::Closed,
                        _ => IssueState::Open,
                    };
                    let priority = Priority::Medium; // Derive from labels if needed
                    let assignees: Vec<String> = issue["assignees"].as_array()
                        .map(|arr| arr.iter().filter_map(|a| a["login"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let labels: Vec<String> = issue["labels"].as_array()
                        .map(|arr| arr.iter().filter_map(|l| l["name"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let story_points = None;
                    let sprint = None;
                    let epic = None;
                    let blocked = labels.iter().any(|l| l.to_lowercase().contains("blocked"));
                    let blockers = Vec::new();
                    let time_estimate = None;
                    let time_spent = None;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Issue {
                            id,
                            title,
                            state,
                            priority,
                            assignees,
                            labels,
                            story_points,
                            sprint,
                            epic,
                            blocked,
                            blockers,
                            time_estimate,
                            time_spent,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback to regex-based parsing for non-JSON content
                let pr_regex = Regex::new(r"PR #(\d+)").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
                for cap in pr_regex.captures_iter(&content) {
                    let pr_number: u32 = cap[1].parse().unwrap_or(0);
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::PullRequest {
                            number: pr_number,
                            title: format!("Pull Request #{}", pr_number),
                            state: PullRequestState::Open,
                            files_changed: 0,
                            additions: 0,
                            deletions: 0,
                            merged: false,
                            draft: false,
                            base_branch: "main".to_string(),
                            head_branch: "feature".to_string(),
                            author: "unknown".to_string(),
                            reviewers: Vec::new(),
                            labels: Vec::new(),
                            merge_conflict: false,
                            ci_status: None,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
                
                let issue_regex = Regex::new(r"#(\d+)").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
                for cap in issue_regex.captures_iter(&content) {
                    let issue_id = cap[1].to_string();
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Issue {
                            id: issue_id.clone(),
                            title: format!("Issue #{}", issue_id),
                            state: IssueState::Open,
                            priority: Priority::Medium,
                            assignees: Vec::new(),
                            labels: Vec::new(),
                            story_points: None,
                            sprint: None,
                            epic: None,
                            blocked: false,
                            blockers: Vec::new(),
                            time_estimate: None,
                            time_spent: None,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}


#[async_trait]
impl Parser for TeamsParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                if value.get("type").and_then(|v| v.as_str()) == Some("message") {
                    let id = value["id"].as_str().unwrap_or("").to_string();
                    let channel = value["channelId"].as_str().unwrap_or("general").to_string();
                    let thread_id = value["replyToId"].as_str().map(ToOwned::to_owned);
                    let author = value["from"]["user"]["displayName"].as_str().unwrap_or("user").to_string();
                    let msg_content = value["body"]["content"].as_str().unwrap_or("").to_string();
                    let mentions: Vec<String> = value["mentions"].as_array()
                        .map(|arr| arr.iter().filter_map(|m| m["mentioned"]["user"]["displayName"].as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let attachments: Vec<Attachment> = value["attachments"].as_array()
                        .map(|arr| arr.iter().filter_map(|att| {
                            att["contentUrl"].as_str().map(|url| Attachment {
                                name: att["name"].as_str().unwrap_or("attachment").to_string(),
                                url: url.to_string(),
                                mime_type: att["contentType"].as_str().unwrap_or("application/octet-stream").to_string(),
                                size: att["size"].as_u64().unwrap_or(0),
                            })
                        }).collect())
                        .unwrap_or_default();
                    let reactions: HashMap<String, u32> = value["reactions"].as_array()
                        .map(|arr| arr.iter().fold(HashMap::new(), |mut map, r| {
                            if let (Some(name), Some(count)) = (r["reactionType"].as_str(), r["count"].as_u64()) {
                                map.insert(name.to_string(), count as u32);
                            }
                            map
                        }))
                        .unwrap_or_default();
                    let sentiment = Sentiment::Neutral;
                    let intent = MessageIntent::Discussion;
                    let is_edited = value.get("lastEditedDateTime").is_some();
                    let reply_count = value["replies"].as_array().map(|arr| arr.len() as u32).unwrap_or(0);
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Message {
                            id,
                            channel,
                            thread_id,
                            author,
                            content: msg_content,
                            mentions,
                            attachments,
                            reactions,
                            sentiment,
                            intent,
                            is_edited,
                            reply_count,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback similar to Slack
                let mention_regex = Regex::new(r"@(\w+)").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
                let mentions: Vec<String> = mention_regex
                    .captures_iter(&content)
                    .map(|cap| cap[1].to_string())
                    .collect();
                
                if !content.is_empty() {
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Message {
                            id: Uuid::new_v4().to_string(),
                            channel: "general".to_string(),
                            thread_id: None,
                            author: "user".to_string(),
                            content: content.clone(),
                            mentions,
                            attachments: Vec::new(),
                            reactions: HashMap::new(),
                            sentiment: Sentiment::Neutral,
                            intent: MessageIntent::Discussion,
                            is_edited: false,
                            reply_count: 0,
                        },
                        platform,
                        content,
                    )?;
                    
                    artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}

#[async_trait]
impl Parser for JiraParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let webhook_event = value["webhookEvent"].as_str().unwrap_or("");
                
                if webhook_event.contains("issue") {
                    let issue = &value["issue"];
                    
                    let id = issue["key"].as_str().unwrap_or("UNKNOWN").to_string();
                    let title = issue["fields"]["summary"].as_str().unwrap_or("Untitled Issue").to_string();
                    let state_str = issue["fields"]["status"]["name"].as_str().unwrap_or("Open").to_lowercase();
                    let state = match state_str.as_str() {
                        "open" | "to do" | "backlog" => IssueState::Open,
                        "in progress" => IssueState::InProgress,
                        "done" | "closed" | "resolved" => IssueState::Closed,
                        _ => IssueState::Open,
                    };
                    let priority_str = issue["fields"]["priority"]["name"].as_str().unwrap_or("Medium").to_lowercase();
                    let priority = match priority_str.as_str() {
                        "low" | "lowest" => Priority::Low,
                        "medium" => Priority::Medium,
                        "high" | "highest" => Priority::High,
                        _ => Priority::Medium,
                    };
                    let assignees: Vec<String> = issue["fields"]["assignee"].as_object()
                        .map(|a| vec![a["displayName"].as_str().unwrap_or("unknown").to_string()])
                        .unwrap_or_default();
                    let labels: Vec<String> = issue["fields"]["labels"].as_array()
                        .map(|arr| arr.iter().filter_map(|l| l.as_str().map(str::to_string)).collect())
                        .unwrap_or_default();
                    let story_points = issue["fields"]["customfield_10010"].as_f64().map(|f| f as u32);
                    let sprint = issue["fields"]["customfield_10007"].as_str().map(str::to_string);
                    let epic = issue["fields"]["customfield_10015"].as_str().map(str::to_string);
                    let blocked = labels.iter().any(|l| l.to_lowercase().contains("blocked"));
                    let blockers = Vec::new();
                    let time_estimate = issue["fields"]["timeestimate"].as_u64().map(|t| Duration::seconds(t as i64));
                    let time_spent = issue["fields"]["timespent"].as_u64().map(|t| Duration::seconds(t as i64));
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Issue {
                            id,
                            title,
                            state,
                            priority,
                            assignees,
                            labels,
                            story_points,
                            sprint,
                            epic,
                            blocked,
                            blockers,
                            time_estimate,
                            time_spent,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback to regex-based parsing
                let jira_regex = Regex::new(r"([A-Z]{2,}-\d+)").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
                for cap in jira_regex.captures_iter(&content) {
                let issue_key = cap[1].to_string();
                
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Issue {
                        id: issue_key.clone(),
                        title: format!("Jira Issue {}", issue_key),
                        state: IssueState::Open,
                        priority: Priority::Medium,
                        assignees: Vec::new(),
                        labels: Vec::new(),
                        story_points: Some(3),
                        sprint: None,
                        epic: None,
                        blocked: false,
                        blockers: Vec::new(),
                        time_estimate: None,
                        time_spent: None,
                    },
                    platform,
                        content.clone(),
                )?;
                
                artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}


#[async_trait]
impl Parser for SlackParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let type_ = value["type"].as_str().unwrap_or("");
                
                if type_ == "message" {
                    let id = value["ts"].as_str().unwrap_or(Uuid::new_v4().to_string().as_str()).to_string();
                    let channel = value["channel"].as_str().unwrap_or("general").to_string();
                    let thread_id = value["thread_ts"].as_str().map(str::to_string);
                    let author = value["user"].as_str().unwrap_or("user").to_string();
                    let msg_content = value["text"].as_str().unwrap_or("").to_string();
                    let mention_regex = Regex::new(r"<@(\w+)>").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
                    let mentions: Vec<String> = mention_regex
                        .captures_iter(&msg_content)
                        .map(|cap| cap[1].to_string())
                        .collect();
                    let attachments: Vec<Attachment> = value["attachments"].as_array()
                        .map(|arr| arr.iter().filter_map(|att| {
                            att["original_url"].as_str().or(att["fallback"].as_str()).map(|url| Attachment {
                                name: att["title"].as_str().unwrap_or("attachment").to_string(),
                                url: url.to_string(),
                                mime_type: att["mimetype"].as_str().unwrap_or("application/octet-stream").to_string(),
                                size: att["size"].as_u64().unwrap_or(0),
                            })
                        }).collect())
                        .unwrap_or_default();
                    let reactions: HashMap<String, u32> = value["reactions"].as_array()
                        .map(|arr| arr.iter().fold(HashMap::new(), |mut map, r| {
                            if let Some(name) = r["name"].as_str() {
                                let count = r["count"].as_u64().unwrap_or(0) as u32;
                                map.insert(name.to_string(), count);
                            }
                            map
                        }))
                        .unwrap_or_default();
                    let sentiment = Sentiment::Neutral;
                    let intent = MessageIntent::Discussion;
                    let is_edited = value.get("edited").is_some();
                    let reply_count = value["reply_count"].as_u64().unwrap_or(0) as u32;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Message {
                            id,
                            channel,
                            thread_id,
                            author,
                            content: msg_content,
                            mentions,
                            attachments,
                            reactions,
                            sentiment,
                            intent,
                            is_edited,
                            reply_count,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback to text-based parsing
                let mention_regex = Regex::new(r"@(\w+)").map_err(|e| ArtifactError::ParseError(e.to_string()))?;
            let mentions: Vec<String> = mention_regex
                    .captures_iter(&content)
                .map(|cap| cap[1].to_string())
                .collect();
            
            if !content.is_empty() {
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Message {
                        id: Uuid::new_v4().to_string(),
                        channel: "general".to_string(),
                        thread_id: None,
                        author: "user".to_string(),
                            content: content.clone(),
                        mentions,
                        attachments: Vec::new(),
                        reactions: HashMap::new(),
                        sentiment: Sentiment::Neutral,
                        intent: MessageIntent::Discussion,
                        is_edited: false,
                        reply_count: 0,
                    },
                    platform,
                        content,
                )?;
                
                artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}


#[async_trait]
impl Parser for NotionParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let event_type = value["event_type"].as_str().unwrap_or("");
                
                if event_type == "page.created" || event_type == "page.updated" {
                    let page_id = value["payload"]["page_id"].as_str().unwrap_or("").to_string();
                    let title = "Notion Page".to_string(); // Title not in webhook; would require API call
                    let doc_type = DocumentType::Wiki;
                    let url = Some(format!("https://www.notion.so/{}", page_id.replace("-", "")));
                    let author = value["triggered_by"]["user_id"].as_str().unwrap_or("user").to_string();
                    let collaborators = Vec::new();
                    let word_count = Some(content.split_whitespace().count() as u32); // Approximate if text content
                    let last_modified = Utc::now(); // Parse from timestamp if available
                    let version = 1;
                    let is_template = false;
                    let access_level = AccessLevel::Internal;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id: page_id,
                            title,
                            doc_type,
                            url,
                            author,
                            collaborators,
                            word_count,
                            last_modified,
                            version,
                            is_template,
                            access_level,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback to text-based parsing
            if !content.is_empty() {
                let word_count = content.split_whitespace().count() as u32;
                
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Document {
                        id: Uuid::new_v4().to_string(),
                        title: "Notion Document".to_string(),
                        doc_type: DocumentType::Wiki,
                        url: None,
                        author: "user".to_string(),
                        collaborators: Vec::new(),
                        word_count: Some(word_count),
                        last_modified: Utc::now(),
                        version: 1,
                        is_template: false,
                        access_level: AccessLevel::Internal,
                    },
                    platform,
                        content.clone(),
                )?;
                
                artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}


#[async_trait]
impl Parser for AsanaParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                if let Some(events) = value.get("events").and_then(|e| e.as_array()) {
                    for event in events {
                        if event["resource"]["resource_type"].as_str() == Some("task") {
                            let id = event["resource"]["gid"].as_str().unwrap_or("").to_string();
                            let title = "Asana Task".to_string(); // No title in webhook; API call needed for full details
                            let status = TaskStatus::Todo;
                            let assignee = event["user"]["gid"].as_str().map(str::to_string);
                            let due_date = None;
                            let completed_at = None;
                            let checklist_items = Vec::new();
                            let dependencies = Vec::new();
                            let tags = Vec::new();
                            let recurring = false;
                            let parent_task = event["parent"].as_object().and_then(|p| p["gid"].as_str().map(str::to_string));
                            let subtasks = Vec::new();
                            
                            let artifact = Artifact::new(
                                WorkspaceId::new(),
                                ArtifactType::Task {
                                    id,
                                    title,
                                    status,
                                    assignee,
                                    due_date,
                                    completed_at,
                                    checklist_items,
                                    dependencies,
                                    tags,
                                    recurring,
                                    parent_task,
                                    subtasks,
                                },
                                platform,
                                content.clone(),
                            )?;
                            
                            artifacts.push(artifact);
                        }
                    }
                }
            } else {
                // Fallback to text-based parsing
            if content.contains("task") || content.contains("TODO") {
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Task {
                        id: Uuid::new_v4().to_string(),
                        title: "Asana Task".to_string(),
                        status: TaskStatus::Todo,
                        assignee: None,
                        due_date: None,
                        completed_at: None,
                        checklist_items: Vec::new(),
                        dependencies: Vec::new(),
                        tags: Vec::new(),
                        recurring: false,
                        parent_task: None,
                        subtasks: Vec::new(),
                    },
                    platform,
                        content.clone(),
                )?;
                
                artifacts.push(artifact);
                }
            }
            
            Ok(artifacts)
    }
}

async fn task_fallback_parse(content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
    let mut artifacts = Vec::new();

    if content.contains("task") || content.contains("TODO") {
        let artifact = Artifact::new(
            WorkspaceId::new(),
            ArtifactType::Task {
                id: Uuid::new_v4().to_string(),
                title: format!("{} Task", platform.to_string()),
                status: TaskStatus::Todo,
                assignee: None,
                due_date: None,
                completed_at: None,
                checklist_items: Vec::new(),
                dependencies: Vec::new(),
                tags: Vec::new(),
                recurring: false,
                parent_task: None,
                subtasks: Vec::new(),
            },
            platform,
            content.clone(),
        )?;

        artifacts.push(artifact);
    }

    Ok(artifacts)
}

#[async_trait]
impl Parser for MondayParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
        let mut artifacts = Vec::new();
        
        if let Ok(value) = serde_json::from_str::<Value>(&content) {
            let event = &value["event"];
            
            if event.get("pulseId").is_some() {
                let id = event["pulseId"].as_str().unwrap_or("").to_string();
                let title = "Monday Item".to_string(); // No title in payload
                let status = TaskStatus::Todo;
                let assignee = None;
                let due_date = None;
                let completed_at = None;
                let checklist_items = Vec::new();
                let dependencies = Vec::new();
                let tags = Vec::new();
                let recurring = false;
                let parent_task = None;
                let subtasks = Vec::new();
                
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Task {
                        id,
                        title,
                        status,
                        assignee,
                        due_date,
                        completed_at,
                        checklist_items,
                        dependencies,
                        tags,
                        recurring,
                        parent_task,
                        subtasks,
                    },
                    platform,
                    content.clone(),
                )?;
                
                artifacts.push(artifact);
            }
        } else {
            artifacts.extend(task_fallback_parse(content.clone(), platform).await?);
        }
        
        Ok(artifacts)
    }
}

#[async_trait]
impl Parser for TrelloParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
        let mut artifacts = Vec::new();
        
        if let Ok(value) = serde_json::from_str::<Value>(&content) {
            let action_type = value["action"]["type"].as_str().unwrap_or("");
            
            if action_type.contains("Card") {
                let card = &value["action"]["data"]["card"];
                
                let id = card["id"].as_str().unwrap_or("").to_string();
                let title = card["name"].as_str().unwrap_or("Trello Card").to_string();
                let status = TaskStatus::Todo; // Derive from list name if available
                let assignee = None;
                                 let due_date = card["due"].as_str().map(|d| chrono::DateTime::parse_from_rfc3339(d).ok()).flatten().map(|dt| dt.with_timezone(&Utc));
                let completed_at = None;
                let checklist_items = Vec::new();
                let dependencies = Vec::new();
                let tags = value["action"]["data"]["labels"].as_array()
                    .map(|arr| arr.iter().filter_map(|l| l["name"].as_str().map(str::to_string)).collect())
                    .unwrap_or_default();
                let recurring = false;
                let parent_task = None;
                let subtasks = Vec::new();
                
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Task {
                        id,
                        title,
                        status,
                        assignee,
                        due_date,
                        completed_at,
                        checklist_items,
                        dependencies,
                        tags,
                        recurring,
                        parent_task,
                        subtasks,
                    },
                    platform,
                    content.clone(),
                )?;
                
                artifacts.push(artifact);
            }
        } else {
            artifacts.extend(task_fallback_parse(content.clone(), platform).await?);
        }
        
        Ok(artifacts)
    }
}

#[async_trait]
impl Parser for VSCodeParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            // Assuming content is text or JSON from extension events; fallback to simple parsing
            if content.contains("file") || content.contains("code") {
                let word_count = content.split_whitespace().count() as u32;
                
                let artifact = Artifact::new(
                    WorkspaceId::new(),
                    ArtifactType::Document {
                        id: Uuid::new_v4().to_string(),
                        title: "VSCode File".to_string(),
                        doc_type: DocumentType::Other("Code".to_string()),
                        url: None,
                        author: "developer".to_string(),
                        collaborators: Vec::new(),
                        word_count: Some(word_count),
                        last_modified: Utc::now(),
                        version: 1,
                        is_template: false,
                        access_level: AccessLevel::Internal,
                    },
                    platform,
                    content.clone(),
                )?;
                
                artifacts.push(artifact);
            }
            
        Ok(artifacts)
    }
}

#[async_trait]
impl Parser for GoogleWorkspaceParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let kind = value["kind"].as_str().unwrap_or("");
                
                if kind == "drive#change" {
                    let id = value["id"].as_str().unwrap_or("").to_string();
                    let title = value["name"].as_str().unwrap_or("Google Document").to_string();
                    let mime_type = value["mimeType"].as_str().unwrap_or("");
                    let doc_type = match mime_type {
                        "application/vnd.google-apps.document" => DocumentType::Other("Document".to_string()),
                        "application/vnd.google-apps.spreadsheet" => DocumentType::Other("Spreadsheet".to_string()),
                        "application/vnd.google-apps.presentation" => DocumentType::Other("Presentation".to_string()),
                        _ => DocumentType::Other("Unknown".to_string()),
                    };
                    let url = None; // Construct if fileId available
                    let author = "user".to_string();
                    let collaborators = Vec::new();
                    let word_count = None;
                    let last_modified = Utc::now(); // Parse value["time"]
                    let version = 1;
                    let is_template = false;
                    let access_level = AccessLevel::Internal;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id,
                            title,
                            doc_type,
                            url,
                            author,
                            collaborators,
                            word_count,
                            last_modified,
                            version,
                            is_template,
                            access_level,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback if needed
                if !content.is_empty() {
                    let word_count = content.split_whitespace().count() as u32;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id: Uuid::new_v4().to_string(),
                            title: "Google Workspace Document".to_string(),
                            doc_type: DocumentType::Other("Document".to_string()),
                            url: None,
                            author: "user".to_string(),
                            collaborators: Vec::new(),
                            word_count: Some(word_count),
                            last_modified: Utc::now(),
                            version: 1,
                            is_template: false,
                            access_level: AccessLevel::Internal,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            }
            
        Ok(artifacts)
    }
}

#[async_trait]
impl Parser for ZoomParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let event = value["event"].as_str().unwrap_or("");
                
                if event.contains("meeting") {
                    let obj = &value["payload"]["object"];
                    
                    let id = obj["id"].as_str().unwrap_or("").to_string();
                    let title = obj["topic"].as_str().unwrap_or("Zoom Meeting").to_string();
                    let doc_type = DocumentType::Other("Meeting".to_string()); // Or add Meeting if enum allows
                    let url = obj["join_url"].as_str().map(str::to_string);
                    let author = obj["host_id"].as_str().unwrap_or("host").to_string();
                    let collaborators = Vec::new();
                    let word_count = None;
                    let last_modified = Utc::now();
                    let version = 1;
                    let is_template = false;
                    let access_level = AccessLevel::Internal;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id,
                            title,
                            doc_type,
                            url,
                            author,
                            collaborators,
                            word_count,
                            last_modified,
                            version,
                            is_template,
                            access_level,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback if needed
                if content.contains("meeting") || content.contains("zoom") {
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id: Uuid::new_v4().to_string(),
                            title: "Zoom Meeting".to_string(),
                            doc_type: DocumentType::Other("Meeting".to_string()),
                            url: None,
                            author: "host".to_string(),
                            collaborators: Vec::new(),
                            word_count: None,
                            last_modified: Utc::now(),
                            version: 1,
                            is_template: false,
                            access_level: AccessLevel::Internal,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            }
            
        Ok(artifacts)
    }
}

#[async_trait]
impl Parser for FigmaParser {
    async fn parse(&self, content: String, platform: Platform) -> Result<Vec<Artifact>, ArtifactError> {
            let mut artifacts = Vec::new();
            
            if let Ok(value) = serde_json::from_str::<Value>(&content) {
                let event_type = value["event_type"].as_str().unwrap_or("");
                
                if event_type == "file_update" {
                    let id = value["file_id"].as_str().unwrap_or("").to_string();
                    let title = value["file_name"].as_str().unwrap_or("Figma File").to_string();
                    let doc_type = DocumentType::Design; // Assume enum has Design
                    let url = None;
                    let author = value["triggered_by"]["handle"].as_str().unwrap_or("designer").to_string();
                    let collaborators = Vec::new();
                    let word_count = None;
                    let last_modified = Utc::now();
                    let version = 1;
                    let is_template = false;
                    let access_level = AccessLevel::Internal;
                    
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id,
                            title,
                            doc_type,
                            url,
                            author,
                            collaborators,
                            word_count,
                            last_modified,
                            version,
                            is_template,
                            access_level,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            } else {
                // Fallback if needed
                if content.contains("figma") || content.contains("design") {
                    let artifact = Artifact::new(
                        WorkspaceId::new(),
                        ArtifactType::Document {
                            id: Uuid::new_v4().to_string(),
                            title: "Figma Design".to_string(),
                            doc_type: DocumentType::Design,
                            url: None,
                            author: "designer".to_string(),
                            collaborators: Vec::new(),
                            word_count: None,
                            last_modified: Utc::now(),
                            version: 1,
                            is_template: false,
                            access_level: AccessLevel::Internal,
                        },
                        platform,
                        content.clone(),
                    )?;
                    
                    artifacts.push(artifact);
                }
            }
            
        Ok(artifacts)
    }
}

/// Artifact processor with ML capabilities
pub struct ArtifactProcessor {
    ml_predictor: Option<Arc<dyn MLPredictor>>,
    cache: Arc<parking_lot::RwLock<HashMap<String, ProcessingResult>>>,
}

impl ArtifactProcessor {
    pub fn new(ml_predictor: Option<Arc<dyn MLPredictor>>) -> Self {
        Self {
            ml_predictor,
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }
    
    /// Process artifact and predict outcomes
    #[instrument(skip(self, artifact))]
    pub async fn process(&self, artifact: &Artifact) -> Result<ProcessingResult, ArtifactError> {
        let signature = artifact.signature();
        
        // Check cache
        if let Some(cached) = self.cache.read().get(&signature) {
            debug!("Cache hit for artifact {}", artifact.id);
            return Ok(cached.clone());
        }
        
        let mut result = ProcessingResult {
            artifact_id: artifact.id,
            predictions: Vec::new(),
            quality_metrics: artifact.quality_metrics.clone(),
            recommendations: Vec::new(),
            risk_assessment: RiskAssessment::default(),
        };
        
        // Get ML predictions if available
        if let Some(predictor) = &self.ml_predictor {
            match predictor.predict_outcomes(&[artifact.clone()]).await {
                Ok(predictions) => {
                    result.predictions = predictions;
                }
                Err(e) => {
                    warn!("ML prediction failed: {}", e);
                }
            }
        }
        
        // Generate recommendations based on artifact type and state
        result.recommendations = self.generate_recommendations(artifact);
        
        // Assess risk
        result.risk_assessment = self.assess_risk(artifact);
        
        // Update quality metrics with ML insights
        if !result.predictions.is_empty() {
            let avg_confidence: f64 = result.predictions.iter()
                .map(|p| p.confidence as f64)
                .sum::<f64>() / result.predictions.len() as f64;
            result.quality_metrics.confidence = avg_confidence;
        }
        
        // Cache result
        self.cache.write().insert(signature, result.clone());
        
        Ok(result)
    }
    
    fn generate_recommendations(&self, artifact: &Artifact) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match &artifact.artifact_type {
            ArtifactType::PullRequest { files_changed, merge_conflict, ci_status, .. } => {
                if *files_changed > 50 {
                    recommendations.push("Consider breaking this PR into smaller chunks".to_string());
                }
                if *merge_conflict {
                    recommendations.push("Resolve merge conflicts before review".to_string());
                }
                if let Some(CIStatus::Failed) = ci_status {
                    recommendations.push("Fix CI failures before merging".to_string());
                }
            }
            ArtifactType::Issue { blocked, priority, blockers, .. } => {
                if *blocked {
                    recommendations.push("Address blockers to unblock progress".to_string());
                }
                if matches!(priority, Priority::Critical | Priority::High) && blockers.is_empty() {
                    recommendations.push("High priority issue - consider immediate action".to_string());
                }
            }
            ArtifactType::Task { due_date, status, dependencies, .. } => {
                if let Some(due) = due_date {
                    let days_until = (*due - Utc::now()).num_days();
                    if days_until < 2 && !matches!(status, TaskStatus::Done) {
                        recommendations.push("Task due soon - prioritize completion".to_string());
                    }
                }
                if !dependencies.is_empty() && matches!(status, TaskStatus::Todo) {
                    recommendations.push("Check if dependencies are completed".to_string());
                }
            }
            ArtifactType::Alert { severity, resolved, .. } => {
                if !resolved && matches!(severity, Severity::Critical) {
                    recommendations.push("Critical alert - immediate action required".to_string());
                    recommendations.push("Consider paging on-call engineer".to_string());
                }
            }
            ArtifactType::Deployment { rollback, success, .. } => {
                if *rollback {
                    recommendations.push("Investigate rollback cause and implement fixes".to_string());
                }
                if !success {
                    recommendations.push("Review deployment logs for failure details".to_string());
                }
            }
            _ => {}
        }
        
        recommendations
    }
    
    fn assess_risk(&self, artifact: &Artifact) -> RiskAssessment {
        let mut assessment = RiskAssessment::default();
        
        match &artifact.artifact_type {
            ArtifactType::PullRequest { additions, deletions, files_changed, .. } => {
                let total_changes = additions + deletions;
                if total_changes > 1000 || *files_changed > 50 {
                    assessment.level = RiskLevel::High;
                    assessment.factors.push("Large changeset".to_string());
                } else if total_changes > 500 || *files_changed > 20 {
                    assessment.level = RiskLevel::Medium;
                    assessment.factors.push("Moderate changeset".to_string());
                }
            }
            ArtifactType::Deployment { environment, rollback, .. } => {
                if matches!(environment, Environment::Production) {
                    assessment.level = RiskLevel::High;
                    assessment.factors.push("Production deployment".to_string());
                }
                if *rollback {
                    assessment.level = RiskLevel::High;
                    assessment.factors.push("Previous rollback occurred".to_string());
                }
            }
            ArtifactType::Alert { severity, .. } => {
                assessment.level = match severity {
                    Severity::Critical => RiskLevel::Critical,
                    Severity::High => RiskLevel::High,
                    Severity::Medium => RiskLevel::Medium,
                    _ => RiskLevel::Low,
                };
                assessment.factors.push(format!("{:?} severity alert", severity));
            }
            _ => {}
        }
        
        assessment.score = match assessment.level {
            RiskLevel::Critical => 1.0,
            RiskLevel::High => 0.8,
            RiskLevel::Medium => 0.5,
            RiskLevel::Low => 0.2,
        };
        
        assessment
    }
}

/// Processing result for an artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub artifact_id: Uuid,
    pub predictions: Vec<OutcomePrediction>,
    pub quality_metrics: QualityMetrics,
    pub recommendations: Vec<String>,
    pub risk_assessment: RiskAssessment,
}

/// Risk assessment for artifacts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub score: f64, // 0.0 to 1.0
    pub factors: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    #[default]
    Low,
}

/// Artifact query builder for flexible filtering
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArtifactQuery {
    pub workspace_id: Option<WorkspaceId>,
    pub platforms: Option<Vec<Platform>>,
    pub artifact_types: Option<Vec<String>>,
    pub states: Option<Vec<ArtifactState>>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
    pub updated_after: Option<DateTime<Utc>>,
    pub updated_before: Option<DateTime<Utc>>,
    pub participants: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub min_complexity: Option<f64>,
    pub max_complexity: Option<f64>,
    pub min_relevance: Option<f64>,
    pub actionable_only: bool,
    pub include_archived: bool,
    pub text_search: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_by: SortField,
    pub sort_order: SortOrder,
}

impl ArtifactQuery {
    /// Convert ArtifactQuery to ArtifactFilters for storage queries
    pub fn to_filters(&self) -> crate::storage::ArtifactFilters {
        crate::storage::ArtifactFilters {
            platforms: self.platforms.clone(),
            artifact_types: self.artifact_types.clone(),
            created_after: self.created_after,
            created_before: self.created_before,
            has_outcome: None, // ArtifactQuery doesn't have this field
            tags: self.tags.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortField {
    CreatedAt,
    UpdatedAt,
    Complexity,
    Relevance,
    Priority,
}

impl Default for SortField {
    fn default() -> Self {
        Self::CreatedAt
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

impl Default for SortOrder {
    fn default() -> Self {
        Self::Descending
    }
}

impl ArtifactQuery {
    pub fn builder() -> ArtifactQueryBuilder {
        ArtifactQueryBuilder::default()
    }
    
    /// Apply query filters to a collection of artifacts
    pub fn filter(&self, artifacts: &[Artifact]) -> Vec<Artifact> {
        let mut filtered: Vec<Artifact> = artifacts
            .iter()
            .filter(|a| self.matches(a))
            .cloned()
            .collect();
        
        // Sort
        filtered.sort_by(|a, b| {
            let cmp = match self.sort_by {
                SortField::CreatedAt => a.created_at.cmp(&b.created_at),
                SortField::UpdatedAt => a.updated_at.cmp(&b.updated_at),
                SortField::Complexity => {
                    let a_complex = a.artifact_type.complexity_score();
                    let b_complex = b.artifact_type.complexity_score();
                    a_complex.partial_cmp(&b_complex).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortField::Relevance => {
                    let a_rel = a.relevance_score();
                    let b_rel = b.relevance_score();
                    a_rel.partial_cmp(&b_rel).unwrap_or(std::cmp::Ordering::Equal)
                }
                SortField::Priority => {
                    let a_pri = a.artifact_type.priority_level();
                    let b_pri = b.artifact_type.priority_level();
                    a_pri.cmp(&b_pri)
                }
            };
            
            match self.sort_order {
                SortOrder::Ascending => cmp,
                SortOrder::Descending => cmp.reverse(),
            }
        });
        
        // Apply pagination
        let offset = self.offset.unwrap_or(0);
        let limit = self.limit.unwrap_or(filtered.len());
        
        filtered.into_iter()
            .skip(offset)
            .take(limit)
            .collect()
    }
    
    fn matches(&self, artifact: &Artifact) -> bool {
        // Workspace filter
        if let Some(ws_id) = &self.workspace_id {
            if artifact.workspace_id != *ws_id {
                return false;
            }
        }
        
        // Platform filter
        if let Some(platforms) = &self.platforms {
            if !platforms.contains(&artifact.platform) {
                return false;
            }
        }
        
        // Type filter
        if let Some(types) = &self.artifact_types {
            if !types.contains(&artifact.artifact_type.type_name().to_string()) {
                return false;
            }
        }
        
        // State filter
        if let Some(states) = &self.states {
            if !states.contains(&artifact.state) {
                return false;
            }
        }
        
        // Archived filter
        if !self.include_archived && artifact.state == ArtifactState::Archived {
            return false;
        }
        
        // Date filters
        if let Some(after) = self.created_after {
            if artifact.created_at < after {
                return false;
            }
        }
        
        if let Some(before) = self.created_before {
            if artifact.created_at > before {
                return false;
            }
        }
        
        if let Some(after) = self.updated_after {
            if artifact.updated_at < after {
                return false;
            }
        }
        
        if let Some(before) = self.updated_before {
            if artifact.updated_at > before {
                return false;
            }
        }
        
        // Participant filter
        if let Some(participants) = &self.participants {
            let artifact_participants = artifact.artifact_type.participants();
            if !participants.iter().any(|p| artifact_participants.contains(p)) {
                return false;
            }
        }
        
        // Tag filter
        if let Some(tags) = &self.tags {
            if !tags.iter().any(|t| artifact.tags.contains(t)) {
                return false;
            }
        }
        
        // Complexity filter
        let complexity = artifact.artifact_type.complexity_score();
        if let Some(min) = self.min_complexity {
            if complexity < min {
                return false;
            }
        }
        
        if let Some(max) = self.max_complexity {
            if complexity > max {
                return false;
            }
        }
        
        // Relevance filter
        if let Some(min) = self.min_relevance {
            if artifact.relevance_score() < min {
                return false;
            }
        }
        
        // Actionable filter
        if self.actionable_only && !artifact.artifact_type.is_actionable() {
            return false;
        }
        
        // Text search
        if let Some(search) = &self.text_search {
            let search_lower = search.to_lowercase();
            let content_lower = artifact.content.to_lowercase();
            let type_name_lower = artifact.artifact_type.type_name().to_lowercase();
            
            if !content_lower.contains(&search_lower) && !type_name_lower.contains(&search_lower) {
                return false;
            }
        }
        
        true
    }
}

/// Builder for ArtifactQuery
#[derive(Default)]
pub struct ArtifactQueryBuilder {
    query: ArtifactQuery,
}

impl ArtifactQueryBuilder {
    pub fn workspace(mut self, id: WorkspaceId) -> Self {
        self.query.workspace_id = Some(id);
        self
    }
    
    pub fn platforms(mut self, platforms: Vec<Platform>) -> Self {
        self.query.platforms = Some(platforms);
        self
    }
    
    pub fn types(mut self, types: Vec<String>) -> Self {
        self.query.artifact_types = Some(types);
        self
    }
    
    pub fn created_between(mut self, after: DateTime<Utc>, before: DateTime<Utc>) -> Self {
        self.query.created_after = Some(after);
        self.query.created_before = Some(before);
        self
    }
    
    pub fn actionable_only(mut self) -> Self {
        self.query.actionable_only = true;
        self
    }
    
    pub fn with_participants(mut self, participants: Vec<String>) -> Self {
        self.query.participants = Some(participants);
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.query.tags = Some(tags);
        self
    }
    
    pub fn search(mut self, text: String) -> Self {
        self.query.text_search = Some(text);
        self
    }
    
    pub fn limit(mut self, limit: usize) -> Self {
        self.query.limit = Some(limit);
        self
    }
    
    pub fn sort_by(mut self, field: SortField, order: SortOrder) -> Self {
        self.query.sort_by = field;
        self.query.sort_order = order;
        self
    }
    
    pub fn build(self) -> ArtifactQuery {
        self.query
    }
}

/// Artifact statistics with advanced metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStats {
    pub total_count: u64,
    pub by_platform: HashMap<Platform, u64>,
    pub by_type: HashMap<String, u64>,
    pub by_state: HashMap<String, u64>,
    pub average_complexity: f64,
    pub median_complexity: f64,
    pub total_participants: u64,
    pub unique_participants: u64,
    pub created_today: u64,
    pub created_this_week: u64,
    pub created_this_month: u64,
    pub actionable_count: u64,
    pub high_priority_count: u64,
    pub blocked_count: u64,
    pub average_age_days: f64,
    pub velocity_metrics: VelocityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityMetrics {
    pub daily_creation_rate: f64,
    pub weekly_completion_rate: f64,
    pub average_resolution_time: Duration,
    pub throughput_trend: Trend,
}

impl ArtifactStats {
    pub fn calculate(artifacts: &[Artifact]) -> Self {
        if artifacts.is_empty() {
            return Self::empty();
        }
        
        let mut by_platform: HashMap<Platform, u64> = HashMap::new();
        let mut by_type: HashMap<String, u64> = HashMap::new();
        let mut by_state: HashMap<String, u64> = HashMap::new();
        let mut complexities: Vec<f64> = Vec::new();
        let mut unique_participants = HashSet::new();
        let mut total_age_days = 0.0;
        
        let now = Utc::now();
        let today = now.date_naive();
        let week_ago = now - Duration::weeks(1);
        let month_ago = now - Duration::days(30);
        
        let mut created_today = 0;
        let mut created_this_week = 0;
        let mut created_this_month = 0;
        let mut actionable_count = 0;
        let mut high_priority_count = 0;
        let mut blocked_count = 0;
        
        for artifact in artifacts {
            // Platform stats
            *by_platform.entry(artifact.platform).or_insert(0) += 1;
            
            // Type stats
            let type_name = artifact.artifact_type.type_name().to_string();
            *by_type.entry(type_name).or_insert(0) += 1;
            
            // State stats
            let state_name = format!("{:?}", artifact.state);
            *by_state.entry(state_name).or_insert(0) += 1;
            
            // Complexity
            let complexity = artifact.artifact_type.complexity_score();
            complexities.push(complexity);
            
            // Participants
            for participant in artifact.artifact_type.participants() {
                unique_participants.insert(participant);
            }
            
            // Age
            let age_days = (now - artifact.created_at).num_days() as f64;
            total_age_days += age_days;
            
            // Time-based stats
            if artifact.created_at.date_naive() == today {
                created_today += 1;
            }
            if artifact.created_at >= week_ago {
                created_this_week += 1;
            }
            if artifact.created_at >= month_ago {
                created_this_month += 1;
            }
            
            // Actionable
            if artifact.artifact_type.is_actionable() {
                actionable_count += 1;
            }
            
            // Priority
            if matches!(
                artifact.artifact_type.priority_level(),
                Priority::Critical | Priority::High
            ) {
                high_priority_count += 1;
            }
            
            // Blocked
            if let ArtifactType::Issue { blocked, .. } = &artifact.artifact_type {
                if *blocked {
                    blocked_count += 1;
                }
            }
            if let ArtifactType::Task { status, .. } = &artifact.artifact_type {
                if matches!(status, TaskStatus::Blocked) {
                    blocked_count += 1;
                }
            }
        }
        
        // Calculate average and median complexity
        let average_complexity = complexities.iter().sum::<f64>() / complexities.len() as f64;
        complexities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_complexity = if complexities.len() % 2 == 0 {
            let mid = complexities.len() / 2;
            (complexities[mid - 1] + complexities[mid]) / 2.0
        } else {
            complexities[complexities.len() / 2]
        };
        
        // Calculate velocity metrics
        let daily_creation_rate = created_today as f64;
        let weekly_completion_rate = artifacts.iter()
            .filter(|a| {
                if let ArtifactType::Task { completed_at, .. } = &a.artifact_type {
                    completed_at.map_or(false, |t| t >= week_ago)
                } else {
                    false
                }
            })
            .count() as f64 / 7.0;
        
        // Determine throughput trend
        let recent_count = artifacts.iter()
            .filter(|a| a.created_at >= week_ago)
            .count() as f64;
        let previous_week = now - Duration::weeks(2);
        let previous_count = artifacts.iter()
            .filter(|a| a.created_at >= previous_week && a.created_at < week_ago)
            .count() as f64;
        
        let throughput_trend = if recent_count > previous_count * 1.1 {
            Trend::Rising
        } else if recent_count < previous_count * 0.9 {
            Trend::Falling
        } else {
            Trend::Stable
        };
        
        Self {
            total_count: artifacts.len() as u64,
            by_platform,
            by_type,
            by_state,
            average_complexity,
            median_complexity,
            total_participants: unique_participants.len() as u64,
            unique_participants: unique_participants.len() as u64,
            created_today,
            created_this_week,
            created_this_month,
            actionable_count,
            high_priority_count,
            blocked_count,
            average_age_days: total_age_days / artifacts.len() as f64,
            velocity_metrics: VelocityMetrics {
                daily_creation_rate,
                weekly_completion_rate,
                average_resolution_time: Duration::days(3), // Placeholder
                throughput_trend,
            },
        }
    }
    
    fn empty() -> Self {
        Self {
            total_count: 0,
            by_platform: HashMap::new(),
            by_type: HashMap::new(),
            by_state: HashMap::new(),
            average_complexity: 0.0,
            median_complexity: 0.0,
            total_participants: 0,
            unique_participants: 0,
            created_today: 0,
            created_this_week: 0,
            created_this_month: 0,
            actionable_count: 0,
            high_priority_count: 0,
            blocked_count: 0,
            average_age_days: 0.0,
            velocity_metrics: VelocityMetrics {
                daily_creation_rate: 0.0,
                weekly_completion_rate: 0.0,
                average_resolution_time: Duration::seconds(0),
                throughput_trend: Trend::Stable,
            },
        }
    }
}

/// Batch processing for multiple artifacts
pub struct ArtifactBatch {
    artifacts: Vec<Artifact>,
    processor: Arc<ArtifactProcessor>,
}

impl ArtifactBatch {
    pub fn new(artifacts: Vec<Artifact>, processor: Arc<ArtifactProcessor>) -> Self {
        Self {
            artifacts,
            processor,
        }
    }
    
    /// Process all artifacts in parallel
    pub async fn process_all(&self) -> Vec<Result<ProcessingResult, ArtifactError>> {
        use futures::future::join_all;
        
        let futures = self.artifacts.iter().map(|artifact| {
            let processor = self.processor.clone();
            let artifact = artifact.clone();
            async move {
                processor.process(&artifact).await
            }
        });
        
        join_all(futures).await
    }
    
    /// Get statistics for the batch
    pub fn stats(&self) -> ArtifactStats {
        ArtifactStats::calculate(&self.artifacts)
    }
    
    /// Filter batch by query
    pub fn filter(&self, query: &ArtifactQuery) -> Vec<Artifact> {
        query.filter(&self.artifacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_artifact_creation() {
        let artifact = Artifact::new(
            WorkspaceId::new(),
            ArtifactType::PullRequest {
                number: 123,
                title: "Test PR".to_string(),
                state: PullRequestState::Open,
                files_changed: 5,
                additions: 100,
                deletions: 50,
                merged: false,
                draft: false,
                base_branch: "main".to_string(),
                head_branch: "feature".to_string(),
                author: "user1".to_string(),
                reviewers: vec!["user2".to_string()],
                labels: vec!["enhancement".to_string()],
                merge_conflict: false,
                ci_status: Some(CIStatus::Success),
            },
            Platform::GitHub,
            "Test pull request content".to_string(),
        ).unwrap();
        
        assert_eq!(artifact.platform, Platform::GitHub);
        assert!(matches!(artifact.artifact_type, ArtifactType::PullRequest { .. }));
        assert_eq!(artifact.state, ArtifactState::Pending);
    }
    
    #[test]
    fn test_artifact_extractor() {
        let extractor = ArtifactExtractor::new();
        let content = "Working on PR #123 and issue #456";
        
        // Test GitHub extraction
        let rt = tokio::runtime::Runtime::new().unwrap();
        let artifacts = rt.block_on(extractor.extract(content, Platform::GitHub)).unwrap();
        
        assert!(!artifacts.is_empty());
        // Should extract both PR and issue
        assert!(artifacts.iter().any(|a| matches!(a.artifact_type, ArtifactType::PullRequest { .. })));
        assert!(artifacts.iter().any(|a| matches!(a.artifact_type, ArtifactType::Issue { .. })));
    }
    
    #[test]
    fn test_complexity_scoring() {
        let pr = ArtifactType::PullRequest {
            number: 1,
            title: "Test".to_string(),
            state: PullRequestState::Open,
            files_changed: 10,
            additions: 500,
            deletions: 200,
            merged: false,
            draft: false,
            base_branch: "main".to_string(),
            head_branch: "feature".to_string(),
            author: "user1".to_string(),
            reviewers: vec!["user2".to_string(), "user3".to_string()],
            labels: Vec::new(),
            merge_conflict: true,
            ci_status: Some(CIStatus::Failed),
        };
        
        let score = pr.complexity_score();
        assert!(score > 0.0 && score <= 10.0);
    }
    
    #[tokio::test]
    async fn test_artifact_processor() {
        let processor = Arc::new(ArtifactProcessor::new(None));
        
        let artifact = Artifact::new(
            WorkspaceId::new(),
            ArtifactType::PullRequest {
                number: 42,
                title: "Feature: Add new capability".to_string(),
                state: PullRequestState::Open,
                files_changed: 100,
                additions: 2000,
                deletions: 500,
                merged: false,
                draft: false,
                base_branch: "main".to_string(),
                head_branch: "feature/new-capability".to_string(),
                author: "developer".to_string(),
                reviewers: vec!["reviewer1".to_string(), "reviewer2".to_string()],
                labels: vec!["enhancement".to_string(), "needs-review".to_string()],
                merge_conflict: true,
                ci_status: Some(CIStatus::Failed),
            },
            Platform::GitHub,
            "This PR adds a new capability to the system".to_string(),
        ).unwrap();
        
        let result = processor.process(&artifact).await.unwrap();
        
        // Check recommendations were generated
        assert!(!result.recommendations.is_empty());
        assert!(result.recommendations.iter().any(|r| r.contains("smaller chunks")));
        assert!(result.recommendations.iter().any(|r| r.contains("merge conflicts")));
        
        // Check risk assessment
        assert_eq!(result.risk_assessment.level, RiskLevel::High);
    }
}