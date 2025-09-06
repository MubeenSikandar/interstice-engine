//! # Core Types Module
//! 
//! Comprehensive type definitions for the INTERSTICE-ENGINE WorkOS.
//! Provides all shared types, identifiers, and data structures used across the system.
//interstice-core/src/types.rs
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};
use std::str::FromStr;
use std::time::Duration;

use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};
use strum::{Display as StrumDisplay, EnumString};
use uuid::Uuid;

/// Supported platforms for integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, StrumDisplay, EnumString)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum Platform {
    Slack,
    Teams,
    Jira,
    Asana,
    Notion,
    GitHub,
    VSCode,
    GoogleWorkspace,
    Monday,
    Trello,
    Zoom,
    Figma,
}

impl Platform {
    /// Get display name for the platform
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Slack => "Slack",
            Self::Teams => "Microsoft Teams",
            Self::Jira => "Jira",
            Self::Asana => "Asana",
            Self::Notion => "Notion",
            Self::GitHub => "GitHub",
            Self::VSCode => "VS Code",
            Self::GoogleWorkspace => "Google Workspace",
            Self::Monday => "Monday.com",
            Self::Trello => "Trello",
            Self::Zoom => "Zoom",
            Self::Figma => "Figma",
        }
    }
    
    /// Check if platform supports real-time updates
    pub fn supports_realtime(&self) -> bool {
        matches!(self, 
            Self::Slack | Self::Teams  | 
            Self::GitHub  
        )
    }
    
    /// Get rate limit for the platform (requests per minute)
    pub fn rate_limit(&self) -> u32 {
        match self {
            Self::Slack => 60,
            Self::Teams => 30,
            Self::GitHub => 5000,
            Self::Jira => 100,
            Self::Asana => 150,
            Self::Notion => 180,
            _ => 60,
        }
    }
}

/// Workspace identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkspaceId(pub Uuid);

impl WorkspaceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for WorkspaceId {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for WorkspaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for WorkspaceId {
    type Err = uuid::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// User identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub String);

impl UserId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for UserId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for UserId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl Display for UserId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Team identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TeamId(pub Uuid);

impl TeamId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TeamId {
    fn default() -> Self {
        Self::new()
    }
}

/// Project identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectId(pub Uuid);

impl ProjectId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ProjectId {
    fn default() -> Self {
        Self::new()
    }
}

/// Channel/Room identifier (platform-specific)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelId {
    pub platform: Platform,
    pub id: String,
}

impl ChannelId {
    pub fn new(platform: Platform, id: impl Into<String>) -> Self {
        Self {
            platform,
            id: id.into(),
        }
    }
}

/// Message identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId {
    pub platform: Platform,
    pub channel_id: String,
    pub message_id: String,
    pub thread_id: Option<String>,
}

/// Task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Integration identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntegrationId {
    pub workspace_id: WorkspaceId,
    pub platform: Platform,
    pub external_id: String,
}

/// Session identifier for user sessions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// Metric value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Duration(Duration),
    Timestamp(DateTime<Utc>),
    Json(serde_json::Value),
    Array(Vec<MetricValue>),
    Map(HashMap<String, MetricValue>),
}

impl MetricValue {
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }
    
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Integer(_) => "integer",
            Self::Float(_) => "float",
            Self::Boolean(_) => "boolean",
            Self::String(_) => "string",
            Self::Duration(_) => "duration",
            Self::Timestamp(_) => "timestamp",
            Self::Json(_) => "json",
            Self::Array(_) => "array",
            Self::Map(_) => "map",
        }
    }
}

/// Time range for queries and analytics
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

impl TimeRange {
    pub fn new(start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        Self { start, end }
    }
    
    pub fn duration(&self) -> chrono::Duration {
        self.end - self.start
    }
    
    pub fn contains(&self, timestamp: &DateTime<Utc>) -> bool {
        *timestamp >= self.start && *timestamp <= self.end
    }
    
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && self.end > other.start
    }
    
    /// Create a time range for the last N hours
    pub fn last_hours(hours: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::hours(hours);
        Self { start, end }
    }
    
    /// Create a time range for the last N days
    pub fn last_days(days: i64) -> Self {
        let end = Utc::now();
        let start = end - chrono::Duration::days(days);
        Self { start, end }
    }
    
    /// Create a time range for today
    pub fn today() -> Self {
        let now = Utc::now();
        let start = now.date_naive().and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let end = start + chrono::Duration::days(1);
        Self { start, end }
    }
    
    /// Create a time range for this week
    pub fn this_week() -> Self {
        let now = Utc::now();
        let weekday = now.weekday().num_days_from_monday();
        let start = (now - chrono::Duration::days(weekday as i64))
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let end = start + chrono::Duration::weeks(1);
        Self { start, end }
    }
    
    /// Create a time range for this month
    pub fn this_month() -> Self {
        let now = Utc::now();
        let start = now
            .date_naive()
            .with_day(1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        let end = if now.month() == 12 {
            start.with_year(now.year() + 1)
                .unwrap()
                .with_month(1)
                .unwrap()
        } else {
            start.with_month(now.month() + 1).unwrap()
        };
        Self { start, end }
    }
}

/// User activity status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UserStatus {
    Active,
    Away,
    Busy,
    DoNotDisturb,
    Offline,
    InMeeting,
    OnBreak,
    OutOfOffice,
}

impl UserStatus {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Active | Self::Away)
    }
    
    pub fn display_emoji(&self) -> &'static str {
        match self {
            Self::Active => "🟢",
            Self::Away => "🟡",
            Self::Busy => "🔴",
            Self::DoNotDisturb => "⛔",
            Self::Offline => "⚫",
            Self::InMeeting => "📅",
            Self::OnBreak => "☕",
            Self::OutOfOffice => "🏖️",
        }
    }
}

/// Priority levels for tasks and outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
    None,
}

impl Priority {
    pub fn weight(&self) -> u8 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Medium => 2,
            Self::Low => 1,
            Self::None => 0,
        }
    }
    
    pub fn color_hex(&self) -> &'static str {
        match self {
            Self::Critical => "#FF0000",
            Self::High => "#FF6B00",
            Self::Medium => "#FFC700",
            Self::Low => "#0066FF",
            Self::None => "#999999",
        }
    }
}

/// Event types for system events
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    // User events
    UserCreated,
    UserUpdated,
    UserDeleted,
    UserStatusChanged,
    UserLogin,
    UserLogout,
    
    // Workspace events
    WorkspaceCreated,
    WorkspaceUpdated,
    WorkspaceDeleted,
    WorkspaceMemberAdded,
    WorkspaceMemberRemoved,
    
    // Integration events
    IntegrationConnected,
    IntegrationDisconnected,
    IntegrationError,
    IntegrationRateLimited,
    
    // Task events
    TaskCreated,
    TaskUpdated,
    TaskCompleted,
    TaskDeleted,
    TaskAssigned,
    TaskCommented,
    
    // Outcome events
    OutcomeCreated,
    OutcomeUpdated,
    OutcomeCompleted,
    OutcomeDeleted,
    OutcomeProgressUpdated,
    
    // Message events
    MessageReceived,
    MessageSent,
    MessageUpdated,
    MessageDeleted,
    MessageReacted,
    
    // Analytics events
    MetricRecorded,
    AnomalyDetected,
    ReportGenerated,
    
    // System events
    SystemStartup,
    SystemShutdown,
    SystemError,
    SystemWarning,
    
    // Custom events
    Custom(String),
}

impl EventType {
    pub fn category(&self) -> EventCategory {
        match self {
            Self::UserCreated | Self::UserUpdated | Self::UserDeleted |
            Self::UserStatusChanged | Self::UserLogin | Self::UserLogout => EventCategory::User,
            
            Self::WorkspaceCreated | Self::WorkspaceUpdated | Self::WorkspaceDeleted |
            Self::WorkspaceMemberAdded | Self::WorkspaceMemberRemoved => EventCategory::Workspace,
            
            Self::IntegrationConnected | Self::IntegrationDisconnected |
            Self::IntegrationError | Self::IntegrationRateLimited => EventCategory::Integration,
            
            Self::TaskCreated | Self::TaskUpdated | Self::TaskCompleted |
            Self::TaskDeleted | Self::TaskAssigned | Self::TaskCommented => EventCategory::Task,
            
            Self::OutcomeCreated | Self::OutcomeUpdated | Self::OutcomeCompleted |
            Self::OutcomeDeleted | Self::OutcomeProgressUpdated => EventCategory::Outcome,
            
            Self::MessageReceived | Self::MessageSent | Self::MessageUpdated |
            Self::MessageDeleted | Self::MessageReacted => EventCategory::Message,
            
            Self::MetricRecorded | Self::AnomalyDetected | Self::ReportGenerated => EventCategory::Analytics,
            
            Self::SystemStartup | Self::SystemShutdown | Self::SystemError |
            Self::SystemWarning => EventCategory::System,
            
            Self::Custom(_) => EventCategory::Custom,
        }
    }
}

/// Event categories for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventCategory {
    User,
    Workspace,
    Integration,
    Task,
    Outcome,
    Message,
    Analytics,
    System,
    Custom,
}

/// System event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    pub id: Uuid,
    pub event_type: EventType,
    pub workspace_id: Option<WorkspaceId>,
    pub user_id: Option<UserId>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub correlation_id: Option<Uuid>,
    pub platform: Option<Platform>,
}

impl SystemEvent {
    pub fn new(event_type: EventType) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            workspace_id: None,
            user_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
            correlation_id: None,
            platform: None,
        }
    }
    
    pub fn with_workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }
    
    pub fn with_user(mut self, user_id: UserId) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn with_correlation_id(mut self, id: Uuid) -> Self {
        self.correlation_id = Some(id);
        self
    }
    
    pub fn with_platform(mut self, platform: Platform) -> Self {
        self.platform = Some(platform);
        self
    }
}

/// Context object for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub workspace_id: WorkspaceId,
    pub user_id: Option<UserId>,
    pub team_id: Option<TeamId>,
    pub project_id: Option<ProjectId>,
    pub platform: Option<Platform>,
    pub session_id: Option<SessionId>,
    pub correlation_id: Uuid,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

impl Context {
    pub fn new(workspace_id: WorkspaceId) -> Self {
        Self {
            workspace_id,
            user_id: None,
            team_id: None,
            project_id: None,
            platform: None,
            session_id: None,
            correlation_id: Uuid::new_v4(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }
    
    pub fn with_user(mut self, user_id: UserId) -> Self {
        self.user_id = Some(user_id);
        self
    }
    
    pub fn with_team(mut self, team_id: TeamId) -> Self {
        self.team_id = Some(team_id);
        self
    }
    
    pub fn with_project(mut self, project_id: ProjectId) -> Self {
        self.project_id = Some(project_id);
        self
    }
    
    pub fn with_platform(mut self, platform: Platform) -> Self {
        self.platform = Some(platform);
        self
    }
    
    pub fn with_session(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }
    
    pub fn add_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Resource limits for workspaces
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_users: Option<usize>,
    pub max_integrations: Option<usize>,
    pub max_storage_gb: Option<f64>,
    pub max_api_calls_per_day: Option<usize>,
    pub max_outcomes_per_month: Option<usize>,
    pub max_ai_tokens_per_month: Option<usize>,
    pub retention_days: Option<u32>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_users: Some(100),
            max_integrations: Some(10),
            max_storage_gb: Some(100.0),
            max_api_calls_per_day: Some(10_000),
            max_outcomes_per_month: Some(1_000),
            max_ai_tokens_per_month: Some(1_000_000),
            retention_days: Some(90),
        }
    }
}

/// Workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub id: WorkspaceId,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub owner_id: UserId,
    pub timezone: String,
    pub locale: String,
    pub enabled_platforms: HashSet<Platform>,
    pub resource_limits: ResourceLimits,
    pub features: HashSet<Feature>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Feature flags for workspaces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    Analytics,
    AnomalyDetection,
    AutomatedOutcomes,
    CustomIntegrations,
    AdvancedReporting,
    RealTimeSync,
    AiSuggestions,
    TeamCollaboration,
    ApiAccess,
    DataExport,
    CustomBranding,
    SsoAuthentication,
    AuditLogs,
    ComplianceMode,
}

/// Permissions for access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    // Workspace permissions
    WorkspaceView,
    WorkspaceEdit,
    WorkspaceDelete,
    WorkspaceManageMembers,
    WorkspaceManageIntegrations,
    WorkspaceManageBilling,
    
    // User permissions
    UserView,
    UserEdit,
    UserDelete,
    UserInvite,
    UserManageRoles,
    
    // Outcome permissions
    OutcomeView,
    OutcomeCreate,
    OutcomeEdit,
    OutcomeDelete,
    OutcomeAssign,
    
    // Task permissions
    TaskView,
    TaskCreate,
    TaskEdit,
    TaskDelete,
    TaskAssign,
    
    // Analytics permissions
    AnalyticsView,
    AnalyticsExport,
    AnalyticsManage,
    
    // System permissions
    SystemAdmin,
    SystemDebug,
}

/// Role definition for RBAC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub permissions: HashSet<Permission>,
    pub is_system: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// User role assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserRole {
    pub user_id: UserId,
    pub role_id: Uuid,
    pub workspace_id: WorkspaceId,
    pub assigned_at: DateTime<Utc>,
    pub assigned_by: UserId,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Authentication token types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    Access,
    Refresh,
    ApiKey,
    WebhookSecret,
    IntegrationToken,
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub id: Uuid,
    pub token_type: TokenType,
    pub user_id: Option<UserId>,
    pub workspace_id: Option<WorkspaceId>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Health status for system components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl HealthStatus {
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
}

/// Component health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub component: String,
    pub status: HealthStatus,
    pub latency_ms: Option<u64>,
    pub message: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub checked_at: DateTime<Utc>,
}

/// Notification channel types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannel {
    Email,
    Slack,
    Teams,
    InApp,
    Webhook,
    Sms,
    Push,
}

/// Notification preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreference {
    pub user_id: UserId,
    pub workspace_id: WorkspaceId,
    pub channel: NotificationChannel,
    pub event_types: HashSet<EventType>,
    pub enabled: bool,
    pub settings: HashMap<String, serde_json::Value>,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub workspace_id: WorkspaceId,
    pub user_id: Option<UserId>,
    pub action: String,
    pub resource_type: String,
    pub resource_id: String,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Data retention policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionPolicy {
    Days(u32),
    Months(u32),
    Years(u32),
    Forever,
}

impl RetentionPolicy {
    pub fn to_days(&self) -> Option<u32> {
        match self {
            Self::Days(d) => Some(*d),
            Self::Months(m) => Some(m * 30),
            Self::Years(y) => Some(y * 365),
            Self::Forever => None,
        }
    }
}

/// Batch operation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
    PartiallyCompleted,
}

impl BatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// Batch operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult<T> {
    pub id: Uuid,
    pub status: BatchStatus,
    pub total_items: usize,
    pub processed_items: usize,
    pub successful_items: usize,
    pub failed_items: usize,
    pub results: Vec<Result<T, String>>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_summary: Option<String>,
}

/// Rate limit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInfo {
    pub limit: u32,
    pub remaining: u32,
    pub reset_at: DateTime<Utc>,
    pub window_seconds: u64,
}

/// Pagination parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    pub page: u32,
    pub per_page: u32,
    pub sort_by: Option<String>,
    pub sort_order: SortOrder,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: 1,
            per_page: 50,
            sort_by: None,
            sort_order: SortOrder::Desc,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Asc,
    Desc,
}

/// Paginated response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub data: Vec<T>,
    pub pagination: PaginationMeta,
}

/// Pagination metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationMeta {
    pub page: u32,
    pub per_page: u32,
    pub total_pages: u32,
    pub total_items: usize,
    pub has_next: bool,
    pub has_prev: bool,
}

impl PaginationMeta {
    pub fn new(page: u32, per_page: u32, total_items: usize) -> Self {
        let total_pages = ((total_items as f64) / (per_page as f64)).ceil() as u32;
        Self {
            page,
            per_page,
            total_pages,
            total_items,
            has_next: page < total_pages,
            has_prev: page > 1,
        }
    }
}

/// Cache control directives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    pub max_age: Option<Duration>,
    pub must_revalidate: bool,
    pub no_cache: bool,
    pub no_store: bool,
    pub private: bool,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            max_age: Some(Duration::from_secs(300)),
            must_revalidate: false,
            no_cache: false,
            no_store: false,
            private: false,
        }
    }
}

/// Encryption type for sensitive data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncryptionType {
    Aes256Gcm,
    ChaCha20Poly1305,
    RsaOaep,
    None,
}

/// Sensitive data wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitiveData {
    pub encryption_type: EncryptionType,
    pub encrypted_data: Vec<u8>,
    pub nonce: Option<Vec<u8>>,
    pub key_id: Option<String>,
}

/// File attachment metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAttachment {
    pub id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub sha256_hash: String,
    pub storage_path: String,
    pub uploaded_by: UserId,
    pub uploaded_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Webhook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookConfig {
    pub id: Uuid,
    pub workspace_id: WorkspaceId,
    pub url: String,
    pub events: HashSet<EventType>,
    pub secret: SensitiveData,
    pub enabled: bool,
    pub retry_config: RetryConfig,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Retry configuration for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_backoff: bool,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            exponential_backoff: true,
            jitter: true,
        }
    }
}

/// Geographic location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy_meters: Option<f64>,
    pub altitude_meters: Option<f64>,
    pub country_code: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
}

/// IP address information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpInfo {
    pub ip: String,
    pub location: Option<GeoLocation>,
    pub isp: Option<String>,
    pub organization: Option<String>,
    pub is_vpn: Option<bool>,
    pub is_proxy: Option<bool>,
}

/// User agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAgentInfo {
    pub raw: String,
    pub browser: Option<String>,
    pub browser_version: Option<String>,
    pub os: Option<String>,
    pub os_version: Option<String>,
    pub device_type: Option<DeviceType>,
}

/// Device type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceType {
    Desktop,
    Mobile,
    Tablet,
    Tv,
    Console,
    Wearable,
    Unknown,
}

/// Request context for API calls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestContext {
    pub request_id: Uuid,
    pub workspace_id: Option<WorkspaceId>,
    pub user_id: Option<UserId>,
    pub session_id: Option<SessionId>,
    pub ip_info: Option<IpInfo>,
    pub user_agent_info: Option<UserAgentInfo>,
    pub timestamp: DateTime<Utc>,
    pub path: String,
    pub method: HttpMethod,
}

/// HTTP method types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Patch,
    Delete,
    Head,
    Options,
}

/// Feature usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureUsage {
    pub workspace_id: WorkspaceId,
    pub feature: Feature,
    pub usage_count: u64,
    pub last_used_at: DateTime<Utc>,
    pub first_used_at: DateTime<Utc>,
    pub unique_users: HashSet<UserId>,
}

/// Cost tracking for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostTracking {
    pub workspace_id: WorkspaceId,
    pub operation_type: String,
    pub resource_type: String,
    pub quantity: f64,
    pub unit_cost: f64,
    pub total_cost: f64,
    pub currency: Currency,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Currency types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Currency {
    Usd,
    Eur,
    Gbp,
    Jpy,
    Cad,
    Aud,
    Cny,
    Inr,
}

impl Currency {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Usd => "$",
            Self::Eur => "€",
            Self::Gbp => "£",
            Self::Jpy => "¥",
            Self::Cad => "C$",
            Self::Aud => "A$",
            Self::Cny => "¥",
            Self::Inr => "₹",
        }
    }
}

/// ML model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub model_id: String,
    pub model_type: ModelType,
    pub version: String,
    pub provider: String,
    pub capabilities: HashSet<String>,
    pub context_window: Option<usize>,
    pub max_tokens: Option<usize>,
    pub cost_per_token: Option<f64>,
}

/// ML model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    TextGeneration,
    TextEmbedding,
    ImageGeneration,
    ImageAnalysis,
    AudioTranscription,
    AudioGeneration,
    Classification,
    Regression,
    Clustering,
    Custom,
}

/// Deployment environment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Environment {
    Development,
    Staging,
    Production,
    Testing,
}

impl Environment {
    pub fn is_production(&self) -> bool {
        matches!(self, Self::Production)
    }
    
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Development => "dev",
            Self::Staging => "stage",
            Self::Production => "prod",
            Self::Testing => "test",
        }
    }
}

/// Service mesh configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMeshConfig {
    pub service_name: String,
    pub namespace: String,
    pub version: String,
    pub replicas: u32,
    pub health_check_path: String,
    pub timeout_seconds: u64,
    pub circuit_breaker: CircuitBreakerConfig,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_seconds: u64,
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 2,
            timeout_seconds: 60,
            half_open_max_calls: 3,
        }
    }
}

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub export_interval_seconds: u64,
    pub max_batch_size: usize,
    pub endpoints: TelemetryEndpoints,
}

/// Telemetry export endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEndpoints {
    pub traces: Option<String>,
    pub metrics: Option<String>,
    pub logs: Option<String>,
}

/// Data classification for compliance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    Pii,
    Phi,
}

impl DataClassification {
    pub fn requires_encryption(&self) -> bool {
        matches!(self, Self::Confidential | Self::Restricted | Self::Pii | Self::Phi)
    }
    
    pub fn retention_days(&self) -> u32 {
        match self {
            Self::Public => 365,
            Self::Internal => 180,
            Self::Confidential => 90,
            Self::Restricted => 30,
            Self::Pii => 30,
            Self::Phi => 7,
        }
    }
}

/// Compliance standard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ComplianceStandard {
    Gdpr,
    Ccpa,
    Hipaa,
    Sox,
    Pci,
    Iso27001,
    Soc2,
}

/// Data residency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResidency {
    pub workspace_id: WorkspaceId,
    pub required_regions: HashSet<String>,
    pub prohibited_regions: HashSet<String>,
    pub compliance_standards: HashSet<ComplianceStandard>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workspace_id_generation() {
        let id1 = WorkspaceId::new();
        let id2 = WorkspaceId::new();
        assert_ne!(id1, id2);
        assert_eq!(id1, id1);
    }
    
    #[test]
    fn test_time_range_operations() {
        let range = TimeRange::last_days(7);
        assert_eq!(range.duration().num_days(), 7);
        
        let now = Utc::now();
        assert!(range.contains(&now));
        
        let past = now - chrono::Duration::days(10);
        assert!(!range.contains(&past));
    }
    
    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
        assert!(Priority::Low > Priority::None);
    }
    
    #[test]
    fn test_platform_capabilities() {
        assert!(Platform::Slack.supports_realtime());
        assert!(!Platform::Jira.supports_realtime());
        assert_eq!(Platform::GitHub.rate_limit(), 5000);
    }
    
    #[test]
    fn test_metric_value_conversions() {
        let float_val = MetricValue::Float(42.5);
        assert_eq!(float_val.as_float(), Some(42.5));
        assert_eq!(float_val.as_integer(), Some(42));
        
        let int_val = MetricValue::Integer(100);
        assert_eq!(int_val.as_float(), Some(100.0));
        assert_eq!(int_val.as_integer(), Some(100));
        
        let str_val = MetricValue::String("test".to_string());
        assert_eq!(str_val.as_string(), Some("test"));
        assert_eq!(str_val.as_float(), None);
    }
    
    #[test]
    fn test_event_type_categories() {
        assert_eq!(EventType::UserCreated.category(), EventCategory::User);
        assert_eq!(EventType::TaskCompleted.category(), EventCategory::Task);
        assert_eq!(EventType::AnomalyDetected.category(), EventCategory::Analytics);
    }
    
    #[test]
    fn test_context_builder() {
        let workspace_id = WorkspaceId::new();
        let user_id = UserId::from("user123");
        
        let context = Context::new(workspace_id)
            .with_user(user_id.clone())
            .with_platform(Platform::Slack)
            .add_metadata("key".to_string(), serde_json::json!("value"));
        
        assert_eq!(context.workspace_id, workspace_id);
        assert_eq!(context.user_id, Some(user_id));
        assert_eq!(context.platform, Some(Platform::Slack));
        assert!(context.metadata.contains_key("key"));
    }
    
    #[test]
    fn test_pagination_meta() {
        let meta = PaginationMeta::new(2, 20, 100);
        assert_eq!(meta.total_pages, 5);
        assert!(meta.has_next);
        assert!(meta.has_prev);
        
        let first_page = PaginationMeta::new(1, 20, 100);
        assert!(first_page.has_next);
        assert!(!first_page.has_prev);
        
        let last_page = PaginationMeta::new(5, 20, 100);
        assert!(!last_page.has_next);
        assert!(last_page.has_prev);
    }
    
    #[test]
    fn test_retention_policy() {
        assert_eq!(RetentionPolicy::Days(30).to_days(), Some(30));
        assert_eq!(RetentionPolicy::Months(3).to_days(), Some(90));
        assert_eq!(RetentionPolicy::Years(1).to_days(), Some(365));
        assert_eq!(RetentionPolicy::Forever.to_days(), None);
    }
    
    #[test]
    fn test_batch_status() {
        assert!(BatchStatus::Completed.is_terminal());
        assert!(BatchStatus::Failed.is_terminal());
        assert!(!BatchStatus::Processing.is_terminal());
        assert!(!BatchStatus::Pending.is_terminal());
    }
}