//! # Platform Adapter Traits
//! 
//! Core trait definitions for platform-agnostic adapter implementations.
//! Provides a unified interface for integrating with various work platforms.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

use interstice_core::{
    Artifact, Platform, ProcessedData, Result as CoreResult, WorkspaceId,
};

/// Adapter-specific error types
#[derive(Error, Debug)]
pub enum AdapterError {
    /// Platform is not supported by this adapter
    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),
    
    /// Failed to parse incoming event data
    #[error("Event parsing failed: {0}")]
    EventParsingError(String),
    
    /// Authentication credentials are invalid or expired
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    /// API rate limit has been exceeded
    #[error("Rate limit exceeded: retry after {retry_after} seconds")]
    RateLimitExceeded { 
        /// Number of seconds to wait before retrying
        retry_after: u64 
    },
    
    /// Webhook signature verification failed
    #[error("Webhook verification failed")]
    WebhookVerificationFailed,
    
    /// Network connectivity or HTTP transport error
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// JSON serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    /// Adapter configuration is invalid or incomplete
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Platform API returned an error response
    #[error("Platform API error: {status} - {message}")]
    PlatformApiError { 
        /// HTTP status code from the platform API
        status: u16, 
        /// Error message from the platform API
        message: String 
    },
}

/// Core trait for platform adapters
#[async_trait]
pub trait PlatformAdapter: Send + Sync {
    /// Get the platform this adapter handles
    fn platform(&self) -> Platform;
    
    /// Get adapter metadata
    fn metadata(&self) -> AdapterMetadata;
    
    /// Check if the adapter is properly configured and ready
    async fn health_check(&self) -> Result<HealthStatus, AdapterError>;
    
    /// Process an incoming webhook/event
    async fn process_event(&self, event: PlatformEvent) -> CoreResult<ProcessedData>;
    
    /// Send a response back to the platform
    async fn send_response(&self, response: PlatformResponse) -> CoreResult<()>;
    
    /// Fetch historical data from the platform
    async fn fetch_history(
        &self,
        params: HistoryParams,
    ) -> CoreResult<Vec<Artifact>>;
    
    /// Subscribe to real-time updates (if supported)
    async fn subscribe(
        &self,
        subscription: Subscription,
    ) -> CoreResult<SubscriptionHandle>;
    
    /// Unsubscribe from updates
    async fn unsubscribe(&self, handle: SubscriptionHandle) -> CoreResult<()>;
    
    /// Get current rate limit status
    async fn rate_limit_status(&self) -> Result<RateLimitStatus, AdapterError>;
    
    /// Authenticate/refresh credentials
    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, AdapterError>;
    
    /// Validate webhook signature (if applicable)
    fn verify_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<bool, AdapterError>;
}

/// Extended adapter capabilities
#[async_trait]
pub trait ExtendedAdapter: PlatformAdapter {
    /// Create a new item on the platform
    async fn create_item(&self, item: CreateItemRequest) -> CoreResult<ItemResponse>;
    
    /// Update an existing item
    async fn update_item(&self, item: UpdateItemRequest) -> CoreResult<ItemResponse>;
    
    /// Delete an item
    async fn delete_item(&self, id: ItemId) -> CoreResult<()>;
    
    /// Search for items
    async fn search(&self, query: SearchQuery) -> CoreResult<SearchResults>;
    
    /// Get user information
    async fn get_user(&self, user_id: &str) -> CoreResult<UserInfo>;
    
    /// List available channels/spaces
    async fn list_channels(&self) -> CoreResult<Vec<ChannelInfo>>;
    
    /// Get platform-specific configuration options
    async fn get_config_schema(&self) -> CoreResult<ConfigSchema>;
}

/// Comprehensive metadata describing an adapter's identity and capabilities.
///
/// This structure provides essential information about a platform adapter, including
/// its version, supported features, and documentation references. Used for runtime
/// discovery, compatibility checks, and administrative interfaces.
///
/// # Examples
///
/// ```rust
/// use interstice_adapters::AdapterMetadata;
/// use interstice_core::Platform;
///
/// let metadata = AdapterMetadata {
///     name: "Slack Official Adapter".to_string(),
///     version: "2.1.0".to_string(),
///     platform: Platform::Slack,
///     capabilities: AdapterCapabilities::default(),
///     author: "Interstice Team".to_string(),
///     description: "Production-ready Slack integration with ML support".to_string(),
///     documentation_url: Some("https://docs.interstice.dev/adapters/slack".to_string()),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    /// Human-readable adapter name
    pub name: String,
    /// Semantic version string (e.g., "1.2.3")
    pub version: String,
    /// Target platform for this adapter
    pub platform: Platform,
    /// Feature capabilities supported by this adapter
    pub capabilities: AdapterCapabilities,
    /// Adapter author or organization
    pub author: String,
    /// Brief description of adapter functionality
    pub description: String,
    /// Optional URL to adapter documentation
    pub documentation_url: Option<String>,
}
///AdapterCapabilities struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    /// Supports real-time event streaming
    pub real_time: bool,
    /// Supports webhook event delivery
    pub webhooks: bool,
    /// Supports polling-based event retrieval
    pub polling: bool,
    /// Can send responses back to platform
    pub bidirectional: bool,
    /// Supports file upload operations
    pub file_upload: bool,
    /// Supports rich text and block formatting
    pub rich_formatting: bool,
    /// Supports threaded conversations
    pub threading: bool,
    /// Supports emoji reactions and interactions
    pub reactions: bool,
    /// Supports content search functionality
    pub search: bool,
    /// Can track user presence and status
    pub user_presence: bool,
    /// Supports custom metadata fields
    pub custom_fields: bool,
    /// Supports bulk operations for efficiency
    pub bulk_operations: bool,
}

impl Default for AdapterCapabilities {
    fn default() -> Self {
        Self {
            real_time: false,
            webhooks: true,
            polling: true,
            bidirectional: false,
            file_upload: false,
            rich_formatting: true,
            threading: false,
            reactions: false,
            search: false,
            user_presence: false,
            custom_fields: false,
            bulk_operations: false,
        }
    }
}

/// platform Event Struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformEvent {
    /// Unique identifier for event tracking and deduplication
    pub id: Uuid,
    /// Source platform that generated this event
    pub platform: Platform,
    /// Classification of the event type
    pub event_type: EventType,
    /// Optional workspace context for multi-tenant systems
    pub workspace_id: Option<WorkspaceId>,
    /// When the event occurred (UTC)
    pub timestamp: DateTime<Utc>,
    /// Original platform-specific event data
    pub raw_data: Value,
    /// Additional contextual information and processing metadata
    pub metadata: EventMetadata,
}

impl PlatformEvent {
    /// Create a new platform event
    pub fn new(platform: Platform, event_type: EventType, raw_data: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            platform,
            event_type,
            workspace_id: None,
            timestamp: Utc::now(),
            raw_data,
            metadata: EventMetadata::default(),
        }
    }
    
    /// Add workspace context to the event
    pub fn with_workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, metadata: EventMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    // Message events

    ///Message New Enum
    MessageNew,
    ///Message Updated Enum
    MessageUpdated,
    ///Message Deleted Enum
    MessageDeleted,
    ///Message Reaction Enum
    MessageReaction,
    
    // Channel events
    ///Channel Created Enum
    ChannelCreated,
    ///Channel Updated Enum
    ChannelUpdated,
    ///Channel Deleted Enum
    ChannelDeleted,
    ///Channel Joined Enum
    ChannelJoined,
    ///Channel Left Enum
    ChannelLeft,
    
    // User events
    /// User Joined Enum
    UserJoined,
    /// User Left Enum
    UserLeft,
    /// User Updated Enum
    UserUpdated,
    /// User Presence Changed Enum
    UserPresenceChanged,
    
    // File events
    /// File Shared Enum
    FileShared,
    /// File Deleted Enum
    FileDeleted,
    /// File Commented Enum
    FileCommented,
    
    // Task/Issue events
    /// Task Created Enum
    TaskCreated,

    /// Task Updated Enum
    TaskUpdated,
    /// Task Completed Enum
    TaskCompleted,
    /// Task Assigned Enum
    TaskAssigned,
    /// Task Commented Enum
    TaskCommented,
    
    // Workflow events
    /// Workflow Started Enum
    WorkflowStarted,
    /// Workflow Completed Enum
    WorkflowCompleted,
    /// Workflow Failed Enum
    WorkflowFailed,
    
    // Integration events
    /// App Mention Enum
    AppMention,
    /// Slash Command Enum
    SlashCommand,
    /// Interactive Action Enum
    InteractiveAction,
    
    // System events
    /// Rate Limited Enum
    RateLimited,
    /// Connection Lost Enum
    ConnectionLost,
    /// Connection Restored Enum
    ConnectionRestored,
    
    // Custom events
    /// Custom Enum
    Custom(String),
}

/// Event Metadata Struct
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    /// User ID
    pub user_id: Option<String>,
    /// Channel ID
    pub channel_id: Option<String>,
    /// Thread ID   
    pub thread_id: Option<String>,
    /// Retry Count
    pub retry_count: u32,
    /// Is Duplicate
    pub is_duplicate: bool,
    /// Correlation ID
    pub correlation_id: Option<Uuid>,
    /// Source IP
    pub source_ip: Option<String>,
    /// Custom
    pub custom: HashMap<String, Value>,
}

/// Platform Response Struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformResponse {
    /// Target
    pub target: ResponseTarget,
    /// Content
    pub content: ResponseContent,
    /// Options
    pub options: ResponseOptions,
}

impl PlatformResponse {
    /// Create a new platform response
    pub fn new(target: ResponseTarget, content: ResponseContent) -> Self {
        Self {
            target,
            content,
            options: ResponseOptions::default(),
        }
    }
    
    /// Add options to the response
    pub fn with_options(mut self, options: ResponseOptions) -> Self {
        self.options = options;
        self
    }
}

/// Response target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseTarget {
    /// Send response to a specific channel
    Channel {
        /// Channel ID
         id: String },
    /// Send response to a specific user
    User { 
        /// User ID
        id: String },
    
    /// Send response to a specific thread within a channel
    Thread { 
        /// Channel ID
        channel_id: String, 
        /// Thread ID
        thread_id: String 
    },
    /// Send response to multiple channels simultaneously
    Broadcast { 
        /// Channel IDs
        channel_ids: Vec<String> },
}

/// Response content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseContent {
    /// Text content
    Text(String),
    /// Markdown content
    Markdown(String),
    /// HTML content
    Html(String),
    /// Blocks content
    Blocks(Vec<BlockElement>),
    /// File content
    File(FileContent),
    /// Custom content
    Custom(Value),
}

/// Block elements for rich formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockElement {
    /// Section block
    Section {
        /// Text
        text: String,
        /// Fields
        fields: Option<Vec<TextField>>,
        /// Accessory
        accessory: Option<Box<BlockElement>>,
    },
    /// Header block
    Header {
        /// Text
        text: String,
    },
    /// Divider block
    Divider,
    /// Image block
    Image {
        /// URL
        url: String,
        /// Alt text
        alt_text: String,
        /// Title
        title: Option<String>,
    },
    /// Actions block
    Actions {
        /// Elements
        elements: Vec<ActionElement>,
    },
    /// Context block
    Context {
        /// Elements
        elements: Vec<ContextElement>,
    },
    /// Input block
    Input {
        /// Label
        label: String,
        /// Element
        element: InputElement,
        /// Hint
        hint: Option<String>,
        /// Optional
        optional: bool,
    },
}

/// Text field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextField {
    /// Title
    pub title: String,
    /// Value
    pub value: String,
    /// Short
    pub short: bool,
}

/// Action elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionElement {
    /// Button element
    Button {
        /// Text
        text: String,
        /// Action ID
        action_id: String,
        /// URL
        url: Option<String>,
        /// Value
        value: Option<String>,
        /// Style
        style: Option<ButtonStyle>,
    },
    /// Select element
    Select {
        /// Placeholder
        placeholder: String,
        /// Action ID
        action_id: String,
        /// Options
        options: Vec<SelectOption>,
    },
    /// Date picker element
    DatePicker {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: String,
        /// Initial date
        initial_date: Option<String>,
    },
    /// Overflow element
    Overflow {
        /// Action ID
        action_id: String,
        /// Options
        options: Vec<SelectOption>,
    },
}

/// Button styles
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ButtonStyle {
    /// Primary style
    Primary,
    /// Danger style
    Danger,
    /// Default style
    Default,
}
/// Select option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectOption {
    /// Text
    pub text: String,
    /// Value
    pub value: String,
    /// Description
    pub description: Option<String>,
}

/// Context elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextElement {
    /// Text element
    Text(String),
    /// Image element
    Image { 
        /// URL
        url: String, 
        /// Alt text
        alt_text: String 
    },
}

/// Input elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputElement {
    /// Plain text input
    PlainTextInput {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: Option<String>,
        /// Initial value
        initial_value: Option<String>,
        /// Multiline
        multiline: bool,
        /// Min length
        min_length: Option<u32>,
        /// Max length
        max_length: Option<u32>,
    },
    /// Select input
    Select {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: String,
        /// Options
        options: Vec<SelectOption>,
    },
    /// Multi-select input
    MultiSelect {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: String,
        /// Options
        options: Vec<SelectOption>,
        /// Max selected items
        max_selected_items: Option<u32>,
    },
    /// Date picker input
    DatePicker {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: Option<String>,
        /// Initial date
        initial_date: Option<String>,
    },
    /// Time picker input
    TimePicker {
        /// Action ID
        action_id: String,
        /// Placeholder
        placeholder: Option<String>,
        /// Initial time
        initial_time: Option<String>,
    },
}

/// File content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    /// Filename
    pub filename: String,
    /// MIME type
    pub mime_type: String,
    /// Data
    pub data: Vec<u8>,
    /// Title
    pub title: Option<String>,
    /// Description
    pub description: Option<String>,
}
/// Response options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseOptions {
    /// Ephemeral
    pub ephemeral: bool,
    /// In thread
    pub in_thread: bool,
    /// Replace original
    pub replace_original: bool,
    /// Delete original
    pub delete_original: bool,
    /// Unfurl links
    pub unfurl_links: bool,
    /// Unfurl media
    pub unfurl_media: bool,
    /// Notification text
    pub notification_text: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// History params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryParams {
    /// Channel ID
    pub channel_id: Option<String>,
    /// Start date
    pub start_date: DateTime<Utc>,
    /// End date
    pub end_date: DateTime<Utc>,
    /// Limit
    pub limit: Option<usize>,
    /// Include deleted
    pub include_deleted: bool,
    /// Artifact types
    pub artifact_types: Option<Vec<String>>,
}

/// Subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    /// ID
    pub id: Uuid,
    /// Events
    pub events: Vec<EventType>,
    /// Filters
    pub filters: SubscriptionFilters,
    /// Callback URL
    pub callback_url: Option<String>,
}

impl Subscription {
    /// Create a new subscription
    pub fn new(events: Vec<EventType>) -> Self {
        Self {
            id: Uuid::new_v4(),
            events,
            filters: SubscriptionFilters::default(),
            callback_url: None,
        }
    }
}

/// Subscription filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubscriptionFilters {
    /// Channels
    pub channels: Option<Vec<String>>,
    /// Users
    pub users: Option<Vec<String>>,
    /// Keywords
    pub keywords: Option<Vec<String>>,
    /// Exclude bots
    pub exclude_bots: bool,
}

/// Subscription handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionHandle {
    /// ID
    pub id: Uuid,
    /// Platform
    pub platform: Platform,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Events
    pub events: Vec<EventType>,
}

/// Rate limit status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    /// Limit
    pub limit: u32,
    /// Remaining
    pub remaining: u32,
    /// Reset at
    pub reset_at: DateTime<Utc>,
    /// Window seconds
    pub window_seconds: u64,
}

impl RateLimitStatus {
    /// Is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.remaining == 0
    }
    
    /// Seconds until reset
    pub fn seconds_until_reset(&self) -> i64 {
        (self.reset_at - Utc::now()).num_seconds().max(0)
    }
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Status
    pub status: HealthState,
    /// Message
    pub message: Option<String>,
    /// Last successful event
    pub last_successful_event: Option<DateTime<Utc>>,
    /// Error count
    pub error_count: u32,
    /// Metrics
    pub metrics: HashMap<String, f64>,
}

/// Health state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    /// Healthy
    Healthy,
    /// Degraded
    Degraded,
    /// Unhealthy
    Unhealthy,
    /// Unknown
    Unknown,
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthCredentials {
    /// OAuth
    OAuth {
        /// Client ID   
        client_id: String,
        /// Client secret
        client_secret: String,
        /// Redirect URI
        redirect_uri: String,
        /// Scopes
        scopes: Vec<String>,
    },
    /// API key
    ApiKey {
        /// Key
        key: String,
    },
    /// Bot token
    BotToken {
        /// Token
        token: String,
    },
    /// User token
    UserToken {
        /// Access token
        access_token: String,
        /// Refresh token
        refresh_token: Option<String>,
    },
    /// Custom
    Custom(HashMap<String, String>),
}

/// Auth token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Access token
    pub access_token: String,
    /// Token type
    pub token_type: String,
    /// Expires at
    pub expires_at: Option<DateTime<Utc>>,
    /// Refresh token
    pub refresh_token: Option<String>,
    /// Scopes
    pub scopes: Vec<String>,
}

/// Item id
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemId {
    /// Platform
    pub platform: Platform,
    /// ID
    pub id: String,
    /// Item type
    pub item_type: ItemType,
}

/// Item types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemType {
    /// Message
    Message,
    /// Task
    Task,
    /// Issue
    Issue,
    /// Document
    Document,
    /// File
    File,
    /// User
    User,
    /// Channel
    Channel,
    /// Project
    Project,
    /// Custom
    Custom(String),
}

/// Create item request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateItemRequest {
    /// Item type
    pub item_type: ItemType,
    /// Title
    pub title: String,
    /// Content
    pub content: Option<String>,
    /// Parent ID
    pub parent_id: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Update item request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateItemRequest {
    /// ID
    pub id: ItemId,
    /// Title
    pub title: Option<String>,
    /// Content
    pub content: Option<String>,
    /// Status
    pub status: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Item response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResponse {
    /// ID
    pub id: ItemId,
    /// Title
    pub title: String,
    /// Content
    pub content: Option<String>,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Updated at
    pub updated_at: DateTime<Utc>,
    /// Created by
    pub created_by: Option<String>,
    /// URL
    pub url: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query
    pub query: String,
    /// Filters
    pub filters: SearchFilters,
    /// Limit
    pub limit: Option<usize>,
    /// Offset
    pub offset: Option<usize>,
    /// Sort by
    pub sort_by: Option<SortField>,
    /// Sort order
    pub sort_order: Option<SortOrder>,
}

/// Search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Item types
    pub item_types: Option<Vec<ItemType>>,
    /// Date range
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Authors
    pub authors: Option<Vec<String>>,
    /// Channels
    pub channels: Option<Vec<String>>,
    /// Tags
    pub tags: Option<Vec<String>>,
}

/// Sort field
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortField {
    /// Relevance
    Relevance,
    /// Date
    Date,
    /// Title
    Title,
    /// Author
    Author,
}

/// Sort order
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    /// Ascending
    Asc,
    /// Descending
    Desc,
}

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Items
    pub items: Vec<ItemResponse>,
    /// Total count
    pub total_count: usize,
    /// Has more
    pub has_more: bool,
    /// Next offset
    pub next_offset: Option<usize>,
}

/// User info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// ID
    pub id: String,
    /// Username
    pub username: String,
    /// Display name
    pub display_name: Option<String>,
    /// Email
    pub email: Option<String>,
    /// Avatar URL
    pub avatar_url: Option<String>,
    /// Status
    pub status: Option<UserStatus>,
    /// Timezone
    pub timezone: Option<String>,
    /// Is bot
    pub is_bot: bool,
    /// Is admin
    pub is_admin: bool,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// User status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStatus {
    /// Presence    
    pub presence: Presence,
    /// Status text
    pub status_text: Option<String>,
    /// Status emoji
    pub status_emoji: Option<String>,
    /// Until
    pub until: Option<DateTime<Utc>>,
}

/// Presence
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Presence {
    /// Active
    Active,
    /// Away
    Away,
    /// Do not disturb
    DoNotDisturb,
    /// Offline
    Offline,
}

/// Channel info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    /// ID
    pub id: String,
    /// Name
    pub name: String,
    /// Topic
    pub topic: Option<String>,
    /// Purpose
    pub purpose: Option<String>,
    /// Is private
    pub is_private: bool,
    /// Is archived
    pub is_archived: bool,
    /// Member count
    pub member_count: Option<usize>,
    /// Metadata
    pub metadata: HashMap<String, Value>,
}

/// Config schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    /// Fields
    pub fields: Vec<ConfigField>,
    /// Required
    pub required: Vec<String>,
    /// Sections
    pub sections: Option<Vec<ConfigSection>>,
}

/// Config field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigField {
    /// Name
    pub name: String,
    /// Label
    pub label: String,
    /// Field type
    pub field_type: FieldType,
    /// Description
    pub description: Option<String>,
    /// Default value
    pub default_value: Option<Value>,
    /// Validation
    pub validation: Option<FieldValidation>,
    /// Depends on
    pub depends_on: Option<String>,
}

/// Field types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    /// Text
    Text,
    /// Number
    Number,
    /// Boolean
    Boolean,
    /// Select
    Select,
    /// Multi-select
    MultiSelect,
    /// Date
    Date,
    /// Time
    Time,
    /// DateTime
    DateTime,
    /// Url
    Url,
    /// Email
    Email,
    /// Password
    Password,
    /// Json
    Json,
}

/// Field validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldValidation {
    /// Min length
    pub min_length: Option<usize>,
    /// Max length
    pub max_length: Option<usize>,
    /// Pattern
    pub pattern: Option<String>,
    /// Min value
    pub min_value: Option<f64>,
    /// Max value
    pub max_value: Option<f64>,
    /// Options
    pub options: Option<Vec<String>>,
}

/// Config section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSection {
    /// Name
    pub name: String,
    /// Label
    pub label: String,
    /// Fields
    pub fields: Vec<String>,
    /// Description
    pub description: Option<String>,
}

/// Adapter manager
pub struct AdapterManager {
    /// Adapters
    adapters: HashMap<Platform, Arc<dyn PlatformAdapter>>,
}



impl AdapterManager {
    /// Create a new adapter manager
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
    
    /// Register an adapter
    pub fn register(&mut self, adapter: Arc<dyn PlatformAdapter>) {
        self.adapters.insert(adapter.platform(), adapter);
    }
    
    /// Get an adapter
    pub fn get(&self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.get(&platform).cloned()
    }
    
    /// Remove an adapter
    pub fn remove(&mut self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.remove(&platform)
    }
    
    /// Get all platforms
    pub fn platforms(&self) -> Vec<Platform> {
        self.adapters.keys().copied().collect()
    }
    
    /// Get the count of adapters
    pub fn count(&self) -> usize {
        self.adapters.len()
    }
    
    /// Perform health check on all adapters
    pub async fn health_check_all(&self) -> HashMap<Platform, Result<HealthStatus, AdapterError>> {
        let mut results = HashMap::new();
        
        for (platform, adapter) in &self.adapters {
            results.insert(*platform, adapter.health_check().await);
        }
        
        results
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_platform_event_creation() {
        let event = PlatformEvent::new(
            Platform::Slack,
            EventType::MessageNew,
            serde_json::json!({"text": "hello"}),
        );
        
        assert_eq!(event.platform, Platform::Slack);
        assert_eq!(event.event_type, EventType::MessageNew);
    }
    
    #[test]
    fn test_response_builder() {
        let response = PlatformResponse::new(
            ResponseTarget::Channel { id: "C123".to_string() },
            ResponseContent::Text("Hello!".to_string()),
        )
        .with_options(ResponseOptions {
            ephemeral: true,
            ..Default::default()
        });
        
        assert!(response.options.ephemeral);
    }
    
    #[test]
    fn test_rate_limit_status() {
        let status = RateLimitStatus {
            limit: 100,
            remaining: 0,
            reset_at: Utc::now() + chrono::Duration::seconds(60),
            window_seconds: 3600,
        };
        
        assert!(status.is_exhausted());
        assert!(status.seconds_until_reset() > 0);
    }
}