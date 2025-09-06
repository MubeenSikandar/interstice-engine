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
    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),
    
    #[error("Event parsing failed: {0}")]
    EventParsingError(String),
    
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    #[error("Rate limit exceeded: retry after {retry_after} seconds")]
    RateLimitExceeded { retry_after: u64 },
    
    #[error("Webhook verification failed")]
    WebhookVerificationFailed,
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Platform API error: {status} - {message}")]
    PlatformApiError { status: u16, message: String },
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

/// Adapter metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub name: String,
    pub version: String,
    pub platform: Platform,
    pub capabilities: AdapterCapabilities,
    pub author: String,
    pub description: String,
    pub documentation_url: Option<String>,
}

/// Adapter capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    pub real_time: bool,
    pub webhooks: bool,
    pub polling: bool,
    pub bidirectional: bool,
    pub file_upload: bool,
    pub rich_formatting: bool,
    pub threading: bool,
    pub reactions: bool,
    pub search: bool,
    pub user_presence: bool,
    pub custom_fields: bool,
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

/// Platform event wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformEvent {
    pub id: Uuid,
    pub platform: Platform,
    pub event_type: EventType,
    pub workspace_id: Option<WorkspaceId>,
    pub timestamp: DateTime<Utc>,
    pub raw_data: Value,
    pub metadata: EventMetadata,
}

impl PlatformEvent {
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
    
    pub fn with_workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }
    
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
    MessageNew,
    MessageUpdated,
    MessageDeleted,
    MessageReaction,
    
    // Channel events
    ChannelCreated,
    ChannelUpdated,
    ChannelDeleted,
    ChannelJoined,
    ChannelLeft,
    
    // User events
    UserJoined,
    UserLeft,
    UserUpdated,
    UserPresenceChanged,
    
    // File events
    FileShared,
    FileDeleted,
    FileCommented,
    
    // Task/Issue events
    TaskCreated,
    TaskUpdated,
    TaskCompleted,
    TaskAssigned,
    TaskCommented,
    
    // Workflow events
    WorkflowStarted,
    WorkflowCompleted,
    WorkflowFailed,
    
    // Integration events
    AppMention,
    SlashCommand,
    InteractiveAction,
    
    // System events
    RateLimited,
    ConnectionLost,
    ConnectionRestored,
    
    // Custom events
    Custom(String),
}

/// Event metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventMetadata {
    pub user_id: Option<String>,
    pub channel_id: Option<String>,
    pub thread_id: Option<String>,
    pub retry_count: u32,
    pub is_duplicate: bool,
    pub correlation_id: Option<Uuid>,
    pub source_ip: Option<String>,
    pub custom: HashMap<String, Value>,
}

/// Platform response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformResponse {
    pub target: ResponseTarget,
    pub content: ResponseContent,
    pub options: ResponseOptions,
}

impl PlatformResponse {
    pub fn new(target: ResponseTarget, content: ResponseContent) -> Self {
        Self {
            target,
            content,
            options: ResponseOptions::default(),
        }
    }
    
    pub fn with_options(mut self, options: ResponseOptions) -> Self {
        self.options = options;
        self
    }
}

/// Response target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseTarget {
    Channel { id: String },
    User { id: String },
    Thread { channel_id: String, thread_id: String },
    Broadcast { channel_ids: Vec<String> },
}

/// Response content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseContent {
    Text(String),
    Markdown(String),
    Html(String),
    Blocks(Vec<BlockElement>),
    File(FileContent),
    Custom(Value),
}

/// Block elements for rich formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockElement {
    Section {
        text: String,
        fields: Option<Vec<TextField>>,
        accessory: Option<Box<BlockElement>>,
    },
    Header {
        text: String,
    },
    Divider,
    Image {
        url: String,
        alt_text: String,
        title: Option<String>,
    },
    Actions {
        elements: Vec<ActionElement>,
    },
    Context {
        elements: Vec<ContextElement>,
    },
    Input {
        label: String,
        element: InputElement,
        hint: Option<String>,
        optional: bool,
    },
}

/// Text field for sections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextField {
    pub title: String,
    pub value: String,
    pub short: bool,
}

/// Action elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionElement {
    Button {
        text: String,
        action_id: String,
        url: Option<String>,
        value: Option<String>,
        style: Option<ButtonStyle>,
    },
    Select {
        placeholder: String,
        action_id: String,
        options: Vec<SelectOption>,
    },
    DatePicker {
        action_id: String,
        placeholder: String,
        initial_date: Option<String>,
    },
    Overflow {
        action_id: String,
        options: Vec<SelectOption>,
    },
}

/// Button styles
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ButtonStyle {
    Primary,
    Danger,
    Default,
}

/// Select options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectOption {
    pub text: String,
    pub value: String,
    pub description: Option<String>,
}

/// Context elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextElement {
    Text(String),
    Image { url: String, alt_text: String },
}

/// Input elements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputElement {
    PlainTextInput {
        action_id: String,
        placeholder: Option<String>,
        initial_value: Option<String>,
        multiline: bool,
        min_length: Option<u32>,
        max_length: Option<u32>,
    },
    Select {
        action_id: String,
        placeholder: String,
        options: Vec<SelectOption>,
    },
    MultiSelect {
        action_id: String,
        placeholder: String,
        options: Vec<SelectOption>,
        max_selected_items: Option<u32>,
    },
    DatePicker {
        action_id: String,
        placeholder: Option<String>,
        initial_date: Option<String>,
    },
    TimePicker {
        action_id: String,
        placeholder: Option<String>,
        initial_time: Option<String>,
    },
}

/// File content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContent {
    pub filename: String,
    pub mime_type: String,
    pub data: Vec<u8>,
    pub title: Option<String>,
    pub description: Option<String>,
}

/// Response options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseOptions {
    pub ephemeral: bool,
    pub in_thread: bool,
    pub replace_original: bool,
    pub delete_original: bool,
    pub unfurl_links: bool,
    pub unfurl_media: bool,
    pub notification_text: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// History fetch parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryParams {
    pub channel_id: Option<String>,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub limit: Option<usize>,
    pub include_deleted: bool,
    pub artifact_types: Option<Vec<String>>,
}

/// Subscription request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub id: Uuid,
    pub events: Vec<EventType>,
    pub filters: SubscriptionFilters,
    pub callback_url: Option<String>,
}

impl Subscription {
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
    pub channels: Option<Vec<String>>,
    pub users: Option<Vec<String>>,
    pub keywords: Option<Vec<String>>,
    pub exclude_bots: bool,
}

/// Subscription handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionHandle {
    pub id: Uuid,
    pub platform: Platform,
    pub created_at: DateTime<Utc>,
    pub events: Vec<EventType>,
}

/// Rate limit status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub limit: u32,
    pub remaining: u32,
    pub reset_at: DateTime<Utc>,
    pub window_seconds: u64,
}

impl RateLimitStatus {
    pub fn is_exhausted(&self) -> bool {
        self.remaining == 0
    }
    
    pub fn seconds_until_reset(&self) -> i64 {
        (self.reset_at - Utc::now()).num_seconds().max(0)
    }
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub message: Option<String>,
    pub last_successful_event: Option<DateTime<Utc>>,
    pub error_count: u32,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthCredentials {
    OAuth {
        client_id: String,
        client_secret: String,
        redirect_uri: String,
        scopes: Vec<String>,
    },
    ApiKey {
        key: String,
    },
    BotToken {
        token: String,
    },
    UserToken {
        access_token: String,
        refresh_token: Option<String>,
    },
    Custom(HashMap<String, String>),
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_at: Option<DateTime<Utc>>,
    pub refresh_token: Option<String>,
    pub scopes: Vec<String>,
}

/// Item ID wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemId {
    pub platform: Platform,
    pub id: String,
    pub item_type: ItemType,
}

/// Item types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemType {
    Message,
    Task,
    Issue,
    Document,
    File,
    User,
    Channel,
    Project,
    Custom(String),
}

/// Create item request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateItemRequest {
    pub item_type: ItemType,
    pub title: String,
    pub content: Option<String>,
    pub parent_id: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Update item request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateItemRequest {
    pub id: ItemId,
    pub title: Option<String>,
    pub content: Option<String>,
    pub status: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Item response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResponse {
    pub id: ItemId,
    pub title: String,
    pub content: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub created_by: Option<String>,
    pub url: Option<String>,
    pub metadata: HashMap<String, Value>,
}

/// Search query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: String,
    pub filters: SearchFilters,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub sort_by: Option<SortField>,
    pub sort_order: Option<SortOrder>,
}

/// Search filters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub item_types: Option<Vec<ItemType>>,
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub authors: Option<Vec<String>>,
    pub channels: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
}

/// Sort field
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortField {
    Relevance,
    Date,
    Title,
    Author,
}

/// Sort order
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortOrder {
    Asc,
    Desc,
}

/// Search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    pub items: Vec<ItemResponse>,
    pub total_count: usize,
    pub has_more: bool,
    pub next_offset: Option<usize>,
}

/// User information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub username: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub avatar_url: Option<String>,
    pub status: Option<UserStatus>,
    pub timezone: Option<String>,
    pub is_bot: bool,
    pub is_admin: bool,
    pub metadata: HashMap<String, Value>,
}

/// User status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserStatus {
    pub presence: Presence,
    pub status_text: Option<String>,
    pub status_emoji: Option<String>,
    pub until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Presence {
    Active,
    Away,
    DoNotDisturb,
    Offline,
}

/// Channel information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelInfo {
    pub id: String,
    pub name: String,
    pub topic: Option<String>,
    pub purpose: Option<String>,
    pub is_private: bool,
    pub is_archived: bool,
    pub member_count: Option<usize>,
    pub metadata: HashMap<String, Value>,
}

/// Configuration schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSchema {
    pub fields: Vec<ConfigField>,
    pub required: Vec<String>,
    pub sections: Option<Vec<ConfigSection>>,
}

/// Configuration field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigField {
    pub name: String,
    pub label: String,
    pub field_type: FieldType,
    pub description: Option<String>,
    pub default_value: Option<Value>,
    pub validation: Option<FieldValidation>,
    pub depends_on: Option<String>,
}

/// Field types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    Text,
    Number,
    Boolean,
    Select,
    MultiSelect,
    Date,
    Time,
    DateTime,
    Url,
    Email,
    Password,
    Json,
}

/// Field validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldValidation {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub options: Option<Vec<String>>,
}

/// Configuration section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSection {
    pub name: String,
    pub label: String,
    pub fields: Vec<String>,
    pub description: Option<String>,
}

/// Adapter manager for managing multiple adapters
pub struct AdapterManager {
    adapters: HashMap<Platform, Arc<dyn PlatformAdapter>>,
}

impl AdapterManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
    
    pub fn register(&mut self, adapter: Arc<dyn PlatformAdapter>) {
        self.adapters.insert(adapter.platform(), adapter);
    }
    
    pub fn get(&self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.get(&platform).cloned()
    }
    
    pub fn remove(&mut self, platform: Platform) -> Option<Arc<dyn PlatformAdapter>> {
        self.adapters.remove(&platform)
    }
    
    pub fn platforms(&self) -> Vec<Platform> {
        self.adapters.keys().copied().collect()
    }
    
    pub fn count(&self) -> usize {
        self.adapters.len()
    }
    
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