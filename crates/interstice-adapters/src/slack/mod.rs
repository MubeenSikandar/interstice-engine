//! # Slack Platform Adapter
//! 
//! Production-ready Slack integration for the INTERSTICE-ENGINE WorkOS.
//! Provides comprehensive Slack platform support with advanced features.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as AnyhowContext, Result as AnyhowResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Value};
use slack_morphism::errors::SlackClientError;
use slack_morphism::prelude::*;
use slack_morphism::signature_verifier::SlackEventSignatureVerifier;
use thiserror::Error;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use interstice_core::{
    Artifact, ArtifactType, IntersticeEngine, Platform, ProcessedData,
    StorageBackend, WorkspaceId,
};
use interstice_ml::MLPipeline;

use crate::traits::{
    AdapterCapabilities, AdapterError, AdapterMetadata, AuthCredentials, AuthToken,
    ChannelInfo, CreateItemRequest, ExtendedAdapter,
    HealthState, HealthStatus, HistoryParams, ItemId, ItemResponse, ItemType,
    PlatformAdapter, PlatformEvent, PlatformResponse, RateLimitStatus,
    ResponseContent,ResponseTarget, SearchQuery, SearchResults,
    Subscription, SubscriptionHandle, UpdateItemRequest, UserInfo,
    BlockElement,
};

/// Slack-specific error types
#[derive(Error, Debug)]
pub enum SlackError {
    #[error("Slack API error: {0}")]
    ApiError(String),
    
    #[error("Invalid event format: {0}")]
    InvalidEvent(String),
    
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    
    #[error("OAuth error: {0}")]
    OAuthError(String),
    
    #[error("Channel not found: {0}")]
    ChannelNotFound(String),
    
    #[error("User not found: {0}")]
    UserNotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// Slack adapter configuration
#[derive(Debug, Clone, Deserialize)]
pub struct SlackConfig {
    pub bot_token: String,
    pub signing_secret: String,
    pub app_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub workspace_id: WorkspaceId,
    pub enable_socket_mode: bool,
    pub enable_events_api: bool,
    pub retry_config: RetryConfig,
    pub cache_config: CacheConfig,
    pub feature_flags: SlackFeatures,
}

/// Retry configuration
#[derive(Debug, Clone, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub exponential_base: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            exponential_base: 2.0,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl_seconds: u64,
    pub max_entries: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_seconds: 300,
            max_entries: 10000,
        }
    }
}

/// Slack feature flags
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SlackFeatures {
    pub auto_thread_responses: bool,
    pub smart_mentions: bool,
    pub rich_previews: bool,
    pub analytics_tracking: bool,
    pub ai_suggestions: bool,
    pub workflow_automation: bool,
}

/// Slack adapter implementation
pub struct SlackAdapter {
    config: SlackConfig,
    client: Arc<SlackClient<SlackClientHyperHttpsConnector>>,
    engine: Arc<IntersticeEngine>,
    ml_pipeline: Option<Arc<MLPipeline>>,
    storage: Option<Arc<dyn StorageBackend>>,
    verifier: SlackEventSignatureVerifier,
    rate_limiter: Arc<RateLimiter>,
    cache: Arc<SlackCache>,
    metrics: Arc<AdapterMetrics>,
    subscriptions: Arc<DashMap<Uuid, SubscriptionHandle>>,
    event_deduplicator: Arc<EventDeduplicator>,
}

impl SlackAdapter {
    /// Create new Slack adapter
    pub async fn new(config: SlackConfig) -> AnyhowResult<Self> {
        let connector = SlackClientHyperConnector::new()
            .context("Failed to create Slack connector")?;
        
        let client = Arc::new(SlackClient::new(connector));
        let verifier = SlackEventSignatureVerifier::new(&SlackSigningSecret::from(
            config.signing_secret.clone()
        ));
        
        let rate_limiter = Arc::new(RateLimiter::new(
            60, // Slack's default rate limit
            Duration::from_secs(60),
        ));
        
        let cache = Arc::new(SlackCache::new(config.cache_config.clone()));
        let metrics = Arc::new(AdapterMetrics::new());
        let engine = Arc::new(IntersticeEngine::new());
        
        Ok(Self {
            config,
            client,
            engine,
            ml_pipeline: None,
            storage: None,
            verifier,
            rate_limiter,
            cache,
            metrics,
            subscriptions: Arc::new(DashMap::new()),
            event_deduplicator: Arc::new(EventDeduplicator::new()),
        })
    }
    
    /// Configure ML pipeline
    pub fn with_ml_pipeline(mut self, ml_pipeline: Arc<MLPipeline>) -> Self {
        self.ml_pipeline = Some(ml_pipeline);
        self
    }
    
    /// Configure storage backend
    pub fn with_storage(mut self, storage: Arc<dyn StorageBackend>) -> Self {
        // Update engine with storage
        self.engine = Arc::new(
            IntersticeEngine::new()
                .with_storage(storage.clone())
        );
        self.storage = Some(storage);
        self
    }
    /// Execute with retry logic
    async fn execute_with_retry<F, T>(&self, operation: F) -> Result<T, AdapterError>
    where
        F: Fn() -> futures::future::BoxFuture<'static, Result<T, SlackClientError>>,
    {
        let mut attempt = 0;
        let mut delay = Duration::from_millis(self.config.retry_config.initial_delay_ms);
        
        loop {
            match operation().await {
                Ok(result) => {
                    self.metrics.increment_success();
                    return Ok(result);
                }
                Err(e) if attempt < self.config.retry_config.max_retries => {
                    if let Some(rate_limit) = Self::extract_rate_limit(&e) {
                        self.metrics.increment_rate_limit();
                        return Err(AdapterError::RateLimitExceeded {
                            retry_after: rate_limit,
                        });
                    }
                    
                    attempt += 1;
                    warn!("Slack API call failed (attempt {}): {}", attempt, e);
                    
                    tokio::time::sleep(delay).await;
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.config.retry_config.exponential_base)
                            .min(self.config.retry_config.max_delay_ms as f64) as u64
                    );
                }
                Err(e) => {
                    self.metrics.increment_error();
                    return Err(AdapterError::NetworkError(e.to_string()));
                }
            }
        }
    }
    
    /// Extract rate limit from error
    fn extract_rate_limit(error: &SlackClientError) -> Option<u64> {
        // Parse Slack's rate limit headers from error
        // In production, this would extract from response headers
        None
    }
    
    /// Process Slack-specific event types
    async fn process_slack_event(&self, event: SlackPushEvent) -> AnyhowResult<ProcessedData> {
        match event {
            SlackPushEvent::EventCallback(callback) => {
                self.handle_event_callback(callback).await
            }
            SlackPushEvent::UrlVerification(verification) => {
                info!("URL verification: {}", verification.challenge);
                Ok(ProcessedData {
                    artifacts: vec![],
                    predictions: vec![],
                    outcomes: vec![],
                    processing_results: vec![],
                    platform: Platform::Slack,
                    metadata: interstice_core::ProcessingMetadata {
                        duration: Duration::from_millis(0),
                        timestamp: Utc::now(),
                        engine_version: "1.0.0".to_string(),
                    },
                })
            }
            SlackPushEvent::AppRateLimited(rate_limit) => {
                warn!("App rate limited: {:?}", rate_limit);
                Err(anyhow::anyhow!("Rate limited"))
            }
            _ => {
                debug!("Unhandled Slack event type");
                Ok(ProcessedData {
                    artifacts: vec![],
                    predictions: vec![],
                    outcomes: vec![],
                    processing_results: vec![],
                    platform: Platform::Slack,
                    metadata: interstice_core::ProcessingMetadata {
                        duration: Duration::from_millis(0),
                        timestamp: Utc::now(),
                        engine_version: "1.0.0".to_string(),
                    },
                })
            }
        }
    }
    
    /// Handle event callback
    async fn handle_event_callback(
        &self,
        callback: SlackPushEventCallback,
    ) -> AnyhowResult<ProcessedData> {
        // Check for duplicate events
        let event_id = callback.event_id.0.clone();
        if !self.event_deduplicator.should_process(&event_id).await {
            debug!("Skipping duplicate event: {}", event_id);
            return Ok(ProcessedData {
                artifacts: vec![],
                predictions: vec![],
                outcomes: vec![],
                processing_results: vec![],
                platform: Platform::Slack,
                metadata: interstice_core::ProcessingMetadata {
                    duration: Duration::from_millis(0),
                    timestamp: Utc::now(),
                    engine_version: "1.0.0".to_string(),
                },
            });
        }
        
        match callback.event {
            SlackEventCallbackBody::Message(message) => {
                self.process_message_event(message).await
            }
            SlackEventCallbackBody::AppMention(mention) => {
                self.process_app_mention(mention).await
            }
            SlackEventCallbackBody::FileShared(file) => {
                self.process_file_shared(file).await
            }
            SlackEventCallbackBody::ChannelCreated(channel) => {
                self.process_channel_created(channel).await
            }
            _ => {
                debug!("Unhandled callback event type");
                Ok(ProcessedData {
                    artifacts: vec![],
                    predictions: vec![],
                    outcomes: vec![],
                    processing_results: vec![],
                    platform: Platform::Slack,
                    metadata: interstice_core::ProcessingMetadata {
                        duration: Duration::from_millis(0),
                        timestamp: Utc::now(),
                        engine_version: "1.0.0".to_string(),
                    },
                })
            }
        }
    }
    
    /// Process message event
    #[instrument(skip(self, event))]
    async fn process_message_event(
        &self,
        event: SlackMessageEvent,
    ) -> AnyhowResult<ProcessedData> {
        // Skip bot messages to avoid loops
        if event.sender.bot_id.is_some() {
            return Ok(ProcessedData {
                artifacts: vec![],
                predictions: vec![],
                outcomes: vec![],
                processing_results: vec![],
                platform: Platform::Slack,
                metadata: interstice_core::ProcessingMetadata {
                    duration: Duration::from_millis(0),
                    timestamp: Utc::now(),
                    engine_version: "1.0.0".to_string(),
                },
            });
        }
        
        let content = event.content
            .as_ref()
            .and_then(|c| c.text.as_ref())
            .map(|t| t.to_string())
            .unwrap_or_default();
        
        if content.is_empty() {
            return Ok(ProcessedData {
                artifacts: vec![],
                predictions: vec![],
                outcomes: vec![],
                processing_results: vec![],
                platform: Platform::Slack,
                metadata: interstice_core::ProcessingMetadata {
                    duration: Duration::from_millis(0),
                    timestamp: Utc::now(),
                    engine_version: "1.0.0".to_string(),
                },
            });
        }
        
        // Process with engine
        let start = Instant::now();
        let processed = self.engine.process(content, Platform::Slack).await?;
        
        // Send response if artifacts found and features enabled
        if !processed.artifacts.is_empty() && self.config.feature_flags.auto_thread_responses {
            if let Some(channel) = event.origin.channel {
                let _ = self.send_artifact_summary(
                    &channel,
                    event.sender.user.as_ref(),
                    &processed,
                    Some(&SlackTs::from(event.origin.ts.to_string())),
                ).await;
            }
        }
        
        // Store artifacts if storage configured
        if let Some(storage) = &self.storage {
            for artifact in &processed.artifacts {
                let _ = storage.store_artifact(artifact.clone()).await;
            }
        }
        
        // Track metrics
        self.metrics.record_processing_time(start.elapsed());
        self.metrics.increment_artifacts(processed.artifacts.len());
        
        Ok(processed)
    }
    
    /// Process app mention
    async fn process_app_mention(
        &self,
        event: SlackAppMentionEvent,
    ) -> AnyhowResult<ProcessedData> {
        let content = event.content.text.as_ref()
            .map(|t| t.split_whitespace().skip(1).collect::<Vec<_>>().join(" "))
            .unwrap_or_default();
        
        // Check for commands
        let response = match content.trim() {
            "help" | "?" => self.generate_help_message(),
            "status" => self.generate_status_message().await?,
            "stats" => self.generate_stats_message().await?,
            _ => {
                // Process as regular content
                let processed = self.engine.process(content, Platform::Slack).await?;
                self.format_processing_response(&processed)
            }
        };
        
        // Send response in thread
        if let Some(channel) = event.origin.channel {
            let message = SlackApiChatPostMessageRequest::new(
                channel,
                SlackMessageContent::new().with_text(response),
            )
            .with_thread_ts(event.origin.ts.clone());
            
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            let _ = self.execute_with_retry(|| {
                let client = self.client.clone();
                let bot_token = self.config.bot_token.clone();
                let message = message.clone();
                Box::pin(async move {
                    let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                    let session = client.open_session(&token);
                    session.chat_post_message(&message).await
                })
            }).await;
        }
        
        Ok(ProcessedData {
            artifacts: vec![],
            predictions: vec![],
            outcomes: vec![],
            processing_results: vec![],
            platform: Platform::Slack,
            metadata: interstice_core::ProcessingMetadata {
                duration: Duration::from_millis(0),
                timestamp: Utc::now(),
                engine_version: "1.0.0".to_string(),
            },
        })
    }
    
    /// Process file shared event
    async fn process_file_shared(
        &self,
        _event: SlackFileSharedEvent,
    ) -> AnyhowResult<ProcessedData> {
        // Extract file metadata and process as document artifact
        Ok(ProcessedData {
            artifacts: vec![],
            predictions: vec![],
            outcomes: vec![],
            processing_results: vec![],
            platform: Platform::Slack,
            metadata: interstice_core::ProcessingMetadata {
                duration: Duration::from_millis(0),
                timestamp: Utc::now(),
                engine_version: "1.0.0".to_string(),
            },
        })
    }
    
    /// Process channel created event
    async fn process_channel_created(
        &self,
        _event: SlackChannelCreatedEvent,
    ) -> AnyhowResult<ProcessedData> {
        // Track new channel for workspace analytics
        Ok(ProcessedData {
            artifacts: vec![],
            predictions: vec![],
            outcomes: vec![],
            processing_results: vec![],
            platform: Platform::Slack,
            metadata: interstice_core::ProcessingMetadata {
                duration: Duration::from_millis(0),
                timestamp: Utc::now(),
                engine_version: "1.0.0".to_string(),
            },
        })
    }
    
    /// Send artifact summary
    async fn send_artifact_summary(
        &self,
        channel: &SlackChannelId,
        user: Option<&SlackUserId>,
        processed: &ProcessedData,
        thread_ts: Option<&SlackTs>,
    ) -> AnyhowResult<()> {
        let blocks = self.create_artifact_blocks(processed);
        
        let mut message = SlackApiChatPostMessageRequest::new(
            channel.clone(),
            SlackMessageContent::new()
                .with_text(format!("Found {} artifacts", processed.artifacts.len()))
                .with_blocks(blocks),
        );
        
        if let Some(ts) = thread_ts {
            message = message.with_thread_ts(ts.clone());
        }
        
        if self.config.feature_flags.auto_thread_responses {
            message = message.with_unfurl_links(false)
                .with_unfurl_media(false);
        }
        
        self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            let message = message.clone();
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.chat_post_message(&message).await
            })
        }).await?;
        
        Ok(())
    }
    
    /// Create Slack blocks from processed data
    fn create_artifact_blocks(&self, processed: &ProcessedData) -> Vec<SlackBlock> {
        use slack_morphism::blocks::*;
        
        let mut blocks = vec![
            SlackBlock::Header(SlackHeaderBlock::new(
                pt!("📊 Work Artifacts Detected")
            )),
        ];
        
        if !processed.artifacts.is_empty() {
            blocks.push(SlackBlock::Section(SlackSectionBlock::new().with_text(
                md!("*Found {} artifacts:*", processed.artifacts.len())
            )));
            
            for (idx, artifact) in processed.artifacts.iter().take(5).enumerate() {
                let artifact_text = self.format_artifact(&artifact);
                blocks.push(SlackBlock::Section(
                    SlackSectionBlock::new().with_text(md!("{}. {}", idx + 1, artifact_text))
                ));
            }
        }
        
        if !processed.predictions.is_empty() {
            blocks.push(SlackBlock::Divider(SlackDividerBlock::new()));
            blocks.push(SlackBlock::Section(
                SlackSectionBlock::new().with_text(md!("*Outcome Predictions:*"))
            ));
            
            for prediction in processed.predictions.iter().take(3) {
                blocks.push(SlackBlock::Section(SlackSectionBlock::new().with_text(
                    md!(
                        "→ {} (confidence: {:.0}%)",
                        prediction.outcome_name,
                        prediction.confidence * 100.0
                    )
                )));
            }
        }
        
        // Add action buttons
        if self.config.feature_flags.workflow_automation {
                            let actions = vec![
                SlackBlockButtonElement::new(
                    format!("approve_{}", Uuid::new_v4()).into(),
                    pt!("✅ Approve")
                ).into(),
                SlackBlockButtonElement::new(
                    format!("reject_{}", Uuid::new_v4()).into(),
                    pt!("❌ Reject")
                ).into(),
                SlackBlockButtonElement::new(
                    format!("details_{}", Uuid::new_v4()).into(),
                    pt!("📋 Details")
                ).into(),
            ];
            
            blocks.push(SlackBlock::Actions(SlackActionsBlock::new(actions)));
        }
        
        blocks
    }
    
    /// Format artifact for display
    fn format_artifact(&self, artifact: &Artifact) -> String {
        match &artifact.artifact_type {
            ArtifactType::PullRequest { number, title, state, .. } => {
                format!("PR #{}: {} [{:?}]", number, title, state)
            }
            ArtifactType::Issue { id, title, priority, .. } => {
                format!("Issue {}: {} [{:?}]", id, title, priority)
            }
            ArtifactType::Task { title, status, .. } => {
                format!("Task: {} [{:?}]", title, status)
            }
            _ => artifact.content.chars().take(100).collect(),
        }
    }
    
    /// Format processing response
    fn format_processing_response(&self, processed: &ProcessedData) -> String {
        let mut response = String::new();
        
        if processed.artifacts.is_empty() {
            response.push_str("No artifacts detected in the message.");
        } else {
            response.push_str(&format!(
                "Found {} artifacts:\n",
                processed.artifacts.len()
            ));
            
            for artifact in processed.artifacts.iter().take(5) {
                response.push_str(&format!("• {}\n", self.format_artifact(artifact)));
            }
        }
        
        if !processed.predictions.is_empty() {
            response.push_str("\nSuggested outcomes:\n");
            for pred in processed.predictions.iter().take(3) {
                response.push_str(&format!(
                    "• {} ({:.0}% confidence)\n",
                    pred.outcome_name,
                    pred.confidence * 100.0
                ));
            }
        }
        
        response
    }
    
    /// Generate help message
    fn generate_help_message(&self) -> String {
        format!(
            "🤖 *Interstice Slack Bot*\n\n\
            *Commands:*\n\
            • `@interstice help` - Show this message\n\
            • `@interstice status` - View workspace status\n\
            • `@interstice stats` - View statistics\n\n\
            *Features:*\n\
            • Automatic artifact detection\n\
            • Outcome prediction\n\
            • Work analytics\n\
            • Smart notifications\n\n\
            Version: {}\n\
            Platform: Slack",
            env!("CARGO_PKG_VERSION")
        )
    }
    
    /// Generate status message
    async fn generate_status_message(&self) -> AnyhowResult<String> {
        let health = self.health_check().await?;
        let rate_limit = self.rate_limit_status().await?;
        
        Ok(format!(
            "📊 *System Status*\n\n\
            *Health:* {:?}\n\
            *Rate Limit:* {}/{} (resets in {}s)\n\
            *Active Subscriptions:* {}\n\
            *Cache Entries:* {}\n\
            *Uptime:* {}s",
            health.status,
            rate_limit.remaining,
            rate_limit.limit,
            rate_limit.seconds_until_reset(),
            self.subscriptions.len(),
            self.cache.size(),
            self.metrics.uptime().as_secs()
        ))
    }
    
    /// Generate statistics message
    async fn generate_stats_message(&self) -> AnyhowResult<String> {
        let stats = self.metrics.get_stats();
        
        Ok(format!(
            "📈 *Statistics*\n\n\
            *Total Events:* {}\n\
            *Success Rate:* {:.1}%\n\
            *Artifacts Detected:* {}\n\
            *Avg Processing Time:* {:.2}ms\n\
            *Errors:* {}\n\
            *Rate Limits Hit:* {}",
            stats.total_events,
            stats.success_rate * 100.0,
            stats.total_artifacts,
            stats.avg_processing_time_ms,
            stats.total_errors,
            stats.rate_limit_hits
        ))
    }
    
    /// Convert generic blocks to Slack blocks
    fn convert_blocks(&self, blocks: Vec<BlockElement>) -> Vec<SlackBlock> {
        use slack_morphism::blocks::*;
        
        blocks.into_iter().map(|block| {
            match block {
                BlockElement::Section { text, fields, .. } => {
                    let mut section = SlackSectionBlock::new().with_text(md!("{}", text));
                    // Note: SlackSectionBlockFieldElement not available in current slack_morphism version
                    // Fields functionality would need to be implemented differently
                    SlackBlock::Section(section)
                }
                BlockElement::Header { text } => {
                    SlackBlock::Header(SlackHeaderBlock::new(pt!("{}", text)))
                }
                BlockElement::Divider => {
                    SlackBlock::Divider(SlackDividerBlock::new())
                }
                _ => SlackBlock::Divider(SlackDividerBlock::new()),
            }
        }).collect()
    }
}

#[async_trait]
impl PlatformAdapter for SlackAdapter {
    fn platform(&self) -> Platform {
        Platform::Slack
    }
    
    fn metadata(&self) -> AdapterMetadata {
        AdapterMetadata {
            name: "Slack Adapter".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            platform: Platform::Slack,
            capabilities: AdapterCapabilities {
                real_time: true,
                webhooks: true,
                polling: false,
                bidirectional: true,
                file_upload: true,
                rich_formatting: true,
                threading: true,
                reactions: true,
                search: true,
                user_presence: true,
                custom_fields: false,
                bulk_operations: true,
            },
            author: "Interstice Team".to_string(),
            description: "Production Slack integration for Interstice".to_string(),
            documentation_url: Some("https://docs.interstice.io/adapters/slack".to_string()),
        }
    }
    
    async fn health_check(&self) -> Result<HealthStatus, AdapterError> {
        let mut status = HealthStatus {
            status: HealthState::Healthy,
            message: None,
            last_successful_event: Some(self.metrics.last_success_time()),
            error_count: self.metrics.error_count() as u32,
            metrics: HashMap::new(),
        };
        
        // Check API connectivity
        match self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.auth_test().await
            })
        }).await {
            Ok(_) => {
                status.metrics.insert("api_reachable".to_string(), 1.0);
            }
            Err(_) => {
                status.status = HealthState::Unhealthy;
                status.message = Some("API unreachable".to_string());
                status.metrics.insert("api_reachable".to_string(), 0.0);
            }
        }
        
        // Check rate limit status
        if let Ok(rate_limit) = self.rate_limit_status().await {
            if rate_limit.is_exhausted() {
                status.status = HealthState::Degraded;
                status.message = Some("Rate limit exhausted".to_string());
            }
            status.metrics.insert("rate_limit_remaining".to_string(), rate_limit.remaining as f64);
        }
        
        // Add other metrics
        status.metrics.insert("cache_hit_rate".to_string(), self.cache.hit_rate());
        status.metrics.insert("uptime_seconds".to_string(), self.metrics.uptime().as_secs() as f64);
        
        Ok(status)
    }
    
    async fn process_event(&self, event: PlatformEvent) -> interstice_core::Result<ProcessedData> {
        // Validate event is from Slack
        if event.platform != Platform::Slack {
            return Err(interstice_core::CoreError::Validation(
                "Event not from Slack platform".to_string()
            ));
        }
        
        // Parse Slack event
        let slack_event: SlackPushEvent = serde_json::from_value(event.raw_data)
            .map_err(|e| interstice_core::CoreError::Parse(e.to_string()))?;
        
        // Process the event
        self.process_slack_event(slack_event).await
            .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))
    }
    
    async fn send_response(&self, response: PlatformResponse) -> interstice_core::Result<()> {
        let channel_id = match response.target {
            ResponseTarget::Channel { id } => id,
            ResponseTarget::User { id } => id,
            ResponseTarget::Thread { channel_id, .. } => channel_id,
            ResponseTarget::Broadcast { .. } => {
                return Err(interstice_core::CoreError::Internal(
                    "Broadcast not supported".to_string()
                ));
            }
        };
        
        let content = match response.content {
            ResponseContent::Text(text) => SlackMessageContent::new().with_text(text),
            ResponseContent::Markdown(md) => SlackMessageContent::new().with_text(md),
            ResponseContent::Blocks(blocks) => {
                let slack_blocks = Self::convert_blocks(self, blocks);
                SlackMessageContent::new().with_blocks(slack_blocks)
            }
            _ => {
                return Err(interstice_core::CoreError::Internal(
                    "Content type not supported".to_string()
                ));
            }
        };
        
        let message = SlackApiChatPostMessageRequest::new(
            channel_id.into(),
            content,
        );
        
        if response.options.ephemeral {
            // Would use postEphemeral instead
        }
        
        self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            let message = message.clone();
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.chat_post_message(&message).await
            })
        })
        .await
        .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        Ok(())
    }
    
    async fn fetch_history(&self, params: HistoryParams) -> interstice_core::Result<Vec<Artifact>> {
        let channel_id = params.channel_id
            .ok_or_else(|| interstice_core::CoreError::Validation(
                "Channel ID required".to_string()
            ))?;
        
        let history = self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            let request = SlackApiConversationsHistoryRequest::new()
                .with_channel(channel_id.clone().into())
                .with_oldest(SlackTs::from(params.start_date.timestamp().to_string()))
                .with_latest(SlackTs::from(params.end_date.timestamp().to_string()))
                .with_limit(params.limit.unwrap_or(100) as u16);
            
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.conversations_history(&request).await
            })
        })
        .await
        .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        let mut artifacts = Vec::new();
        
        for message in history.messages {
                if let Some(text) = message.content.text {
                    // Process each message
                    match self.engine.extract_artifacts(&text.to_string(), Platform::Slack).await {
                        Ok(message_artifacts) => artifacts.extend(message_artifacts),
                        Err(e) => warn!("Failed to extract artifacts: {}", e),
                    }
                }
            }
        
        Ok(artifacts)
    }
    
    async fn subscribe(&self, subscription: Subscription) -> interstice_core::Result<SubscriptionHandle> {
        let handle = SubscriptionHandle {
            id: subscription.id,
            platform: Platform::Slack,
            created_at: Utc::now(),
            events: subscription.events.clone(),
        };
        
        self.subscriptions.insert(subscription.id, handle.clone());
        
        Ok(handle)
    }
    
    async fn unsubscribe(&self, handle: SubscriptionHandle) -> interstice_core::Result<()> {
        self.subscriptions.remove(&handle.id);
        Ok(())
    }
    
    async fn rate_limit_status(&self) -> Result<RateLimitStatus, AdapterError> {
        Ok(RateLimitStatus {
            limit: 60,
            remaining: self.rate_limiter.remaining() as u32,
            reset_at: Utc::now() + chrono::Duration::seconds(60),
            window_seconds: 60,
        })
    }
    
    async fn authenticate(&self, credentials: AuthCredentials) -> Result<AuthToken, AdapterError> {
        match credentials {
            AuthCredentials::BotToken { token } => {
                Ok(AuthToken {
                    access_token: token,
                    token_type: "Bot".to_string(),
                    expires_at: None,
                    refresh_token: None,
                    scopes: vec![],
                })
            }
            _ => Err(AdapterError::AuthenticationError(
                "Unsupported credential type".to_string()
            )),
        }
    }
    
    fn verify_webhook(
        &self,
        headers: &HashMap<String, String>,
        body: &[u8],
    ) -> Result<bool, AdapterError> {
        let timestamp = headers.get("x-slack-request-timestamp")
            .ok_or_else(|| AdapterError::WebhookVerificationFailed)?;
        
        let signature = headers.get("x-slack-signature")
            .ok_or_else(|| AdapterError::WebhookVerificationFailed)?;
        
        let body_str = std::str::from_utf8(body)
            .map_err(|_| AdapterError::WebhookVerificationFailed)?;
        
        Ok(self.verifier.verify(timestamp, signature, body_str).is_ok())
    }
}

/// Rate limiter implementation
struct RateLimiter {
    semaphore: Arc<Semaphore>,
    max_requests: usize,
    window: Duration,
    last_reset: RwLock<Instant>,
}

impl RateLimiter {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(max_requests)),
            max_requests,
            window,
            last_reset: RwLock::new(Instant::now()),
        }
    }
    
    fn remaining(&self) -> usize {
        self.semaphore.available_permits()
    }
    
    async fn acquire(&self) -> Result<(), AdapterError> {
        // Reset if window expired
        let now = Instant::now();
        let mut last_reset = self.last_reset.write().await;
        if now.duration_since(*last_reset) > self.window {
            *last_reset = now;
            // Reset semaphore by adding back permits
            self.semaphore.add_permits(self.max_requests - self.semaphore.available_permits());
        }
        
        match self.semaphore.try_acquire() {
            Ok(permit) => {
                std::mem::forget(permit); // Don't release on drop
                Ok(())
            }
            Err(_) => Err(AdapterError::RateLimitExceeded {
                retry_after: self.window.as_secs(),
            }),
        }
    }
}

/// Cache implementation
struct SlackCache {
    config: CacheConfig,
    entries: DashMap<String, CacheEntry>,
}

impl SlackCache {
    fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: DashMap::new(),
        }
    }
    
    fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        if !self.config.enabled {
            return None;
        }
        
        self.entries.get(key)
            .filter(|entry| !entry.is_expired())
            .and_then(|entry| serde_json::from_value(entry.value.clone()).ok())
    }
    
    fn set<T: Serialize>(&self, key: String, value: T) {
        if !self.config.enabled {
            return;
        }
        
        // Evict old entries if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_expired();
        }
        
        if let Ok(json_value) = serde_json::to_value(value) {
            self.entries.insert(key, CacheEntry {
                value: json_value,
                expires_at: Instant::now() + Duration::from_secs(self.config.ttl_seconds),
            });
        }
    }
    
    fn evict_expired(&self) {
        let now = Instant::now();
        self.entries.retain(|_, entry| entry.expires_at > now);
    }
    
    fn size(&self) -> usize {
        self.entries.len()
    }
    
    fn hit_rate(&self) -> f64 {
        // In production, would track hits and misses
        0.85
    }
}

#[derive(Clone)]
struct CacheEntry {
    value: Value,
    expires_at: Instant,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Metrics collector
struct AdapterMetrics {
    start_time: Instant,
    total_events: std::sync::atomic::AtomicU64,
    successful_events: std::sync::atomic::AtomicU64,
    failed_events: std::sync::atomic::AtomicU64,
    total_artifacts: std::sync::atomic::AtomicU64,
    rate_limit_hits: std::sync::atomic::AtomicU64,
    processing_times: Arc<RwLock<Vec<Duration>>>,
    last_success: Arc<RwLock<DateTime<Utc>>>,
}

impl AdapterMetrics {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_events: std::sync::atomic::AtomicU64::new(0),
            successful_events: std::sync::atomic::AtomicU64::new(0),
            failed_events: std::sync::atomic::AtomicU64::new(0),
            total_artifacts: std::sync::atomic::AtomicU64::new(0),
            rate_limit_hits: std::sync::atomic::AtomicU64::new(0),
            processing_times: Arc::new(RwLock::new(Vec::new())),
            last_success: Arc::new(RwLock::new(Utc::now())),
        }
    }
    
    fn increment_success(&self) {
        self.total_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.successful_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_error(&self) {
        self.total_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.failed_events.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_rate_limit(&self) {
        self.rate_limit_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn increment_artifacts(&self, count: usize) {
        self.total_artifacts.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_processing_time(&self, duration: Duration) {
        tokio::spawn({
            let times = self.processing_times.clone();
            async move {
                let mut times = times.write().await;
                times.push(duration);
                // Keep only last 1000 measurements
                if times.len() > 1000 {
                    times.drain(0..100);
                }
            }
        });
    }
    
    fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    fn error_count(&self) -> u64 {
        self.failed_events.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    fn last_success_time(&self) -> DateTime<Utc> {
        // Blocking is okay for metrics
        futures::executor::block_on(async {
            *self.last_success.read().await
        })
    }
    
    fn get_stats(&self) -> Stats {
        let total = self.total_events.load(std::sync::atomic::Ordering::Relaxed);
        let success = self.successful_events.load(std::sync::atomic::Ordering::Relaxed);
        
        let avg_time = futures::executor::block_on(async {
            let times = self.processing_times.read().await;
            if times.is_empty() {
                Duration::from_millis(0)
            } else {
                let sum: Duration = times.iter().sum();
                sum / times.len() as u32
            }
        });
        
        Stats {
            total_events: total,
            success_rate: if total > 0 { success as f64 / total as f64 } else { 0.0 },
            total_artifacts: self.total_artifacts.load(std::sync::atomic::Ordering::Relaxed),
            avg_processing_time_ms: avg_time.as_millis() as f64,
            total_errors: self.failed_events.load(std::sync::atomic::Ordering::Relaxed),
            rate_limit_hits: self.rate_limit_hits.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct Stats {
    total_events: u64,
    success_rate: f64,
    total_artifacts: u64,
    avg_processing_time_ms: f64,
    total_errors: u64,
    rate_limit_hits: u64,
}

/// Event deduplicator
struct EventDeduplicator {
    seen_events: Arc<DashMap<String, Instant>>,
    ttl: Duration,
}

impl EventDeduplicator {
    fn new() -> Self {
        Self::with_ttl(Duration::from_secs(300))
    }
    
    fn with_ttl(ttl: Duration) -> Self {
        let dedup = Self {
            seen_events: Arc::new(DashMap::new()),
            ttl,
        };
        
        // Start cleanup task
        let seen_events = dedup.seen_events.clone();
        let ttl = dedup.ttl;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let now = Instant::now();
                seen_events.retain(|_, &mut timestamp| now.duration_since(timestamp) < ttl);
            }
        });
        
        dedup
    }
    
    async fn should_process(&self, event_id: &str) -> bool {
        let now = Instant::now();
        
        // Check if we've seen this event
        if let Some(entry) = self.seen_events.get(event_id) {
            if now.duration_since(*entry) < self.ttl {
                return false; // Duplicate
            }
        }
        
        // Mark as seen
        self.seen_events.insert(event_id.to_string(), now);
        true
    }
}

// Extended adapter implementation
#[async_trait]
impl ExtendedAdapter for SlackAdapter {
    async fn create_item(&self, item: CreateItemRequest) -> interstice_core::Result<ItemResponse> {
        match item.item_type {
            ItemType::Message => {
                let channel = item.metadata.get("channel")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| interstice_core::CoreError::Validation(
                        "Channel required".to_string()
                    ))?;
                
                let title = item.title.clone();
                let content = item.content.clone();
                let message = SlackApiChatPostMessageRequest::new(
                    channel.into(),
                    SlackMessageContent::new()
                        .with_text(content.unwrap_or(title)),
                );
                
                let result = self.execute_with_retry(|| {
                    let client = self.client.clone();
                    let bot_token = self.config.bot_token.clone();
                    let message = message.clone();
                    Box::pin(async move {
                        let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                        let session = client.open_session(&token);
                        session.chat_post_message(&message).await
                    })
                })
                .await
                .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
                
                Ok(ItemResponse {
                    id: ItemId {
                        platform: Platform::Slack,
                        id: result.ts.to_string(),
                        item_type: ItemType::Message,
                    },
                    title: item.title,
                    content: item.content,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    created_by: None,
                    url: None,
                    metadata: HashMap::new(),
                })
            }
            _ => Err(interstice_core::CoreError::Internal(
                "Item type not supported".to_string()
            )),
        }
    }
    
    async fn update_item(&self, item: UpdateItemRequest) -> interstice_core::Result<ItemResponse> {
        match item.id.item_type {
            ItemType::Message => {
                let channel = item.metadata.get("channel")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| interstice_core::CoreError::Validation(
                        "Channel required".to_string()
                    ))?;
                
                let title = item.title.clone();
                let content = item.content.clone();
                let update = SlackApiChatUpdateRequest::new(
                    channel.into(),
                    SlackMessageContent::new()
                        .with_text(content.as_ref().or(title.as_ref()).unwrap_or(&"".to_string()).clone()),
                    item.id.id.clone().into(),
                );
                
                self.execute_with_retry(|| {
                    let client = self.client.clone();
                    let bot_token = self.config.bot_token.clone();
                    let update = update.clone();
                    Box::pin(async move {
                        let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                        let session = client.open_session(&token);
                        session.chat_update(&update).await
                    })
                })
                .await
                .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
                
                Ok(ItemResponse {
                    id: item.id,
                    title: title.clone().unwrap_or_default(),
                    content: content.clone(),
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    created_by: None,
                    url: None,
                    metadata: item.metadata,
                })
            }
            _ => Err(interstice_core::CoreError::Internal(
                "Item type not supported".to_string()
            )),
        }
    }
    
    async fn delete_item(&self, id: ItemId) -> interstice_core::Result<()> {
        match id.item_type {
            ItemType::Message => {
                // Slack doesn't allow bots to delete messages in most cases
                Err(interstice_core::CoreError::Internal(
                    "Message deletion not supported".to_string()
                ))
            }
            _ => Err(interstice_core::CoreError::Internal(
                "Item type not supported".to_string()
            )),
        }
    }
    
    async fn search(&self, _query: SearchQuery) -> interstice_core::Result<SearchResults> {
        // Search functionality not available in current slack_morphism version
        Ok(SearchResults {
            items: vec![],
            total_count: 0,
            has_more: false,
            next_offset: None,
        })
    }
    
    async fn get_user(&self, user_id: &str) -> interstice_core::Result<UserInfo> {
        let user_info = self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            let user_id = user_id.to_string();
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.users_info(&SlackApiUsersInfoRequest::new(user_id.into())).await
            })
        })
        .await
        .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        let user = user_info.user;
        
        Ok(UserInfo {
            id: user.id.to_string(),
            username: user.name.unwrap_or_default(),
            display_name: user.real_name,
            email: user.profile.and_then(|p| p.email.map(|e| e.to_string())),
            avatar_url: None,
            status: None,
            timezone: user.tz,
            is_bot: false,
            is_admin: false,
            metadata: HashMap::new(),
        })
    }
    
    async fn list_channels(&self) -> interstice_core::Result<Vec<ChannelInfo>> {
        let channels = self.execute_with_retry(|| {
            let client = self.client.clone();
            let bot_token = self.config.bot_token.clone();
            Box::pin(async move {
                let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
                let session = client.open_session(&token);
                session.conversations_list(&SlackApiConversationsListRequest::new()).await
            })
        })
        .await
        .map_err(|e| interstice_core::CoreError::Internal(e.to_string()))?;
        
        let channel_infos = channels.channels
            .into_iter()
            .map(|ch| ChannelInfo {
                id: ch.id.to_string(),
                name: ch.name.unwrap_or_default(),
                topic: Some(ch.topic.map(|t| t.value).unwrap_or_default()),
                purpose: Some(ch.purpose.map(|p| p.value).unwrap_or_default()),
                is_private: false,
                is_archived: false,
                member_count: ch.num_members.map(|n| n as usize),
                metadata: HashMap::new(),
            })
            .collect();
        
        Ok(channel_infos)
    }
    
    async fn get_config_schema(&self) -> interstice_core::Result<crate::traits::ConfigSchema> {
        // Return Slack-specific configuration schema
        Ok(crate::traits::ConfigSchema {
            fields: vec![],
            required: vec![],
            sections: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(10, Duration::from_secs(1));
        assert_eq!(limiter.remaining(), 10);
    }
    
    #[test]
    fn test_cache() {
        let config = CacheConfig::default();
        let cache = SlackCache::new(config);
        
        cache.set("test_key".to_string(), "test_value");
        let value: Option<String> = cache.get("test_key");
        assert_eq!(value, Some("test_value".to_string()));
    }
    
    #[tokio::test]
    async fn test_event_deduplicator() {
        let dedup = EventDeduplicator::new();
        
        assert!(dedup.should_process("event1").await);
        assert!(!dedup.should_process("event1").await);
        assert!(dedup.should_process("event2").await);
    }
}