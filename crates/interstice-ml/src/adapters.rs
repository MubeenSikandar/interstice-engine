// interstice-ml/src/adapters.rs

//! Production-ready ML adapter layer for bridging core system with ML components
//! 
//! This module provides high-performance, observable, and fault-tolerant adapters
//! that seamlessly integrate ML predictions with the core business logic.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use chrono::{Datelike, Local, Timelike};
use interstice_core::types::Priority;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize};
use tokio::sync::{RwLock, Mutex};
use tracing::{info, instrument, warn};
use uuid::Uuid;

use interstice_core::{
    Artifact as CoreArtifact,
    ArtifactType as CoreArtifactType,
    Platform as CorePlatform,
};
use interstice_core::traits::{MLPredictor, OutcomePrediction as CorePrediction};

use crate::inference::{
    DevicePreference, ModelConfig, OutcomePredictor, TextEmbedder
};
use crate::types::{
    Artifact, ArtifactType, OutcomePrediction, Platform, PredictionContext
};

static HASHTAG_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"#\w+").expect("Invalid regex pattern")
});

// Configuration
// -----------------------------------------------------------------------------

/// Comprehensive configuration for ML adapter layer
#[derive(Debug, Clone, Deserialize)]
pub struct AdapterConfig {
    /// Model configuration
    pub models: ModelConfig,
    
    /// Performance settings
    pub performance: PerformanceConfig,
    
    /// Context enrichment settings
    pub context: ContextConfig,
    
    /// Observability configuration
    pub observability: ObservabilityConfig,
}

// Update the new() method implementation:
#[derive(Debug, Clone, Deserialize)]
pub struct FallbackModels {
    pub embedding: String,
    pub predictor: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum concurrent predictions
    pub max_concurrent_predictions: usize,
    
    /// Request timeout
    #[serde(with = "humantime_serde")]
    pub prediction_timeout: Duration,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Circuit breaker settings
    pub circuit_breaker: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CacheConfig {
    /// Enable prediction caching
    pub enabled: bool,
    
    /// Cache TTL
    #[serde(with = "humantime_serde")]
    pub ttl: Duration,
    
    /// Maximum cache size
    pub max_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold before opening circuit
    pub failure_threshold: u32,
    
    /// Success threshold to close circuit
    pub success_threshold: u32,
    
    /// Timeout in open state
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextConfig {
    /// Include temporal features
    pub include_temporal: bool,
    
    /// Include user activity signals
    pub include_user_activity: bool,
    
    /// Include team dynamics
    pub include_team_dynamics: bool,
    
    /// Default values for missing context
    pub defaults: ContextDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextDefaults {
    pub user_activity_level: f32,
    pub user_expertise_score: f32,
    pub team_size: u32,
    pub workspace_activity_level: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable metrics collection
    pub metrics_enabled: bool,
    
    /// Enable distributed tracing
    pub tracing_enabled: bool,
    
    /// Sample rate for tracing (0.0 - 1.0)
    pub trace_sample_rate: f32,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            models: ModelConfig {
                onnx_model_path: Some(PathBuf::from("models/embeddings.onnx")),
                bert_model_path: Some(PathBuf::from("models/predictor.onnx")),
                tokenizer_path: None,
                device_preference: DevicePreference::Auto,
                max_sequence_length: 512,
                embedding_dim: 768,
                confidence_threshold: 0.3,
                cache_embeddings: true,
            },
            performance: PerformanceConfig {
                max_concurrent_predictions: 100,
                prediction_timeout: Duration::from_secs(5),
                batch_size: 32,
                cache: CacheConfig {
                    enabled: true,
                    ttl: Duration::from_secs(300),
                    max_size: 10000,
                },
                circuit_breaker: CircuitBreakerConfig {
                    failure_threshold: 5,
                    success_threshold: 2,
                    timeout: Duration::from_secs(60),
                },
            },
            context: ContextConfig {
                include_temporal: true,
                include_user_activity: true,
                include_team_dynamics: true,
                defaults: ContextDefaults {
                    user_activity_level: 0.5,
                    user_expertise_score: 0.5,
                    team_size: 5,
                    workspace_activity_level: 0.5,
                },
            },
            observability: ObservabilityConfig {
                metrics_enabled: true,
                tracing_enabled: true,
                trace_sample_rate: 0.1,
            },
        }
    }
}

// Core Adapter Implementation
// -----------------------------------------------------------------------------

/// Production-ready ML predictor adapter with comprehensive error handling and monitoring
pub struct MLPredictorAdapter {
    /// Text embedder for converting artifacts to embeddings
    embedder: Arc<TextEmbedder>,
    
    /// Outcome predictor for inference
    predictor: Arc<OutcomePredictor>,
    
    /// Fallback predictor for resilience
    fallback_predictor: Option<Arc<OutcomePredictor>>,
    
    /// Configuration
    config: AdapterConfig,
    
    /// Prediction cache
    cache: Arc<PredictionCache>,
    
    /// Circuit breaker for fault tolerance
    circuit_breaker: Arc<CircuitBreaker>,
    
    /// Metrics collector
    metrics: Arc<AdapterMetrics>,
    
    /// Rate limiter
    rate_limiter: Arc<RateLimiter>
}

pub struct RateLimiter {
    permits: Arc<Mutex<usize>>,  // This will now use tokio::sync::Mutex
    max_permits: usize,
}


impl RateLimiter {
    pub fn new(max_permits: usize) -> Self {
        Self {
            permits: Arc::new(Mutex::new(max_permits)),
            max_permits,
        }
    }
    
    pub async fn acquire(&self) -> RateLimitGuard {
        loop {
            let mut permits = self.permits.lock().await;
            if *permits > 0 {
                *permits -= 1;
                return RateLimitGuard {
                    permits: self.permits.clone(),
                };
            }
            drop(permits);
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }
}

pub struct RateLimitGuard {
    permits: Arc<Mutex<usize>>,  // This will also use tokio::sync::Mutex
}


impl Drop for RateLimitGuard {
    fn drop(&mut self) {
        // Simple approach: just increment without async
        // This avoids all Send issues
        let permits = self.permits.clone();
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                let mut p = permits.lock().await;
                *p += 1;
            });
        }
    }
}



impl MLPredictorAdapter {
    /// Create a new adapter with the given configuration
    #[instrument(skip(config))]
    pub async fn new(config: AdapterConfig) -> Result<Self> {
        // Create model configuration from adapter config
        let model_config = ModelConfig {
            onnx_model_path: config.models.onnx_model_path.clone(),
            bert_model_path: config.models.bert_model_path.clone(),
            tokenizer_path: config.models.tokenizer_path.clone(),
            device_preference: config.models.device_preference.clone(),
            max_sequence_length: config.models.max_sequence_length,
            embedding_dim: config.models.embedding_dim,
            confidence_threshold: config.models.confidence_threshold,
            cache_embeddings: config.models.cache_embeddings,
        };

        let embedder = Arc::new(
            TextEmbedder::new(model_config.clone())
                .await
                .context("Failed to initialize text embedder")?
        );
        
        let predictor = Arc::new(
            OutcomePredictor::new(model_config.clone())
                .await
                .context("Failed to initialize outcome predictor")?
        );
        
        // Initialize fallback predictor (not supported in current config)
        let fallback_predictor = None;
        // Initialize cache
        let cache = Arc::new(PredictionCache::new(
            config.performance.cache.max_size,
            config.performance.cache.ttl,
        ));
        
        let circuit_breaker = Arc::new(CircuitBreaker::new(
            config.performance.circuit_breaker.failure_threshold,
            config.performance.circuit_breaker.success_threshold,
            config.performance.circuit_breaker.timeout,
        ));
        
        let metrics = Arc::new(AdapterMetrics::new());
        
        let rate_limiter = Arc::new(RateLimiter::new(
            config.performance.max_concurrent_predictions
        ));
        
        info!("ML predictor adapter initialized");
        
        Ok(Self {
            embedder,
            predictor,
            fallback_predictor,
            config,
            cache,
            circuit_breaker,
            metrics,
            rate_limiter,
        })
    }

    
    /// Create adapter with default configuration
    pub async fn with_defaults() -> Result<Self> {
        Self::new(AdapterConfig::default()).await
    }
    
    /// Convert artifacts with validation and enrichment
    #[instrument(skip(self, artifacts))]
    fn convert_artifacts(&self, artifacts: &[CoreArtifact]) -> Result<Vec<Artifact>> {
        artifacts
            .iter()
            .enumerate()
            .map(|(idx, artifact)| {
                self.convert_artifact(artifact)
                    .with_context(|| format!("Failed to convert artifact at index {}", idx))
            })
            .collect()
    }
    
    /// Convert a single artifact with comprehensive mapping
    fn convert_artifact(&self, artifact: &CoreArtifact) -> Result<Artifact> {
        // Validate artifact
        if artifact.content.is_empty() {
            bail!("Artifact has empty content");
        }
        
        if artifact.content.len() > 1_000_000 {
            bail!("Artifact content exceeds maximum size limit");
        }
        
        Ok(Artifact {
            id: artifact.id.to_string(),
            version: 1,
            content: self.preprocess_content(&artifact.content),
            platform: self.convert_platform(&artifact.platform)?,
            artifact_type: self.convert_artifact_type(&artifact.artifact_type),
            metadata: self.extract_metadata(artifact),
            created_at: artifact.created_at,
            embedding: None, // Will be computed by engine
            parent_id: None,
            tags: self.extract_tags(artifact),
        })
    }
    
    /// Preprocess content for optimal model performance
    fn preprocess_content(&self, content: &str) -> String {
        // Normalize whitespace
        let normalized = content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Truncate if necessary (preserve semantic meaning)
        if normalized.len() > 10000 {
            let truncated = normalized.chars().take(9997).collect::<String>();
            format!("{}...", truncated)
        } else {
            normalized
        }
    }
    
    /// Convert platform with validation
    fn convert_platform(&self, platform: &CorePlatform) -> Result<Platform> {
        Ok(match platform {
            CorePlatform::Slack => Platform::Slack,
            CorePlatform::GitHub => Platform::GitHub,
            CorePlatform::Jira => Platform::Jira,
            CorePlatform::Teams => Platform::Teams,
            CorePlatform::Asana => Platform::Asana,
            CorePlatform::VSCode => Platform::VSCode,
            CorePlatform::GoogleWorkspace => Platform::GoogleWorkspace,
            CorePlatform::Monday => Platform::Monday,
            CorePlatform::Trello => Platform::Trello,
            CorePlatform::Zoom => Platform::Zoom,
            CorePlatform::Figma => Platform::Figma,
            CorePlatform::Notion => Platform::Notion,
        })
    }
    
    /// Convert artifact type with metadata preservation
    fn convert_artifact_type(&self, artifact_type: &CoreArtifactType) -> ArtifactType {
        match artifact_type {
            CoreArtifactType::PullRequest { .. } => ArtifactType::PullRequest,
            CoreArtifactType::Issue { .. } => ArtifactType::Issue,
            CoreArtifactType::Commit { .. } => ArtifactType::Commit,
            CoreArtifactType::Document { .. } => ArtifactType::Document,
            CoreArtifactType::Message { .. } => ArtifactType::Message,
            CoreArtifactType::Meeting { .. } => ArtifactType::Meeting,
            CoreArtifactType::Task { .. } => ArtifactType::Task,
            CoreArtifactType::Review { .. } => ArtifactType::Review,
            CoreArtifactType::Deployment { .. } => ArtifactType::Deployment,
            CoreArtifactType::Metric { .. } => ArtifactType::Report,
            CoreArtifactType::Alert { .. } => ArtifactType::Alert,
            CoreArtifactType::Custom { .. } => ArtifactType::Comment,
            CoreArtifactType::Design { .. } => ArtifactType::Design,
            CoreArtifactType::TestResult { .. } => ArtifactType::Report,
        }
    }
    
    /// Extract metadata from artifact
    fn extract_metadata(&self, artifact: &CoreArtifact) -> Option<serde_json::Value> {
        if artifact.metadata.is_null() {
            return None;
        }
        
        Some(artifact.metadata.clone())
    }
    
    /// Extract tags from artifact content and metadata
    fn extract_tags(&self, artifact: &CoreArtifact) -> Option<Vec<String>> {
        let mut tags = Vec::new();
        
        // Extract from artifact type
        match &artifact.artifact_type {
            CoreArtifactType::PullRequest { title, .. } => {
                if title.contains("/") {
                    tags.push(format!("repo:{}", title.split("/").next().unwrap_or("unknown")));
                }
            }
            CoreArtifactType::Issue { id, .. } => {
                if id.contains("-") {
                    tags.push(format!("project:{}", id.split("-").next().unwrap_or("unknown")));
                }
            }
            _ => {}
        }
        
        // Use the static regex
        for capture in HASHTAG_REGEX.captures_iter(&artifact.content) {
            if let Some(tag) = capture.get(0) {
                tags.push(tag.as_str().to_string());
            }
        }
        
        if tags.is_empty() {
            None
        } else {
            Some(tags)
        }
    }
    
    /// Create enriched prediction context
    #[instrument(skip(self))]
    fn create_prediction_context(&self) -> PredictionContext {
        let now = Local::now();
        let config = &self.config.context;
        
        let mut context = PredictionContext {
            hour_of_day: if config.include_temporal { now.hour() } else { 12 },
            day_of_week: if config.include_temporal { 
                now.weekday().num_days_from_monday() 
            } else { 
                2 // Wednesday as neutral default
            },
            days_until_deadline: Some(7.0), // Default sprint length
            user_activity_level: config.defaults.user_activity_level,
            user_expertise_score: config.defaults.user_expertise_score,
            team_size: config.defaults.team_size,
            sprint_progress: Some(0.5), // Mid-sprint default
            related_artifacts_count: 0,
            workspace_activity_level: config.defaults.workspace_activity_level,
            platform_signals: None,
        };
        
        // Adjust for time zones and work patterns
        if config.include_temporal {
            context.platform_signals = Some(self.generate_platform_signals(&now));
        }
        
        context
    }
    
    /// Generate platform-specific signals
    fn generate_platform_signals(&self, time: &chrono::DateTime<Local>) -> serde_json::Value {
        serde_json::json!({
            "is_business_hours": time.hour() >= 9 && time.hour() < 18,
            "is_weekday": time.weekday().num_days_from_monday() < 5,
            "quarter": (time.month() - 1) / 3 + 1,
            "week_of_year": time.iso_week().week(),
        })
    }
    
    /// Execute prediction with comprehensive error handling
    async fn execute_prediction(
        &self,
        artifacts: Vec<Artifact>,
        context: PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        // Check circuit breaker
        if !self.circuit_breaker.can_proceed().await {
            return self.execute_fallback_prediction(artifacts, context).await;
        }
        
        // Generate embeddings for artifacts
        let mut combined_embedding = vec![0.0; 768]; // Or get from config
        for artifact in &artifacts {
            let embedding = self.embedder.embed_text(&artifact.content).await?;
            // Combine embeddings (e.g., average them)
            for (i, val) in embedding.iter().enumerate() {
                if i < combined_embedding.len() {
                    combined_embedding[i] += val / artifacts.len() as f32;
                }
            }
        }
        
        // Convert artifacts to core artifacts for predictor
        let core_artifacts: Vec<interstice_core::Artifact> = artifacts.iter()
            .map(|a| self.convert_to_core_artifact(a))
            .collect::<Result<Vec<_>>>()?;
        
        // Try primary predictor
        let start = Instant::now();
        match tokio::time::timeout(
            self.config.performance.prediction_timeout,
            self.predictor.predict(combined_embedding, &core_artifacts)
        ).await {
            Ok(Ok(predictions)) => {
                self.circuit_breaker.record_success().await;
                self.metrics.record_prediction_success(start.elapsed());
                Ok(predictions)
            }
            Ok(Err(e)) => {
                warn!("Primary predictor failed: {}", e);
                self.circuit_breaker.record_failure().await;
                self.metrics.record_prediction_failure();
                self.execute_fallback_prediction(artifacts, context).await
            }
            Err(_) => {
                warn!("Primary predictor timed out");
                self.circuit_breaker.record_failure().await;
                self.metrics.record_prediction_timeout();
                self.execute_fallback_prediction(artifacts, context).await
            }
        }
    }

    fn convert_to_core_artifact(&self, artifact: &Artifact) -> Result<interstice_core::Artifact> {
        Ok(interstice_core::Artifact {
            id: Uuid::parse_str(&artifact.id)?,
            workspace_id: interstice_core::WorkspaceId::new(),
            artifact_type: self.convert_to_core_artifact_type(&artifact.artifact_type),
            platform: self.convert_to_core_platform(&artifact.platform),
            content: artifact.content.clone(),
            metadata: artifact.metadata.clone().unwrap_or(serde_json::Value::Null),
            created_at: artifact.created_at,
            updated_at: artifact.created_at,
            version: 1,
            state: interstice_core::artifact::ArtifactState::Pending,
            quality_metrics: interstice_core::artifact::QualityMetrics::default(),
            related_artifacts: Vec::new(),
            tags: std::collections::HashSet::new(),
        })
    }

    fn convert_to_core_platform(&self, platform: &Platform) -> interstice_core::Platform {
        match platform {
            Platform::Slack => interstice_core::Platform::Slack,
            Platform::GitHub => interstice_core::Platform::GitHub,
            Platform::Jira => interstice_core::Platform::Jira,
            Platform::Teams => interstice_core::Platform::Teams,
            Platform::Asana => interstice_core::Platform::Asana,
            Platform::VSCode => interstice_core::Platform::VSCode,
            Platform::GoogleWorkspace => interstice_core::Platform::GoogleWorkspace,
            Platform::Monday => interstice_core::Platform::Monday,
            Platform::Trello => interstice_core::Platform::Trello,
            Platform::Zoom => interstice_core::Platform::Zoom,
            Platform::Figma => interstice_core::Platform::Figma,
            Platform::Notion => interstice_core::Platform::Notion,
        }
    }

    fn convert_to_core_artifact_type(&self, artifact_type: &ArtifactType) -> interstice_core::ArtifactType {
        match artifact_type {
            ArtifactType::PullRequest => interstice_core::ArtifactType::PullRequest { 
                number: 0,
                title: "Unknown PR".to_string(),
                state: interstice_core::artifact::PullRequestState::Open,
                files_changed: 0,
                additions: 0,
                deletions: 0,
                merged: false,
                draft: false,
                base_branch: "main".to_string(),
                head_branch: "feature".to_string(),
                author: "unknown".to_string(),
                reviewers: vec![],
                labels: vec![],
                merge_conflict: false,
                ci_status: None,
            },
            ArtifactType::Issue => interstice_core::ArtifactType::Issue { 
                id: "unknown".to_string(),
                title: "Unknown Issue".to_string(),
                state: interstice_core::artifact::IssueState::Open,
                priority: interstice_core::artifact::Priority::Medium,
                assignees: vec![],
                labels: vec![],
                story_points: None,
                sprint: None,
                epic: None,
                blocked: false,
                blockers: vec![],
                time_estimate: None,
                time_spent: None,
            },
            ArtifactType::Commit => interstice_core::ArtifactType::Commit { 
                sha: "unknown".to_string(),
                message: "Unknown commit".to_string(),
                author: "unknown".to_string(),
                committer: "unknown".to_string(),
                files_changed: 0,
                additions: 0,
                deletions: 0,
                branch: "main".to_string(),
                is_merge: false,
                signed: false,
                verified: false,
            },
            ArtifactType::Document => interstice_core::ArtifactType::Document { 
                id: "unknown".to_string(),
                title: "Unknown Document".to_string(),
                doc_type: interstice_core::artifact::DocumentType::Other("Unknown".to_string()),
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: chrono::Utc::now(),
                version: 1,
                is_template: false,
                access_level: interstice_core::artifact::AccessLevel::Internal,
            },
            ArtifactType::Message => interstice_core::ArtifactType::Message { 
                id: "unknown".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "unknown".to_string(),
                content: "Unknown message".to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            // Map additional ML types to closest core equivalents
            ArtifactType::Comment => interstice_core::ArtifactType::Message { 
                id: "unknown".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "unknown".to_string(),
                content: "comment".to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            ArtifactType::Review => interstice_core::ArtifactType::PullRequest { 
                number: 0,
                title: "Review".to_string(),
                state: interstice_core::artifact::PullRequestState::Open,
                files_changed: 0,
                additions: 0,
                deletions: 0,
                merged: false,
                draft: false,
                base_branch: "main".to_string(),
                head_branch: "feature".to_string(),
                author: "unknown".to_string(),
                reviewers: vec![],
                labels: vec![],
                merge_conflict: false,
                ci_status: None,
            },
            ArtifactType::Meeting => interstice_core::ArtifactType::Message { 
                id: "unknown".to_string(),
                channel: "general".to_string(),
                thread_id: None,
                author: "unknown".to_string(),
                content: "meeting".to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            ArtifactType::Task => interstice_core::ArtifactType::Issue { 
                id: "task".to_string(),
                title: "Task".to_string(),
                state: interstice_core::artifact::IssueState::Open,
                priority: interstice_core::artifact::Priority::Medium,
                assignees: vec![],
                labels: vec![],
                story_points: None,
                sprint: None,
                epic: None,
                blocked: false,
                blockers: vec![],
                time_estimate: None,
                time_spent: None,
            },
            ArtifactType::Epic => interstice_core::ArtifactType::Issue { 
                id: "epic".to_string(),
                title: "Epic".to_string(),
                state: interstice_core::artifact::IssueState::Open,
                priority: interstice_core::artifact::Priority::Medium,
                assignees: vec![],
                labels: vec![],
                story_points: None,
                sprint: None,
                epic: None,
                blocked: false,
                blockers: vec![],
                time_estimate: None,
                time_spent: None,
            },
            ArtifactType::Design => interstice_core::ArtifactType::Document { 
                id: "unknown".to_string(),
                title: "design".to_string(),
                doc_type: interstice_core::artifact::DocumentType::Design,
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: chrono::Utc::now(),
                version: 1,
                is_template: false,
                access_level: interstice_core::artifact::AccessLevel::Internal,
            },
            ArtifactType::Deployment => interstice_core::ArtifactType::Commit { 
                sha: "deployment".to_string(),
                message: "Deployment".to_string(),
                author: "unknown".to_string(),
                committer: "unknown".to_string(),
                files_changed: 0,
                additions: 0,
                deletions: 0,
                branch: "main".to_string(),
                is_merge: false,
                signed: false,
                verified: false,
            },
            ArtifactType::Alert => interstice_core::ArtifactType::Message { 
                id: "unknown".to_string(),
                channel: "alerts".to_string(),
                thread_id: None,
                author: "system".to_string(),
                content: "alert".to_string(),
                mentions: vec![],
                attachments: vec![],
                reactions: std::collections::HashMap::new(),
                sentiment: interstice_core::artifact::Sentiment::Neutral,
                intent: interstice_core::artifact::MessageIntent::Other,
                is_edited: false,
                reply_count: 0,
            },
            ArtifactType::Report => interstice_core::ArtifactType::Document { 
                id: "unknown".to_string(),
                title: "report".to_string(),
                doc_type: interstice_core::artifact::DocumentType::Other("Report".to_string()),
                url: None,
                author: "unknown".to_string(),
                collaborators: vec![],
                word_count: None,
                last_modified: chrono::Utc::now(),
                version: 1,
                is_template: false,
                access_level: interstice_core::artifact::AccessLevel::Internal,
            },
        }
    }
    
    /// Execute fallback prediction
    async fn execute_fallback_prediction(
        &self,
        artifacts: Vec<Artifact>,
        _context: PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        if let Some(fallback) = &self.fallback_predictor {
            info!("Using fallback engine for prediction");
            let _start = Instant::now();
            
            // Generate embeddings for fallback
            let mut combined_embedding = vec![0.0; 768];
            for artifact in &artifacts {
                let embedding = self.embedder.embed_text(&artifact.content).await?;
                for (i, val) in embedding.iter().enumerate() {
                    if i < combined_embedding.len() {
                        combined_embedding[i] += val / artifacts.len() as f32;
                    }
                }
            }
            
            // Convert artifacts to core artifacts
            let core_artifacts: Vec<interstice_core::Artifact> = artifacts.iter()
                .map(|a| self.convert_to_core_artifact(a))
                .collect::<Result<Vec<_>>>()?;
            
            match tokio::time::timeout(
                self.config.performance.prediction_timeout,
                fallback.predict(combined_embedding, &core_artifacts)
            ).await {
                Ok(result) => {
                    self.metrics.record_fallback_used();
                    result.context("Fallback prediction failed")
                }
                Err(_) => {
                    bail!("Both primary and fallback engines timed out")
                }
            }
        } else {
            bail!("Primary engine unavailable and no fallback configured")
        }
    }
}


#[async_trait]
impl MLPredictor for MLPredictorAdapter {
    async fn predict_outcomes(
        &self,
        artifacts: &[CoreArtifact],
    ) -> Result<Vec<CorePrediction>> {
        let _guard = self.rate_limiter.acquire().await;
        
        let start = Instant::now();
        
        // Input validation
        if artifacts.is_empty() {
            return Ok(Vec::new());
        }
        
        if artifacts.len() > 1000 {
            bail!("Too many artifacts for single prediction request");
        }
        
        // Rest of the implementation remains the same...
        // Check cache if enabled
        if self.config.performance.cache.enabled {
            if let Some(cached) = self.cache.get(artifacts).await {
                self.metrics.record_cache_hit();
                return Ok(cached);
            }
            self.metrics.record_cache_miss();
        }
        
        // Convert artifacts
        let ml_artifacts = self.convert_artifacts(artifacts)?;
        
        // Create context
        let context = self.create_prediction_context();
        
        // Process in batches if necessary
        let predictions = if ml_artifacts.len() > self.config.performance.batch_size {
            self.predict_in_batches(ml_artifacts, context).await?
        } else {
            self.execute_prediction(ml_artifacts, context).await?
        };
        
        // Convert to core predictions
        let core_predictions: Vec<CorePrediction> = predictions
            .into_iter()
            .filter_map(|p| {
                                    match Uuid::parse_str(&p.outcome_id) {
                        Ok(id) => Some(CorePrediction {
                            outcome_id: id,
                            outcome_name: p.outcome_name,
                            confidence: p.confidence,
                            reasoning: p.reasoning,
                            suggested_targets: vec![],
                            estimated_impact: 0.5,
                            recommended_priority: Priority::Medium,
                        }),
                    Err(e) => {
                        warn!("Invalid outcome ID {}: {}", p.outcome_id, e);
                        None
                    }
                }
            })
            .collect();
        
        // Update cache
        if self.config.performance.cache.enabled {
            self.cache.put(artifacts, core_predictions.clone()).await;
        }
        
        // Record metrics
        self.metrics.record_prediction_latency(start.elapsed());
        
        Ok(core_predictions)
    }
}


impl MLPredictorAdapter {
    /// Predict in batches for large inputs
    async fn predict_in_batches(
        &self,
        artifacts: Vec<Artifact>,
        context: PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        let mut all_predictions = Vec::new();
        
        for chunk in artifacts.chunks(self.config.performance.batch_size) {
            let batch_predictions = self.execute_prediction(
                chunk.to_vec(),
                context.clone()
            ).await?;
            
            all_predictions.extend(batch_predictions);
        }
        
        // Deduplicate and merge predictions
        self.merge_predictions(all_predictions)
    }
    
    /// Merge and deduplicate predictions
    fn merge_predictions(
        &self,
        predictions: Vec<OutcomePrediction>,
    ) -> Result<Vec<OutcomePrediction>> {
        let mut merged: HashMap<String, OutcomePrediction> = HashMap::new();
        
        for pred in predictions {
            merged
                .entry(pred.outcome_id.clone())
                .and_modify(|existing| {
                    // Keep the prediction with higher confidence
                    if pred.confidence > existing.confidence {
                        *existing = pred.clone();
                    }
                })
                .or_insert(pred);
        }
        
        let mut result: Vec<_> = merged.into_values().collect();
        result.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        result.truncate(10); // Limit to top 10 predictions
        
        Ok(result)
    }
}

// Cache Implementation
// -----------------------------------------------------------------------------

struct PredictionCache {
    cache: Arc<RwLock<lru::LruCache<String, CachedPrediction>>>,
    ttl: Duration,
}

struct CachedPrediction {
    predictions: Vec<CorePrediction>,
    timestamp: Instant,
}

impl PredictionCache {
    fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(max_size).unwrap()
            ))),
            ttl,
        }
    }
    
    async fn get(&self, artifacts: &[CoreArtifact]) -> Option<Vec<CorePrediction>> {
        let key = self.generate_key(artifacts);
        let mut cache = self.cache.write().await;
        
        if let Some(cached) = cache.get_mut(&key) {
            if cached.timestamp.elapsed() < self.ttl {
                return Some(cached.predictions.clone());
            }
        }
        
        None
    }
    
    async fn put(&self, artifacts: &[CoreArtifact], predictions: Vec<CorePrediction>) {
        let key = self.generate_key(artifacts);
        let mut cache = self.cache.write().await;
        
        cache.put(key, CachedPrediction {
            predictions,
            timestamp: Instant::now(),
        });
    }
    
    fn generate_key(&self, artifacts: &[CoreArtifact]) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        for artifact in artifacts {
            artifact.id.hash(&mut hasher);
            artifact.content.hash(&mut hasher);
        }
        
        format!("{:x}", hasher.finish())
    }
}

// Circuit Breaker Implementation
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    success_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
}


impl CircuitBreaker {
    fn new(failure_threshold: u32, success_threshold: u32, timeout: Duration) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            failure_threshold,
            success_threshold,
            timeout,
        }
    }
    
    async fn can_proceed(&self) -> bool {
        let mut state = self.state.write().await;
        
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let last_failure = self.last_failure_time.read().await;
                if let Some(time) = *last_failure {
                    if time.elapsed() >= self.timeout {
                        *state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }
    
    async fn record_success(&self) {
        let mut state = self.state.write().await;
        let mut success_count = self.success_count.write().await;
        let mut failure_count = self.failure_count.write().await;
        
        match *state {
            CircuitState::HalfOpen => {
                *success_count += 1;
                if *success_count >= self.success_threshold {
                    *state = CircuitState::Closed;
                    *success_count = 0;
                    *failure_count = 0;
                }
            }
            _ => {
                *failure_count = 0;
            }
        }
    }
    
    async fn record_failure(&self) {
        let mut state = self.state.write().await;
        let mut failure_count = self.failure_count.write().await;
        let mut last_failure_time = self.last_failure_time.write().await;
        
        *failure_count += 1;
        *last_failure_time = Some(Instant::now());
        
        match *state {
            CircuitState::Closed => {
                if *failure_count >= self.failure_threshold {
                    *state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
            }
            _ => {}
        }
    }
}

// Metrics Collection
// -----------------------------------------------------------------------------

struct AdapterMetrics {
    prediction_count: Arc<std::sync::atomic::AtomicU64>,
    success_count: Arc<std::sync::atomic::AtomicU64>,
    failure_count: Arc<std::sync::atomic::AtomicU64>,
    timeout_count: Arc<std::sync::atomic::AtomicU64>,
    cache_hits: Arc<std::sync::atomic::AtomicU64>,
    cache_misses: Arc<std::sync::atomic::AtomicU64>,
    fallback_used: Arc<std::sync::atomic::AtomicU64>,
    total_latency_ms: Arc<std::sync::atomic::AtomicU64>,
}

impl AdapterMetrics {
    fn new() -> Self {
        Self {
            prediction_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            success_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            failure_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            timeout_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cache_hits: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            cache_misses: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            fallback_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_latency_ms: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    fn record_prediction_success(&self, duration: Duration) {
        self.success_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_latency_ms.fetch_add(
            duration.as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed
        );
    }
    
    fn record_prediction_failure(&self) {
        self.failure_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_prediction_timeout(&self) {
        self.timeout_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_fallback_used(&self) {
        self.fallback_used.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn record_prediction_latency(&self, duration: Duration) {
        self.prediction_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.total_latency_ms.fetch_add(
            duration.as_millis() as u64,
            std::sync::atomic::Ordering::Relaxed
        );
    }
}