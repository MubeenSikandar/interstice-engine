//interstice-ml/src/feedback/mod.rs
use anyhow::{Result, Context};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Transaction, Postgres};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tracing::{info, debug, warn, error, instrument};
use uuid::Uuid;


use crate::types::{UserAction, ActionType, TrainingExample, TrainingExampleRow};

/// Feedback processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackConfig {
    /// Batch size for processing feedback
    pub batch_size: usize,
    /// Interval for flushing feedback buffer (seconds)
    pub flush_interval_secs: u64,
    /// Minimum confidence threshold for auto-acceptance
    pub auto_accept_threshold: f32,
    /// Maximum retries for database operations
    pub max_retries: u32,
    /// Enable real-time model updates
    pub enable_realtime_updates: bool,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            flush_interval_secs: 30,
            auto_accept_threshold: 0.95,
            max_retries: 3,
            enable_realtime_updates: true,
        }
    }
}

/// Feedback event for real-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackEvent {
    pub workspace_id: Uuid,
    pub artifact_id: Uuid,
    pub outcome_id: Option<Uuid>,
    pub action_type: ActionType,
    pub confidence: f32,
    pub user_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub workspace_id: Uuid,
    pub date: NaiveDate,
    pub predictions_made: i64,
    pub predictions_accepted: i64,
    pub predictions_rejected: i64,
    pub predictions_corrected: i64,
    pub avg_confidence: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
}

/// Feedback statistics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackStats {
    pub total_feedback: i64,
    pub acceptance_rate: f64,
    pub rejection_rate: f64,
    pub correction_rate: f64,
    pub avg_response_time_ms: f64,
    pub top_accepted_outcomes: serde_json::Value,
    pub top_rejected_outcomes: serde_json::Value,
}

/// Production-ready feedback processor with real-time learning
pub struct FeedbackProcessor {
    db: Arc<PgPool>,
    config: FeedbackConfig,
    /// Buffer for batching feedback events
    event_buffer: Arc<RwLock<Vec<FeedbackEvent>>>,
    /// Channel for real-time processing
    event_sender: mpsc::Sender<FeedbackEvent>,
    /// Metrics cache for performance
    metrics_cache: Arc<RwLock<lru::LruCache<(Uuid, NaiveDate), PerformanceMetrics>>>,
}

impl FeedbackProcessor {
    /// Create new processor with configuration
    #[instrument(skip(database_url, config))]
    pub async fn new(database_url: &str, config: FeedbackConfig) -> Result<Self> {
        let pool = PgPool::connect(database_url)
            .await
            .context("Failed to connect to database")?;
        
        let db = Arc::new(pool);
        let event_buffer = Arc::new(RwLock::new(Vec::with_capacity(config.batch_size)));
        let (event_sender, event_receiver) = mpsc::channel::<FeedbackEvent>(1000);
        
        let metrics_cache = Arc::new(RwLock::new(
            lru::LruCache::new(std::num::NonZeroUsize::new(100).unwrap())
        ));
        
        let processor = Self {
            db: db.clone(),
            config: config.clone(),
            event_buffer: event_buffer.clone(),
            event_sender,
            metrics_cache: metrics_cache.clone(),
        };
        
        // Start background processing task
        processor.start_background_processor(event_receiver);
        
        // Start periodic flush task
        processor.start_flush_task();
        
        info!("Feedback processor initialized with config: {:?}", config);
        
        Ok(processor)
    }
    
    /// Process user action with comprehensive tracking
    #[instrument(skip(self, action), fields(workspace_id = %workspace_id))]
    pub async fn process_user_action(
        &self,
        workspace_id: Uuid,
        action: UserAction,
    ) -> Result<()> {
        let start = std::time::Instant::now();
        
        // Validate and parse IDs
        let artifact_id = Uuid::parse_str(&action.artifact_id)
            .context("Invalid artifact ID")?;
        let outcome_id = Uuid::parse_str(&action.outcome_id)
            .context("Invalid outcome ID")?;
        
        // Create feedback event
        let event = FeedbackEvent {
            workspace_id,
            artifact_id,
            outcome_id: Some(outcome_id),
            action_type: action.action_type.clone(),
            confidence: action.confidence.unwrap_or(1.0),
            user_id: action.user_id.clone(),
            metadata: action.metadata.clone(),
            timestamp: Utc::now(),
        };
        
        // Send to real-time processor
        if self.config.enable_realtime_updates {
            self.event_sender.send(event.clone()).await
                .context("Failed to send event to processor")?;
        }
        
        // Add to buffer for batch processing
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.push(event);
            
            // Flush if buffer is full
            if buffer.len() >= self.config.batch_size {
                let events = std::mem::replace(&mut *buffer, Vec::with_capacity(self.config.batch_size));
                drop(buffer); // Release lock before processing
                self.process_batch(events).await?;
            }
        }
        
        let duration = start.elapsed();
        debug!("Processed user action in {:?}", duration);
        
        // Update response time metrics
        self.update_response_metrics(workspace_id, duration.as_millis() as f64).await;
        
        Ok(())
    }
    
    /// Process a batch of feedback events efficiently
    #[instrument(skip(self, events))]
    async fn process_batch(&self, events: Vec<FeedbackEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        
        let mut tx = self.db.begin().await?;
        let batch_size = events.len();
        
        for event in events {
            self.record_feedback_transaction(&mut tx, &event).await?;
        }
        
        tx.commit().await.context("Failed to commit feedback batch")?;
        
        info!("Processed batch of {} feedback events", batch_size);
        Ok(())
    }
    
    /// Record feedback within a transaction
    async fn record_feedback_transaction(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        event: &FeedbackEvent,
    ) -> Result<()> {
        let feedback_type = match event.action_type {
            ActionType::Accept => "accepted",
            ActionType::Reject => "rejected",
            ActionType::Correct => "corrected",
            _ => "unknown",
        };
        
        // Insert or update feedback record
        sqlx::query!(
            r#"
            INSERT INTO feedback_events (
                id, workspace_id, artifact_id, outcome_id, 
                feedback_type, confidence, user_id, metadata, timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (workspace_id, artifact_id, outcome_id) 
            DO UPDATE SET 
                feedback_type = EXCLUDED.feedback_type,
                confidence = EXCLUDED.confidence,
                user_id = EXCLUDED.user_id,
                metadata = EXCLUDED.metadata,
                timestamp = EXCLUDED.timestamp,
                update_count = feedback_events.update_count + 1
            "#,
            Uuid::new_v4(),
            event.workspace_id,
            event.artifact_id,
            event.outcome_id,
            feedback_type,
            event.confidence as f64,
            event.user_id,
            event.metadata,
            event.timestamp
        )
        .execute(&mut **tx)
        .await?;
        
        // Update training examples if exists
        sqlx::query!(
            r#"
            UPDATE training_examples 
            SET 
                user_feedback = $1,
                feedback_confidence = $2,
                feedback_timestamp = $3,
                is_validated = true
            WHERE workspace_id = $4 AND artifact_id = $5
            "#,
            feedback_type,
            event.confidence as f64,
            event.timestamp,
            event.workspace_id,
            event.artifact_id
        )
        .execute(&mut **tx)
        .await?;
        
        Ok(())
    }
    
    /// Update performance metrics with exponential smoothing
    #[instrument(skip(self))]
    pub async fn update_performance_metrics(
        &self,
        workspace_id: Uuid,
        event: &FeedbackEvent,
    ) -> Result<()> {
        let today = Utc::now().date_naive();
        let cache_key = (workspace_id, today);
        
        // Try to get from cache first
        let cached_metrics = {
            let cache = self.metrics_cache.read().await;
            cache.peek(&cache_key).cloned()
        };
        
        let metrics = if let Some(mut metrics) = cached_metrics {
            // Update cached metrics
            metrics.predictions_made += 1;
            match event.action_type {
                ActionType::Accept => metrics.predictions_accepted += 1,
                ActionType::Reject => metrics.predictions_rejected += 1,
                ActionType::Correct => metrics.predictions_corrected += 1,
                _ => {}
            }
            
            // Exponential moving average for confidence
            let alpha = 0.1; // Smoothing factor
            metrics.avg_confidence = alpha * event.confidence as f64 + 
                                    (1.0 - alpha) * metrics.avg_confidence;
            
            // Recalculate accuracy metrics
            metrics.accuracy = if metrics.predictions_made > 0 {
                metrics.predictions_accepted as f64 / metrics.predictions_made as f64
            } else {
                0.0
            };
            
            metrics
        } else {
            // Create new metrics
            let mut metrics = PerformanceMetrics {
                workspace_id,
                date: today,
                predictions_made: 1,
                predictions_accepted: 0,
                predictions_rejected: 0,
                predictions_corrected: 0,
                avg_confidence: event.confidence as f64,
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
            };
            
            match event.action_type {
                ActionType::Accept => metrics.predictions_accepted = 1,
                ActionType::Reject => metrics.predictions_rejected = 1,
                ActionType::Correct => metrics.predictions_corrected = 1,
                _ => {}
            }
            
            metrics
        };
        
        // Update cache
        {
            let mut cache = self.metrics_cache.write().await;
            cache.put(cache_key, metrics.clone());
        }
        
        // Persist to database with retry logic
        self.persist_metrics_with_retry(&metrics).await?;
        
        Ok(())
    }
    
    /// Persist metrics with exponential backoff retry
    async fn persist_metrics_with_retry(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let mut retries = 0;
        let mut backoff = tokio::time::Duration::from_millis(100);
        
        loop {
            match self.persist_metrics(metrics).await {
                Ok(_) => return Ok(()),
                Err(e) if retries < self.config.max_retries => {
                    warn!("Failed to persist metrics (attempt {}): {}", retries + 1, e);
                    tokio::time::sleep(backoff).await;
                    backoff *= 2;
                    retries += 1;
                }
                Err(e) => {
                    error!("Failed to persist metrics after {} retries", self.config.max_retries);
                    return Err(e);
                }
            }
        }
    }
    
    /// Persist metrics to database
    async fn persist_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO model_performance (
                workspace_id, date, predictions_made, predictions_accepted,
                predictions_rejected, predictions_corrected, avg_confidence,
                accuracy, precision, recall
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (workspace_id, date) 
            DO UPDATE SET 
                predictions_made = EXCLUDED.predictions_made,
                predictions_accepted = EXCLUDED.predictions_accepted,
                predictions_rejected = EXCLUDED.predictions_rejected,
                predictions_corrected = EXCLUDED.predictions_corrected,
                avg_confidence = EXCLUDED.avg_confidence,
                accuracy = EXCLUDED.accuracy,
                precision = EXCLUDED.precision,
                recall = EXCLUDED.recall,
                updated_at = NOW()
            "#,
            metrics.workspace_id,
            metrics.date,
            metrics.predictions_made as i32,
            metrics.predictions_accepted as i32,
            metrics.predictions_rejected as i32,
            metrics.predictions_corrected,
            metrics.avg_confidence,
            metrics.accuracy,
            metrics.precision,
            metrics.recall
        )
        .execute(&*self.db)
        .await?;
        
        Ok(())
    }
    
    /// Get comprehensive feedback statistics
    #[instrument(skip(self))]
    pub async fn get_feedback_stats(
        &self,
        workspace_id: Uuid,
        days: i32,
    ) -> Result<FeedbackStats> {
        let since = Utc::now() - chrono::Duration::days(days as i64);
        
        let stats = sqlx::query_as!(
            FeedbackStats,
            r#"
            WITH feedback_summary AS (
    SELECT 
        COUNT(*) as total_feedback,
        COUNT(*) FILTER (WHERE feedback_type = 'accepted') as accepted,
        COUNT(*) FILTER (WHERE feedback_type = 'rejected') as rejected,
        COUNT(*) FILTER (WHERE feedback_type = 'corrected') as corrected,
        AVG(EXTRACT(EPOCH FROM (timestamp - created_at)) * 1000)::float8 as avg_response_time_ms
    FROM feedback_events
    WHERE workspace_id = $1 AND timestamp > $2
),
top_accepted AS (
    SELECT COALESCE(
        json_agg(
            json_build_object(
                'outcome_id', outcome_id,
                'count', count
            )
        ),
        '[]'
    ) as data
    FROM (
        SELECT outcome_id, COUNT(*) as count
        FROM feedback_events
        WHERE workspace_id = $1 AND timestamp > $2 AND feedback_type = 'accepted'
        GROUP BY outcome_id
        ORDER BY count DESC
        LIMIT 10
    ) sub
),
top_rejected AS (
    SELECT COALESCE(
        json_agg(
            json_build_object(
                'outcome_id', outcome_id,
                'count', count
            )
        ),
        '[]'
    ) as data
    FROM (
        SELECT outcome_id, COUNT(*) as count
        FROM feedback_events
        WHERE workspace_id = $1 AND timestamp > $2 AND feedback_type = 'rejected'
        GROUP BY outcome_id
        ORDER BY count DESC
        LIMIT 10
    ) sub
)
SELECT 
    fs.total_feedback as "total_feedback!",
    COALESCE(fs.accepted::float8 / NULLIF(fs.total_feedback, 0), 0.0) as "acceptance_rate!",
    COALESCE(fs.rejected::float8 / NULLIF(fs.total_feedback, 0), 0.0) as "rejection_rate!",
    COALESCE(fs.corrected::float8 / NULLIF(fs.total_feedback, 0), 0.0) as "correction_rate!",
    COALESCE(fs.avg_response_time_ms, 0.0) as "avg_response_time_ms!",
    ta.data as "top_accepted_outcomes!",
    tr.data as "top_rejected_outcomes!"
FROM feedback_summary fs
CROSS JOIN top_accepted ta
CROSS JOIN top_rejected tr
            "#,
            workspace_id,
            since
        )
        .fetch_one(&*self.db)
        .await?;
        
        Ok(stats)
    }
    
    /// Get training examples with feedback for model retraining
    #[instrument(skip(self))]
    pub async fn get_training_examples(
        &self,
        workspace_id: Uuid,
        limit: i32,
        only_validated: bool,
    ) -> Result<Vec<TrainingExample>> {
        let query = if only_validated {
            r#"
            SELECT 
                id,
                input_text,
                input_embedding::text::float4[] as input_embedding,
                suggested_outcome_id,
                actual_outcome_id,
                user_feedback,
                feedback_score,
                context::jsonb as context,
                created_at,
                is_validated,
                validation_method::text as validation_method
            FROM training_examples
            WHERE workspace_id = $1 
                AND is_validated = true
                AND user_feedback IS NOT NULL
            ORDER BY created_at DESC
            LIMIT $2
            "#
        } else {
            r#"
            SELECT 
                id,
                input_text,
                input_embedding::text::float4[] as input_embedding,
                suggested_outcome_id,
                actual_outcome_id,
                user_feedback,
                feedback_score,
                context::jsonb as context,
                created_at,
                is_validated,
                validation_method::text as validation_method
            FROM training_examples
            WHERE workspace_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            "#
        };

        let rows = sqlx::query_as::<_, TrainingExampleRow>(query)
            .bind(workspace_id)
            .bind(limit as i64)
            .fetch_all(&*self.db)
            .await
            .context("Failed to fetch training examples")?;

        let examples: Vec<TrainingExample> = rows.into_iter()
            .map(TrainingExample::from)
            .collect();

        Ok(examples)
    }
    
    /// Start background processor for real-time events
    fn start_background_processor(&self, mut receiver: mpsc::Receiver<FeedbackEvent>) {
        let db = self.db.clone();
        let metrics_cache = self.metrics_cache.clone();
        
        tokio::spawn(async move {
            while let Some(event) = receiver.recv().await {
                // Process event in real-time
                if let Err(e) = Self::process_realtime_event(&db, &metrics_cache, event).await {
                    error!("Failed to process real-time event: {}", e);
                }
            }
        });
    }
    
    /// Process single event in real-time
    async fn process_realtime_event(
        db: &PgPool,
        metrics_cache: &Arc<RwLock<lru::LruCache<(Uuid, NaiveDate), PerformanceMetrics>>>,
        event: FeedbackEvent,
    ) -> Result<()> {
        // Trigger model update if confidence is low
        if event.confidence < 0.5 {
            Self::trigger_model_update(db, event.workspace_id).await?;
        }
        
        // Update cache for real-time metrics
        let cache_key = (event.workspace_id, Utc::now().date_naive());
        let mut cache = metrics_cache.write().await;
        
        if let Some(metrics) = cache.get_mut(&cache_key) {
            metrics.predictions_made += 1;
            match event.action_type {
                ActionType::Accept => metrics.predictions_accepted += 1,
                ActionType::Reject => metrics.predictions_rejected += 1,
                ActionType::Correct => metrics.predictions_corrected += 1,
                _ => {}
            }
        }
        
        Ok(())
    }
    
    /// Trigger model update for workspace
    async fn trigger_model_update(db: &PgPool, workspace_id: Uuid) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO model_update_queue (workspace_id, priority, created_at)
            VALUES ($1, 1, NOW())
            ON CONFLICT (workspace_id) 
            DO UPDATE SET priority = LEAST(model_update_queue.priority - 1, -10)
            "#,
            workspace_id
        )
        .execute(db)
        .await?;
        
        info!("Triggered model update for workspace {}", workspace_id);
        Ok(())
    }
    
    /// Start periodic flush task
    fn start_flush_task(&self) {
        let buffer = self.event_buffer.clone();
        let interval = self.config.flush_interval_secs;
        let processor = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                tokio::time::Duration::from_secs(interval)
            );
            
            loop {
                interval.tick().await;
                
                let events = {
                    let mut buffer = buffer.write().await;
                    if buffer.is_empty() {
                        continue;
                    }
                    std::mem::replace(&mut *buffer, Vec::with_capacity(processor.config.batch_size))
                };
                
                if let Err(e) = processor.process_batch(events).await {
                    error!("Failed to flush feedback buffer: {}", e);
                }
            }
        });
    }
    
    /// Update response time metrics
    async fn update_response_metrics(&self, workspace_id: Uuid, response_time_ms: f64) {
        // This could be extended to track response times in a time-series database
        debug!("Response time for workspace {}: {}ms", workspace_id, response_time_ms);
    }
}

impl Clone for FeedbackProcessor {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            config: self.config.clone(),
            event_buffer: self.event_buffer.clone(),
            event_sender: self.event_sender.clone(),
            metrics_cache: self.metrics_cache.clone(),
        }
    }
}