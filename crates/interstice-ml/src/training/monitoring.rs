// interstice-ml/src/training/monitoring.rs
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use prometheus::{
    CounterVec,  GaugeVec, Histogram, HistogramOpts, HistogramVec, 
    IntGauge, IntGaugeVec, Registry, TextEncoder
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{error, info};
use uuid::Uuid;
use warp::Filter;

// Metrics Collection
// -----------------------------------------------------------------------------

pub struct MetricsCollector {
    registry: Registry,
    
    // Training metrics
    training_duration: HistogramVec,
    training_examples_used: HistogramVec,
    training_accuracy: GaugeVec,
    training_loss: GaugeVec,
    training_attempts: CounterVec,
    training_failures: CounterVec,
    
    // Model metrics
    model_size_bytes: GaugeVec,
    model_versions: IntGaugeVec,
    model_rollbacks: CounterVec,
    
    // Queue metrics
    queue_size: IntGauge,
    queue_processing_time: Histogram,
    queue_wait_time: Histogram,
    
    // System metrics
    active_workspaces: IntGauge,
    total_examples: IntGauge,
    database_connections: IntGauge,
    memory_usage_bytes: IntGauge,
}

impl MetricsCollector {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        // Training metrics
        let training_duration = HistogramVec::new(
            HistogramOpts::new(
                "training_duration_seconds",
                "Time taken to train a model"
            ).buckets(vec![10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0]),
            &["workspace_id", "status"]
        )?;
        registry.register(Box::new(training_duration.clone()))?;
        
        let training_examples_used = HistogramVec::new(
            HistogramOpts::new(
                "training_examples_count",
                "Number of examples used in training"
            ).buckets(vec![100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]),
            &["workspace_id"]
        )?;
        registry.register(Box::new(training_examples_used.clone()))?;
        
        let training_accuracy = GaugeVec::new(
            prometheus::opts!(
                "training_accuracy",
                "Model accuracy after training"
            ),
            &["workspace_id", "model_version"]
        )?;
        registry.register(Box::new(training_accuracy.clone()))?;
        
        let training_loss = GaugeVec::new(
            prometheus::opts!(
                "training_loss",
                "Model loss after training"
            ),
            &["workspace_id", "model_version"]
        )?;
        registry.register(Box::new(training_loss.clone()))?;
        
        let training_attempts = CounterVec::new(
            prometheus::opts!(
                "training_attempts_total",
                "Total number of training attempts"
            ),
            &["workspace_id"]
        )?;
        registry.register(Box::new(training_attempts.clone()))?;
        
        let training_failures = CounterVec::new(
            prometheus::opts!(
                "training_failures_total",
                "Total number of training failures"
            ),
            &["workspace_id", "reason"]
        )?;
        registry.register(Box::new(training_failures.clone()))?;
        
        // Model metrics
        let model_size_bytes = GaugeVec::new(
            prometheus::opts!(
                "model_size_bytes",
                "Size of the model in bytes"
            ),
            &["workspace_id", "version"]
        )?;
        registry.register(Box::new(model_size_bytes.clone()))?;
        
        let model_versions = IntGaugeVec::new(
            prometheus::opts!(
                "model_versions_count",
                "Number of model versions"
            ),
            &["workspace_id"]
        )?;
        registry.register(Box::new(model_versions.clone()))?;
        
        let model_rollbacks = CounterVec::new(
            prometheus::opts!(
                "model_rollbacks_total",
                "Total number of model rollbacks"
            ),
            &["workspace_id", "reason"]
        )?;
        registry.register(Box::new(model_rollbacks.clone()))?;
        
        // Queue metrics
        let queue_size = IntGauge::new(
            "training_queue_size",
            "Number of items in training queue"
        )?;
        registry.register(Box::new(queue_size.clone()))?;
        
        let queue_processing_time = Histogram::with_opts(
            HistogramOpts::new(
                "queue_processing_seconds",
                "Time to process queue items"
            ).buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0])
        )?;
        registry.register(Box::new(queue_processing_time.clone()))?;
        
        let queue_wait_time = Histogram::with_opts(
            HistogramOpts::new(
                "queue_wait_seconds",
                "Time items spend waiting in queue"
            ).buckets(vec![1.0, 10.0, 60.0, 300.0, 600.0, 3600.0])
        )?;
        registry.register(Box::new(queue_wait_time.clone()))?;
        
        // System metrics
        let active_workspaces = IntGauge::new(
            "active_workspaces",
            "Number of active ML workspaces"
        )?;
        registry.register(Box::new(active_workspaces.clone()))?;
        
        let total_examples = IntGauge::new(
            "total_training_examples",
            "Total number of training examples"
        )?;
        registry.register(Box::new(total_examples.clone()))?;
        
        let database_connections = IntGauge::new(
            "database_connections_active",
            "Number of active database connections"
        )?;
        registry.register(Box::new(database_connections.clone()))?;
        
        let memory_usage_bytes = IntGauge::new(
            "memory_usage_bytes",
            "Memory usage in bytes"
        )?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;
        
        Ok(Self {
            registry,
            training_duration,
            training_examples_used,
            training_accuracy,
            training_loss,
            training_attempts,
            training_failures,
            model_size_bytes,
            model_versions,
            model_rollbacks,
            queue_size,
            queue_processing_time,
            queue_wait_time,
            active_workspaces,
            total_examples,
            database_connections,
            memory_usage_bytes,
        })
    }
    
    pub fn record_training_start(&self, workspace_id: Uuid) {
        self.training_attempts
            .with_label_values(&[&workspace_id.to_string()])
            .inc();
    }
    
    pub fn record_training_complete(
        &self,
        workspace_id: Uuid,
        duration: Duration,
        examples: usize,
        accuracy: f64,
        loss: f64,
        version: &str,
    ) {
        let workspace_str = workspace_id.to_string();
        
        self.training_duration
            .with_label_values(&[&workspace_str, &"success".to_string()])
            .observe(duration.as_secs_f64());
        
        self.training_examples_used
            .with_label_values(&[&workspace_str])
            .observe(examples as f64);
        
        self.training_accuracy
            .with_label_values(&[&workspace_str, &version.to_string()])
            .set(accuracy);
        
        self.training_loss
            .with_label_values(&[&workspace_str, &version.to_string()])
            .set(loss);
    }
    
    pub fn record_training_failure(&self, workspace_id: Uuid, reason: &str) {
        self.training_failures
            .with_label_values(&[&workspace_id.to_string(), &reason.to_string()])
            .inc();
    }
    
    pub fn record_model_saved(&self, workspace_id: Uuid, version: &str, size_bytes: u64) {
        self.model_size_bytes
            .with_label_values(&[&workspace_id.to_string(), &version.to_string()])
            .set(size_bytes as f64);
    }
    
    pub fn record_model_rollback(&self, workspace_id: Uuid, reason: &str) {
        self.model_rollbacks
            .with_label_values(&[&workspace_id.to_string(), &reason.to_string()])
            .inc();
    }
    
    pub fn update_queue_size(&self, size: i64) {
        self.queue_size.set(size);
    }
    
    pub fn record_queue_processing(&self, duration: Duration) {
        self.queue_processing_time.observe(duration.as_secs_f64());
    }
    
    pub fn record_queue_wait(&self, duration: Duration) {
        self.queue_wait_time.observe(duration.as_secs_f64());
    }
    
    pub fn update_system_metrics(
        &self,
        active_workspaces: i64,
        total_examples: i64,
        db_connections: i64,
        memory_bytes: i64,
    ) {
        self.active_workspaces.set(active_workspaces);
        self.total_examples.set(total_examples);
        self.database_connections.set(db_connections);
        self.memory_usage_bytes.set(memory_bytes);
    }
    
    pub async fn start_metrics_server(self: Arc<Self>, port: u16) -> Result<()> {
        let metrics = warp::path!("metrics")
            .map(move || {
                let encoder = TextEncoder::new();
                let metric_families = self.registry.gather();
                
                match encoder.encode_to_string(&metric_families) {
                    Ok(body) => warp::reply::with_status(
                        body,
                        warp::http::StatusCode::OK,
                    ),
                    Err(e) => {
                        error!("Failed to encode metrics: {}", e);
                        warp::reply::with_status(
                            "Internal Server Error".to_string(),
                            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
                        )
                    }
                }
            });
        
        let health = warp::path!("health")
            .map(|| warp::reply::json(&HealthStatus::healthy()));
        
        let routes = metrics.or(health);
        
        info!("Starting metrics server on port {}", port);
        
        warp::serve(routes)
            .run(([0, 0, 0, 0], port))
            .await;
        
        Ok(())
    }
}

// Health Checks
// -----------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: i64,
    pub components: Vec<ComponentHealth>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: String,
    pub message: Option<String>,
    pub latency_ms: Option<u64>,
}

impl HealthStatus {
    pub fn healthy() -> Self {
        Self {
            status: "healthy".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            components: vec![],
        }
    }
    
    pub fn with_components(components: Vec<ComponentHealth>) -> Self {
        let overall_status = if components.iter().all(|c| c.status == "healthy") {
            "healthy"
        } else if components.iter().any(|c| c.status == "critical") {
            "critical"
        } else {
            "degraded"
        };
        
        Self {
            status: overall_status.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            components,
        }
    }
}

// Alerting
// -----------------------------------------------------------------------------

pub struct AlertManager {
    rules: Arc<RwLock<Vec<AlertRule>>>,
    notifiers: Vec<Arc<dyn AlertNotifier>>,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub message_template: String,
    pub cooldown: Duration,
    pub last_triggered: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    TrainingFailureRate { threshold: f64, window: Duration },
    ModelAccuracyDrop { threshold: f64 },
    QueueBacklog { threshold: usize, duration: Duration },
    MemoryUsage { threshold: f64 },
    DatabaseConnections { threshold: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[async_trait::async_trait]
pub trait AlertNotifier: Send + Sync {
    async fn send_alert(&self, alert: Alert) -> Result<()>;
}

#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: serde_json::Value,
}

// Slack Notifier
pub struct SlackNotifier {
    webhook_url: String,
    channel: String,
}

impl SlackNotifier {
    pub fn new(webhook_url: String, channel: String) -> Self {
        Self { webhook_url, channel }
    }
}

#[async_trait::async_trait]
impl AlertNotifier for SlackNotifier {
    async fn send_alert(&self, alert: Alert) -> Result<()> {
        let emoji = match alert.severity {
            AlertSeverity::Info => ":information_source:",
            AlertSeverity::Warning => ":warning:",
            AlertSeverity::Error => ":x:",
            AlertSeverity::Critical => ":rotating_light:",
        };
        
        let payload = serde_json::json!({
            "channel": self.channel,
            "username": "ML Training Monitor",
            "icon_emoji": ":robot_face:",
            "attachments": [{
                "color": match alert.severity {
                    AlertSeverity::Info => "good",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Error | AlertSeverity::Critical => "danger",
                },
                "title": format!("{} {}", emoji, alert.rule_name),
                "text": alert.message,
                "fields": [
                    {
                        "title": "Severity",
                        "value": format!("{:?}", alert.severity),
                        "short": true
                    },
                    {
                        "title": "Time",
                        "value": alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                        "short": true
                    }
                ],
                "footer": "ML Training System",
                "ts": alert.timestamp.timestamp()
            }]
        });
        
        reqwest::Client::new()
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await?;
        
        Ok(())
    }
}

// PagerDuty Notifier
pub struct PagerDutyNotifier {
    integration_key: String,
}

impl PagerDutyNotifier {
    pub fn new(integration_key: String) -> Self {
        Self { integration_key }
    }
}

#[async_trait::async_trait]
impl AlertNotifier for PagerDutyNotifier {
    async fn send_alert(&self, alert: Alert) -> Result<()> {
        if !matches!(alert.severity, AlertSeverity::Critical | AlertSeverity::Error) {
            return Ok(()); // Only send critical/error to PagerDuty
        }
        
        let severity = match alert.severity {
            AlertSeverity::Critical => "critical",
            AlertSeverity::Error => "error",
            AlertSeverity::Warning => "warning",
            AlertSeverity::Info => "info",
        };
        
        let payload = serde_json::json!({
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.message,
                "source": "ml-training-system",
                "severity": severity,
                "timestamp": alert.timestamp.to_rfc3339(),
                "custom_details": alert.context
            }
        });
        
        reqwest::Client::new()
            .post("https://events.pagerduty.com/v2/enqueue")
            .json(&payload)
            .send()
            .await?;
        
        Ok(())
    }
}

impl AlertManager {
    pub fn new(notifiers: Vec<Arc<dyn AlertNotifier>>) -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            notifiers,
        }
    }
    
    pub async fn add_rule(&self, rule: AlertRule) {
        let mut rules = self.rules.write().await;
        rules.push(rule);
    }
    
    pub async fn check_and_alert(&self, metrics: &MetricsSnapshot) {
        let mut rules = self.rules.write().await;
        
        for rule in rules.iter_mut() {
            // Check cooldown
            if let Some(last_triggered) = rule.last_triggered {
                if last_triggered.elapsed() < rule.cooldown {
                    continue;
                }
            }
            
            // Check condition
            let should_alert = match &rule.condition {
                AlertCondition::TrainingFailureRate { threshold, window } => {
                    metrics.training_failure_rate > *threshold
                }
                AlertCondition::ModelAccuracyDrop { threshold } => {
                    metrics.accuracy_delta.map_or(false, |delta| delta < -*threshold)
                }
                AlertCondition::QueueBacklog { threshold, duration } => {
                    metrics.queue_size > *threshold
                }
                AlertCondition::MemoryUsage { threshold } => {
                    metrics.memory_usage_percent > *threshold
                }
                AlertCondition::DatabaseConnections { threshold } => {
                    metrics.database_connections > *threshold
                }
            };
            
            if should_alert {
                rule.last_triggered = Some(Instant::now());
                
                let alert = Alert {
                    rule_name: rule.name.clone(),
                    severity: rule.severity.clone(),
                    message: rule.message_template.clone(), // Could template with actual values
                    timestamp: chrono::Utc::now(),
                    context: serde_json::to_value(metrics).unwrap_or(serde_json::Value::Null),
                };
                
                // Send to all notifiers
                for notifier in &self.notifiers {
                    if let Err(e) = notifier.send_alert(alert.clone()).await {
                        error!("Failed to send alert: {}", e);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub training_failure_rate: f64,
    pub accuracy_delta: Option<f64>,
    pub queue_size: usize,
    pub memory_usage_percent: f64,
    pub database_connections: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}