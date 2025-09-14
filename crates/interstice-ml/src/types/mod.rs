//interstice-ml/src/types/mod.rs

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use sqlx::types::Json;
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

/// Core artifact representation with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Artifact {
    pub id: String,
    pub version: u32,
    pub content: String,
    pub platform: Platform,
    pub artifact_type: ArtifactType,
    pub metadata: Option<serde_json::Value>,
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
}

impl Artifact {
    /// Create a new artifact with minimal required fields
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        platform: Platform,
        artifact_type: ArtifactType,
    ) -> Self {
        Self {
            id: id.into(),
            version: 1,
            content: content.into(),
            platform,
            artifact_type,
            metadata: None,
            created_at: Utc::now(),
            embedding: None,
            parent_id: None,
            tags: None,
        }
    }

    /// Create artifact with builder pattern
    pub fn builder() -> ArtifactBuilder {
        ArtifactBuilder::default()
    }

    /// Check if artifact is a derivative of another
    pub fn is_derivative(&self) -> bool {
        self.parent_id.is_some()
    }

    /// Get artifact age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Validate artifact content
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.id.is_empty() {
            return Err(ValidationError::EmptyId);
        }
        if self.content.is_empty() {
            return Err(ValidationError::EmptyContent);
        }
        if self.version == 0 {
            return Err(ValidationError::InvalidVersion);
        }
        Ok(())
    }
}

/// Builder pattern for Artifact construction
#[derive(Default)]
pub struct ArtifactBuilder {
    id: Option<String>,
    version: Option<u32>,
    content: Option<String>,
    platform: Option<Platform>,
    artifact_type: Option<ArtifactType>,
    metadata: Option<serde_json::Value>,
    embedding: Option<Vec<f32>>,
    parent_id: Option<String>,
    tags: Option<Vec<String>>,
}

impl ArtifactBuilder {
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn version(mut self, version: u32) -> Self {
        self.version = Some(version);
        self
    }

    pub fn content(mut self, content: impl Into<String>) -> Self {
        self.content = Some(content.into());
        self
    }

    pub fn platform(mut self, platform: Platform) -> Self {
        self.platform = Some(platform);
        self
    }

    pub fn artifact_type(mut self, artifact_type: ArtifactType) -> Self {
        self.artifact_type = Some(artifact_type);
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn parent_id(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    pub fn build(self) -> Result<Artifact, ValidationError> {
        let artifact = Artifact {
            id: self.id.ok_or(ValidationError::MissingField("id"))?,
            version: self.version.unwrap_or(1),
            content: self
                .content
                .ok_or(ValidationError::MissingField("content"))?,
            platform: self
                .platform
                .ok_or(ValidationError::MissingField("platform"))?,
            artifact_type: self
                .artifact_type
                .ok_or(ValidationError::MissingField("artifact_type"))?,
            metadata: self.metadata,
            created_at: Utc::now(),
            embedding: self.embedding,
            parent_id: self.parent_id,
            tags: self.tags,
        };
        artifact.validate()?;
        Ok(artifact)
    }
}

/// Platform enumeration with comprehensive coverage
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Platform {
    Slack = 0,
    GitHub = 1,
    Jira = 2,
    Teams = 3,
    Asana = 4,
    VSCode = 5,
    GoogleWorkspace = 6,
    Monday = 7,
    Trello = 8,
    Zoom = 9,
    Figma = 10,
    Notion = 11,
}

impl Platform {
    /// Get all available platforms
    pub fn all() -> &'static [Platform] {
        use Platform::*;
        &[
            Slack,
            GitHub,
            Jira,
            Teams,
            Asana,
            VSCode,
            GoogleWorkspace,
            Monday,
            Trello,
            Zoom,
            Figma,
            Notion,
        ]
    }

    /// Get platform display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Platform::Slack => "Slack",
            Platform::GitHub => "GitHub",
            Platform::Jira => "Jira",
            Platform::Teams => "Microsoft Teams",
            Platform::Asana => "Asana",
            Platform::VSCode => "VS Code",
            Platform::GoogleWorkspace => "Google Workspace",
            Platform::Monday => "Monday.com",
            Platform::Trello => "Trello",
            Platform::Zoom => "Zoom",
            Platform::Figma => "Figma",
            Platform::Notion => "Notion",
        }
    }

    /// Check if platform supports real-time events
    pub fn supports_realtime(&self) -> bool {
        matches!(self, Platform::Slack | Platform::Teams | Platform::GitHub)
    }
}

impl fmt::Display for Platform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

impl FromStr for Platform {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "slack" => Ok(Platform::Slack),
            "github" => Ok(Platform::GitHub),
            "jira" => Ok(Platform::Jira),
            "teams" | "microsoft teams" => Ok(Platform::Teams),
            "asana" => Ok(Platform::Asana),
            "vscode" | "vs code" => Ok(Platform::VSCode),
            "google" | "google workspace" => Ok(Platform::GoogleWorkspace),
            "monday" | "monday.com" => Ok(Platform::Monday),
            "trello" => Ok(Platform::Trello),
            "zoom" => Ok(Platform::Zoom),
            "figma" => Ok(Platform::Figma),
            "notion" => Ok(Platform::Notion),
            _ => Err(ParseError::UnknownPlatform(s.to_string())),
        }
    }
}

/// Artifact type enumeration with rich categorization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ArtifactType {
    PullRequest = 0,
    Issue = 1,
    Commit = 2,
    Document = 3,
    Message = 4,
    Comment = 5,
    Review = 6,
    Meeting = 7,
    Task = 8,
    Epic = 9,
    Design = 10,
    Deployment = 11,
    Alert = 12,
    Report = 13,
}

impl ArtifactType {
    /// Get all artifact types
    pub fn all() -> &'static [ArtifactType] {
        use ArtifactType::*;
        &[
            PullRequest,
            Issue,
            Commit,
            Document,
            Message,
            Comment,
            Review,
            Meeting,
            Task,
            Epic,
            Design,
            Deployment,
            Alert,
            Report,
        ]
    }

    /// Get artifact priority weight for ML
    pub fn priority_weight(&self) -> f32 {
        match self {
            ArtifactType::PullRequest => 1.0,
            ArtifactType::Deployment => 0.95,
            ArtifactType::Issue => 0.9,
            ArtifactType::Epic => 0.85,
            ArtifactType::Commit => 0.8,
            ArtifactType::Review => 0.75,
            ArtifactType::Task => 0.7,
            ArtifactType::Document => 0.65,
            ArtifactType::Design => 0.6,
            ArtifactType::Meeting => 0.55,
            ArtifactType::Report => 0.5,
            ArtifactType::Comment => 0.4,
            ArtifactType::Message => 0.3,
            ArtifactType::Alert => 0.25,
        }
    }

    /// Check if artifact type represents completed work
    pub fn is_completion_indicator(&self) -> bool {
        matches!(
            self,
            ArtifactType::PullRequest
                | ArtifactType::Deployment
                | ArtifactType::Commit
                | ArtifactType::Review
        )
    }
}

impl fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ArtifactType::PullRequest => "Pull Request",
            ArtifactType::Issue => "Issue",
            ArtifactType::Commit => "Commit",
            ArtifactType::Document => "Document",
            ArtifactType::Message => "Message",
            ArtifactType::Comment => "Comment",
            ArtifactType::Review => "Review",
            ArtifactType::Meeting => "Meeting",
            ArtifactType::Task => "Task",
            ArtifactType::Epic => "Epic",
            ArtifactType::Design => "Design",
            ArtifactType::Deployment => "Deployment",
            ArtifactType::Alert => "Alert",
            ArtifactType::Report => "Report",
        };
        write!(f, "{}", s)
    }
}

/// Rich prediction context for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContext {
    pub hour_of_day: u32,
    pub day_of_week: u32,
    pub days_until_deadline: Option<f32>,
    pub user_activity_level: f32,
    pub user_expertise_score: f32,
    pub team_size: u32,
    pub sprint_progress: Option<f32>,
    pub related_artifacts_count: u32,
    pub workspace_activity_level: f32,
    pub platform_signals: Option<serde_json::Value>,
    pub historical_accuracy: Option<f32>,
}

impl PredictionContext {
    /// Create context from current environment
    pub fn from_environment() -> Self {
        let now = Utc::now();
        Self {
            hour_of_day: now.hour(),
            day_of_week: now.weekday().num_days_from_monday(),
            days_until_deadline: None,
            user_activity_level: 0.5,
            user_expertise_score: 0.5,
            team_size: 1,
            sprint_progress: None,
            related_artifacts_count: 0,
            workspace_activity_level: 0.5,
            platform_signals: None,
            historical_accuracy: None,
        }
    }

    /// Enrich context with workspace data
    pub fn with_workspace_data(mut self, data: serde_json::Value) -> Self {
        if let Some(team_size) = data.get("team_size").and_then(|v| v.as_u64()) {
            self.team_size = team_size as u32;
        }
        if let Some(activity) = data.get("activity_level").and_then(|v| v.as_f64()) {
            self.workspace_activity_level = activity as f32;
        }
        self.platform_signals = Some(data);
        self
    }

    /// Convert to feature vector for ML
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.hour_of_day as f32 / 24.0,
            self.day_of_week as f32 / 7.0,
            self.days_until_deadline.unwrap_or(30.0) / 30.0,
            self.user_activity_level,
            self.user_expertise_score,
            (self.team_size as f32).ln() / 10.0,
            self.sprint_progress.unwrap_or(0.5),
            (self.related_artifacts_count as f32).ln() / 10.0,
            self.workspace_activity_level,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionReasoning {
    pub summary: String,
    pub confidence_factors: Vec<String>,
    pub uncertainty_factors: Vec<String>,
    pub key_assumptions: Vec<String>,
    pub data_quality_score: f32,
    pub model_version: String,
}

impl PredictionReasoning {
    pub fn new(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            confidence_factors: Vec::new(),
            uncertainty_factors: Vec::new(),
            key_assumptions: Vec::new(),
            data_quality_score: 1.0,
            model_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
    
    pub fn with_confidence_factor(mut self, factor: impl Into<String>) -> Self {
        self.confidence_factors.push(factor.into());
        self
    }
    
    pub fn with_uncertainty_factor(mut self, factor: impl Into<String>) -> Self {
        self.uncertainty_factors.push(factor.into());
        self
    }
}

/// Outcome prediction with rich metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePrediction {
    pub outcome_id: String,
    pub outcome_name: String,
    pub confidence: f32,
    pub reasoning: Option<String>,
    pub contributing_factors: Vec<ContributingFactor>,
    pub alternative_outcomes: Vec<AlternativeOutcome>,
    pub predicted_impact: ImpactLevel,
    pub time_to_completion: Option<Duration>,
}

impl OutcomePrediction {
    /// Create a simple prediction
    pub fn simple(
        outcome_id: impl Into<String>,
        outcome_name: impl Into<String>,
        confidence: f32,
    ) -> Self {
        Self {
            outcome_id: outcome_id.into(),
            outcome_name: outcome_name.into(),
            confidence,
            reasoning: None,
            contributing_factors: Vec::new(),
            alternative_outcomes: Vec::new(),
            predicted_impact: ImpactLevel::Medium,
            time_to_completion: None,
        }
    }

    /// Check if prediction meets confidence threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get prediction quality score
    pub fn quality_score(&self) -> f32 {
        let base_score = self.confidence;
        let factor_bonus = (self.contributing_factors.len() as f32 * 0.05).min(0.2);
        let reasoning_bonus = if self.reasoning.is_some() { 0.1 } else { 0.0 };
        (base_score + factor_bonus + reasoning_bonus).min(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerOutcomeMetrics {
    pub outcome_name: String,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub support: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub true_negatives: usize,
}

impl PerOutcomeMetrics {
    pub fn new(outcome_name: impl Into<String>) -> Self {
        Self {
            outcome_name: outcome_name.into(),
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            support: 0,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            true_negatives: 0,
        }
    }

    /// Calculate metrics from confusion matrix values
    pub fn calculate(&mut self) {
        let tp = self.true_positives as f32;
        let fp = self.false_positives as f32;
        let fn_val = self.false_negatives as f32;

        // Precision = TP / (TP + FP)
        self.precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };

        // Recall = TP / (TP + FN)
        self.recall = if tp + fn_val > 0.0 {
            tp / (tp + fn_val)
        } else {
            0.0
        };

        // F1 = 2 * (precision * recall) / (precision + recall)
        self.f1_score = if self.precision + self.recall > 0.0 {
            2.0 * (self.precision * self.recall) / (self.precision + self.recall)
        } else {
            0.0
        };

        // Support = TP + FN (total actual positives)
        self.support = self.true_positives + self.false_negatives;
    }

    /// Get accuracy for this outcome
    pub fn accuracy(&self) -> f32 {
        let total = (self.true_positives
            + self.false_positives
            + self.false_negatives
            + self.true_negatives) as f32;
        if total > 0.0 {
            (self.true_positives + self.true_negatives) as f32 / total
        } else {
            0.0
        }
    }
}

/// Confusion matrix for multi-class classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Matrix dimensions (n_classes x n_classes)
    pub matrix: Vec<Vec<usize>>,
    /// Class labels
    pub labels: Vec<String>,
    /// Total predictions
    pub total: usize,
}

impl ConfusionMatrix {
    /// Create new confusion matrix with given labels
    pub fn new(labels: Vec<String>) -> Self {
        let n = labels.len();
        Self {
            matrix: vec![vec![0; n]; n],
            labels,
            total: 0,
        }
    }

    /// Add a prediction to the matrix
    pub fn add_prediction(&mut self, actual: usize, predicted: usize) {
        if actual < self.matrix.len() && predicted < self.matrix.len() {
            self.matrix[actual][predicted] += 1;
            self.total += 1;
        }
    }

    /// Get accuracy from confusion matrix
    pub fn accuracy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }

        let correct: usize = (0..self.matrix.len()).map(|i| self.matrix[i][i]).sum();

        correct as f32 / self.total as f32
    }

    /// Get per-class metrics
    pub fn per_class_metrics(&self) -> HashMap<String, PerOutcomeMetrics> {
        let mut metrics = HashMap::new();

        for (i, label) in self.labels.iter().enumerate() {
            let mut metric = PerOutcomeMetrics::new(label.clone());

            // True positives: diagonal element
            metric.true_positives = self.matrix[i][i];

            // False positives: sum of column i excluding diagonal
            metric.false_positives = (0..self.matrix.len())
                .filter(|&j| j != i)
                .map(|j| self.matrix[j][i])
                .sum();

            // False negatives: sum of row i excluding diagonal
            metric.false_negatives = (0..self.matrix.len())
                .filter(|&j| j != i)
                .map(|j| self.matrix[i][j])
                .sum();

            // True negatives: all correct predictions except for this class
            metric.true_negatives = (0..self.matrix.len())
                .flat_map(|j| (0..self.matrix.len()).map(move |k| (j, k)))
                .filter(|&(j, k)| j != i && k != i)
                .map(|(j, k)| self.matrix[j][k])
                .sum();

            metric.calculate();
            metrics.insert(label.clone(), metric);
        }

        metrics
    }

    /// Get macro-averaged metrics
    pub fn macro_metrics(&self) -> (f32, f32, f32) {
        let per_class = self.per_class_metrics();
        let n = per_class.len() as f32;

        if n == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let precision = per_class.values().map(|m| m.precision).sum::<f32>() / n;
        let recall = per_class.values().map(|m| m.recall).sum::<f32>() / n;
        let f1 = per_class.values().map(|m| m.f1_score).sum::<f32>() / n;

        (precision, recall, f1)
    }

    /// Get weighted-averaged metrics
    pub fn weighted_metrics(&self) -> (f32, f32, f32) {
        let per_class = self.per_class_metrics();
        let total_support: usize = per_class.values().map(|m| m.support).sum();

        if total_support == 0 {
            return (0.0, 0.0, 0.0);
        }

        let total_f = total_support as f32;

        let precision = per_class
            .values()
            .map(|m| m.precision * m.support as f32)
            .sum::<f32>()
            / total_f;

        let recall = per_class
            .values()
            .map(|m| m.recall * m.support as f32)
            .sum::<f32>()
            / total_f;

        let f1 = per_class
            .values()
            .map(|m| m.f1_score * m.support as f32)
            .sum::<f32>()
            / total_f;

        (precision, recall, f1)
    }

    /// Generate classification report
    pub fn classification_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!(
            "{:<20} {:>10} {:>10} {:>10} {:>10}\n",
            "Class", "Precision", "Recall", "F1-Score", "Support"
        ));
        report.push_str(&"-".repeat(70));
        report.push('\n');

        let metrics = self.per_class_metrics();
        for label in &self.labels {
            if let Some(m) = metrics.get(label) {
                report.push_str(&format!(
                    "{:<20} {:>10.3} {:>10.3} {:>10.3} {:>10}\n",
                    label, m.precision, m.recall, m.f1_score, m.support
                ));
            }
        }

        report.push_str(&"-".repeat(70));
        report.push('\n');

        let (macro_p, macro_r, macro_f1) = self.macro_metrics();
        report.push_str(&format!(
            "{:<20} {:>10.3} {:>10.3} {:>10.3}\n",
            "Macro avg", macro_p, macro_r, macro_f1
        ));

        let (weighted_p, weighted_r, weighted_f1) = self.weighted_metrics();
        report.push_str(&format!(
            "{:<20} {:>10.3} {:>10.3} {:>10.3}\n",
            "Weighted avg", weighted_p, weighted_r, weighted_f1
        ));

        report.push_str(&format!("\nAccuracy: {:.3}\n", self.accuracy()));
        report.push_str(&format!("Total predictions: {}\n", self.total));

        report
    }

    /// Export as CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("Actual\\Predicted");
        for label in &self.labels {
            csv.push(',');
            csv.push_str(label);
        }
        csv.push('\n');

        // Matrix rows
        for (i, row) in self.matrix.iter().enumerate() {
            csv.push_str(&self.labels[i]);
            for val in row {
                csv.push(',');
                csv.push_str(&val.to_string());
            }
            csv.push('\n');
        }

        csv
    }
}

/// Contributing factor for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub factor_type: String,
    pub weight: f32,
    pub description: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeOutcome {
    pub outcome_id: Uuid,
    pub outcome_name: String,
    pub probability: f32,
    pub relative_likelihood: f32,
    pub key_differences: Vec<String>,
}

/// Impact level enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ImpactLevel {
    Critical,
    High,
    Medium,
    Low,
    Negligible,
}

impl ImpactLevel {
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s >= 0.9 => ImpactLevel::Critical,
            s if s >= 0.7 => ImpactLevel::High,
            s if s >= 0.5 => ImpactLevel::Medium,
            s if s >= 0.3 => ImpactLevel::Low,
            _ => ImpactLevel::Negligible,
        }
    }
}

/// Duration for completion estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Duration {
    pub min_hours: f32,
    pub max_hours: f32,
    pub likely_hours: f32,
}

impl Duration {
    /// Create from hours with uncertainty
    pub fn from_hours_with_uncertainty(likely: f32, uncertainty_factor: f32) -> Self {
        Self {
            min_hours: likely * (1.0 - uncertainty_factor),
            max_hours: likely * (1.0 + uncertainty_factor),
            likely_hours: likely,
        }
    }

    /// Get as standard Duration
    pub fn as_std_duration(&self) -> std::time::Duration {
        std::time::Duration::from_secs((self.likely_hours * 3600.0) as u64)
    }

    /// Format as human-readable string
    pub fn to_human_string(&self) -> String {
        if self.likely_hours < 1.0 {
            format!("{} minutes", (self.likely_hours * 60.0) as u32)
        } else if self.likely_hours < 24.0 {
            format!("{:.1} hours", self.likely_hours)
        } else if self.likely_hours < 168.0 {
            format!("{:.1} days", self.likely_hours / 24.0)
        } else {
            format!("{:.1} weeks", self.likely_hours / 168.0)
        }
    }
}

/// Comprehensive model metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub correct_predictions: u64,
    pub total_predictions: u64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub auc_roc: Option<f64>,
    pub mean_confidence: f64,
    pub prediction_latency_ms: f64,
    pub cache_hit_rate: f32,
    pub last_updated: DateTime<Utc>,
    pub per_outcome_metrics: Option<serde_json::Value>,
}

impl ModelMetrics {
    pub fn default() -> Self {
        Self {
            correct_predictions: 0,
            total_predictions: 0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            auc_roc: None,
            mean_confidence: 0.0,
            prediction_latency_ms: 0.0,
            cache_hit_rate: 0.0,
            last_updated: Utc::now(),
            per_outcome_metrics: None,
        }
    }

    /// Update with confusion matrix
    pub fn from_confusion_matrix(matrix: &ConfusionMatrix) -> Self {
        let (precision, recall, f1) = matrix.weighted_metrics();
        let per_outcome = matrix.per_class_metrics();

        Self {
            correct_predictions: (0..matrix.matrix.len())
                .map(|i| matrix.matrix[i][i])
                .sum::<usize>() as u64,
            total_predictions: matrix.total as u64,
            accuracy: matrix.accuracy() as f64,
            precision: precision as f64,
            recall: recall as f64,
            f1_score: f1 as f64,
            auc_roc: None,
            mean_confidence: 0.0,
            prediction_latency_ms: 0.0,
            last_updated: Utc::now(),
            per_outcome_metrics: Some(serde_json::to_value(per_outcome).unwrap()),
            cache_hit_rate: 0.0,
        }
    }
    /// Update metrics with new prediction result
    pub fn update(&mut self, was_correct: bool, confidence: f32, latency_ms: f64) {
        self.total_predictions += 1;
        if was_correct {
            self.correct_predictions += 1;
        }

        // Update accuracy
        self.accuracy = self.correct_predictions as f64 / self.total_predictions as f64;

        // Update mean confidence (exponential moving average)
        let alpha = 0.1;
        self.mean_confidence = alpha * confidence as f64 + (1.0 - alpha) * self.mean_confidence;

        // Update latency (exponential moving average)
        self.prediction_latency_ms =
            alpha * latency_ms + (1.0 - alpha) * self.prediction_latency_ms;

        self.last_updated = Utc::now();
    }

    /// Calculate F1 score from precision and recall
    pub fn calculate_f1(&mut self) {
        if self.precision > 0.0 && self.recall > 0.0 {
            self.f1_score = 2.0 * (self.precision * self.recall) / (self.precision + self.recall);
        }
    }

    /// Get model performance rating
    pub fn performance_rating(&self) -> PerformanceRating {
        match self.accuracy {
            a if a >= 0.95 => PerformanceRating::Excellent,
            a if a >= 0.85 => PerformanceRating::Good,
            a if a >= 0.70 => PerformanceRating::Fair,
            a if a >= 0.50 => PerformanceRating::Poor,
            _ => PerformanceRating::Unacceptable,
        }
    }
}

/// Performance rating enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PerformanceRating {
    Excellent,
    Good,
    Fair,
    Poor,
    Unacceptable,
}

/// User action enumeration with rich semantics
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActionType {
    Accept,
    Reject,
    Modify,
    Defer,
    Correct,
    Skip,
    RequestMoreInfo,
    Escalate,
}

impl ActionType {
    /// Get feedback weight for ML training
    pub fn feedback_weight(&self) -> f32 {
        match self {
            ActionType::Accept => 1.0,
            ActionType::Correct => 0.8,
            ActionType::Modify => 0.6,
            ActionType::RequestMoreInfo => 0.4,
            ActionType::Defer => 0.2,
            ActionType::Skip => 0.0,
            ActionType::Reject => -0.5,
            ActionType::Escalate => -0.3,
        }
    }

    /// Check if action represents positive feedback
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            ActionType::Accept | ActionType::Correct | ActionType::Modify
        )
    }
}

/// User action with comprehensive metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAction {
    pub action_type: ActionType,
    pub artifact_id: String,
    pub outcome_id: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub confidence: Option<f32>,
    pub feedback_text: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub session_id: Option<Uuid>,
    pub platform: Option<Platform>,
}

impl UserAction {
    /// Create a new user action
    pub fn new(
        action_type: ActionType,
        artifact_id: impl Into<String>,
        outcome_id: impl Into<String>,
    ) -> Self {
        Self {
            action_type,
            artifact_id: artifact_id.into(),
            outcome_id: outcome_id.into(),
            timestamp: Utc::now(),
            user_id: None,
            confidence: None,
            feedback_text: None,
            metadata: None,
            session_id: None,
            platform: None,
        }
    }

    /// Calculate action quality score
    pub fn quality_score(&self) -> f32 {
        let mut score = 0.5;
        if self.confidence.is_some() {
            score += 0.2;
        }
        if self.feedback_text.is_some() {
            score += 0.2;
        }
        if self.user_id.is_some() {
            score += 0.1;
        }
        score
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub version: String,
    pub accuracy: f64,
    pub last_trained: DateTime<Utc>,
    pub training_runs: u64,
}

/// Training example for ML model updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: Uuid,
    pub input_text: String,
    pub input_embedding: Option<Vec<f32>>, // Changed from Vector to Vec<f32>
    pub suggested_outcome_id: Option<Uuid>,
    pub actual_outcome_id: Option<Uuid>,
    pub user_feedback: Option<String>,
    pub feedback_score: Option<f32>,
    pub context: Option<PredictionContext>,
    pub created_at: DateTime<Utc>,
    pub is_validated: bool,
    pub validation_method: Option<ValidationMethod>,
}

#[derive(Debug, sqlx::FromRow)]
pub struct TrainingExampleRow {
    id: Uuid,
    input_text: String,
    input_embedding: Option<Vec<f32>>,
    suggested_outcome_id: Option<Uuid>,
    actual_outcome_id: Option<Uuid>,
    user_feedback: Option<String>,
    feedback_score: Option<f32>,
    context: Option<Json<PredictionContext>>,
    created_at: DateTime<Utc>,
    is_validated: bool,
    validation_method: Option<String>,
}

impl From<TrainingExampleRow> for TrainingExample {
    fn from(row: TrainingExampleRow) -> Self {
        Self {
            id: row.id,
            input_text: row.input_text,
            input_embedding: row.input_embedding,
            suggested_outcome_id: row.suggested_outcome_id,
            actual_outcome_id: row.actual_outcome_id,
            user_feedback: row.user_feedback,
            feedback_score: row.feedback_score,
            context: row.context.map(|j| j.0), // Extract from Json wrapper
            created_at: row.created_at,
            is_validated: row.is_validated,
            validation_method: row.validation_method.map(ValidationMethod::from),
        }
    }
}

impl TrainingExample {
    /// Create new training example
    pub fn new(input_text: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            input_text: input_text.into(),
            input_embedding: None,
            suggested_outcome_id: None,
            actual_outcome_id: None,
            user_feedback: None,
            feedback_score: None,
            context: None,
            created_at: Utc::now(),
            is_validated: false,
            validation_method: None,
        }
    }

    pub fn importance(&self) -> f32 {
        let base = self.training_weight();

        // Boost importance if user provided explicit feedback
        let feedback_boost = if self.user_feedback.is_some() {
            0.2
        } else {
            0.0
        };

        // Boost if feedback score indicates strong signal
        let score_boost = self
            .feedback_score
            .map(|s| (s - 0.5).abs() * 0.2)
            .unwrap_or(0.0);

        (base + feedback_boost + score_boost).min(1.0)
    }

    /// Check if example is ready for training
    pub fn is_complete(&self) -> bool {
        self.actual_outcome_id.is_some()
            && (self.user_feedback.is_some() || self.feedback_score.is_some())
    }

    /// Get training weight based on validation
    pub fn training_weight(&self) -> f32 {
        if !self.is_validated {
            return 0.5;
        }
        match self.validation_method {
            Some(ValidationMethod::Human) => 1.0,
            Some(ValidationMethod::Automated) => 0.8,
            Some(ValidationMethod::Heuristic) => 0.6,
            None => 0.5,
        }
    }
}

/// Validation method for training examples
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
pub enum ValidationMethod {
    #[sqlx(rename = "human")]
    Human,
    #[sqlx(rename = "automated")]
    Automated,
    #[sqlx(rename = "heuristic")]
    Heuristic,
}

impl From<String> for ValidationMethod {
    fn from(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "human" => ValidationMethod::Human,
            "automated" => ValidationMethod::Automated,
            "heuristic" => ValidationMethod::Heuristic,
            _ => ValidationMethod::Heuristic, // Default fallback
        }
    }
}

/// Error types for validation
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    EmptyId,
    EmptyContent,
    InvalidVersion,
    MissingField(&'static str),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyId => write!(f, "Artifact ID cannot be empty"),
            ValidationError::EmptyContent => write!(f, "Artifact content cannot be empty"),
            ValidationError::InvalidVersion => write!(f, "Artifact version must be greater than 0"),
            ValidationError::MissingField(field) => {
                write!(f, "Required field '{}' is missing", field)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Parse error for string conversions
#[derive(Debug, Clone)]
pub enum ParseError {
    UnknownPlatform(String),
    UnknownArtifactType(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnknownPlatform(s) => write!(f, "Unknown platform: {}", s),
            ParseError::UnknownArtifactType(s) => write!(f, "Unknown artifact type: {}", s),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_builder() {
        let artifact = Artifact::builder()
            .id("test-123")
            .content("Test content")
            .platform(Platform::GitHub)
            .artifact_type(ArtifactType::PullRequest)
            .build()
            .unwrap();

        assert_eq!(artifact.id, "test-123");
        assert_eq!(artifact.version, 1);
        assert!(artifact.validate().is_ok());
    }

    #[test]
    fn test_platform_parsing() {
        assert_eq!("github".parse::<Platform>().unwrap(), Platform::GitHub);
        assert_eq!(
            "Microsoft Teams".parse::<Platform>().unwrap(),
            Platform::Teams
        );
        assert!("unknown".parse::<Platform>().is_err());
    }

    #[test]
    fn test_prediction_quality() {
        let mut prediction = OutcomePrediction::simple("out-1", "Outcome 1", 0.85);
        prediction.contributing_factors.push(ContributingFactor {
            factor_type: "historical".to_string(),
            weight: 0.5,
            description: "Historical pattern match".to_string(),
        });

        assert!(prediction.is_confident(0.8));
        assert!(prediction.quality_score() > 0.85);
    }

    #[test]
    fn test_model_metrics_update() {
        let mut metrics = ModelMetrics::default();
        metrics.update(true, 0.9, 50.0);
        metrics.update(false, 0.7, 45.0);

        assert_eq!(metrics.total_predictions, 2);
        assert_eq!(metrics.correct_predictions, 1);
        assert_eq!(metrics.accuracy, 0.5);
    }

    #[test]
    fn test_training_example_completeness() {
        let mut example = TrainingExample::new("Test input");
        assert!(!example.is_complete());

        example.actual_outcome_id = Some(Uuid::new_v4());
        example.user_feedback = Some("accepted".to_string());
        assert!(example.is_complete());
    }
}
