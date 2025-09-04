//interstice-ml/src/inference/engine.rs
//! Advanced Outcome Prediction System with ML Integration
//! 
//! This module provides a comprehensive prediction system for outcomes,
//! featuring confidence scoring, reasoning, impact analysis, and rich context handling.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration};

use chrono::{DateTime, Datelike, Timelike, Utc};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use thiserror::Error;

// Custom error types for the prediction system
#[derive(Error, Debug)]
pub enum PredictionError {
    #[error("Invalid confidence value {0}: must be between 0.0 and 1.0")]
    InvalidConfidence(f32),
    
    #[error("Invalid feature vector: {0}")]
    InvalidFeatureVector(String),
    
    #[error("Context validation failed: {0}")]
    InvalidContext(String),
    
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Time computation error: {0}")]
    TimeError(String),
}

pub type Result<T> = std::result::Result<T, PredictionError>;

/// Impact level enumeration with ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImpactLevel {
    Minimal,
    Low,
    Medium,
    High,
    Critical,
}

impl ImpactLevel {
    /// Get numeric value for calculations
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Minimal => 0.2,
            Self::Low => 0.4,
            Self::Medium => 0.6,
            Self::High => 0.8,
            Self::Critical => 1.0,
        }
    }
    
    /// Create from numeric value
    pub fn from_score(score: f32) -> Self {
        match score {
            s if s < 0.3 => Self::Minimal,
            s if s < 0.5 => Self::Low,
            s if s < 0.7 => Self::Medium,
            s if s < 0.9 => Self::High,
            _ => Self::Critical,
        }
    }
    
    /// Get color for visualization
    pub fn color(&self) -> &'static str {
        match self {
            Self::Minimal => "#94a3b8",
            Self::Low => "#3b82f6",
            Self::Medium => "#eab308",
            Self::High => "#f97316",
            Self::Critical => "#ef4444",
        }
    }
}

impl fmt::Display for ImpactLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Minimal => "Minimal",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Critical => "Critical",
        };
        write!(f, "{}", s)
    }
}

/// Contributing factor to a prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributingFactor {
    pub factor_id: String,
    pub name: String,
    pub weight: f32,
    pub description: Option<String>,
    pub category: FactorCategory,
    pub evidence: Vec<Evidence>,
}

impl ContributingFactor {
    pub fn new(name: impl Into<String>, weight: f32, category: FactorCategory) -> Self {
        let name_str = name.into();
        Self {
            factor_id: format!("factor_{}", uuid::Uuid::new_v4()),
            name: name_str,
            weight: weight.clamp(0.0, 1.0),
            description: None,
            category,
            evidence: Vec::new(),
        }
    }
    
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
    
    pub fn with_evidence(mut self, evidence: Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }
    
    /// Calculate the effective contribution
    pub fn contribution(&self) -> f32 {
        self.weight * self.evidence_strength()
    }
    
    fn evidence_strength(&self) -> f32 {
        if self.evidence.is_empty() {
            0.5 // Default strength
        } else {
            let sum: f32 = self.evidence.iter().map(|e| e.strength).sum();
            (sum / self.evidence.len() as f32).min(1.0)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactorCategory {
    Historical,
    Contextual,
    Environmental,
    Behavioral,
    Technical,
    External,
    Custom(String),
}

/// Evidence supporting a factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub source: String,
    pub strength: f32,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, JsonValue>,
}

/// Alternative outcome with probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeOutcome {
    pub outcome_id: String,
    pub outcome_name: String,
    pub probability: f32,
    pub relative_likelihood: f32,
    pub key_differences: Vec<String>,
}

impl AlternativeOutcome {
    pub fn new(
        outcome_id: impl Into<String>,
        outcome_name: impl Into<String>,
        probability: f32,
    ) -> Self {
        Self {
            outcome_id: outcome_id.into(),
            outcome_name: outcome_name.into(),
            probability: probability.clamp(0.0, 1.0),
            relative_likelihood: 1.0,
            key_differences: Vec::new(),
        }
    }
    
    pub fn with_differences(mut self, differences: Vec<String>) -> Self {
        self.key_differences = differences;
        self
    }
}

/// Reasoning for a prediction
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
    
    pub fn with_assumption(mut self, assumption: impl Into<String>) -> Self {
        self.key_assumptions.push(assumption.into());
        self
    }
    
    /// Calculate overall reasoning quality
    pub fn quality_score(&self) -> f32 {
        let base = self.data_quality_score;
        let confidence_boost = (self.confidence_factors.len() as f32 * 0.05).min(0.2);
        let uncertainty_penalty = (self.uncertainty_factors.len() as f32 * 0.03).min(0.15);
        
        (base + confidence_boost - uncertainty_penalty).clamp(0.0, 1.0)
    }
}

impl fmt::Display for PredictionReasoning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary)?;
        
        if !self.confidence_factors.is_empty() {
            write!(f, "\nStrengths: {}", self.confidence_factors.join(", "))?;
        }
        
        if !self.uncertainty_factors.is_empty() {
            write!(f, "\nUncertainties: {}", self.uncertainty_factors.join(", "))?;
        }
        
        Ok(())
    }
}

/// Main outcome prediction structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomePrediction {
    pub outcome_id: String,
    pub outcome_name: String,
    pub confidence: f32,
    pub reasoning: Option<PredictionReasoning>,
    pub contributing_factors: Vec<ContributingFactor>,
    pub alternative_outcomes: Vec<AlternativeOutcome>,
    pub predicted_impact: ImpactLevel,
    pub time_to_completion: Option<Duration>,
    pub metadata: PredictionMetadata,
    
    #[serde(skip)]
    pub cached_quality_score: Option<f32>,
}

impl OutcomePrediction {
    /// Create a simple prediction
    pub fn simple(
        outcome_id: impl Into<String>,
        outcome_name: impl Into<String>,
        confidence: f32,
    ) -> Result<Self> {
        let confidence = Self::validate_confidence(confidence)?;
        
        Ok(Self {
            outcome_id: outcome_id.into(),
            outcome_name: outcome_name.into(),
            confidence,
            reasoning: None,
            contributing_factors: Vec::new(),
            alternative_outcomes: Vec::new(),
            predicted_impact: ImpactLevel::Medium,
            time_to_completion: None,
            metadata: PredictionMetadata::default(),
            cached_quality_score: None,
        })
    }
    
    /// Create a detailed prediction with builder pattern
    pub fn builder(
        outcome_id: impl Into<String>,
        outcome_name: impl Into<String>,
    ) -> PredictionBuilder {
        PredictionBuilder::new(outcome_id, outcome_name)
    }
    
    /// Validate confidence value
    fn validate_confidence(confidence: f32) -> Result<f32> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(PredictionError::InvalidConfidence(confidence));
        }
        Ok(confidence)
    }
    
    /// Check if prediction meets confidence threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
    
    /// Get prediction quality score with caching
    pub fn quality_score(&mut self) -> f32 {
        if let Some(score) = self.cached_quality_score {
            return score;
        }
        
        let score = self.calculate_quality_score();
        self.cached_quality_score = Some(score);
        score
    }
    
    fn calculate_quality_score(&self) -> f32 {
        let mut score = self.confidence;
        
        // Factor contribution (up to 20% bonus)
        let factor_score = self.contributing_factors
            .iter()
            .map(|f| f.contribution())
            .sum::<f32>()
            .min(0.2);
        score += factor_score;
        
        // Reasoning quality (up to 15% bonus)
        if let Some(ref reasoning) = self.reasoning {
            score += reasoning.quality_score() * 0.15;
        }
        
        // Alternative outcomes consideration (up to 10% bonus)
        let alt_score = (self.alternative_outcomes.len() as f32 * 0.02).min(0.1);
        score += alt_score;
        
        // Metadata completeness (up to 5% bonus)
        score += self.metadata.completeness_score() * 0.05;
        
        score.min(1.0)
    }
    
    /// Get the most likely alternative
    pub fn best_alternative(&self) -> Option<&AlternativeOutcome> {
        self.alternative_outcomes
            .iter()
            .max_by(|a, b| {
                OrderedFloat(a.probability)
                    .cmp(&OrderedFloat(b.probability))
            })
    }
    
    /// Calculate uncertainty level
    pub fn uncertainty(&self) -> f32 {
        1.0 - self.confidence
    }
    
    /// Get estimated completion time in human-readable format
    pub fn completion_estimate(&self) -> String {
        match self.time_to_completion {
            None => "Unknown".to_string(),
            Some(d) if d < Duration::from_secs(3600) => {
                format!("{} minutes", d.as_secs() / 60)
            }
            Some(d) if d < Duration::from_secs(86400) => {
                format!("{} hours", d.as_secs() / 3600)
            }
            Some(d) => {
                format!("{} days", d.as_secs() / 86400)
            }
        }
    }
    
    /// Merge with another prediction
    pub fn merge(&mut self, other: &OutcomePrediction) -> Result<()> {
        if self.outcome_id != other.outcome_id {
            return Err(PredictionError::PredictionFailed(
                "Cannot merge predictions for different outcomes".to_string()
            ));
        }
        
        // Weighted average of confidence
        let total_weight = self.confidence + other.confidence;
        self.confidence = (self.confidence * self.confidence + 
                          other.confidence * other.confidence) / total_weight;
        
        // Merge factors
        for factor in &other.contributing_factors {
            if !self.contributing_factors.iter().any(|f| f.factor_id == factor.factor_id) {
                self.contributing_factors.push(factor.clone());
            }
        }
        
        // Merge alternatives
        for alt in &other.alternative_outcomes {
            if !self.alternative_outcomes.iter().any(|a| a.outcome_id == alt.outcome_id) {
                self.alternative_outcomes.push(alt.clone());
            }
        }
        
        // Update impact to higher of the two
        if other.predicted_impact > self.predicted_impact {
            self.predicted_impact = other.predicted_impact;
        }
        
        // Clear cached score
        self.cached_quality_score = None;
        
        Ok(())
    }
}

impl PartialEq for OutcomePrediction {
    fn eq(&self, other: &Self) -> bool {
        self.outcome_id == other.outcome_id
    }
}

impl Eq for OutcomePrediction {}

impl PartialOrd for OutcomePrediction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OutcomePrediction {
    fn cmp(&self, other: &Self) -> Ordering {
        OrderedFloat(self.confidence)
            .cmp(&OrderedFloat(other.confidence))
            .reverse()
    }
}

impl Hash for OutcomePrediction {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.outcome_id.hash(state);
    }
}

impl fmt::Display for OutcomePrediction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}% confidence, {} impact)",
            self.outcome_name,
            (self.confidence * 100.0) as u32,
            self.predicted_impact
        )
    }
}

/// Metadata for predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    pub created_at: DateTime<Utc>,
    pub model_version: String,
    pub computation_time_ms: Option<u64>,
    pub data_sources: HashSet<String>,
    pub tags: HashSet<String>,
    pub confidence_interval: Option<(f32, f32)>,
    pub feature_importance: Option<HashMap<String, f32>>,
}

impl Default for PredictionMetadata {
    fn default() -> Self {
        Self {
            created_at: Utc::now(),
            model_version: env!("CARGO_PKG_VERSION").to_string(),
            computation_time_ms: None,
            data_sources: HashSet::new(),
            tags: HashSet::new(),
            confidence_interval: None,
            feature_importance: None,
        }
    }
}

impl PredictionMetadata {
    pub fn completeness_score(&self) -> f32 {
        let mut score = 0.0;
        let mut max_score = 0.0;
        
        // Check each optional field
        max_score += 1.0;
        if self.computation_time_ms.is_some() {
            score += 1.0;
        }
        
        max_score += 1.0;
        if !self.data_sources.is_empty() {
            score += 1.0;
        }
        
        max_score += 1.0;
        if !self.tags.is_empty() {
            score += 1.0;
        }
        
        max_score += 1.0;
        if self.confidence_interval.is_some() {
            score += 1.0;
        }
        
        max_score += 1.0;
        if self.feature_importance.is_some() {
            score += 1.0;
        }
        
        if max_score > 0.0 {
            score / max_score
        } else {
            1.0
        }
    }
}

/// Builder for creating predictions
pub struct PredictionBuilder {
    outcome_id: String,
    outcome_name: String,
    confidence: Option<f32>,
    reasoning: Option<PredictionReasoning>,
    contributing_factors: Vec<ContributingFactor>,
    alternative_outcomes: Vec<AlternativeOutcome>,
    predicted_impact: Option<ImpactLevel>,
    time_to_completion: Option<Duration>,
    metadata: PredictionMetadata,
}

impl PredictionBuilder {
    pub fn new(outcome_id: impl Into<String>, outcome_name: impl Into<String>) -> Self {
        Self {
            outcome_id: outcome_id.into(),
            outcome_name: outcome_name.into(),
            confidence: None,
            reasoning: None,
            contributing_factors: Vec::new(),
            alternative_outcomes: Vec::new(),
            predicted_impact: None,
            time_to_completion: None,
            metadata: PredictionMetadata::default(),
        }
    }
    
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
    
    pub fn reasoning(mut self, reasoning: PredictionReasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }
    
    pub fn add_factor(mut self, factor: ContributingFactor) -> Self {
        self.contributing_factors.push(factor);
        self
    }
    
    pub fn add_alternative(mut self, alternative: AlternativeOutcome) -> Self {
        self.alternative_outcomes.push(alternative);
        self
    }
    
    pub fn impact(mut self, impact: ImpactLevel) -> Self {
        self.predicted_impact = Some(impact);
        self
    }
    
    pub fn completion_time(mut self, duration: Duration) -> Self {
        self.time_to_completion = Some(duration);
        self
    }
    
    pub fn add_tag(mut self, tag: impl Into<String>) -> Self {
        self.metadata.tags.insert(tag.into());
        self
    }
    
    pub fn add_data_source(mut self, source: impl Into<String>) -> Self {
        self.metadata.data_sources.insert(source.into());
        self
    }
    
    pub fn computation_time(mut self, ms: u64) -> Self {
        self.metadata.computation_time_ms = Some(ms);
        self
    }
    
    pub fn confidence_interval(mut self, lower: f32, upper: f32) -> Self {
        self.metadata.confidence_interval = Some((lower, upper));
        self
    }
    
    pub fn build(self) -> Result<OutcomePrediction> {
        let confidence = self.confidence
            .ok_or_else(|| PredictionError::PredictionFailed(
                "Confidence is required".to_string()
            ))?;
        
        OutcomePrediction::simple(self.outcome_id, self.outcome_name, confidence)
            .map(|mut pred| {
                pred.reasoning = self.reasoning;
                pred.contributing_factors = self.contributing_factors;
                pred.alternative_outcomes = self.alternative_outcomes;
                pred.predicted_impact = self.predicted_impact.unwrap_or(ImpactLevel::Medium);
                pred.time_to_completion = self.time_to_completion;
                pred.metadata = self.metadata;
                pred
            })
    }
}

/// Platform-specific signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformSignals {
    pub platform_name: String,
    pub signals: HashMap<String, JsonValue>,
    pub timestamp: DateTime<Utc>,
}

impl PlatformSignals {
    pub fn new(platform_name: impl Into<String>) -> Self {
        Self {
            platform_name: platform_name.into(),
            signals: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
    
    pub fn add_signal(mut self, key: impl Into<String>, value: JsonValue) -> Self {
        self.signals.insert(key.into(), value);
        self
    }
    
    pub fn get_signal<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.signals.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

/// Prediction context with rich environmental data
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
    pub platform_signals: Option<PlatformSignals>,
    
    // Additional context fields
    pub user_timezone: Option<String>,
    pub is_business_hours: bool,
    pub concurrent_tasks: u32,
    pub historical_accuracy: Option<f32>,
    pub context_quality_score: f32,
}

impl PredictionContext {
    /// Create context from current environment
    pub fn from_environment() -> Self {
        let now = Utc::now();
        let hour = now.hour();
        let is_business_hours = (9..=17).contains(&hour);
        
        Self {
            hour_of_day: hour,
            day_of_week: now.weekday().num_days_from_monday(),
            days_until_deadline: None,
            user_activity_level: 0.5,
            user_expertise_score: 0.5,
            team_size: 1,
            sprint_progress: None,
            related_artifacts_count: 0,
            workspace_activity_level: 0.5,
            platform_signals: None,
            user_timezone: None,
            is_business_hours,
            concurrent_tasks: 0,
            historical_accuracy: None,
            context_quality_score: 0.5,
        }
    }
    
    /// Create with builder pattern
    pub fn builder() -> ContextBuilder {
        ContextBuilder::new()
    }
    
    /// Enrich context with workspace data
    pub fn with_workspace_data(mut self, data: JsonValue) -> Self {
        if let Some(team_size) = data.get("team_size").and_then(|v| v.as_u64()) {
            self.team_size = team_size as u32;
        }
        
        if let Some(activity) = data.get("activity_level").and_then(|v| v.as_f64()) {
            self.workspace_activity_level = activity as f32;
        }
        
        if let Some(sprint) = data.get("sprint_progress").and_then(|v| v.as_f64()) {
            self.sprint_progress = Some(sprint as f32);
        }
        
        if let Some(tasks) = data.get("concurrent_tasks").and_then(|v| v.as_u64()) {
            self.concurrent_tasks = tasks as u32;
        }
        
        // Store full data as platform signals
        let mut signals = PlatformSignals::new("workspace");
        signals.signals.insert("raw_data".to_string(), data);
        self.platform_signals = Some(signals);
        
        self
    }
    
    /// Convert to feature vector for ML
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            // Temporal features
            self.hour_of_day as f32 / 24.0,
            self.day_of_week as f32 / 7.0,
            self.days_until_deadline.unwrap_or(30.0) / 30.0,
            if self.is_business_hours { 1.0 } else { 0.0 },
            
            // User features
            self.user_activity_level,
            self.user_expertise_score,
            self.historical_accuracy.unwrap_or(0.5),
            
            // Team features
            (self.team_size as f32).ln() / 10.0,
            self.sprint_progress.unwrap_or(0.5),
            
            // Workload features
            (self.related_artifacts_count as f32).ln() / 10.0,
            (self.concurrent_tasks as f32).ln() / 5.0,
            self.workspace_activity_level,
            
            // Quality indicator
            self.context_quality_score,
        ]
    }
    
    /// Validate context integrity
    pub fn validate(&self) -> Result<()> {
        if self.hour_of_day >= 24 {
            return Err(PredictionError::InvalidContext(
                format!("Invalid hour: {}", self.hour_of_day)
            ));
        }
        
        if self.day_of_week >= 7 {
            return Err(PredictionError::InvalidContext(
                format!("Invalid day of week: {}", self.day_of_week)
            ));
        }
        
        for value in [
            self.user_activity_level,
            self.user_expertise_score,
            self.workspace_activity_level,
            self.context_quality_score,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(PredictionError::InvalidContext(
                    format!("Invalid normalized value: {}", value)
                ));
            }
        }
        
        if let Some(progress) = self.sprint_progress {
            if !progress.is_finite() || !(0.0..=1.0).contains(&progress) {
                return Err(PredictionError::InvalidContext(
                    format!("Invalid sprint progress: {}", progress)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Calculate context completeness
    pub fn completeness(&self) -> f32 {
        let mut score = 0.0;
        let mut max_score = 0.0;
        
        // Check optional fields
        let optional_fields = [
            self.days_until_deadline.is_some(),
            self.sprint_progress.is_some(),
            self.platform_signals.is_some(),
            self.user_timezone.is_some(),
            self.historical_accuracy.is_some(),
        ];
        
        for field_present in optional_fields {
            max_score += 1.0;
            if field_present {
                score += 1.0;
            }
        }
        
        // Check data quality
        max_score += 1.0;
        score += self.context_quality_score;
        
        score / max_score
    }
}

/// Builder for PredictionContext
pub struct ContextBuilder {
    context: PredictionContext,
}

impl ContextBuilder {
    pub fn new() -> Self {
        Self {
            context: PredictionContext::from_environment(),
        }
    }
    
    pub fn deadline_days(mut self, days: f32) -> Self {
        self.context.days_until_deadline = Some(days);
        self
    }
    
    pub fn user_activity(mut self, level: f32) -> Self {
        self.context.user_activity_level = level.clamp(0.0, 1.0);
        self
    }
    
    pub fn user_expertise(mut self, score: f32) -> Self {
        self.context.user_expertise_score = score.clamp(0.0, 1.0);
        self
    }
    
    pub fn team_size(mut self, size: u32) -> Self {
        self.context.team_size = size;
        self
    }
    
    pub fn sprint_progress(mut self, progress: f32) -> Self {
        self.context.sprint_progress = Some(progress.clamp(0.0, 1.0));
        self
    }
    
    pub fn timezone(mut self, tz: impl Into<String>) -> Self {
        self.context.user_timezone = Some(tz.into());
        self
    }
    
    pub fn concurrent_tasks(mut self, count: u32) -> Self {
        self.context.concurrent_tasks = count;
        self
    }
    
    pub fn historical_accuracy(mut self, accuracy: f32) -> Self {
        self.context.historical_accuracy = Some(accuracy.clamp(0.0, 1.0));
        self
    }
    
    pub fn platform_signals(mut self, signals: PlatformSignals) -> Self {
        self.context.platform_signals = Some(signals);
        self
    }
    
    pub fn build(self) -> Result<PredictionContext> {
        self.context.validate()?;
        Ok(self.context)
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Prediction ensemble for combining multiple predictions
#[derive(Debug, Clone)]
pub struct PredictionEnsemble {
    predictions: Vec<OutcomePrediction>,
    weights: Vec<f32>,
    aggregation_method: AggregationMethod,
    confidence_threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    WeightedAverage,
    MaxConfidence,
    Voting,
    BayesianCombination,
    StackedGeneralization,
}

impl PredictionEnsemble {
    pub fn new(aggregation_method: AggregationMethod) -> Self {
        Self {
            predictions: Vec::new(),
            weights: Vec::new(),
            aggregation_method,
            confidence_threshold: 0.5,
        }
    }
    
    pub fn add_prediction(&mut self, prediction: OutcomePrediction, weight: f32) {
        self.predictions.push(prediction);
        self.weights.push(weight);
    }
    
    pub fn set_threshold(&mut self, threshold: f32) {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
    }
    
    /// Aggregate predictions into a single outcome
    pub fn aggregate(&self) -> Result<OutcomePrediction> {
        if self.predictions.is_empty() {
            return Err(PredictionError::PredictionFailed(
                "Cannot aggregate empty ensemble".to_string()
            ));
        }
        
        match self.aggregation_method {
            AggregationMethod::WeightedAverage => self.weighted_average(),
            AggregationMethod::MaxConfidence => self.max_confidence(),
            AggregationMethod::Voting => self.voting(),
            AggregationMethod::BayesianCombination => self.bayesian_combination(),
            AggregationMethod::StackedGeneralization => self.stacked_generalization(),
        }
    }
    
    fn weighted_average(&self) -> Result<OutcomePrediction> {
        // Group predictions by outcome_id
        let mut grouped: HashMap<String, Vec<(f32, f32)>> = HashMap::new();
        
        for (pred, &weight) in self.predictions.iter().zip(&self.weights) {
            grouped.entry(pred.outcome_id.clone())
                .or_insert_with(Vec::new)
                .push((pred.confidence, weight));
        }
        
        // Find outcome with highest weighted confidence
        let best_outcome = grouped
            .iter()
            .map(|(id, confs)| {
                let weighted_sum: f32 = confs.iter()
                    .map(|(conf, weight)| conf * weight)
                    .sum();
                let weight_sum: f32 = confs.iter()
                    .map(|(_, weight)| weight)
                    .sum();
                let avg_confidence = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    0.0
                };
                (id.clone(), avg_confidence)
            })
            .max_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)))
            .ok_or_else(|| PredictionError::PredictionFailed(
                "Failed to find best outcome".to_string()
            ))?;
        
        // Get the first prediction with this outcome_id as template
        let template = self.predictions
            .iter()
            .find(|p| p.outcome_id == best_outcome.0)
            .ok_or_else(|| PredictionError::PredictionFailed(
                "Failed to find template prediction".to_string()
            ))?;
        
        let mut result = template.clone();
        result.confidence = best_outcome.1;
        
        // Merge all factors from predictions with same outcome
        for pred in &self.predictions {
            if pred.outcome_id == best_outcome.0 {
                for factor in &pred.contributing_factors {
                    if !result.contributing_factors.iter().any(|f| f.factor_id == factor.factor_id) {
                        result.contributing_factors.push(factor.clone());
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    fn max_confidence(&self) -> Result<OutcomePrediction> {
        self.predictions
            .iter()
            .max_by(|a, b| OrderedFloat(a.confidence).cmp(&OrderedFloat(b.confidence)))
            .cloned()
            .ok_or_else(|| PredictionError::PredictionFailed(
                "No predictions available".to_string()
            ))
    }
    
    fn voting(&self) -> Result<OutcomePrediction> {
        let mut votes: HashMap<String, (usize, f32, OutcomePrediction)> = HashMap::new();
        
        for pred in &self.predictions {
            if pred.confidence >= self.confidence_threshold {
                let entry = votes.entry(pred.outcome_id.clone())
                    .or_insert((0, 0.0, pred.clone()));
                entry.0 += 1; // Increment vote count
                entry.1 += pred.confidence; // Sum confidence
            }
        }
        
        let best = votes
            .into_iter()
            .max_by(|a, b| a.1.0.cmp(&b.1.0))
            .ok_or_else(|| PredictionError::PredictionFailed(
                "No predictions met threshold".to_string()
            ))?;
        
        let mut result = best.1.2;
        result.confidence = best.1.1 / best.1.0 as f32; // Average confidence
        
        Ok(result)
    }
    
    fn bayesian_combination(&self) -> Result<OutcomePrediction> {
        // Simplified Bayesian combination
        // In production, implement proper Bayesian inference
        
        let mut posterior: HashMap<String, f32> = HashMap::new();
        
        for (pred, &weight) in self.predictions.iter().zip(&self.weights) {
            let prior = posterior.get(&pred.outcome_id).copied().unwrap_or(0.5);
            
            // Bayesian update
            let likelihood = pred.confidence;
            let evidence = weight;
            let updated = (likelihood * prior * evidence) / 
                         ((likelihood * prior * evidence) + ((1.0 - likelihood) * (1.0 - prior) * evidence));
            
            posterior.insert(pred.outcome_id.clone(), updated);
        }
        
        let best_outcome = posterior
            .iter()
            .max_by(|a, b| OrderedFloat(*a.1).cmp(&OrderedFloat(*b.1)))
            .ok_or_else(|| PredictionError::PredictionFailed(
                "Bayesian combination failed".to_string()
            ))?;
        
        let template = self.predictions
            .iter()
            .find(|p| &p.outcome_id == best_outcome.0)
            .ok_or_else(|| PredictionError::PredictionFailed(
                "Failed to find template".to_string()
            ))?;
        
        let mut result = template.clone();
        result.confidence = *best_outcome.1;
        
        Ok(result)
    }
    
    fn stacked_generalization(&self) -> Result<OutcomePrediction> {
        // Simplified stacking - in production, use trained meta-model
        
        // Extract meta-features
        let mean_confidence = self.predictions.iter()
            .map(|p| p.confidence)
            .sum::<f32>() / self.predictions.len() as f32;
        
        let confidence_std = {
            let variance = self.predictions.iter()
                .map(|p| (p.confidence - mean_confidence).powi(2))
                .sum::<f32>() / self.predictions.len() as f32;
            variance.sqrt()
        };
        
        // Simple meta-model: adjust based on agreement
        let agreement_factor = 1.0 - confidence_std;
        
        let mut result = self.weighted_average()?;
        result.confidence *= agreement_factor;
        
        Ok(result)
    }
}

/// Prediction history for tracking and analysis
#[derive(Debug, Clone)]
pub struct PredictionHistory {
    predictions: Arc<RwLock<Vec<HistoricalPrediction>>>,
    max_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalPrediction {
    pub prediction: OutcomePrediction,
    pub actual_outcome: Option<String>,
    pub accuracy: Option<f32>,
    pub feedback: Option<PredictionFeedback>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionFeedback {
    pub rating: i32, // 1-5 scale
    pub was_helpful: bool,
    pub comments: Option<String>,
    pub correction: Option<String>,
}

impl PredictionHistory {
    pub fn new(max_size: usize) -> Self {
        Self {
            predictions: Arc::new(RwLock::new(Vec::new())),
            max_size,
        }
    }
    
    pub fn add(&self, prediction: OutcomePrediction) {
        let mut history = self.predictions.write();
        
        history.push(HistoricalPrediction {
            prediction,
            actual_outcome: None,
            accuracy: None,
            feedback: None,
            timestamp: Utc::now(),
        });
        
        // Maintain size limit
        if history.len() > self.max_size {
            history.remove(0);
        }
    }
    
    pub fn update_outcome(&self, prediction_id: &str, actual_outcome: String, accuracy: f32) {
        let mut history = self.predictions.write();
        
        if let Some(entry) = history.iter_mut()
            .find(|h| h.prediction.outcome_id == prediction_id) 
        {
            entry.actual_outcome = Some(actual_outcome);
            entry.accuracy = Some(accuracy);
        }
    }
    
    pub fn add_feedback(&self, prediction_id: &str, feedback: PredictionFeedback) {
        let mut history = self.predictions.write();
        
        if let Some(entry) = history.iter_mut()
            .find(|h| h.prediction.outcome_id == prediction_id)
        {
            entry.feedback = Some(feedback);
        }
    }
    
    pub fn calculate_accuracy(&self) -> f32 {
        let history = self.predictions.read();
        
        let accuracies: Vec<f32> = history.iter()
            .filter_map(|h| h.accuracy)
            .collect();
        
        if accuracies.is_empty() {
            return 0.5; // Default
        }
        
        accuracies.iter().sum::<f32>() / accuracies.len() as f32
    }
    
    pub fn get_recent(&self, count: usize) -> Vec<HistoricalPrediction> {
        let history = self.predictions.read();
        let start = if history.len() > count {
            history.len() - count
        } else {
            0
        };
        
        history[start..].to_vec()
    }
    
    pub fn get_by_outcome(&self, outcome_id: &str) -> Vec<HistoricalPrediction> {
        let history = self.predictions.read();
        history.iter()
            .filter(|h| h.prediction.outcome_id == outcome_id)
            .cloned()
            .collect()
    }
}

/// Prediction validator for quality assurance
pub struct PredictionValidator {
    rules: Vec<Box<dyn ValidationRule>>,
}

impl PredictionValidator {
    pub fn new() -> Self {
        Self {
            rules: vec![
                Box::new(ConfidenceRule { min: 0.0, max: 1.0 }),
                Box::new(FactorWeightRule { max_weight: 1.0 }),
                Box::new(AlternativeConsistencyRule {}),
                Box::new(TimeConsistencyRule {}),
            ],
        }
    }
    
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }
    
    pub fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        for rule in &self.rules {
            match rule.validate(prediction) {
                ValidationResult::Valid => {}
                ValidationResult::Invalid(err) => errors.push(err),
                ValidationResult::Warning(warn) => warnings.push(warn),
            }
        }
        
        if !errors.is_empty() {
            ValidationResult::Invalid(errors.join("; "))
        } else if !warnings.is_empty() {
            ValidationResult::Warning(warnings.join("; "))
        } else {
            ValidationResult::Valid
        }
    }
}

impl Default for PredictionValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum ValidationResult {
    Valid,
    Invalid(String),
    Warning(String),
}

pub trait ValidationRule: Send + Sync {
    fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult;
}

struct ConfidenceRule {
    min: f32,
    max: f32,
}

impl ValidationRule for ConfidenceRule {
    fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult {
        if prediction.confidence < self.min || prediction.confidence > self.max {
            ValidationResult::Invalid(format!(
                "Confidence {} out of range [{}, {}]",
                prediction.confidence, self.min, self.max
            ))
        } else {
            ValidationResult::Valid
        }
    }
}

struct FactorWeightRule {
    max_weight: f32,
}

impl ValidationRule for FactorWeightRule {
    fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult {
        for factor in &prediction.contributing_factors {
            if factor.weight > self.max_weight {
                return ValidationResult::Invalid(format!(
                    "Factor '{}' weight {} exceeds maximum {}",
                    factor.name, factor.weight, self.max_weight
                ));
            }
        }
        ValidationResult::Valid
    }
}

struct AlternativeConsistencyRule;

impl ValidationRule for AlternativeConsistencyRule {
    fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult {
        let total_prob: f32 = prediction.alternative_outcomes
            .iter()
            .map(|a| a.probability)
            .sum();
        
        if total_prob > 1.01 { // Allow small floating point error
            ValidationResult::Warning(format!(
                "Alternative outcome probabilities sum to {}, exceeds 1.0",
                total_prob
            ))
        } else {
            ValidationResult::Valid
        }
    }
}

struct TimeConsistencyRule;

impl ValidationRule for TimeConsistencyRule {
    fn validate(&self, prediction: &OutcomePrediction) -> ValidationResult {
        if let Some(duration) = prediction.time_to_completion {
            if duration > Duration::from_secs(365 * 24 * 3600) {
                ValidationResult::Warning(
                    "Completion time exceeds one year".to_string()
                )
            } else {
                ValidationResult::Valid
            }
        } else {
            ValidationResult::Valid
        }
    }
}

/// Performance monitoring for predictions
#[derive(Debug, Clone)]
pub struct PredictionMonitor {
    metrics: Arc<RwLock<PredictionMetrics>>,
    alerts: Arc<RwLock<Vec<PerformanceAlert>>>,
    thresholds: MonitorThresholds,
}

#[derive(Debug, Clone, Default)]
pub struct PredictionMetrics {
    pub total_predictions: u64,
    pub avg_confidence: f32,
    pub avg_quality_score: f32,
    pub avg_computation_time_ms: f64,
    pub error_rate: f32,
    pub cache_hit_rate: f32,
}

#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub metric_value: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct MonitorThresholds {
    pub min_confidence: f32,
    pub max_computation_time_ms: u64,
    pub max_error_rate: f32,
    pub min_cache_hit_rate: f32,
}

impl Default for MonitorThresholds {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            max_computation_time_ms: 1000,
            max_error_rate: 0.05,
            min_cache_hit_rate: 0.7,
        }
    }
}

impl PredictionMonitor {
    pub fn new(thresholds: MonitorThresholds) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(PredictionMetrics::default())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            thresholds,
        }
    }
    
    pub fn record_prediction(&self, prediction: &OutcomePrediction, computation_time_ms: u64) {
        let mut metrics = self.metrics.write();
        
        metrics.total_predictions += 1;
        
        // Update running averages
        let n = metrics.total_predictions as f32;
        metrics.avg_confidence = (metrics.avg_confidence * (n - 1.0) + prediction.confidence) / n;
        
        let quality_score = prediction.clone().quality_score();
        metrics.avg_quality_score = (metrics.avg_quality_score * (n - 1.0) + quality_score) / n;
        
        metrics.avg_computation_time_ms = (metrics.avg_computation_time_ms * (n as f64 - 1.0) 
            + computation_time_ms as f64) / n as f64;
        
        // Check thresholds
        drop(metrics); // Release lock before checking
        self.check_thresholds();
    }
    
    pub fn record_error(&self) {
        let mut metrics = self.metrics.write();
        let total = metrics.total_predictions as f32;
        if total > 0.0 {
            metrics.error_rate = (metrics.error_rate * total + 1.0) / (total + 1.0);
        }
    }
    
    pub fn record_cache_hit(&self, hit: bool) {
        let mut metrics = self.metrics.write();
        let total = metrics.total_predictions as f32;
        if total > 0.0 {
            let hit_value = if hit { 1.0 } else { 0.0 };
            metrics.cache_hit_rate = (metrics.cache_hit_rate * (total - 1.0) + hit_value) / total;
        }
    }
    
    fn check_thresholds(&self) {
        let metrics = self.metrics.read();
        let mut alerts = self.alerts.write();
        
        if metrics.avg_confidence < self.thresholds.min_confidence {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "Average confidence {} below threshold {}",
                    metrics.avg_confidence,
                    self.thresholds.min_confidence
                ),
                timestamp: Utc::now(),
                metric_value: metrics.avg_confidence,
            });
        }
        
        if metrics.avg_computation_time_ms > self.thresholds.max_computation_time_ms as f64 {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "Average computation time {}ms exceeds threshold {}ms",
                    metrics.avg_computation_time_ms,
                    self.thresholds.max_computation_time_ms
                ),
                timestamp: Utc::now(),
                metric_value: metrics.avg_computation_time_ms as f32,
            });
        }
        
        if metrics.error_rate > self.thresholds.max_error_rate {
            alerts.push(PerformanceAlert {
                severity: AlertSeverity::Error,
                message: format!(
                    "Error rate {} exceeds threshold {}",
                    metrics.error_rate,
                    self.thresholds.max_error_rate
                ),
                timestamp: Utc::now(),
                metric_value: metrics.error_rate,
            });
        }
        
        // Keep only recent alerts (last 100)
        if alerts.len() > 100 {
            *alerts = alerts[alerts.len() - 100..].to_vec();
        }
    }
    
    pub fn get_metrics(&self) -> PredictionMetrics {
        self.metrics.read().clone()
    }
    
    pub fn get_alerts(&self) -> Vec<PerformanceAlert> {
        self.alerts.read().clone()
    }
}