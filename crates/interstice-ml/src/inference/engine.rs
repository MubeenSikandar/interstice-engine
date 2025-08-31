//interstice-ml/src/inference/engine.rs
use anyhow::Result;
use ndarray::{Array2, CowArray};
use ort::{Environment, Session, SessionBuilder, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rand_distr::{Distribution, Gamma};
use rand::rng;
use crate::types::{Artifact, Platform, ArtifactType, PredictionContext, OutcomePrediction};



pub struct EngineConfig {
    pub embedding_model_path: String,
    pub predictor_model_path: String,
    pub n_outcomes: usize,
}

pub struct OutcomeEngine {
    // Core models
    embedding_session: Arc<Session>,
    outcome_predictor: Arc<Session>,
    
    // Bandit for exploration/exploitation
    bandit: Arc<RwLock<ThompsonSamplingBandit>>,
    
    // Cache for embeddings
    embedding_cache: Arc<RwLock<LRUCache<String, Vec<f32>>>>,
    
    // Performance tracking
    metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl OutcomeEngine {
    pub async fn new(config: EngineConfig) -> Result<Self> {
        // Initialize ONNX environment
        let environment = Arc::new(
            Environment::builder()
                .with_name("outcome-engine")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()?
        );

        // Load embedding model (all-MiniLM-L6-v2)
        let embedding_session = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_model_from_file(&config.embedding_model_path)?
        );

        // Load outcome prediction model
        let outcome_predictor = Arc::new(
            SessionBuilder::new(&environment)?
                .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
                .with_intra_threads(4)?
                .with_model_from_file(&config.predictor_model_path)?
        );

        Ok(Self {
            embedding_session,
            outcome_predictor,
            bandit: Arc::new(RwLock::new(ThompsonSamplingBandit::new(config.n_outcomes))),
            embedding_cache: Arc::new(RwLock::new(LRUCache::new(10000))),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
        })
    }

    /// Main prediction pipeline
    pub async fn predict(
        &self,
        artifacts: Vec<Artifact>,
        context: PredictionContext,
    ) -> Result<Vec<OutcomePrediction>> {
        let start = std::time::Instant::now();

        // Step 1: Generate embeddings (with caching)
        let embeddings = self.generate_embeddings(&artifacts).await?;
        
        // Step 2: Create feature matrix
        let features = self.create_feature_matrix(embeddings, &artifacts, &context)?;
        
        // Step 3: Run neural network prediction
        let nn_predictions = self.run_nn_prediction(features).await?;
        
        // Step 4: Apply bandit for exploration
        let bandit_scores = self.apply_bandit(nn_predictions.clone()).await?;
        
        // Step 5: Combine and rank
        let final_predictions = self.combine_predictions(nn_predictions, bandit_scores)?;
        
        // Update metrics
        self.update_metrics(start.elapsed().as_millis() as f64).await;
        
        Ok(final_predictions)
    }

    /// Generate embeddings with caching
    async fn generate_embeddings(&self, artifacts: &[Artifact]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        let mut cache = self.embedding_cache.write().await;
        
        for artifact in artifacts {
            let cache_key = format!("{}:{}", artifact.id, artifact.version);
            
            if let Some(cached) = cache.get(&cache_key) {
                embeddings.push(cached.clone());
            } else {
                let embedding = self.encode_text(&artifact.content).await?;
                cache.put(cache_key, embedding.clone());
                embeddings.push(embedding);
            }
        }
        
        Ok(embeddings)
    }

    /// Encode text using sentence transformer
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize (simplified - in production use proper tokenizer)
        let tokens = self.simple_tokenize(text, 256)?;
        
        // Create input tensors
        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens)?;
        let attention_mask: Array2<i64> = Array2::ones((1, input_ids.len_of(ndarray::Axis(1))));
        
        // Run model
        let input_ids_dyn = CowArray::from(input_ids.view()).into_dyn();
        let attention_mask_dyn = CowArray::from(attention_mask.view()).into_dyn();

        let input_ids_value = Value::from_array(self.embedding_session.allocator(), &input_ids_dyn)?;
        let attention_mask_value = Value::from_array(self.embedding_session.allocator(), &attention_mask_dyn)?;
        
        let outputs = self.embedding_session.run(vec![input_ids_value, attention_mask_value])?;
        let output = &outputs[0];
        
        // Extract embeddings (mean pooling)
        let embeddings = output.try_extract::<f32>()?;
        let embeddings_view = embeddings.view();
        
        // Mean pooling across sequence length
        let mut pooled = vec![0.0f32; 384]; // all-MiniLM-L6-v2 has 384 dims
        let seq_len = embeddings_view.shape()[1];
        
        for i in 0..384 {
            for j in 0..seq_len {
                pooled[i] += embeddings_view[[0, j, i]];
            }
            pooled[i] /= seq_len as f32;
        }
        
        Ok(pooled)
    }

    /// Create feature matrix for prediction
    fn create_feature_matrix(
        &self,
        embeddings: Vec<Vec<f32>>,
        artifacts: &[Artifact],
        context: &PredictionContext,
    ) -> Result<Array2<f32>> {
        let n_samples = embeddings.len();
        let embedding_dim = 384;
        let context_features = 64;
        let total_features = embedding_dim + context_features;
        
        let mut features = Array2::zeros((n_samples, total_features));
        
        for (i, (embedding, artifact)) in embeddings.iter().zip(artifacts).enumerate() {
            // Add embeddings
            for (j, &val) in embedding.iter().enumerate() {
                features[[i, j]] = val;
            }
            
            // Add context features
            let mut offset = embedding_dim;
            
            // Platform encoding (one-hot)
            features[[i, offset + artifact.platform as usize]] = 1.0;
            offset += 12; // Number of platforms
            
            // Artifact type encoding
            features[[i, offset + artifact.artifact_type as usize]] = 1.0;
            offset += 5; // Number of artifact types
            
            // Temporal features
            features[[i, offset]] = context.hour_of_day as f32 / 24.0;
            features[[i, offset + 1]] = context.day_of_week as f32 / 7.0;
            features[[i, offset + 2]] = context.days_until_deadline as f32 / 30.0;
            offset += 3;
            
            // User features
            features[[i, offset]] = context.user_activity_level;
            features[[i, offset + 1]] = context.user_expertise_score;
            features[[i, offset + 2]] = context.team_size as f32 / 100.0;
            
            // Add remaining statistical features...
        }
        
        Ok(features)
    }

    /// Run neural network prediction
    async fn run_nn_prediction(&self, features: Array2<f32>) -> Result<Vec<f32>> {
        let features_dyn = CowArray::from(features).into_dyn();
        let input = Value::from_array(self.outcome_predictor.allocator(), &features_dyn)?;
        let outputs = self.outcome_predictor.run(vec![input])?;
        
        let output = &outputs[0];
        let predictions = output.try_extract::<f32>()?;
        
        // Convert to Vec
        let predictions_view = predictions.view();
        let pred_slice = predictions_view.as_slice().ok_or_else(|| anyhow::anyhow!("Failed to get slice"))?;
        Ok(pred_slice.to_vec())
    }

    /// Apply Thompson Sampling bandit
    async fn apply_bandit(&self, base_predictions: Vec<f32>) -> Result<Vec<f32>> {
        let mut bandit = self.bandit.write().await;
        bandit.sample(base_predictions)
    }

    /// Combine predictions from different models
    fn combine_predictions(
        &self,
        nn_predictions: Vec<f32>,
        bandit_scores: Vec<f32>,
    ) -> Result<Vec<OutcomePrediction>> {
        let n_outcomes = nn_predictions.len();
        let mut combined_scores = vec![0.0; n_outcomes];
        
        // Weighted combination (70% NN, 30% bandit for exploration)
        for i in 0..n_outcomes {
            combined_scores[i] = 0.7 * nn_predictions[i] + 0.3 * bandit_scores[i];
        }
        
        // Apply softmax for final probabilities
        let max_score = combined_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = combined_scores.iter()
            .map(|&s| (s - max_score).exp())
            .collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        
        let probabilities: Vec<f32> = exp_scores.iter()
            .map(|&s| s / sum_exp)
            .collect();
        
        // Create predictions (top-k with threshold)
        let mut predictions = Vec::new();
        let mut indexed_probs: Vec<(usize, f32)> = probabilities.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (idx, prob) in indexed_probs.iter().take(5) {
            if *prob > 0.1 { // Confidence threshold
                predictions.push(OutcomePrediction {
                    outcome_id: self.get_outcome_id(*idx),
                    outcome_name: self.get_outcome_name(*idx).to_string(),
                    confidence: *prob,
                    reasoning: Some(self.generate_reasoning(*idx, *prob)),
                });
            }
        }
        
        Ok(predictions)
    }

    /// Update performance metrics
    async fn update_metrics(&self, latency_ms: f64) {
        let mut metrics = self.metrics.write().await;
        metrics.add_prediction(latency_ms);
    }

    // Helper methods
    fn simple_tokenize(&self, text: &str, max_length: usize) -> Result<Vec<i64>> {
        // Simplified tokenization - in production use proper tokenizer
        let words: Vec<&str> = text.split_whitespace().take(max_length).collect();
        let mut tokens = vec![101]; // [CLS] token
        
        for word in words {
            // Simple hash-based token ID (replace with real tokenizer)
            let token_id = (word.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32)) % 30000) as i64;
            tokens.push(token_id);
        }
        
        tokens.push(102); // [SEP] token
        
        // Pad to max_length
        while tokens.len() < max_length {
            tokens.push(0);
        }
        
        Ok(tokens)
    }

    fn get_outcome_id(&self, index: usize) -> String {
        // Map index to actual outcome ID
        format!("outcome_{}", index)
    }

    fn get_outcome_name(&self, index: usize) -> String {
        // Map index to actual outcome name
        format!("Outcome {}", index)
    }

    fn generate_reasoning(&self, index: usize, confidence: f32) -> String {
        format!("Predicted with {:.1}% confidence based on artifact patterns", confidence * 100.0)
    }
}

/// Thompson Sampling Bandit for exploration
pub struct ThompsonSamplingBandit {
    alpha: Vec<f32>, // Success counts
    beta: Vec<f32>,  // Failure counts
    n_arms: usize,
}

impl ThompsonSamplingBandit {
    pub fn new(n_arms: usize) -> Self {
        Self {
            alpha: vec![1.0; n_arms],
            beta: vec![1.0; n_arms],
            n_arms,
        }
    }

    pub fn sample(&self, base_predictions: Vec<f32>) -> Result<Vec<f32>> {
        let mut rng = rng();
        let mut samples = Vec::new();
        
        for i in 0..self.n_arms {
            // Sample from Beta distribution using Gamma
            let alpha_dist = Gamma::new(self.alpha[i], 1.0)?;
            let beta_dist = Gamma::new(self.beta[i], 1.0)?;
            
            let alpha_sample = alpha_dist.sample(&mut rng);
            let beta_sample = beta_dist.sample(&mut rng);
            
            let theta = alpha_sample / (alpha_sample + beta_sample);
            
            // Combine with base prediction
            samples.push(theta * 0.3 + base_predictions.get(i).unwrap_or(&0.0) * 0.7);
        }
        
        Ok(samples)
    }

    pub fn update(&mut self, arm: usize, reward: f32) {
        if arm < self.n_arms {
            if reward > 0.5 {
                self.alpha[arm] += reward;
            } else {
                self.beta[arm] += 1.0 - reward;
            }
        }
    }
}

/// LRU Cache implementation
pub struct LRUCache<K: std::hash::Hash + Eq + Clone, V: Clone> {
    capacity: usize,
    map: HashMap<K, V>,
    order: Vec<K>,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<V> {
        if let Some(value) = self.map.get(key) {
            // Move to front
            self.order.retain(|k| k != key);
            self.order.push(key.clone());
            Some(value.clone())
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
            // Evict least recently used
            if let Some(lru_key) = self.order.first() {
                let lru_key = lru_key.clone();
                self.map.remove(&lru_key);
                self.order.remove(0);
            }
        }
        
        self.map.insert(key.clone(), value);
        self.order.retain(|k| k != &key);
        self.order.push(key);
    }
}

/// Performance metrics tracking
#[derive(Default)]
pub struct PerformanceMetrics {
    total_predictions: u64,
    total_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    latencies: Vec<f64>,
}

impl PerformanceMetrics {
    pub fn add_prediction(&mut self, latency_ms: f64) {
        self.total_predictions += 1;
        self.total_latency_ms += latency_ms;
        self.latencies.push(latency_ms);
        
        // Keep only last 1000 for percentile calculation
        if self.latencies.len() > 1000 {
            self.latencies.remove(0);
        }
        
        // Calculate percentiles
        let mut sorted = self.latencies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p95_idx = (sorted.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted.len() as f64 * 0.99) as usize;
        
        self.p95_latency_ms = sorted.get(p95_idx).copied().unwrap_or(0.0);
        self.p99_latency_ms = sorted.get(p99_idx).copied().unwrap_or(0.0);
    }
    
    pub fn avg_latency_ms(&self) -> f64 {
        if self.total_predictions > 0 {
            self.total_latency_ms / self.total_predictions as f64
        } else {
            0.0
        }
    }

}