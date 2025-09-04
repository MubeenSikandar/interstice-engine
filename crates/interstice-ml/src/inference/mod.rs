//interstice-ml/src/inference/mod.rs
use crate::types::{ImpactLevel, ModelMetrics, OutcomePrediction};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use ndarray::{Array1, Array2, CowArray};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{Tokenizer};
use tracing::{debug,info, instrument, warn};
use uuid::Uuid;

mod engine;
mod bandit;
mod cache;
mod edge;

pub use engine::PredictionContext;
pub use bandit::ThompsonSamplingBandit;
pub use cache::{LRUCache, ConcurrentLRUCache};

// Configuration for model loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub onnx_model_path: Option<PathBuf>,
    pub bert_model_path: Option<PathBuf>,
    pub tokenizer_path: Option<PathBuf>,
    pub device_preference: DevicePreference,
    pub max_sequence_length: usize,
    pub embedding_dim: usize,
    pub confidence_threshold: f32,
    pub cache_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DevicePreference {
    Cpu,
    Cuda(usize),
    Metal,
    Auto,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            onnx_model_path: None,
            bert_model_path: None,
            tokenizer_path: None,
            device_preference: DevicePreference::Auto,
            max_sequence_length: 512,
            embedding_dim: 768,
            confidence_threshold: 0.3,
            cache_embeddings: true,
        }
    }
}

/// Enhanced Outcome predictor using ONNX models
pub struct OutcomePredictor {
    session: Arc<RwLock<Option<Arc<Session>>>>,
    environment: Arc<Environment>,
    config: ModelConfig,
    device: Device,
    outcome_mapping: Arc<RwLock<HashMap<usize, OutcomeMetadata>>>,
    performance_tracker: Arc<PerformanceTracker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutcomeMetadata {
    id: String,
    name: String,
    description: String,
    base_confidence: f32,
    impact_level: ImpactLevel,
}

impl OutcomePredictor {
    /// Create a new predictor with configuration
    pub async fn new(config: ModelConfig) -> Result<Self> {
        let device = Self::select_device(&config.device_preference)?;
        
        let environment = Arc::new(
            Environment::builder()
                .with_name("interstice-ml")
                .with_log_level(ort::LoggingLevel::Warning)
                .build()
                .context("Failed to create ONNX environment")?
        );
        
        let session = if let Some(ref model_path) = config.onnx_model_path {
            Some(Arc::new(Self::load_onnx_model(&environment, model_path).await?))
        } else {
            None
        };
        
        Ok(Self {
            session: Arc::new(RwLock::new(session)),
            environment,
            config,
            device,
            outcome_mapping: Arc::new(RwLock::new(Self::load_outcome_mapping()?)),
            performance_tracker: Arc::new(PerformanceTracker::new()),
        })
    }

    /// Select appropriate device based on availability
    fn select_device(preference: &DevicePreference) -> Result<Device> {
        match preference {
            DevicePreference::Cpu => Ok(Device::Cpu),
            DevicePreference::Cuda(idx) => {
                Device::new_cuda(*idx).context("CUDA device not available")
            }
            DevicePreference::Metal => {
                Device::new_metal(0).context("Metal device not available")
            }
            DevicePreference::Auto => {
                // Try CUDA first, then Metal, fallback to CPU
                Ok(Device::new_cuda(0)
                    .or_else(|_| Device::new_metal(0))
                    .unwrap_or(Device::Cpu))
            }
        }
    }

    /// Load ONNX model with optimizations
    async fn load_onnx_model(
        environment: &Arc<Environment>,
        model_path: &Path,
    ) -> Result<Session> {
        info!("Loading ONNX model from: {}", model_path.display());
        
        let mut session_builder = SessionBuilder::new(environment)?;
        
        // Configure optimization
        session_builder = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_cpus::get() as i16)?;
        
        // Try to use CUDA if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(provider) = ExecutionProvider::cuda_provider() {
                session_builder = session_builder.with_execution_providers(&[provider])?;
            }
        }
        
        // Fallback to CPU
        session_builder = session_builder
            .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])?;
        
        let session = session_builder
            .with_model_from_file(model_path)
            .context("Failed to load ONNX model")?;
        
        info!("ONNX model loaded successfully");
        Ok(session)
    }

    /// Load outcome mapping from configuration or database
    fn load_outcome_mapping() -> Result<HashMap<usize, OutcomeMetadata>> {
        // In production, load from database
        let mut mapping = HashMap::new();
        
        mapping.insert(0, OutcomeMetadata {
            id: Uuid::new_v4().to_string(),
            name: "User Activation".to_string(),
            description: "Complete user onboarding and activation".to_string(),
            base_confidence: 0.8,
            impact_level: ImpactLevel::High,
        });
        
        mapping.insert(1, OutcomeMetadata {
            id: Uuid::new_v4().to_string(),
            name: "Performance Optimization".to_string(),
            description: "Optimize system performance".to_string(),
            base_confidence: 0.6,
            impact_level: ImpactLevel::Medium,
        });
        
        mapping.insert(2, OutcomeMetadata {
            id: Uuid::new_v4().to_string(),
            name: "Security Hardening".to_string(),
            description: "Improve security posture".to_string(),
            base_confidence: 0.7,
            impact_level: ImpactLevel::Critical,
        });
        
        Ok(mapping)
    }

    /// Predict outcomes with enhanced error handling
    #[instrument(skip(self, embedding, artifacts))]
    pub async fn predict(
        &self,
        embedding: Vec<f32>,
        artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<OutcomePrediction>> {
        let start = std::time::Instant::now();
        
        // Ensure model is loaded
        let session = self.ensure_model_loaded().await?;
        
        // Prepare and validate features
        let input_features = self.prepare_input_features(embedding, artifacts)?;
        
        // Run inference with retry logic
        let predictions = self.run_inference_with_retry(&session, input_features, 3).await?;
        
        // Convert to outcome predictions
        let outcome_predictions = self.convert_predictions(predictions)?;
        
        // Track performance
        self.performance_tracker.record_prediction(
            start.elapsed().as_millis() as f64,
            outcome_predictions.len(),
        );
        
        Ok(outcome_predictions)
    }

    /// Ensure model is loaded (lazy loading support)
    async fn ensure_model_loaded(&self) -> Result<Arc<Session>> {
        // Check if already loaded
        {
            let session_guard = self.session.read();
            if let Some(session) = session_guard.as_ref() {
                return Ok(session.clone());
            }
        } // session_guard is dropped here
        
        // Load model if path is configured
        let model_path = self.config.onnx_model_path.clone();
        if let Some(model_path) = model_path {
            let session = Arc::new(
                Self::load_onnx_model(&self.environment, &model_path).await?
            );
            
            // Update the session with the loaded model
            let mut session_guard = self.session.write();
            *session_guard = Some(session.clone());
            Ok(session)
        } else {
            Err(anyhow::anyhow!("No model path configured"))
        }
    }

    /// Run inference with retry logic
    async fn run_inference_with_retry(
        &self,
        session: &Session,
        features: Vec<f32>,
        max_retries: usize,
    ) -> Result<Vec<f32>> {
        let mut last_error = None;
        
        for attempt in 0..max_retries {
            match self.run_inference(session, features.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    warn!("Inference attempt {} failed: {}", attempt + 1, e);
                    last_error = Some(e);
                    tokio::time::sleep(tokio::time::Duration::from_millis(100 * (attempt as u64 + 1))).await;
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Inference failed")))
    }

    /// Enhanced feature preparation with validation
    fn prepare_input_features(
        &self,
        embedding: Vec<f32>,
        artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<f32>> {
        // Validate embedding dimension
        if embedding.len() != self.config.embedding_dim {
            warn!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.config.embedding_dim,
                embedding.len()
            );
        }
        
        let mut features = Vec::with_capacity(self.config.embedding_dim + 100);
        features.extend_from_slice(&embedding);
        
        // Aggregate artifact features
        let artifact_features = self.aggregate_artifact_features(artifacts)?;
        features.extend(artifact_features);
        
        // Ensure correct dimensionality
        features.resize(self.config.embedding_dim, 0.0);
        
        Ok(features)
    }

    /// Aggregate features from multiple artifacts
    fn aggregate_artifact_features(&self, artifacts: &[interstice_core::Artifact]) -> Result<Vec<f32>> {
        if artifacts.is_empty() {
            return Ok(vec![0.0; 18]); // Platform (12) + Type (5) + Length (1)
        }
        
        let mut aggregated = vec![0.0; 18];
        
        for artifact in artifacts {
            let features = self.extract_artifact_features(artifact);
            for (i, &feat) in features.iter().enumerate() {
                if i < aggregated.len() {
                    aggregated[i] = f32::max(aggregated[i], feat); // Max pooling
                }
            }
        }
        
        Ok(aggregated)
    }
    
    /// Extract features from a single artifact
    fn extract_artifact_features(&self, _artifact: &interstice_core::Artifact) -> Vec<f32> {
        // Placeholder implementation - extract features from artifact
        vec![0.0; 18] // Platform (12) + Type (5) + Length (1)
    }

    /// Run inference with proper tensor handling
    async fn run_inference(&self, session: &Session, features: Vec<f32>) -> Result<Vec<f32>> {
        // Create input tensor
        let input_array = Array2::from_shape_vec((1, features.len()), features)?;
        let input_dyn = CowArray::from(input_array.view()).into_dyn();
        let input_value = Value::from_array(session.allocator(), &input_dyn)?;
        
        // Run model
        let outputs = session.run(vec![input_value])?;
        let output = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("No output from model"))?;
        
        // Extract and process output
        let output_tensor = output.try_extract::<f32>()?;
        let output_view = output_tensor.view();
        
        // Apply softmax
        let logits = Array1::from_iter(output_view.iter().copied());
        let probabilities = self.softmax(&logits)?;
        
        Ok(probabilities.to_vec())
    }

    /// Numerically stable softmax
    fn softmax(&self, logits: &Array1<f32>) -> Result<Array1<f32>> {
        let max_logit = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        
        if sum_exp == 0.0 {
            return Err(anyhow::anyhow!("Softmax computation resulted in zero sum"));
        }
        
        Ok(exp_logits / sum_exp)
    }

    /// Convert predictions to structured outcomes
    fn convert_predictions(&self, predictions: Vec<f32>) -> Result<Vec<OutcomePrediction>> {
        let mapping = self.outcome_mapping.read();
        let mut outcome_predictions = Vec::new();
        
        for (idx, &confidence) in predictions.iter().enumerate() {
            if confidence < self.config.confidence_threshold {
                continue;
            }
            
            if let Some(metadata) = mapping.get(&idx) {
                let adjusted_confidence = confidence * metadata.base_confidence;
                
                outcome_predictions.push(OutcomePrediction {
                    outcome_id: metadata.id.clone(),
                    outcome_name: metadata.name.clone(),
                    confidence: adjusted_confidence,
                    reasoning: Some(self.generate_reasoning(adjusted_confidence, metadata)),
                    contributing_factors: Vec::new(),
                    alternative_outcomes: Vec::new(),
                    predicted_impact: metadata.impact_level,
                    time_to_completion: Some(self.estimate_completion_time(metadata)),
                });
            }
        }
        
        // Sort by confidence
        outcome_predictions.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(outcome_predictions)
    }

    fn generate_reasoning(&self, confidence: f32, metadata: &OutcomeMetadata) -> String {
        format!(
            "{} with {:.1}% confidence based on pattern analysis",
            metadata.description,
            confidence * 100.0
        )
    }
    
    fn estimate_completion_time(&self, metadata: &OutcomeMetadata) -> crate::types::Duration {
        match metadata.impact_level {
            crate::types::ImpactLevel::Critical => crate::types::Duration {
                min_hours: 2.0,
                max_hours: 6.0,
                likely_hours: 4.0,
            },
            crate::types::ImpactLevel::High => crate::types::Duration {
                min_hours: 12.0,
                max_hours: 36.0,
                likely_hours: 24.0,
            },
            crate::types::ImpactLevel::Medium => crate::types::Duration {
                min_hours: 48.0,
                max_hours: 96.0,
                likely_hours: 72.0,
            },
            crate::types::ImpactLevel::Low => crate::types::Duration {
                min_hours: 120.0,
                max_hours: 240.0,
                likely_hours: 168.0,
            },
            crate::types::ImpactLevel::Negligible => crate::types::Duration {
                min_hours: 240.0,
                max_hours: 480.0,
                likely_hours: 336.0,
            },
        }
    }
}

/// Enhanced Text Embedder with caching and optimization
pub struct TextEmbedder {
    model: Arc<RwLock<Option<BertModel>>>,
    tokenizer: Arc<RwLock<Option<Arc<Tokenizer>>>>,
    device: Device,
    config: ModelConfig,
    embedding_cache: Arc<ConcurrentLRUCache<String, Vec<f32>>>,
}

impl TextEmbedder {
    pub async fn new(config: ModelConfig) -> Result<Self> {
        let device = OutcomePredictor::select_device(&config.device_preference)?;
        
        let (model, tokenizer) = if let (Some(ref model_path), Some(ref tokenizer_path)) = 
            (config.bert_model_path.as_ref(), config.tokenizer_path.as_ref()) 
        {
            let model = Self::load_bert_model(model_path, &device).await?;
            let tokenizer = Self::load_tokenizer(tokenizer_path).await?;
            (Some(model), Some(Arc::new(tokenizer)))
        } else {
            (None, None)
        };
        
        let cache_size = if config.cache_embeddings { 10000 } else { 0 };
        
        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            device,
            config,
            embedding_cache: Arc::new(ConcurrentLRUCache::new(cache_size)?),
        })
    }

    async fn load_bert_model(model_path: &Path, device: &Device) -> Result<BertModel> {
        info!("Loading BERT model from: {}", model_path.display());
        
        let config_path = model_path.join("config.json");
        let weights_path = model_path.join("model.safetensors");
        
        // Load configuration
        let config_str = tokio::fs::read_to_string(&config_path)
            .await
            .context("Failed to read BERT config")?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .context("Failed to parse BERT config")?;
        
        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], candle_core::DType::F32, device)?
        };
        
        let model = BertModel::load(vb, &config)
            .context("Failed to load BERT model")?;
        
        info!("BERT model loaded successfully");
        Ok(model)
    }

    async fn load_tokenizer(tokenizer_path: &Path) -> Result<Tokenizer> {
        info!("Loading tokenizer from: {}", tokenizer_path.display());
        
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        
        info!("Tokenizer loaded successfully");
        Ok(tokenizer)
    }

    #[instrument(skip(self, text))]
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        let cache_key = format!("{:x}", md5::compute(text));
        
        if self.config.cache_embeddings {
            if let Some(cached) = self.embedding_cache.get(&cache_key) {
                debug!("Cache hit for text embedding");
                return Ok(cached);
            }
        }
        
        // Generate embedding
        let embedding = self.generate_embedding(text).await?;
        
        // Cache result
        if self.config.cache_embeddings {
            self.embedding_cache.put(cache_key, embedding.clone());
        }
        
        Ok(embedding)
    }

    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let model_guard = self.model.read();
        let tokenizer_guard = self.tokenizer.read();
        
        match (model_guard.as_ref(), tokenizer_guard.as_ref()) {
            (Some(model), Some(tokenizer)) => {
                // Tokenize
                let encoding = tokenizer
                    .encode(text, true)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                
                let input_ids = Tensor::new(encoding.get_ids(), &self.device)?;
                let attention_mask = Tensor::new(encoding.get_attention_mask(), &self.device)?;
                
                // Run through BERT
                let embeddings = model.forward(&input_ids, &attention_mask, None)?;
                
                // Mean pooling
                let pooled = self.mean_pool(&embeddings, &attention_mask)?;
                
                Ok(pooled.to_vec1::<f32>()?)
            }
            _ => {
                // Fallback to simple embedding
                warn!("Using fallback embedding generation");
                self.generate_fallback_embedding(text)
            }
        }
    }

    fn mean_pool(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mask_expanded = attention_mask.unsqueeze(2)?;
        let masked_embeddings = embeddings.broadcast_mul(&mask_expanded)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let sum_mask = mask_expanded.sum(1)?;
        
        Ok(sum_embeddings.broadcast_div(&sum_mask)?)
    }

    fn generate_fallback_embedding(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut embeddings = Vec::with_capacity(self.config.embedding_dim);
        let mut hasher = DefaultHasher::new();
        
        // Generate deterministic pseudo-random embeddings
        for (i, chunk) in text.as_bytes().chunks(32).enumerate() {
            chunk.hash(&mut hasher);
            i.hash(&mut hasher);
            
            let hash = hasher.finish();
            let value = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0; // Normalize to [-1, 1]
            embeddings.push(value);
            
            if embeddings.len() >= self.config.embedding_dim {
                break;
            }
        }
        
        // Pad with zeros if needed
        embeddings.resize(self.config.embedding_dim, 0.0);
        
        // L2 normalize
        let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embeddings {
                *val /= norm;
            }
        }
        
        Ok(embeddings)
    }
}

/// Performance tracking for model inference
struct PerformanceTracker {
    metrics: Arc<RwLock<ModelMetrics>>,
    latencies: Arc<RwLock<Vec<f64>>>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(ModelMetrics::default())),
            latencies: Arc::new(RwLock::new(Vec::with_capacity(1000))),
        }
    }

    fn record_prediction(&self, latency_ms: f64, num_outcomes: usize) {
        let mut metrics = self.metrics.write();
        let mut latencies = self.latencies.write();
        
        metrics.total_predictions += 1;
        latencies.push(latency_ms);
        
        // Keep only last 1000 latencies
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
        
        // Update latency metrics
        metrics.prediction_latency_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
        metrics.last_updated = chrono::Utc::now();
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            correct_predictions: 0,
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            total_predictions: 0,
            auc_roc: None,
            mean_confidence: 0.0,
            prediction_latency_ms: 0.0,
            last_updated: chrono::Utc::now(),
            per_outcome_metrics: None,
        }
    }
}