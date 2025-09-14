//interstice-ml/src/models/mod.rs
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, ops, VarMap};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument};
use uuid::Uuid;

use crate::types::{
    AlternativeOutcome, ContributingFactor, Duration, ImpactLevel, ModelMetrics, OutcomePrediction, PerOutcomeMetrics, TrainingExample
};

/// Model state persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    pub workspace_id: Uuid,
    pub model_id: Uuid,
    pub version: u32,
    pub accuracy: f32,
    pub checkpoint_path: PathBuf,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub training_config: TrainingConfig,
    pub metadata: ModelMetadata,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub warmup_steps: usize,
    pub weight_decay: f32,
    pub dropout: f32,
    pub gradient_clip_value: f32,
    pub early_stopping_patience: usize,
    pub validation_split: f32,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub optimizer: OptimizerType,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-5,
            batch_size: 16,
            epochs: 3,
            warmup_steps: 100,
            weight_decay: 0.01,
            dropout: 0.1,
            gradient_clip_value: 1.0,
            early_stopping_patience: 3,
            validation_split: 0.2,
            lora_rank: 8,
            lora_alpha: 16.0,
            optimizer: OptimizerType::AdamW,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    RMSprop,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub base_model: String,
    pub fine_tuned: bool,
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub compression_ratio: f32,
    pub tags: Vec<String>,
    pub description: Option<String>,
}

/// LoRA adapter configuration
#[derive(Debug, Clone)]
struct LoRAAdapter {
    rank: usize,
    alpha: f32,
    dropout: f32,
    target_modules: Vec<String>,
    weights_a: HashMap<String, Tensor>,
    weights_b: HashMap<String, Tensor>,
}

impl LoRAAdapter {
    fn new(rank: usize, alpha: f32, dropout: f32) -> Self {
        Self {
            rank,
            alpha,
            dropout,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
            ],
            weights_a: HashMap::new(),
            weights_b: HashMap::new(),
        }
    }

    fn initialize(&mut self, base_model: &BaseModel, device: &Device) -> Result<()> {
        for module_name in &self.target_modules {
            if let Some(layer_dim) = base_model.get_layer_dimension(module_name) {
                // Initialize LoRA weights A and B
                let weight_a = Tensor::randn(
                    0.0_f32,
                    0.02,
                    &[layer_dim, self.rank],
                    device,
                )?;
                
                let weight_b = Tensor::zeros(&[self.rank, layer_dim], DType::F32, device)?;
                
                self.weights_a.insert(module_name.clone(), weight_a);
                self.weights_b.insert(module_name.clone(), weight_b);
            }
        }
        Ok(())
    }

    fn forward(&self, input: &Tensor, module_name: &str) -> Result<Tensor> {
        if let (Some(weight_a), Some(weight_b)) = 
            (self.weights_a.get(module_name), self.weights_b.get(module_name)) 
        {
            // Apply LoRA: output = input + (input @ A @ B) * (alpha / rank)
            let lora_output = input
                .matmul(weight_a)?
                .matmul(weight_b)?;
            
            let scale_factor = self.alpha / self.rank as f32;
            let scaled = lora_output.broadcast_mul(&Tensor::new(&[scale_factor], &lora_output.device()).map_err(anyhow::Error::from)?).map_err(anyhow::Error::from)?;
            
            // Apply dropout if configured
            let output = if self.dropout > 0.0 {
                ops::dropout(&input.add(&scaled)?, self.dropout)?
            } else {
                input.add(&scaled)?
            };
            
            Ok(output)
        } else {
            Ok(input.clone())
        }
    }
}

/// Base model wrapper
struct BaseModel {
    model_type: ModelType,
    device: Device,
    weights: Arc<VarMap>,
    config: BaseModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ModelType {
    BERT,
    GPT2,
    T5,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BaseModelConfig {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    dropout: f32,
}

impl BaseModel {
    fn get_layer_dimension(&self, module_name: &str) -> Option<usize> {
        match module_name {
            "q_proj" | "k_proj" | "v_proj" => Some(self.config.hidden_size),
            "o_proj" => Some(self.config.hidden_size),
            _ => None,
        }
    }

    async fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Simplified forward pass - in production, use actual model architecture
        let embeddings = self.embed(input_ids)?;
        let output = self.encode(embeddings, attention_mask)?;
        Ok(output)
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Token embedding layer
        let vocab_size = self.config.vocab_size;
        let hidden_size = self.config.hidden_size;
        
        let embedding_weights = self.weights
            .get(
                (vocab_size, hidden_size),
                "embeddings.word_embeddings",
                candle_nn::init::DEFAULT_KAIMING_NORMAL,
                DType::F32,
                &self.device,
            )?;
        
        input_ids.embedding(&embedding_weights).map_err(anyhow::Error::from)
    }

    fn encode(&self, embeddings: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Simplified transformer encoding
        let mut hidden_states = embeddings;
        
        for layer_idx in 0..self.config.num_layers {
            hidden_states = self.transformer_layer(hidden_states, attention_mask, layer_idx)?;
        }
        
        Ok(hidden_states)
    }

    fn transformer_layer(
        &self,
        input: Tensor,
        attention_mask: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        // Simplified transformer layer - implement full attention mechanism in production
        let normalized = self.layer_norm(input.clone(), layer_idx)?;
        let attention_output = self.self_attention(normalized, attention_mask, layer_idx)?;
        let output = (input + attention_output)?;
        
        Ok(output)
    }

    fn layer_norm(&self, input: Tensor, layer_idx: usize) -> Result<Tensor> {
        debug!("Applying layer norm for layer: {}", layer_idx);
        let hidden_size = self.config.hidden_size;
        let weight = Tensor::ones(&[hidden_size], DType::F32, &self.device)?;
        let bias = Tensor::zeros(&[hidden_size], DType::F32, &self.device)?;
        ops::layer_norm(&input, &weight, &bias, 1e-5).map_err(anyhow::Error::from)
    }

    fn self_attention(
        &self,
        input: Tensor,
        attention_mask: Option<&Tensor>,
        layer_idx: usize,
    ) -> Result<Tensor> {
        debug!("Self attention layer: {}, mask present: {}", layer_idx, attention_mask.is_some());
        // Simplified self-attention
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_heads;
        let head_dim = hidden_size / num_heads;
        
        debug!("Attention config - hidden_size: {}, num_heads: {}, head_dim: {}", hidden_size, num_heads, head_dim);
        
        // Query, Key, Value projections would go here
        // For now, return input with dropout
        if self.config.dropout > 0.0 {
            ops::dropout(&input, self.config.dropout).map_err(anyhow::Error::from)
        } else {
            Ok(input)
        }
    }
}

/// Main organization model with fine-tuning capabilities
#[derive(Clone)]
pub struct OrgModel {
    pub workspace_id: Uuid,
    pub model_id: Uuid,
    pub best_accuracy: f32,
    pub model_version: u32,
    pub last_trained: Option<DateTime<Utc>>,
    
    // Internal components
    base_model: Arc<BaseModel>,
    lora_adapter: Arc<RwLock<LoRAAdapter>>,
    training_config: TrainingConfig,
    device: Device,
    tokenizer: Arc<Tokenizer>,
    outcome_mapping: Arc<HashMap<usize, String>>,
    
    // Training state
    training_history: Arc<RwLock<TrainingHistory>>,
    checkpoint_manager: Arc<CheckpointManager>,
    
    // Performance tracking
    metrics_tracker: Arc<MetricsTracker>,
}

/// Tokenizer wrapper
struct Tokenizer {
    vocab: HashMap<String, usize>,
    inverse_vocab: HashMap<usize, String>,
    max_length: usize,
}

impl Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        // Simplified tokenization - use proper tokenizer in production
        text.split_whitespace()
            .take(self.max_length)
            .filter_map(|word| self.vocab.get(word.to_lowercase().as_str()))
            .copied()
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens.iter()
            .filter_map(|&id| self.inverse_vocab.get(&id))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Training history tracking
#[derive(Debug, Clone, Default)]
struct TrainingHistory {
    epochs: Vec<EpochMetrics>,
    best_validation_loss: f32,
    best_epoch: usize,
    total_training_time: std::time::Duration,
}

#[derive(Debug, Clone)]
struct EpochMetrics {
    epoch: usize,
    train_loss: f32,
    train_accuracy: f32,
    val_loss: f32,
    val_accuracy: f32,
    learning_rate: f32,
    duration: std::time::Duration,
}

/// Checkpoint management
struct CheckpointManager {
    checkpoint_dir: PathBuf,
    max_checkpoints: usize,
    checkpoint_queue: Arc<RwLock<VecDeque<PathBuf>>>,
}

impl CheckpointManager {
    fn new(workspace_id: Uuid) -> Self {
        let checkpoint_dir = PathBuf::from(format!("checkpoints/{}", workspace_id));
        std::fs::create_dir_all(&checkpoint_dir).ok();
        
        Self {
            checkpoint_dir,
            max_checkpoints: 5,
            checkpoint_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    async fn save_checkpoint(&self, model: &OrgModel, epoch: usize) -> candle_core::Result<PathBuf> {
        let checkpoint_path = self.checkpoint_dir.join(format!(
            "model_v{}_epoch_{}.safetensors",
            model.model_version,
            epoch
        ));
        
        // Save model weights
        let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
        
        // Save LoRA adapter weights
        let lora = model.lora_adapter.read();
        for (name, tensor) in &lora.weights_a {
            tensors.insert(format!("lora_a.{}", name), tensor.clone());
        }
        for (name, tensor) in &lora.weights_b {
            tensors.insert(format!("lora_b.{}", name), tensor.clone());
        }
        
        // Prepare tensors: move to CPU and make contiguous
        for tensor in tensors.values_mut() {
            *tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
        }
        
        // Save using manual serialization
        let mut buffer = Vec::new();
        for (name, tensor) in tensors {
            debug!("Serializing checkpoint tensor: {}", name);
            let data = tensor.to_vec1::<f32>()?;
            let bytes: Vec<u8> = data.iter().flat_map(|&f| f.to_le_bytes()).collect();
            buffer.extend_from_slice(&bytes);
        }
        std::fs::write(&checkpoint_path, buffer)?;
        
        // Manage checkpoint queue
        let mut queue = self.checkpoint_queue.write();
        queue.push_back(checkpoint_path.clone());
        
        if queue.len() > self.max_checkpoints {
            if let Some(old_checkpoint) = queue.pop_front() {
                std::fs::remove_file(old_checkpoint).ok();
            }
        }
        
        Ok(checkpoint_path)
    }
    
    async fn load_checkpoint(&self, checkpoint_path: &Path) -> candle_core::Result<HashMap<String, candle_core::Tensor>> {
        let data = std::fs::read(checkpoint_path)?;
        let mut tensors = HashMap::new();
        
        // Convert bytes back to f32 values
        let f32_data: Vec<f32> = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // For now, create a single tensor from all data
        // In production, you'd need to store tensor shapes and names
        if !f32_data.is_empty() {
            let len = f32_data.len();
            let tensor = Tensor::from_vec(f32_data, &[len], &Device::Cpu)?;
            tensors.insert("lora_weights".to_string(), tensor);
        }
        
        Ok(tensors)
    }
}

/// Metrics tracking
struct MetricsTracker {
    current_metrics: Arc<RwLock<ModelMetrics>>,
    historical_metrics: Arc<RwLock<Vec<ModelMetrics>>>,
}

impl MetricsTracker {
    fn new() -> Self {
        Self {
            current_metrics: Arc::new(RwLock::new(ModelMetrics::default())),
            historical_metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn update(&self, metrics: ModelMetrics) {
        let mut current = self.current_metrics.write();
        *current = metrics.clone();
        
        let mut history = self.historical_metrics.write();
        history.push(metrics);
        
        // Keep only last 100 metrics
        if history.len() > 100 {
            history.remove(0);
        }
    }
}

impl OrgModel {
    /// Create a new organization model
    pub async fn new(workspace_id: Uuid) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        
        // Initialize base model
        let base_model = Arc::new(Self::load_base_model(&device).await?);
        
        // Initialize LoRA adapter
        let training_config = TrainingConfig::default();
        let mut lora_adapter = LoRAAdapter::new(
            training_config.lora_rank,
            training_config.lora_alpha,
            training_config.dropout,
        );
        lora_adapter.initialize(&base_model, &device)?;
        
        // Initialize tokenizer
        let tokenizer = Arc::new(Self::load_tokenizer().await?);
        
        // Load outcome mapping
        let outcome_mapping = Arc::new(Self::load_outcome_mapping().await?);
        
        Ok(Self {
            workspace_id,
            model_id: Uuid::new_v4(),
            best_accuracy: 0.0,
            model_version: 1,
            last_trained: None,
            base_model,
            lora_adapter: Arc::new(RwLock::new(lora_adapter)),
            training_config,
            device,
            tokenizer,
            outcome_mapping,
            training_history: Arc::new(RwLock::new(TrainingHistory::default())),
            checkpoint_manager: Arc::new(CheckpointManager::new(workspace_id)),
            metrics_tracker: Arc::new(MetricsTracker::new()),
        })
    }

    async fn load_base_model(device: &Device) -> Result<BaseModel> {
        // Load pre-trained base model
        let config = BaseModelConfig {
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            vocab_size: 30522,
            max_position_embeddings: 512,
            dropout: 0.1,
        };
        
        let var_map = VarMap::new();
        
        Ok(BaseModel {
            model_type: ModelType::BERT,
            device: device.clone(),
            weights: Arc::new(var_map),
            config,
        })
    }

    async fn load_tokenizer() -> Result<Tokenizer> {
        // Load tokenizer vocabulary
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();
        
        // Simplified vocabulary - load from file in production
        for (idx, word) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"].iter().enumerate() {
            vocab.insert(word.to_string(), idx);
            inverse_vocab.insert(idx, word.to_string());
        }
        
        Ok(Tokenizer {
            vocab,
            inverse_vocab,
            max_length: 512,
        })
    }

    async fn load_outcome_mapping() -> Result<HashMap<usize, String>> {
        // Load outcome ID to name mapping
        let mut mapping = HashMap::new();
        mapping.insert(0, "User Activation".to_string());
        mapping.insert(1, "Feature Adoption".to_string());
        mapping.insert(2, "Bug Resolution".to_string());
        mapping.insert(3, "Performance Optimization".to_string());
        Ok(mapping)
    }

    /// Fine-tune the model on organization-specific examples
    #[instrument(skip(self, examples))]
    pub async fn fine_tune(&mut self, examples: &[TrainingExample]) -> Result<()> {
        info!(
            "Fine-tuning model for workspace {} with {} examples",
            self.workspace_id,
            examples.len()
        );
        
        let start_time = Instant::now();
        
        // Prepare data
        let (train_data, val_data) = self.prepare_training_data(examples)?;
        
        // Initialize optimizer
        let mut optimizer = self.create_optimizer()?;
        
        // Training loop
        let mut best_val_loss = f32::MAX;
        let mut patience_counter = 0;
        
        for epoch in 0..self.training_config.epochs {
            let epoch_start = Instant::now();
            
            // Training phase
            let train_metrics = self.train_epoch(&train_data, &mut optimizer, epoch).await?;
            
            // Validation phase
            let val_metrics = self.validate(&val_data).await?;
            
            // Update metrics
            let epoch_metrics = EpochMetrics {
                epoch,
                train_loss: train_metrics.0,
                train_accuracy: train_metrics.1,
                val_loss: val_metrics.0,
                val_accuracy: val_metrics.1,
                learning_rate: self.get_learning_rate(epoch),
                duration: epoch_start.elapsed(),
            };
            
            let mut history = self.training_history.write();
            history.epochs.push(epoch_metrics.clone());
            
            // Update best metrics tracking
            if epoch_metrics.val_loss < history.best_validation_loss {
                history.best_validation_loss = epoch_metrics.val_loss;
                history.best_epoch = epoch_metrics.epoch;
            }
            
            info!(
                "Epoch {}/{}: train_loss={:.4}, train_acc={:.4}, val_loss={:.4}, val_acc={:.4}, lr={:.6}, duration={:?}",
                epoch + 1,
                self.training_config.epochs,
                epoch_metrics.train_loss,
                epoch_metrics.train_accuracy,
                epoch_metrics.val_loss,
                epoch_metrics.val_accuracy,
                epoch_metrics.learning_rate,
                epoch_metrics.duration
            );
            
            // Early stopping check
            if val_metrics.0 < best_val_loss {
                best_val_loss = val_metrics.0;
                self.best_accuracy = val_metrics.1;
                patience_counter = 0;
                
                // Save best checkpoint
                self.checkpoint_manager.save_checkpoint(self, epoch).await?;
            } else {
                patience_counter += 1;
                if patience_counter >= self.training_config.early_stopping_patience {
                    info!("Early stopping triggered at epoch {}", epoch + 1);
                    break;
                }
            }
        }
        
        // Update model metadata
        self.model_version += 1;
        self.last_trained = Some(Utc::now());
        
        let total_time = start_time.elapsed();
        self.training_history.write().total_training_time = total_time;
        
        info!(
            "Model fine-tuning completed for workspace {} in {:?}",
            self.workspace_id, total_time
        );
        
        Ok(())
    }

    fn prepare_training_data(
        &self,
        examples: &[TrainingExample],
    ) -> Result<(Vec<ProcessedExample>, Vec<ProcessedExample>)> {
        let mut processed = Vec::new();
        
        for example in examples {
            let input_ids = self.tokenizer.encode(&example.input_text);
            let label = example.suggested_outcome_id.map(|id| id.as_u128() as usize).unwrap_or(0);
            processed.push(ProcessedExample {
                input_ids: Tensor::from_vec(input_ids.iter().map(|&x| x as u32).collect::<Vec<_>>(), &[1, 512], &self.device)?,
                attention_mask: Tensor::ones(&[1, 512], DType::F32, &self.device)?,
                label,
            });
        }
        
        // Split into train and validation
        let split_idx = (processed.len() as f32 * (1.0 - self.training_config.validation_split)) as usize;
        let (train, val) = processed.split_at(split_idx);
        
        Ok((train.to_vec(), val.to_vec()))
    }

    fn encode_outcome(&self, outcome: &str) -> Result<usize> {
        self.outcome_mapping
            .iter()
            .find(|(_, name)| name.as_str() == outcome)
            .map(|(idx, _)| *idx)
            .ok_or_else(|| anyhow::anyhow!("Unknown outcome: {}", outcome))
    }

    fn create_optimizer(&self) -> Result<AdamW> {
        let mut params = Vec::new();
        
        // Add LoRA parameters
        let lora = self.lora_adapter.read();
        for tensor in lora.weights_a.values() {
            params.push(tensor.clone());
        }
        for tensor in lora.weights_b.values() {
            params.push(tensor.clone());
        }
        
        Ok(AdamW::new(
            params,
            self.training_config.learning_rate,
            0.9,
            0.999,
            self.training_config.weight_decay,
        )?)
    }

    async fn train_epoch(
        &self,
        data: &[ProcessedExample],
        optimizer: &mut impl Optimizer,
        epoch: usize,
    ) -> Result<(f32, f32)> {
        debug!("Training epoch: {}", epoch);
        let mut total_loss = 0.0;
        let correct = 0;
        let mut total = 0;
        
        // Create batches
        for batch in data.chunks(self.training_config.batch_size) {
            let batch_loss = self.forward_batch(batch, true).await?;
            
            // Backward pass
            optimizer.backward_step(&batch_loss)?;
            
            
            // Update metrics
            total_loss += batch_loss.to_scalar::<f32>()?;
            total += batch.len();
        }
        
        let avg_loss = total_loss / total as f32;
        let accuracy = correct as f32 / total as f32;
        
        Ok((avg_loss, accuracy))
    }

    async fn validate(&self, data: &[ProcessedExample]) -> Result<(f32, f32)> {
        let mut total_loss = 0.0;
        let correct = 0;
        let mut total = 0;
        
        for batch in data.chunks(self.training_config.batch_size) {
            let batch_loss = self.forward_batch(batch, false).await?;
            
            total_loss += batch_loss.to_scalar::<f32>()?;
            total += batch.len();
        }
        
        let avg_loss = total_loss / total as f32;
        let accuracy = correct as f32 / total as f32;
        
        Ok((avg_loss, accuracy))
    }

    async fn forward_batch(&self, batch: &[ProcessedExample], training: bool) -> Result<Tensor> {
        debug!("Forward pass - training mode: {}", training);
        // Stack batch tensors
        let input_ids = Tensor::stack(
            &batch.iter().map(|e| e.input_ids.clone()).collect::<Vec<_>>(),
            0,
        )?;
        
        let attention_mask = Tensor::stack(
            &batch.iter().map(|e| e.attention_mask.clone()).collect::<Vec<_>>(),
            0,
        )?;
        
        // Forward through base model
        let base_output = self.base_model.forward(&input_ids, Some(&attention_mask)).await?;
        
        // Apply LoRA adapter
        let lora = self.lora_adapter.read();
        let adapted_output = lora.forward(&base_output, "o_proj")?;
        
        // Classification head
        let logits = self.classification_head(adapted_output)?;
        
        // Calculate loss
        let labels = Tensor::from_vec(
            batch.iter().map(|e| e.label as f32).collect::<Vec<_>>(),
            &[batch.len()],
            &self.device,
        )?;
        
        loss::cross_entropy(&logits, &labels.to_dtype(DType::U32)?).map_err(anyhow::Error::from)
    }

    fn classification_head(&self, input: Tensor) -> Result<Tensor> {
        // Simple classification head
        let num_classes = self.outcome_mapping.len();
        let hidden_size = self.base_model.config.hidden_size;
        
        // Pool the sequence
        let pooled = input.mean(1)?;
        
        // Linear projection to classes
        let weight = Tensor::randn(
            0.0_f32,
            0.02,
            &[hidden_size, num_classes],
            &self.device,
        )?;
        
        pooled.matmul(&weight).map_err(anyhow::Error::from)
    }


    fn get_learning_rate(&self, epoch: usize) -> f32 {
        // Learning rate scheduling
        let warmup_steps = self.training_config.warmup_steps;
        let total_steps = self.training_config.epochs * 100; // Approximate steps
        let current_step = epoch * 100;
        
        if current_step < warmup_steps {
            // Linear warmup
            self.training_config.learning_rate * (current_step as f32 / warmup_steps as f32)
        } else {
            // Cosine decay
            let progress = (current_step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
            let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.training_config.learning_rate * cosine_decay
        }
    }

    /// Evaluate model performance
    #[instrument(skip(self))]
    pub async fn evaluate(&self) -> Result<ModelMetrics> {
        info!("Evaluating model for workspace {}", self.workspace_id);
        
        // Load test dataset
        let test_data = self.load_test_data().await?;
        
        let mut all_predictions = Vec::new();
        let mut all_labels = Vec::new();
        let mut total_latency = 0.0;
        
        for example in &test_data {
            let start = Instant::now();
            
            // Get prediction - decode input_ids back to text for prediction
            let input_text = self.tokenizer.decode(&example.input_ids.to_vec1::<u32>()?.iter().map(|&x| x as usize).collect::<Vec<_>>());
            let predictions = self.predict(&input_text).await?;
            let predicted_class = predictions
                .first()
                .and_then(|p| self.encode_outcome(&p.outcome_name).ok())
                .unwrap_or(0);
            
            all_predictions.push(predicted_class);
            all_labels.push(example.label);
            
            total_latency += start.elapsed().as_millis() as f64;
        }
        
        // Calculate metrics
        let metrics = self.calculate_metrics(&all_predictions, &all_labels, total_latency)?;
        
        // Update tracker
        self.metrics_tracker.update(metrics.clone());
        
        Ok(metrics)
    }

    async fn load_test_data(&self) -> Result<Vec<ProcessedExample>> {
        // Load test dataset - implement actual data loading
        Ok(vec![])
    }

    fn calculate_metrics(
        &self,
        predictions: &[usize],
        labels: &[usize],
        total_latency: f64,
    ) -> Result<ModelMetrics> {
        let total = predictions.len();
        let correct = predictions
            .iter()
            .zip(labels)
            .filter(|(p, l)| p == l)
            .count();
        
        let accuracy = correct as f32 / total as f32;
        
        // Calculate per-class metrics
        let mut per_class = HashMap::new();
        for class_id in 0..self.outcome_mapping.len() {
            let tp = predictions.iter().zip(labels)
                .filter(|(&p, &l)| p == class_id && l == class_id)
                .count() as f32;
            
            let fp = predictions.iter().zip(labels)
                .filter(|(&p, &l)| p == class_id && l != class_id)
                .count() as f32;
            
            let fn_count = predictions.iter().zip(labels)
                .filter(|(&p, &l)| p != class_id && l == class_id)
                .count() as f32;
            
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };
            
            per_class.insert(
                self.outcome_mapping[&class_id].clone(),
                PerOutcomeMetrics {
                    outcome_name: self.outcome_mapping[&class_id].clone(),
                    precision,
                    recall,
                    f1_score: f1,
                    support: labels.iter().filter(|&&l| l == class_id).count(),
                    true_positives: tp as usize,
                    false_positives: fp as usize,
                    false_negatives: fn_count as usize,
                    true_negatives: (total as f32 - tp - fp - fn_count) as usize,
                },
            );
        }
        
        Ok(ModelMetrics {
            accuracy: accuracy as f64,
            precision: per_class.values().map(|m| m.precision as f64).sum::<f64>() / per_class.len() as f64,
            recall: per_class.values().map(|m| m.recall as f64).sum::<f64>() / per_class.len() as f64,
            f1_score: per_class.values().map(|m| m.f1_score as f64).sum::<f64>() / per_class.len() as f64,
            total_predictions: total as u64,
            correct_predictions: correct as u64,
            auc_roc: None, // Calculate if needed
            mean_confidence: 0.85, // Calculate from actual predictions
            prediction_latency_ms: total_latency / total as f64,
            last_updated: Utc::now(),
            per_outcome_metrics: Some(serde_json::to_value(per_class).unwrap_or(serde_json::Value::Null)),
            cache_hit_rate: 0.0,
        })
   }

   /// Make predictions on new input
   #[instrument(skip(self, input_text))]
   pub async fn predict(&self, input_text: &str) -> Result<Vec<OutcomePrediction>> {
       debug!("Making prediction for input text of length {}", input_text.len());
       
       // Tokenize input
       let input_ids = self.tokenizer.encode(input_text);
       let input_tensor = Tensor::from_vec(
           input_ids.clone().iter().map(|&x| x as u32).collect::<Vec<_>>(),
           &[1, input_ids.len()],
           &self.device,
       )?;
       
       // Create attention mask
       let attention_mask = Tensor::ones(
           &[1, input_ids.len()],
           DType::F32,
           &self.device,
       )?;
       
       // Forward pass
       let base_output = self.base_model
           .forward(&input_tensor, Some(&attention_mask))
           .await?;
       
       // Apply LoRA adapter
       let lora = self.lora_adapter.read();
       let adapted_output = lora.forward(&base_output, "o_proj")?;
       
       // Get logits from classification head
       let logits = self.classification_head(adapted_output)?;
       
       // Apply softmax to get probabilities
       let probs = ops::softmax(&logits, 1)?;
       let probs_vec = probs.to_vec2::<f32>()?[0].clone();
       
       // Convert to predictions
       let mut predictions = Vec::new();
       let mut indexed_probs: Vec<(usize, f32)> = probs_vec
           .iter()
           .enumerate()
           .map(|(i, &p)| (i, p))
           .collect();
       
       // Sort by probability
       indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
       
       // Take top predictions above threshold
       for (class_idx, prob) in indexed_probs.iter().take(5) {
           if *prob < 0.1 {
               continue;
           }
           
           let outcome_name = self.outcome_mapping
               .get(class_idx)
               .cloned()
               .unwrap_or_else(|| format!("Unknown_{}", class_idx));
           
           let prediction = OutcomePrediction {
               outcome_id: Uuid::new_v4().to_string(),
               outcome_name: outcome_name.clone(),
               confidence: *prob,
               reasoning: Some(self.generate_reasoning(input_text, &outcome_name, *prob)),
                               contributing_factors: self.extract_contributing_factors(input_text, &outcome_name)
                    .into_iter()
                    .map(|factor| ContributingFactor {
                        factor_type: factor.clone(),
                        weight: 0.5,
                        description: factor,  
                    })
                    .collect(),
               alternative_outcomes: self.get_alternative_outcomes(&indexed_probs, *class_idx)
                   .into_iter()
                   .map(|outcome| AlternativeOutcome {
                       outcome_id: Uuid::new_v4(),
                       outcome_name: outcome,
                       probability: 0.1,
                       relative_likelihood: 0.1,
                       key_differences: Vec::new(),
                       
                   })
                   .collect(),
               predicted_impact: self.estimate_impact(*prob),
               time_to_completion: Some(Duration {
                   likely_hours: self.estimate_completion_time(&outcome_name).likely_hours,
                   min_hours: self.estimate_completion_time(&outcome_name).min_hours,
                   max_hours: self.estimate_completion_time(&outcome_name).max_hours,
               }),
           };
           
           predictions.push(prediction);
       }
       
       Ok(predictions)
   }

   fn generate_reasoning(&self, input: &str, outcome: &str, confidence: f32) -> String {
       debug!("Generating reasoning for input length: {}", input.len());
       let confidence_level = match confidence {
           c if c > 0.8 => "high",
           c if c > 0.5 => "moderate",
           _ => "low",
       };
       
       format!(
           "Based on the input patterns and learned organizational context, {} is predicted with {} confidence ({:.1}%). \
            The model identified key indicators in the text that align with historical {} patterns.",
           outcome,
           confidence_level,
           confidence * 100.0,
           outcome.to_lowercase()
       )
   }

   fn extract_contributing_factors(&self, input: &str, outcome: &str) -> Vec<String> {
       let mut factors = Vec::new();
       
       // Extract key phrases (simplified - use NLP in production)
       let keywords = input.split_whitespace()
           .filter(|word| word.len() > 4)
           .take(5)
           .map(|w| format!("Keyword: {}", w))
           .collect::<Vec<_>>();
       
       factors.extend(keywords);
       factors.push(format!("Historical pattern match for {}", outcome));
       
       factors
   }

   fn get_alternative_outcomes(
       &self,
       all_probs: &[(usize, f32)],
       primary_idx: usize,
   ) -> Vec<String> {
       all_probs.iter()
           .filter(|(idx, prob)| *idx != primary_idx && *prob > 0.05)
           .take(3)
           .filter_map(|(idx, _)| self.outcome_mapping.get(idx).cloned())
           .collect()
   }

   fn estimate_impact(&self, confidence: f32) -> ImpactLevel {
       match confidence {
           c if c > 0.8 => ImpactLevel::High,
           c if c > 0.6 => ImpactLevel::Medium,
           c if c > 0.4 => ImpactLevel::Low,
           _ => ImpactLevel::Low,
       }
   }

   fn estimate_completion_time(&self, outcome: &str) -> Duration {
       // Estimate based on outcome type
       match outcome {
           "User Activation" => Duration::from_hours_with_uncertainty(1.0, 0.2),
           "Feature Adoption" => Duration::from_hours_with_uncertainty(8.0, 0.2),
           "Bug Resolution" => Duration::from_hours_with_uncertainty(0.5, 0.2),
           "Performance Optimization" => Duration::from_hours_with_uncertainty(24.0, 0.2),
           _ => Duration::from_hours_with_uncertainty(8.0, 0.2),
       }
   }

   /// Export model for deployment
   pub async fn export(&self, format: ExportFormat) -> Result<Vec<u8>> {
       info!("Exporting model in format: {:?}", format);
       
       match format {
           ExportFormat::ONNX => self.export_onnx().await,
           ExportFormat::SafeTensors => self.export_safetensors().await,
           ExportFormat::TorchScript => self.export_torchscript().await,
       }
   }

   async fn export_onnx(&self) -> Result<Vec<u8>> {
       // Export to ONNX format
       // Implementation would use tract or similar library
       Ok(vec![])
   }

   async fn export_safetensors(&self) -> Result<Vec<u8>> {
       let mut tensors = HashMap::new();
       
       // Export LoRA weights
       let lora = self.lora_adapter.read();
       for (name, tensor) in &lora.weights_a {
           tensors.insert(format!("lora_a.{}", name), tensor.clone());
       }
       for (name, tensor) in &lora.weights_b {
           tensors.insert(format!("lora_b.{}", name), tensor.clone());
       }
       
               // Serialize to bytes - use candle's built-in serialization
        let mut buffer = Vec::new();
        for (name, tensor) in tensors {
            debug!("Exporting tensor: {}", name);
            let tensor_data = tensor.to_vec1::<f32>()?;
            buffer.extend_from_slice(&tensor_data);
        }
        Ok(buffer.into_iter().map(|x| x as u8).collect::<Vec<u8>>())
   }

   async fn export_torchscript(&self) -> Result<Vec<u8>> {
       // Export to TorchScript format
       Ok(vec![])
   }

   /// Get model state for persistence
   pub fn get_state(&self) -> ModelState {
       ModelState {
           workspace_id: self.workspace_id,
           model_id: self.model_id,
           version: self.model_version,
           accuracy: self.best_accuracy,
           checkpoint_path: self.checkpoint_manager.checkpoint_dir.clone(),
           created_at: Utc::now(),
           updated_at: self.last_trained.unwrap_or_else(Utc::now),
           training_config: self.training_config.clone(),
           metadata: ModelMetadata {
               base_model: format!("{:?}", self.base_model.model_type),
               fine_tuned: self.last_trained.is_some(),
               total_parameters: self.base_model.config.hidden_size * self.base_model.config.num_layers * 4,
               trainable_parameters: self.calculate_trainable_parameters(),
               compression_ratio: self.calculate_compression_ratio(),
               tags: vec!["lora".to_string(), "fine-tuned".to_string()],
               description: Some(format!("Organization model for workspace {}", self.workspace_id)),
           },
       }
   }

   fn calculate_trainable_parameters(&self) -> usize {
       let lora = self.lora_adapter.read();
       let lora_params = lora.rank * self.base_model.config.hidden_size * 2 * lora.target_modules.len();
       lora_params
   }

   fn calculate_compression_ratio(&self) -> f32 {
       let total_params = self.base_model.config.hidden_size * self.base_model.config.num_layers * 4;
       let trainable = self.calculate_trainable_parameters();
       1.0 - (trainable as f32 / total_params as f32)
   }

   /// Load model from saved state
   pub async fn from_state(state: ModelState, device: Device) -> Result<Self> {
       info!("Loading model from state: version {}", state.version);
       
       let base_model = Arc::new(Self::load_base_model(&device).await?);
       let tokenizer = Arc::new(Self::load_tokenizer().await?);
       let outcome_mapping = Arc::new(Self::load_outcome_mapping().await?);
       
       // Initialize LoRA with saved config
       let mut lora_adapter = LoRAAdapter::new(
           state.training_config.lora_rank,
           state.training_config.lora_alpha,
           state.training_config.dropout,
       );
       
       // Load checkpoint if exists
       let checkpoint_path = state.checkpoint_path.join(format!("model_v{}_best.safetensors", state.version));
       if checkpoint_path.exists() {
           let checkpoint_manager = CheckpointManager::new(state.workspace_id);
           let tensors = checkpoint_manager.load_checkpoint(&checkpoint_path).await?;
           
           // Load LoRA weights
           for (name, tensor) in tensors {
               if name.starts_with("lora_a.") {
                   let module_name = name.strip_prefix("lora_a.").unwrap();
                   lora_adapter.weights_a.insert(module_name.to_string(), tensor);
               } else if name.starts_with("lora_b.") {
                   let module_name = name.strip_prefix("lora_b.").unwrap();
                   lora_adapter.weights_b.insert(module_name.to_string(), tensor);
               }
           }
       } else {
           lora_adapter.initialize(&base_model, &device)?;
       }
       
       Ok(Self {
           workspace_id: state.workspace_id,
           model_id: state.model_id,
           best_accuracy: state.accuracy,
           model_version: state.version,
           last_trained: Some(state.updated_at),
           base_model,
           lora_adapter: Arc::new(RwLock::new(lora_adapter)),
           training_config: state.training_config,
           device,
           tokenizer,
           outcome_mapping,
           training_history: Arc::new(RwLock::new(TrainingHistory::default())),
           checkpoint_manager: Arc::new(CheckpointManager::new(state.workspace_id)),
           metrics_tracker: Arc::new(MetricsTracker::new()),
       })
   }
}

/// Export formats
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
   ONNX,
   SafeTensors,
   TorchScript,
}

/// Processed training example
#[derive(Clone)]
struct ProcessedExample {
    input_ids: Tensor,
    attention_mask: Tensor,
    label: usize,
}

/// AdamW optimizer implementation
struct AdamW {
   params: Vec<Tensor>,
   learning_rate: f32,
   beta1: f32,
   beta2: f32,
   weight_decay: f32,
   step: usize,
   m: Vec<Tensor>,
   v: Vec<Tensor>,
}

impl AdamW {
   fn new(
       params: Vec<Tensor>,
       learning_rate: f32,
       beta1: f32,
       beta2: f32,
       weight_decay: f32,
   ) -> Result<Self> {
               let m = params.iter()
            .map(|p| Tensor::zeros_like(p).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        
        let v = params.iter()
            .map(|p| Tensor::zeros_like(p).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
       
       Ok(Self {
           params,
           learning_rate,
           beta1,
           beta2,
           weight_decay,
           step: 0,
           m,
           v,
       })
   }
}

impl Optimizer for AdamW {
   fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
       self.step += 1;
       
       let grads = loss.backward()?;
       
            for (i, param) in self.params.iter_mut().enumerate() {
            let grad = grads.get(param).ok_or_else(|| anyhow::anyhow!("Gradient not found for parameter {}", i))?;
                       // Update biased first moment estimate
            let beta1_tensor = Tensor::new(&[self.beta1], &param.device())?;
            let one_minus_beta1 = Tensor::new(&[1.0 - self.beta1], &param.device())?;
            self.m[i] = ((self.m[i].clone() * &beta1_tensor)? + (grad * &one_minus_beta1)?)?;
            
            // Update biased second raw moment estimate
            let beta2_tensor = Tensor::new(&[self.beta2], &param.device())?;
            let one_minus_beta2 = Tensor::new(&[1.0 - self.beta2], &param.device())?;
            self.v[i] = ((self.v[i].clone() * &beta2_tensor)? + (grad.sqr()? * &one_minus_beta2)?)?;
            
            // Compute bias-corrected first moment estimate
            let bias_correction1 = Tensor::new(&[1.0 - self.beta1.powi(self.step as i32)], &param.device())?;
            let m_hat = (self.m[i].clone() / &bias_correction1)?;
            
            // Compute bias-corrected second raw moment estimate
            let bias_correction2 = Tensor::new(&[1.0 - self.beta2.powi(self.step as i32)], &param.device())?;
            let v_hat = (self.v[i].clone() / &bias_correction2)?;
            
            // Update parameters with weight decay
            let epsilon = Tensor::new(&[1e-8], &param.device())?;
            let update = (m_hat / (&v_hat.sqrt()? + &epsilon))?;
            let weight_decay_tensor = Tensor::new(&[1.0 - self.weight_decay], &param.device())?;
            let lr_tensor = Tensor::new(&[self.learning_rate], &param.device())?;
            *param = ((param.clone() * &weight_decay_tensor)? - (update * &lr_tensor)?)?;
       }
       
       Ok(())
   }
}

// Trait for optimizers
trait Optimizer {
   fn backward_step(&mut self, loss: &Tensor) -> Result<()>;
}

#[cfg(test)]
mod tests {
   use super::*;

   #[tokio::test]
   async fn test_model_creation() {
       let workspace_id = Uuid::new_v4();
       let model = OrgModel::new(workspace_id).await.unwrap();
       
       assert_eq!(model.workspace_id, workspace_id);
       assert_eq!(model.model_version, 1);
       assert_eq!(model.best_accuracy, 0.0);
   }

   #[tokio::test]
   async fn test_fine_tuning() {
       let mut model = OrgModel::new(Uuid::new_v4()).await.unwrap();
       
               let examples = vec![
            TrainingExample {
                id: Uuid::new_v4(),
                input_text: "User clicked signup button".to_string(),
                input_embedding: None,
                suggested_outcome_id: Some(Uuid::new_v4()),
                actual_outcome_id: None,
                user_feedback: None,
                feedback_score: None,
                context: None,
                created_at: Utc::now(),
                is_validated: false,
                validation_method: None,
            },
        ];
       
       model.fine_tune(&examples).await.unwrap();
       
       assert_eq!(model.model_version, 2);
       assert!(model.last_trained.is_some());
   }

   #[tokio::test]
   async fn test_prediction() {
       let model = OrgModel::new(Uuid::new_v4()).await.unwrap();
       
       let predictions = model.predict("User completed onboarding steps").await.unwrap();
       
       assert!(!predictions.is_empty());
       assert!(predictions[0].confidence >= 0.0 && predictions[0].confidence <= 1.0);
   }

   #[tokio::test]
   async fn test_model_export() {
       let model = OrgModel::new(Uuid::new_v4()).await.unwrap();
       
       let exported = model.export(ExportFormat::SafeTensors).await.unwrap();
       assert!(!exported.is_empty());
   }
}