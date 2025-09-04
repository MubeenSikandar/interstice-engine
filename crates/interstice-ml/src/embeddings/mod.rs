//interstice-ml/src/embeddings/mod.rs

use anyhow::{Result, Context};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::{Tokenizer, PaddingParams, TruncationParams};
use tracing::{info, warn, debug, instrument};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use lru::LruCache;
use serde::{Serialize, Deserialize};
use std::num::NonZeroUsize;

// Model configurations
const DEFAULT_MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const FALLBACK_MODEL_ID: &str = "bert-base-uncased";
const EMBEDDING_DIM: usize = 384; // MiniLM outputs 384d embeddings
const MAX_SEQ_LENGTH: usize = 512;
const BATCH_SIZE: usize = 32;
const CACHE_SIZE: usize = 10000;

/// Configuration for the embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    pub model_id: String,
    pub cache_dir: PathBuf,
    pub max_seq_length: usize,
    pub batch_size: usize,
    pub use_gpu: bool,
    pub normalize_embeddings: bool,
    pub pooling_strategy: PoolingStrategy,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_id: DEFAULT_MODEL_ID.to_string(),
            cache_dir: PathBuf::from(".cache/models"),
            max_seq_length: MAX_SEQ_LENGTH,
            batch_size: BATCH_SIZE,
            use_gpu: true,
            normalize_embeddings: true,
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,
    Max,
    CLS,
    MeanSqrt,
}

/// Production-ready embedder with caching, batching, and fallback
pub struct Embedder {
    model: Arc<RwLock<Option<BertModel>>>,
    tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    device: Device,
    config: EmbedderConfig,
    cache: Arc<RwLock<LruCache<String, Vec<f32>>>>,
    model_loaded: Arc<RwLock<bool>>,
}

impl Embedder {
    /// Create new embedder with default configuration
    #[instrument(skip_all)]
    pub async fn new() -> Result<Self> {
        Self::with_config(EmbedderConfig::default()).await
    }

    /// Create embedder with custom configuration
    #[instrument(skip(config))]
    pub async fn with_config(config: EmbedderConfig) -> Result<Self> {
        let device = if config.use_gpu {
            Device::cuda_if_available(0).unwrap_or_else(|e| {
                warn!("CUDA not available: {}, falling back to CPU", e);
                Device::Cpu
            })
        } else {
            Device::Cpu
        };

        info!("Embedder using device: {:?}", device);

        let cache = LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap());

        Ok(Self {
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            device,
            config,
            cache: Arc::new(RwLock::new(cache)),
            model_loaded: Arc::new(RwLock::new(false)),
        })
    }

    /// Load model from HuggingFace or local cache
    #[instrument(skip(self))]
    pub async fn load_model(&self) -> Result<()> {
        if *self.model_loaded.read().await {
            debug!("Model already loaded");
            return Ok(());
        }

        info!("Loading model: {}", self.config.model_id);
        
        // Try to load from cache first
        let model_path = self.config.cache_dir.join(&self.config.model_id);
        
        if !model_path.exists() {
            info!("Model not in cache, downloading from HuggingFace");
            self.download_model().await
                .context("Failed to download model")?;
        }

        // Load tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {:?}", e))?;
        
        // Configure tokenizer
        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
        
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: self.config.max_seq_length,
            ..Default::default()
        }));

        // Load model weights
        let weights_path = model_path.join("model.safetensors");
        let config_path = model_path.join("config.json");
        
        let config = std::fs::read_to_string(config_path)
            .context("Failed to read model config")?;
        let bert_config: BertConfig = serde_json::from_str(&config)
            .context("Failed to parse model config")?;
        
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path], 
                DType::F32, 
                &self.device
            )?
        };
        
        let model = BertModel::load(vb, &bert_config)
            .context("Failed to load BERT model")?;

        *self.model.write().await = Some(model);
        *self.tokenizer.write().await = Some(tokenizer);
        *self.model_loaded.write().await = true;

        info!("Model loaded successfully");
        Ok(())
    }

    /// Download model from HuggingFace
    async fn download_model(&self) -> Result<()> {
        let api = Api::new()
            .context("Failed to create HuggingFace API")?;
        
        let repo = api.repo(Repo::with_revision(
            self.config.model_id.clone(),
            RepoType::Model,
            "main".to_string(),
        ));

        let model_path = self.config.cache_dir.join(&self.config.model_id);
        std::fs::create_dir_all(&model_path)
            .context("Failed to create cache directory")?;

        // Download required files
        let files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        ];

        for file in files {
            info!("Downloading {}", file);
            let content = repo.get(file).await
                .context(format!("Failed to download {}", file))?;
            
            // Copy file from downloaded location to our cache
            let file_path = model_path.join(file);
            std::fs::copy(content, file_path)
                .context(format!("Failed to save {}", file))?;
        }

        Ok(())
    }

    /// Embed a single text with caching
    #[instrument(skip(self, text))]
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.cache.read().await.peek(text) {
            debug!("Cache hit for text");
            return Ok(cached.clone());
        }

        // Ensure model is loaded
        if !*self.model_loaded.read().await {
            self.load_model().await?;
        }

        let embedding = self.embed_text_uncached(text).await?;
        
        // Cache the result
        self.cache.write().await.put(text.to_string(), embedding.clone());
        
        Ok(embedding)
    }

    /// Internal embedding without cache
    async fn embed_text_uncached(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.read().await;
        let tokenizer = self.tokenizer.read().await;
        
        let model = model.as_ref()
            .context("Model not loaded")?;
        let tokenizer = tokenizer.as_ref()
            .context("Tokenizer not loaded")?;

        // Tokenize
        let encoding = tokenizer.encode(text, true)
            .map_err(|e| anyhow::anyhow!("Failed to tokenize text: {:?}", e))?;
        
        let input_ids = Tensor::new(
            encoding.get_ids(), 
            &self.device
        )?.unsqueeze(0)?; // Add batch dimension
        
        let attention_mask = Tensor::new(
            encoding.get_attention_mask(), 
            &self.device
        )?.unsqueeze(0)?;

        // Forward pass - BertModel.forward() requires 3 arguments
        // The third argument is token_type_ids (usually zeros for single sentence)
        let token_type_ids = Tensor::zeros_like(&input_ids)?;
        let outputs = model.forward(&input_ids, &attention_mask, Some(&token_type_ids))?;
        
        // Apply pooling
        let pooled = self.apply_pooling(&outputs, &attention_mask)?;
        
        // Convert to Vec<f32>
        let embedding = pooled.squeeze(0)?.to_vec1::<f32>()?;
        
        // Normalize if configured
        let embedding = if self.config.normalize_embeddings {
            Self::normalize(&embedding)
        } else {
            embedding
        };

        Ok(embedding)
    }

    /// Batch embed multiple texts efficiently
    #[instrument(skip(self, texts))]
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure model is loaded
        if !*self.model_loaded.read().await {
            self.load_model().await?;
        }

        let mut embeddings = Vec::with_capacity(texts.len());
        
        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let batch_embeddings = self.embed_batch_internal(chunk).await?;
            embeddings.extend(batch_embeddings);
        }
        
        Ok(embeddings)
    }

    /// Internal batch embedding
    async fn embed_batch_internal(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let model = self.model.read().await;
        let tokenizer = self.tokenizer.read().await;
        
        let model = model.as_ref()
            .context("Model not loaded")?;
        let tokenizer = tokenizer.as_ref()
            .context("Tokenizer not loaded")?;

        // Tokenize batch
        let encodings = tokenizer.encode_batch(
            texts.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            true
        ).map_err(|e| anyhow::anyhow!("Failed to tokenize batch: {:?}", e))?;

        // Convert to tensors
        let input_ids: Vec<Vec<u32>> = encodings.iter()
            .map(|e| e.get_ids().to_vec())
            .collect();
        
        let attention_masks: Vec<Vec<u32>> = encodings.iter()
            .map(|e| e.get_attention_mask().to_vec())
            .collect();

        let input_ids = Tensor::new(input_ids, &self.device)?;
        let attention_mask = Tensor::new(attention_masks, &self.device)?;

        // Forward pass with token_type_ids
        let token_type_ids = Tensor::zeros_like(&input_ids)?;
        let outputs = model.forward(&input_ids, &attention_mask, Some(&token_type_ids))?;
        
        // Apply pooling and convert
        let mut embeddings = Vec::new();
        for i in 0..texts.len() {
            let output = outputs.get(i)?;
            let mask = attention_mask.get(i)?;
            
            let pooled = self.apply_pooling(&output.unsqueeze(0)?, &mask.unsqueeze(0)?)?;
            let embedding = pooled.squeeze(0)?.to_vec1::<f32>()?;
            
            let embedding = if self.config.normalize_embeddings {
                Self::normalize(&embedding)
            } else {
                embedding
            };
            
            embeddings.push(embedding);
        }

        Ok(embeddings)
    }

    /// Apply pooling strategy
    fn apply_pooling(&self, outputs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        match self.config.pooling_strategy {
            PoolingStrategy::Mean => {
                let expanded_mask = attention_mask.unsqueeze(2)?
                    .broadcast_as(outputs.shape())?;
                let masked = outputs.mul(&expanded_mask)?;
                let sum = masked.sum_keepdim(1)?;
                let count = expanded_mask.sum_keepdim(1)?;
                Ok((sum / count)?)
            },
            PoolingStrategy::CLS => {
                // Take first token (CLS token)
                Ok(outputs.narrow(1, 0, 1)?)
            },
            PoolingStrategy::Max => {
                let expanded_mask = attention_mask.unsqueeze(2)?
                    .broadcast_as(outputs.shape())?;
                let masked = outputs.where_cond(
                    &expanded_mask.eq(&Tensor::ones_like(&expanded_mask)?)?,
                    &Tensor::full(
                        f32::NEG_INFINITY,
                        outputs.shape(),
                        &self.device
                    )?
                )?;
                Ok(masked.max_keepdim(1)?)
            },
            PoolingStrategy::MeanSqrt => {
                // Mean pooling with sqrt length normalization
                let expanded_mask = attention_mask.unsqueeze(2)?
                    .broadcast_as(outputs.shape())?;
                let masked = outputs.mul(&expanded_mask)?;
                let sum = masked.sum_keepdim(1)?;
                let count = expanded_mask.sum_keepdim(1)?;
                let sqrt_count = count.sqrt()?;
                Ok((sum / sqrt_count)?)
            }
        }
    }

    /// Normalize embedding to unit length
    fn normalize(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        
        if norm > 0.0 {
            embedding.iter()
                .map(|x| x / norm)
                .collect()
        } else {
            embedding.to_vec()
        }
    }

    /// Calculate cosine similarity between two embeddings
    pub fn similarity(&self, embedding1: &[f32], embedding2: &[f32]) -> f32 {
        if embedding1.len() != embedding2.len() {
            warn!("Embedding dimension mismatch: {} vs {}", 
                  embedding1.len(), embedding2.len());
            return 0.0;
        }
        
        // If embeddings are normalized, dot product = cosine similarity
        if self.config.normalize_embeddings {
            embedding1.iter()
                .zip(embedding2.iter())
                .map(|(a, b)| a * b)
                .sum()
        } else {
            // Calculate cosine similarity
            let dot_product: f32 = embedding1.iter()
                .zip(embedding2.iter())
                .map(|(a, b)| a * b)
                .sum();
            
            let norm1: f32 = embedding1.iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            
            let norm2: f32 = embedding2.iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            
            if norm1 == 0.0 || norm2 == 0.0 {
                return 0.0;
            }
            
            dot_product / (norm1 * norm2)
        }
    }

    /// Find most similar text from a collection
    pub async fn find_most_similar(
        &self,
        query: &str,
        candidates: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Embed query
        let query_embedding = self.embed_text(query).await?;
        
        // Embed all candidates
        let candidate_embeddings = self.embed_batch(candidates).await?;
        
        // Calculate similarities
        let mut similarities: Vec<(usize, f32)> = candidate_embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| {
                (idx, self.similarity(&query_embedding, emb))
            })
            .collect();
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top-k
        Ok(similarities.into_iter().take(top_k).collect())
    }

    /// Clear the embedding cache
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
        info!("Embedding cache cleared");
    }

    /// Get cache statistics
    pub async fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.read().await;
        (cache.len(), cache.cap().get())
    }

    /// Preload embeddings for common artifacts
    pub async fn preload_artifact_embeddings(&self) -> Result<()> {
        let common_prefixes = vec![
            "PR #", "Issue #", "Commit", "Document", "Meeting",
            "Sprint", "Feature", "Bug", "Task", "Epic"
        ];
        
        // Fix: Use reference to avoid moving
        for prefix in &common_prefixes {
            self.embed_text(prefix).await?;
        }
        
        info!("Preloaded {} common artifact embeddings", common_prefixes.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedder_creation() {
        let embedder = Embedder::new().await.unwrap();
        assert!(!*embedder.model_loaded.read().await);
    }

    #[tokio::test]
    async fn test_similarity_calculation() {
        let embedder = Embedder::new().await.unwrap();
        
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        let emb3 = vec![0.0, 1.0, 0.0];
        
        assert!((embedder.similarity(&emb1, &emb2) - 1.0).abs() < 0.001);
        assert!((embedder.similarity(&emb1, &emb3) - 0.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_normalization() {
        let embedding = vec![3.0, 4.0];
        let normalized = Embedder::normalize(&embedding);
        
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}