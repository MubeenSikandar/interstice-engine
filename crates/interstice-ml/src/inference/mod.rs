//interstice-ml/src/inference/mod.rs
use crate::types::OutcomePrediction;
use anyhow::Result;
use candle_transformers::models::bert::BertModel;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use ndarray::{Array, CowArray};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;
use uuid::Uuid;
use std::{sync::Arc};
use tracing::{info};

mod engine;
mod bandit;
mod cache;

pub use engine::{OutcomeEngine, EngineConfig};
pub use bandit::ThompsonSamplingBandit;
pub use cache::LRUCache;

/// Outcome predictor using ONNX models
pub struct OutcomePredictor {
    session: Option<Session>,
    model_path: Option<String>,
    device: Device,
}

impl OutcomePredictor {
    /// Create a new predictor with ONNX model
    pub async fn new(model_path: &str) -> Result<Self> {
        let device = Device::Cpu; // Start with CPU, can be upgraded to GPU later
        
        let session = Self::load_onnx_model(model_path).await?;
        
        Ok(Self {
            session: Some(session),
            model_path: Some(model_path.to_string()),
            device,
        })
    }

    /// Create a lazy predictor that will load the model when first used
    pub fn connect_lazy() -> Result<Self> {
        Ok(Self {
            session: None,
            model_path: None,
            device: Device::Cpu,
        })
    }

    /// Load ONNX model from file
    async fn load_onnx_model(model_path: &str) -> Result<Session> {
        info!("Loading ONNX model from: {}", model_path);
        
        let environment = Environment::builder()
            .with_name("interstice-ml")
            .with_log_level(ort::LoggingLevel::Warning)
            .build()?;

        let session = SessionBuilder::new(&Arc::new(environment))?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_execution_providers(&[ExecutionProvider::CPU(Default::default())])?
            .with_model_from_file(model_path)?;

        info!("ONNX model loaded successfully");
        Ok(session)
    }

    /// Predict outcomes for given text and artifacts
    pub async fn predict(
        &self,
        embedding: Vec<f32>,
        artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<OutcomePrediction>> {
        // If no session is loaded, try to load it
        let session = if let Some(session) = &self.session {
            session
        } else if let Some(model_path) = &self.model_path {
            let _session = Self::load_onnx_model(model_path).await?;
            // Note: This is a temporary solution. In production, you'd want to store the session
            // or use a proper model manager
            return Err(anyhow::anyhow!("Model not loaded"));
        } else {
            return Err(anyhow::anyhow!("No model path configured"));
        };

        // Prepare input features
        let input_features = self.prepare_input_features(embedding, artifacts)?;
        
        // Run inference
        let predictions = self.run_inference(session, input_features).await?;
        
        // Convert to outcome predictions
        let outcome_predictions = self.convert_predictions(predictions)?;
        
        Ok(outcome_predictions)
    }

    /// Prepare input features for the model
    fn prepare_input_features(
        &self,
        embedding: Vec<f32>,
        artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<f32>> {
        let mut features = embedding;
        
        // Add artifact type features
        for artifact in artifacts {
            let artifact_features = self.extract_artifact_features(artifact);
            features.extend(artifact_features);
        }
        
        // Pad or truncate to expected input size (e.g., 768 for BERT)
        let target_size = 768;
        if features.len() < target_size {
            features.extend(vec![0.0; target_size - features.len()]);
        } else if features.len() > target_size {
            features.truncate(target_size);
        }
        
        Ok(features)
    }

    /// Extract features from an artifact
    fn extract_artifact_features(&self, artifact: &interstice_core::Artifact) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Platform feature (one-hot encoded) - need unique vectors for each platform
        match artifact.platform {
            interstice_core::Platform::Slack => features.extend(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::GitHub => features.extend(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Jira => features.extend(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Teams => features.extend(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Asana => features.extend(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::VSCode => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::GoogleWorkspace => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Monday => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Trello => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            interstice_core::Platform::Zoom => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            interstice_core::Platform::Figma => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            interstice_core::Platform::Notion => features.extend(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        
        // Artifact type features
        match &artifact.artifact_type {
            interstice_core::ArtifactType::PullRequest { .. } => features.extend(vec![1.0, 0.0, 0.0, 0.0, 0.0]),
            interstice_core::ArtifactType::Issue { .. } => features.extend(vec![0.0, 1.0, 0.0, 0.0, 0.0]),
            interstice_core::ArtifactType::Commit { .. } => features.extend(vec![0.0, 0.0, 1.0, 0.0, 0.0]),
            interstice_core::ArtifactType::Document { .. } => features.extend(vec![0.0, 0.0, 0.0, 1.0, 0.0]),
            interstice_core::ArtifactType::Message { .. } => features.extend(vec![0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        
        // Text length feature (normalized)
        let text_length = artifact.raw_text.len() as f32;
        features.push(text_length / 1000.0); // Normalize to 0-1 range
        
        features
    }

    /// Run inference using the ONNX model
    async fn run_inference(&self, session: &Session, features: Vec<f32>) -> Result<Vec<f32>> {
        // Convert features to ndarray
        let input_array = Array::from_shape_vec((1, features.len()), features.clone())?;
        let input_dyn = CowArray::from(input_array).into_dyn();
        let input_value = Value::from_array(session.allocator(), &input_dyn)?;
        
        // Run inference
        let outputs = session.run(vec![input_value])?;
        let output = outputs.first().ok_or_else(|| anyhow::anyhow!("No output from model"))?;
        
        // Extract output array
        let output_array = output.try_extract::<f32>()?;
        
        // Convert to candle tensor for softmax
        let output_tensor = Tensor::from_slice(
            output_array.view().as_slice().unwrap(), 
            &[output_array.view().len()], // Use view().len() instead of just len()
            &self.device
        )?;
        
        // Apply softmax to get probabilities
        let probabilities = self.apply_softmax(&output_tensor)?;
        
        Ok(probabilities.to_vec1::<f32>()?)
    }

    /// Apply softmax to get probability distribution
    fn apply_softmax(&self, logits: &Tensor) -> Result<Tensor> {
        let max_logits = logits.max(0)?;
        let shifted = (logits - &max_logits)?;
        let exp_logits = shifted.exp()?;
        let sum_exp = exp_logits.sum(0)?;
        Ok((exp_logits / &sum_exp)?)
    }

    /// Convert model predictions to outcome predictions
    fn convert_predictions(
        &self,
        predictions: Vec<f32>,
    ) -> Result<Vec<OutcomePrediction>> {
        // In a real implementation, you would:
        // 1. Map prediction indices to actual outcome IDs from the database
        // 2. Filter predictions below a confidence threshold
        // 3. Apply business logic for outcome selection
        
        let mut outcome_predictions = Vec::new();
        
        // For now, create mock predictions
        // In production, you'd query the database for available outcomes
        let mock_outcomes = vec![
            ("User Activation", 0.8),
            ("Performance Optimization", 0.6),
            ("Security Hardening", 0.7),
            ("Code Quality", 0.5),
        ];
        
        for (i, (name, base_confidence)) in mock_outcomes.into_iter().enumerate() {
            if i < predictions.len() {
                let confidence = predictions[i] * base_confidence;
                if confidence > 0.3 { // Confidence threshold
                    outcome_predictions.push(OutcomePrediction {
                        outcome_id: Uuid::new_v4().to_string(), // In production, use real outcome ID
                        outcome_name: name.to_string(),
                        confidence,
                        reasoning: Some(format!("ML model prediction with {}% confidence", (confidence * 100.0) as i32)),
                    });
                }
            }
        }
        
        Ok(outcome_predictions)
    }

    /// Get model performance metrics
    pub async fn get_model_performance(&self) -> Result<Option<crate::types::ModelMetrics>> {
        // In a real implementation, this would query the database for:
        // - Prediction accuracy
        // - User feedback scores
        // - Model drift metrics
        
        // For now, return mock metrics
        Ok(Some(crate::types::ModelMetrics {
            correct_predictions: 1062,
            accuracy: 0.85,
            precision: 0.82,
            recall: 0.88,
            f1_score: 0.85,
            total_predictions: 1250,
        }))
    }

    pub async fn predict_ml(
        &self,
        embedding: Vec<f32>,
        artifacts: &[crate::types::Artifact], // Use ML's Artifact
    ) -> Result<Vec<crate::types::OutcomePrediction>> {
        // Your existing prediction logic but using ML types
        // ... implementation ...
        Ok(vec![])
    }


}

/// Text embedding generator using BERT
pub struct TextEmbedder {
    model: Option<BertModel>,
    tokenizer: Option<Tokenizer>,
    device: Device,
}

impl TextEmbedder {
    /// Create a new embedder with BERT model
    pub async fn new(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        
        // Load BERT model and tokenizer
        let model = Self::load_bert_model(model_path, &device).await?;
        let tokenizer = Self::load_tokenizer(model_path).await?;
        
        Ok(Self {
            model: Some(model),
            tokenizer: Some(tokenizer),
            device,
        })
    }

    /// Create a lazy embedder
    pub fn connect_lazy() -> Result<Self> {
        Ok(Self {
            model: None,
            tokenizer: None,
            device: Device::Cpu,
        })
    }

    /// Load BERT model from path
    async fn load_bert_model(model_path: &str, _device: &Device) -> Result<BertModel> {
        info!("Loading BERT model from: {}", model_path);
        
        // In a real implementation, you would load the actual BERT model
        // For now, return a placeholder
        Err(anyhow::anyhow!("BERT model loading not yet implemented"))
    }

    /// Load tokenizer from path
    async fn load_tokenizer(model_path: &str) -> Result<Tokenizer> {
        info!("Loading tokenizer from: {}", model_path);
        
        // In a real implementation, you would load the actual tokenizer
        // For now, return a placeholder
        Err(anyhow::anyhow!("Tokenizer loading not yet implemented"))
    }

    /// Generate embeddings for text
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // In a real implementation, this would:
        // 1. Tokenize the input text
        // 2. Run it through the BERT model
        // 3. Extract the [CLS] token embedding or mean pooling
        
        // For now, return mock embeddings
        let mut embeddings = Vec::new();
        let mut hash = 0u32;
        
        for byte in text.bytes() {
            hash = hash.wrapping_add(byte as u32).wrapping_mul(31);
            embeddings.push((hash as f32) / (u32::MAX as f32));
            
            if embeddings.len() >= 768 {
                break;
            }
        }
        
        // Pad to 768 dimensions
        while embeddings.len() < 768 {
            embeddings.push(0.0);
        }
        
        Ok(embeddings)
    }
}