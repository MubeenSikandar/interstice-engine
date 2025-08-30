use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;
use anyhow::Result;
use tracing::{info, debug};

use crate::embeddings::Embedder;
use crate::types::OutcomePrediction;
use crate::models::OrgModel;
use interstice_core::Artifact;

pub struct OutcomePredictor {
    embedder: Arc<Embedder>,
    org_models: Arc<tokio::sync::RwLock<HashMap<Uuid, OrgModel>>>,
}

impl OutcomePredictor {
    pub async fn new() -> Result<Self> {
        let embedder = Arc::new(Embedder::new().await?);
        
        Ok(Self {
            embedder,
            org_models: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    pub fn connect_lazy() -> Result<Self> {
        let embedder = Arc::new(Embedder::connect_lazy()?);
        
        Ok(Self {
            embedder,
            org_models: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }

    pub async fn predict(
        &self,
        workspace_id: Uuid,
        embedding: Vec<f32>,
        artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<OutcomePrediction>> {
        debug!("Predicting outcomes for workspace {} with {} artifacts", 
               workspace_id, artifacts.len());

        // Get or create org model
        let mut models = self.org_models.write().await;
        let model = models.entry(workspace_id)
            .or_insert_with(|| OrgModel::new(workspace_id));

        // Use the org model to make predictions
        let combined_text = artifacts.iter()
            .map(|a| &a.raw_text)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let predictions = model.predict(&combined_text).await?;

        info!("Generated {} predictions for workspace {}", predictions.len(), workspace_id);
        Ok(predictions)
    }

    pub async fn predict_with_context(
        &self,
        workspace_id: Uuid,
        text: &str,
        context_artifacts: &[interstice_core::Artifact],
    ) -> Result<Vec<OutcomePrediction>> {
        // Generate embedding for the input text
        let embedding = self.embedder.embed_text(text).await?;
        
        // Combine text with context artifacts
        let context_text = context_artifacts.iter()
            .map(|a| &a.raw_text)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let combined_text = format!("{} {}", text, context_text);

        // Make prediction
        self.predict(workspace_id, embedding, context_artifacts).await
    }

    pub async fn batch_predict(
        &self,
        workspace_id: Uuid,
        texts: &[String],
    ) -> Result<Vec<Vec<OutcomePrediction>>> {
        let mut all_predictions = Vec::with_capacity(texts.len());
        
        for text in texts {
            let predictions = self.predict_with_context(
                workspace_id, 
                text, 
                &[]
            ).await?;
            all_predictions.push(predictions);
        }
        
        Ok(all_predictions)
    }

    pub async fn get_model_performance(&self, workspace_id: Uuid) -> Result<Option<crate::types::ModelMetrics>> {
        let models = self.org_models.read().await;
        
        if let Some(model) = models.get(&workspace_id) {
            let metrics = model.evaluate().await?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }
}
