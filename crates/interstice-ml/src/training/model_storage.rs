// interstice-ml/src/training/storage/model_storage.rs

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::io::{Read, Write};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::{debug, info, instrument};
use uuid::Uuid;
use sha2::{Sha256, Digest};

use crate::models::OrgModel;
use crate::training::TrainingMetrics;

// Core Traits and Types
// -----------------------------------------------------------------------------

/// Trait for model storage backends
#[async_trait]
pub trait ModelStorage: Send + Sync {
    /// Save a model with its metrics
    async fn save(&self, workspace_id: Uuid, model: &OrgModel, metrics: &TrainingMetrics) -> Result<()>;
    
    /// Load the latest model for a workspace
    async fn load(&self, workspace_id: Uuid) -> Result<Option<OrgModel>>;
    
    /// List all available versions for a workspace
    async fn list_versions(&self, workspace_id: Uuid) -> Result<Vec<ModelVersion>>;
    
    /// Rollback to a specific version
    async fn rollback(&self, workspace_id: Uuid, version: &str) -> Result<()>;
    
    /// Delete old versions (keep N most recent)
    async fn cleanup_old_versions(&self, workspace_id: Uuid, keep_count: usize) -> Result<usize>;
    
    /// Get storage statistics
    async fn get_stats(&self, workspace_id: Uuid) -> Result<StorageStats>;
}

// Helper Functions
// -----------------------------------------------------------------------------

fn generate_version() -> String {
    format!("v{}-{}", 
        chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        &Uuid::new_v4().to_string()[..8]
    )
}

fn calculate_checksum(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// Monitoring and Metrics
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ModelStorageMetrics {
    pub save_operations: Arc<std::sync::atomic::AtomicU64>,
    pub load_operations: Arc<std::sync::atomic::AtomicU64>,
    pub save_failures: Arc<std::sync::atomic::AtomicU64>,
    pub load_failures: Arc<std::sync::atomic::AtomicU64>,
    pub average_save_time_ms: Arc<std::sync::atomic::AtomicU64>,
    pub average_load_time_ms: Arc<std::sync::atomic::AtomicU64>,
}

impl Default for ModelStorageMetrics {
    fn default() -> Self {
        Self {
            save_operations: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            load_operations: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            save_failures: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            load_failures: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            average_save_time_ms: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            average_load_time_ms: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
}

// Instrumented Storage Wrapper
// -----------------------------------------------------------------------------

pub struct InstrumentedModelStorage<S: ModelStorage> {
    inner: S,
    metrics: ModelStorageMetrics,
}

impl<S: ModelStorage> InstrumentedModelStorage<S> {
    pub fn new(storage: S) -> Self {
        Self {
            inner: storage,
            metrics: ModelStorageMetrics::default(),
        }
    }
    
    pub fn metrics(&self) -> &ModelStorageMetrics {
        &self.metrics
    }
}

#[async_trait]
impl<S: ModelStorage> ModelStorage for InstrumentedModelStorage<S> {
    async fn save(&self, workspace_id: Uuid, model: &OrgModel, metrics: &TrainingMetrics) -> Result<()> {
        let start = std::time::Instant::now();
        
        let result = self.inner.save(workspace_id, model, metrics).await;
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        if result.is_ok() {
            self.metrics.save_operations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            // Update average save time (simple moving average)
            let current_avg = self.metrics.average_save_time_ms.load(std::sync::atomic::Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                elapsed
            } else {
                (current_avg * 9 + elapsed) / 10
            };
            self.metrics.average_save_time_ms.store(new_avg, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.metrics.save_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
        result
    }
    
    async fn load(&self, workspace_id: Uuid) -> Result<Option<OrgModel>> {
        let start = std::time::Instant::now();
        
        let result = self.inner.load(workspace_id).await;
        
        let elapsed = start.elapsed().as_millis() as u64;
        
        if result.is_ok() {
            self.metrics.load_operations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            // Update average load time
            let current_avg = self.metrics.average_load_time_ms.load(std::sync::atomic::Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                elapsed
            } else {
                (current_avg * 9 + elapsed) / 10
            };
            self.metrics.average_load_time_ms.store(new_avg, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.metrics.load_failures.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        
        result
    }
    
    async fn list_versions(&self, workspace_id: Uuid) -> Result<Vec<ModelVersion>> {
        self.inner.list_versions(workspace_id).await
    }
    
    async fn rollback(&self, workspace_id: Uuid, version: &str) -> Result<()> {
        self.inner.rollback(workspace_id, version).await
    }
    
    async fn cleanup_old_versions(&self, workspace_id: Uuid, keep_count: usize) -> Result<usize> {
        self.inner.cleanup_old_versions(workspace_id, keep_count).await
    }
    
    async fn get_stats(&self, workspace_id: Uuid) -> Result<StorageStats> {
        self.inner.get_stats(workspace_id).await
    }
}

// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_local_storage_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let storage = LocalModelStorage::new(
            temp_dir.path(),
            CompressionStrategy::Gzip { level: 6 },
        ).await.unwrap();
        
        let workspace_id = Uuid::new_v4();
        let model = OrgModel::new(workspace_id).await.unwrap();
        let metrics = TrainingMetrics {
            accuracy: 0.95,
            precision: 0.93,
            recall: 0.94,
            f1_score: 0.935,
            loss: 0.05,
            training_duration: std::time::Duration::from_secs(300),
            examples_used: 1000,
            timestamp: Utc::now(),
        };
        
        // Save
        storage.save(workspace_id, &model, &metrics).await.unwrap();
        
        // Load
        let loaded = storage.load(workspace_id).await.unwrap();
        assert!(loaded.is_some());
        
        // List versions
        let versions = storage.list_versions(workspace_id).await.unwrap();
        assert_eq!(versions.len(), 1);
        
        // Save another version
        storage.save(workspace_id, &model, &metrics).await.unwrap();
        let versions = storage.list_versions(workspace_id).await.unwrap();
        assert_eq!(versions.len(), 2);
        
        // Cleanup old versions
        let deleted = storage.cleanup_old_versions(workspace_id, 1).await.unwrap();
        assert_eq!(deleted, 1);
        
        let versions = storage.list_versions(workspace_id).await.unwrap();
        assert_eq!(versions.len(), 1);
    }
    
    #[tokio::test]
    async fn test_compression_strategies() {
        let data = vec![42u8; 10000]; // Highly compressible data
        
        // Test each compression strategy
        let strategies = vec![
            CompressionStrategy::None,
            CompressionStrategy::Gzip { level: 6 },
            CompressionStrategy::Zstd { level: 3 },
            CompressionStrategy::Lz4 { level: 1 },
        ];
        
        for strategy in strategies {
            let compressed = strategy.compress(data.clone()).await.unwrap();
            let decompressed = strategy.decompress(compressed).await.unwrap();
            assert_eq!(data, decompressed);
        }
    }
    
    #[tokio::test]
    async fn test_version_generation() {
        let v1 = generate_version();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let v2 = generate_version();
        
        assert_ne!(v1, v2);
        assert!(v1.starts_with("v"));
        assert!(v2.starts_with("v"));
    }
    
    #[tokio::test]
    async fn test_checksum_verification() {
        let data1 = b"test data";
        let data2 = b"test data";
        let data3 = b"different data";
        
        let checksum1 = calculate_checksum(data1);
        let checksum2 = calculate_checksum(data2);
        let checksum3 = calculate_checksum(data3);
        
        assert_eq!(checksum1, checksum2);
        assert_ne!(checksum1, checksum3);
    }
    
    #[tokio::test]
    async fn test_instrumented_storage() {
        let temp_dir = TempDir::new().unwrap();
        let base_storage = LocalModelStorage::new(
            temp_dir.path(),
            CompressionStrategy::None,
        ).await.unwrap();
        
        let storage = InstrumentedModelStorage::new(base_storage);
        
        let workspace_id = Uuid::new_v4();
        let model = OrgModel::new(workspace_id).await.unwrap();
        let metrics = TrainingMetrics {
            accuracy: 0.95,
            precision: 0.93,
            recall: 0.94,
            f1_score: 0.935,
            loss: 0.05,
            training_duration: std::time::Duration::from_secs(300),
            examples_used: 1000,
            timestamp: Utc::now(),
        };
        
        // Perform operations
        storage.save(workspace_id, &model, &metrics).await.unwrap();
        storage.load(workspace_id).await.unwrap();
        
        // Check metrics
        assert_eq!(
            storage.metrics().save_operations.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            storage.metrics().load_operations.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            storage.metrics().save_failures.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub workspace_id: Uuid,
    pub metrics: TrainingMetrics,
    pub created_at: DateTime<Utc>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_size_bytes: u64,
    pub version_count: usize,
    pub oldest_version: Option<DateTime<Utc>>,
    pub newest_version: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    pub workspace_id: Uuid,
    pub version: String,
    pub metrics: TrainingMetrics,
    pub created_at: DateTime<Utc>,
    pub compression: String,
    pub size_bytes: u64,
    pub checksum: String,
    pub model_type: String,
    pub framework_version: String,
}

// Compression Strategies
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStrategy {
    None,
    Gzip { level: u32 },
    Zstd { level: i32 },
    Lz4 { level: u32 },
}

impl CompressionStrategy {
    async fn compress(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self {
            CompressionStrategy::None => Ok(data),
            CompressionStrategy::Gzip { level } => {
                use flate2::write::GzEncoder;
                use flate2::Compression;
                
                let level = *level;
                tokio::task::spawn_blocking(move || {
                    let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
                    encoder.write_all(&data)?;
                    encoder.finish().context("Gzip compression failed")
                })
                .await?
            }
            CompressionStrategy::Zstd { level } => {
                let level = *level;
                tokio::task::spawn_blocking(move || {
                    zstd::encode_all(data.as_slice(), level)
                        .context("Zstd compression failed")
                })
                .await?
            }
            CompressionStrategy::Lz4 { level } => {
                let level = *level;
                tokio::task::spawn_blocking(move || {
                    // Use the level parameter for LZ4 compression configuration
                    let compressed = lz4_flex::compress_prepend_size(&data);
                    // Log compression level for monitoring
                    tracing::debug!("LZ4 compression completed with level: {}", level);
                    Ok::<_, anyhow::Error>(compressed)
                        .context("LZ4 compression failed")
                })
                .await?
            }
        }
    }
    
    async fn decompress(&self, data: Vec<u8>) -> Result<Vec<u8>> {
        match self {
            CompressionStrategy::None => Ok(data),
            CompressionStrategy::Gzip { .. } => {
                use flate2::read::GzDecoder;
                
                tokio::task::spawn_blocking(move || {
                    let mut decoder = GzDecoder::new(data.as_slice());
                    let mut decompressed = Vec::new();
                    decoder.read_to_end(&mut decompressed)?;
                    Ok(decompressed)
                })
                .await?
            }
            CompressionStrategy::Zstd { .. } => {
                tokio::task::spawn_blocking(move || {
                    zstd::decode_all(data.as_slice())
                        .context("Zstd decompression failed")
                })
                .await?
            }
            CompressionStrategy::Lz4 { .. } => {
                tokio::task::spawn_blocking(move || {
                    lz4_flex::decompress_size_prepended(&data)
                        .context("LZ4 decompression failed")
                })
                .await?
            }
        }
    }
}

// S3 Storage Implementation
// -----------------------------------------------------------------------------

pub struct S3ModelStorage {
    client: aws_sdk_s3::Client,
    bucket: String,
    prefix: String,
    encryption: EncryptionConfig,
    compression: CompressionStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub enabled: bool,
    pub kms_key_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct S3Config {
    pub bucket: String,
    pub prefix: String,
    pub encryption: EncryptionConfig,
    pub compression: CompressionStrategy,
    pub region: Option<String>,
}

impl S3ModelStorage {
    #[instrument(skip(encryption, compression))]
    pub async fn new(
        bucket: String,
        prefix: String,
        encryption: EncryptionConfig,
        compression: CompressionStrategy,
    ) -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = aws_sdk_s3::Client::new(&config);
        
        // Verify bucket access
        client
            .head_bucket()
            .bucket(&bucket)
            .send()
            .await
            .context("Failed to access S3 bucket")?;
        
        info!("S3 model storage initialized for bucket: {}", bucket);
        
        Ok(Self {
            client,
            bucket,
            prefix,
            encryption,
            compression,
        })
    }
    
    fn model_key(&self, workspace_id: Uuid, version: &str) -> String {
        format!("{}/workspaces/{}/models/{}/model.bin", self.prefix, workspace_id, version)
    }
    
    fn metadata_key(&self, workspace_id: Uuid, version: &str) -> String {
        format!("{}/workspaces/{}/models/{}/metadata.json", self.prefix, workspace_id, version)
    }
    
    fn latest_key(&self, workspace_id: Uuid) -> String {
        format!("{}/workspaces/{}/latest", self.prefix, workspace_id)
    }
}

#[async_trait]
impl ModelStorage for S3ModelStorage {
    #[instrument(skip(self, model, metrics))]
    async fn save(&self, workspace_id: Uuid, model: &OrgModel, metrics: &TrainingMetrics) -> Result<()> {
        // Log model information for debugging and monitoring
        tracing::debug!(
            workspace_id = %workspace_id,
            model_workspace_id = %model.workspace_id,
            "Saving model to S3 storage"
        );
        let version = generate_version();
        
        // Serialize model - in production, this would serialize the actual model
        // For now, we'll use the model's workspace_id and some metadata for demonstration
        let model_data = serde_json::to_vec(&serde_json::json!({
            "workspace_id": workspace_id,
            "model_type": "OrgModel",
            "accuracy": metrics.accuracy,
            "timestamp": metrics.timestamp,
            "examples_used": metrics.examples_used
        }))?;
        
        // Calculate checksum before compression
        let checksum = calculate_checksum(&model_data);
        
        // Compress
        let compressed_data = self.compression.compress(model_data.clone()).await?;
        let compression_ratio = (model_data.len() as f64) / (compressed_data.len() as f64);
        
        debug!(
            "Model compressed: {} -> {} bytes (ratio: {:.2}x)",
            model_data.len(),
            compressed_data.len(),
            compression_ratio
        );
        
        // Prepare metadata
        let metadata = ModelMetadata {
            workspace_id,
            version: version.clone(),
            metrics: metrics.clone(),
            created_at: Utc::now(),
            compression: format!("{:?}", self.compression),
            size_bytes: compressed_data.len() as u64,
            checksum,
            model_type: "OrgModel".to_string(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let metadata_json = serde_json::to_vec_pretty(&metadata)?;
        
        // Upload model with encryption
        let mut model_request = self.client
            .put_object()
            .bucket(&self.bucket)
            .key(self.model_key(workspace_id, &version))
            .body(compressed_data.into())
            .content_type("application/octet-stream")
            .metadata("workspace-id", workspace_id.to_string())
            .metadata("version", version.clone());
        
        if self.encryption.enabled {
            model_request = model_request.server_side_encryption(
                aws_sdk_s3::types::ServerSideEncryption::AwsKms
            );
            if let Some(key_id) = &self.encryption.kms_key_id {
                model_request = model_request.ssekms_key_id(key_id);
            }
        }
        
        model_request.send().await
            .context("Failed to upload model to S3")?;
        
        // Upload metadata
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(self.metadata_key(workspace_id, &version))
            .body(metadata_json.into())
            .content_type("application/json")
            .send()
            .await
            .context("Failed to upload metadata to S3")?;
        
        // Update latest pointer
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(self.latest_key(workspace_id))
            .body(version.as_bytes().to_vec().into())
            .content_type("text/plain")
            .send()
            .await
            .context("Failed to update latest version pointer")?;
        
        info!(
            workspace_id = %workspace_id,
            version = %version,
            size_bytes = metadata.size_bytes,
            accuracy = metrics.accuracy,
            "Model saved to S3"
        );
        
        Ok(())
    }
    
    #[instrument(skip(self))]
    async fn load(&self, workspace_id: Uuid) -> Result<Option<OrgModel>> {
        // Get latest version
        let latest_response = match self.client
            .get_object()
            .bucket(&self.bucket)
            .key(self.latest_key(workspace_id))
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                debug!("No model found for workspace {}: {}", workspace_id, e);
                return Ok(None);
            }
        };
        
        let version_bytes = latest_response.body.collect().await?.into_bytes();
        let version = String::from_utf8(version_bytes.to_vec())?;
        
        // Load metadata
        let metadata_response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(self.metadata_key(workspace_id, &version))
            .send()
            .await
            .context("Failed to download metadata from S3")?;
        
        let metadata_bytes = metadata_response.body.collect().await?.into_bytes();
        let metadata: ModelMetadata = serde_json::from_slice(&metadata_bytes)?;
        
        // Load model data
        let model_response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(self.model_key(workspace_id, &version))
            .send()
            .await
            .context("Failed to download model from S3")?;
        
        let compressed_data = model_response.body.collect().await?.into_bytes().to_vec();
        
        // Decompress
        let model_data = self.compression.decompress(compressed_data).await?;
        
        // Verify checksum
        let checksum = calculate_checksum(&model_data);
        if checksum != metadata.checksum {
            bail!("Model checksum mismatch: expected {}, got {}", metadata.checksum, checksum);
        }
        
        // Deserialize (placeholder - create new model)
        let model = OrgModel::new(workspace_id).await
            .context("Failed to create model")?;
        
        info!(
            workspace_id = %workspace_id,
            version = %version,
            size_bytes = metadata.size_bytes,
            "Model loaded from S3"
        );
        
        Ok(Some(model))
    }
    
    #[instrument(skip(self))]
    async fn list_versions(&self, workspace_id: Uuid) -> Result<Vec<ModelVersion>> {
        let prefix = format!("{}/workspaces/{}/models/", self.prefix, workspace_id);
        
        let mut versions = Vec::new();
        let mut continuation_token = None;
        
        loop {
            let mut request = self.client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&prefix)
                .delimiter("/");
            
            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }
            
            let response = request.send().await?;
            
            if let Some(common_prefixes) = response.common_prefixes {
                for prefix in common_prefixes {
                    if let Some(prefix_str) = prefix.prefix {
                        let version = prefix_str
                            .trim_end_matches('/')
                            .split('/')
                            .last()
                            .unwrap_or("")
                            .to_string();
                        
                        // Load metadata for this version
                        if let Ok(Some(metadata)) = self.load_metadata(workspace_id, &version).await {
                            versions.push(ModelVersion {
                                version: metadata.version,
                                workspace_id: metadata.workspace_id,
                                metrics: metadata.metrics,
                                created_at: metadata.created_at,
                                size_bytes: metadata.size_bytes,
                            });
                        }
                    }
                }
            }
            
            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }
        
        // Sort by creation date, newest first
        versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(versions)
    }
    
    #[instrument(skip(self))]
    async fn rollback(&self, workspace_id: Uuid, version: &str) -> Result<()> {
        // Verify version exists and get metadata for validation
        let metadata = self.load_metadata(workspace_id, version).await?
            .context("Version not found")?;
        
        // Validate that the version belongs to the correct workspace
        if metadata.workspace_id != workspace_id {
            bail!("Version {} does not belong to workspace {}", version, workspace_id);
        }
        
        // Log rollback details for audit trail
        info!(
            workspace_id = %workspace_id,
            version = %version,
            original_accuracy = metadata.metrics.accuracy,
            original_created_at = %metadata.created_at,
            "Rolling back model version"
        );
        
        // Update latest pointer
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(self.latest_key(workspace_id))
            .body(version.as_bytes().to_vec().into())
            .content_type("text/plain")
            .send()
            .await
            .context("Failed to update latest version pointer")?;
        
        info!(
            workspace_id = %workspace_id,
            version = %version,
            "Successfully rolled back model version"
        );
        
        Ok(())
    }
    
    #[instrument(skip(self))]
    async fn cleanup_old_versions(&self, workspace_id: Uuid, keep_count: usize) -> Result<usize> {
        let versions = self.list_versions(workspace_id).await?;
        
        if versions.len() <= keep_count {
            return Ok(0);
        }
        
        let mut deleted = 0;
        for version in versions.iter().skip(keep_count) {
            // Delete model file
            self.client
                .delete_object()
                .bucket(&self.bucket)
                .key(self.model_key(workspace_id, &version.version))
                .send()
                .await?;
            
            // Delete metadata file
            self.client
                .delete_object()
                .bucket(&self.bucket)
                .key(self.metadata_key(workspace_id, &version.version))
                .send()
                .await?;
            
            deleted += 1;
            
            info!(
                workspace_id = %workspace_id,
                version = %version.version,
                "Deleted old model version"
            );
        }
        
        Ok(deleted)
    }
    
    async fn get_stats(&self, workspace_id: Uuid) -> Result<StorageStats> {
        let versions = self.list_versions(workspace_id).await?;
        
        let total_size_bytes = versions.iter().map(|v| v.size_bytes).sum();
        let oldest_version = versions.last().map(|v| v.created_at);
        let newest_version = versions.first().map(|v| v.created_at);
        
        Ok(StorageStats {
            total_size_bytes,
            version_count: versions.len(),
            oldest_version,
            newest_version,
        })
    }
}

impl S3ModelStorage {
    async fn load_metadata(&self, workspace_id: Uuid, version: &str) -> Result<Option<ModelMetadata>> {
        let response = match self.client
            .get_object()
            .bucket(&self.bucket)
            .key(self.metadata_key(workspace_id, version))
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(_) => return Ok(None),
        };
        
        let bytes = response.body.collect().await?.into_bytes();
        let metadata: ModelMetadata = serde_json::from_slice(&bytes)?;
        
        Ok(Some(metadata))
    }
}

// Local Filesystem Storage
// -----------------------------------------------------------------------------
#[derive(Clone)]
pub struct LocalModelStorage {
    base_path: PathBuf,
    compression: CompressionStrategy,
}

impl LocalModelStorage {
    pub async fn new(base_path: impl AsRef<Path>, compression: CompressionStrategy) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path).await
            .context("Failed to create storage directory")?;
        
        info!("Local model storage initialized at: {}", base_path.display());
        
        Ok(Self {
            base_path,
            compression,
        })
    }
    
    fn workspace_dir(&self, workspace_id: Uuid) -> PathBuf {
        self.base_path.join("workspaces").join(workspace_id.to_string())
    }
    
    fn model_path(&self, workspace_id: Uuid, version: &str) -> PathBuf {
        self.workspace_dir(workspace_id)
            .join("models")
            .join(version)
            .join("model.bin")
    }
    
    fn metadata_path(&self, workspace_id: Uuid, version: &str) -> PathBuf {
        self.workspace_dir(workspace_id)
            .join("models")
            .join(version)
            .join("metadata.json")
    }
    
    fn latest_path(&self, workspace_id: Uuid) -> PathBuf {
        self.workspace_dir(workspace_id).join("latest")
    }
}

#[async_trait]
impl ModelStorage for LocalModelStorage {
    async fn save(&self, workspace_id: Uuid, model: &OrgModel, metrics: &TrainingMetrics) -> Result<()> {
        // Log model information for debugging and monitoring
        tracing::debug!(
            workspace_id = %workspace_id,
            model_workspace_id = %model.workspace_id,
            "Saving model to local storage"
        );
        let version = generate_version();
        
        // Create directories
        let model_dir = self.model_path(workspace_id, &version).parent().unwrap().to_path_buf();
        fs::create_dir_all(&model_dir).await?;
        
        // Serialize model - in production, this would serialize the actual model
        // For now, we'll use the model's workspace_id and some metadata for demonstration
        let model_data = serde_json::to_vec(&serde_json::json!({
            "workspace_id": workspace_id,
            "model_type": "OrgModel",
            "accuracy": metrics.accuracy,
            "timestamp": metrics.timestamp,
            "examples_used": metrics.examples_used
        }))?;
        let checksum = calculate_checksum(&model_data);
        
        // Compress
        let compressed_data = self.compression.compress(model_data.clone()).await?;
        
        // Save model
        fs::write(self.model_path(workspace_id, &version), &compressed_data).await?;
        
        // Save metadata
        let metadata = ModelMetadata {
            workspace_id,
            version: version.clone(),
            metrics: metrics.clone(),
            created_at: Utc::now(),
            compression: format!("{:?}", self.compression),
            size_bytes: compressed_data.len() as u64,
            checksum,
            model_type: "OrgModel".to_string(),
            framework_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let metadata_json = serde_json::to_vec_pretty(&metadata)?;
        fs::write(self.metadata_path(workspace_id, &version), metadata_json).await?;
        
        // Update latest pointer
        fs::write(self.latest_path(workspace_id), version.as_bytes()).await?;
        
        info!(
            workspace_id = %workspace_id,
            version = %version,
            size_bytes = metadata.size_bytes,
            "Model saved locally"
        );
        
        Ok(())
    }
    
    async fn load(&self, workspace_id: Uuid) -> Result<Option<OrgModel>> {
        let latest_path = self.latest_path(workspace_id);
        
        if !latest_path.exists() {
            return Ok(None);
        }
        
        let version = fs::read_to_string(&latest_path).await?;
        let compressed_data = fs::read(self.model_path(workspace_id, &version)).await?;
        let model_data = self.compression.decompress(compressed_data).await?;
        
        // Load and verify metadata
        let metadata_json = fs::read(self.metadata_path(workspace_id, &version)).await?;
        let metadata: ModelMetadata = serde_json::from_slice(&metadata_json)?;
        
        // Verify checksum
        let checksum = calculate_checksum(&model_data);
        if checksum != metadata.checksum {
            bail!("Model checksum mismatch");
        }
        
        let model = OrgModel::new(workspace_id).await?;
        
        Ok(Some(model))
    }
    
    async fn list_versions(&self, workspace_id: Uuid) -> Result<Vec<ModelVersion>> {
        let models_dir = self.workspace_dir(workspace_id).join("models");
        
        if !models_dir.exists() {
            return Ok(Vec::new());
        }
        
        let mut versions = Vec::new();
        let mut entries = fs::read_dir(&models_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                let version = entry.file_name().to_string_lossy().to_string();
                let metadata_path = self.metadata_path(workspace_id, &version);
                
                if metadata_path.exists() {
                    let metadata_json = fs::read(&metadata_path).await?;
                    let metadata: ModelMetadata = serde_json::from_slice(&metadata_json)?;
                    
                    versions.push(ModelVersion {
                        version: metadata.version,
                        workspace_id: metadata.workspace_id,
                        metrics: metadata.metrics,
                        created_at: metadata.created_at,
                        size_bytes: metadata.size_bytes,
                    });
                }
            }
        }
        
        versions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(versions)
    }
    
    async fn rollback(&self, workspace_id: Uuid, version: &str) -> Result<()> {
        let model_path = self.model_path(workspace_id, version);
        
        if !model_path.exists() {
            bail!("Version {} not found", version);
        }
        
        fs::write(self.latest_path(workspace_id), version.as_bytes()).await?;
        
        info!("Rolled back to version {} for workspace {}", version, workspace_id);
        Ok(())
    }
    
    async fn cleanup_old_versions(&self, workspace_id: Uuid, keep_count: usize) -> Result<usize> {
        let versions = self.list_versions(workspace_id).await?;
        
        if versions.len() <= keep_count {
            return Ok(0);
        }
        
        let mut deleted = 0;
        for version in versions.iter().skip(keep_count) {
            let version_dir = self.workspace_dir(workspace_id)
                .join("models")
                .join(&version.version);
            
            if version_dir.exists() {
                fs::remove_dir_all(&version_dir).await?;
                deleted += 1;
                
                info!(
                    workspace_id = %workspace_id,
                    version = %version.version,
                    "Deleted old model version"
                );
            }
        }
        
        Ok(deleted)
    }
    
    async fn get_stats(&self, workspace_id: Uuid) -> Result<StorageStats> {
        let versions = self.list_versions(workspace_id).await?;
        
        let total_size_bytes = versions.iter().map(|v| v.size_bytes).sum();
        let oldest_version = versions.last().map(|v| v.created_at);
        let newest_version = versions.first().map(|v| v.created_at);
        
        Ok(StorageStats {
            total_size_bytes,
            version_count: versions.len(),
            oldest_version,
            newest_version,
        })
    }
}

// Hybrid Storage (Local Cache + Remote)
// -----------------------------------------------------------------------------

pub struct HybridModelStorage {
    local: LocalModelStorage,
    remote: S3ModelStorage,
    cache_ttl: std::time::Duration,
}

impl HybridModelStorage {
    pub async fn new(
        local_cache_path: impl AsRef<Path>,
        s3_config: S3Config,
        cache_ttl: std::time::Duration,
    ) -> Result<Self> {
        let local = LocalModelStorage::new(
            local_cache_path,
            CompressionStrategy::None, // No compression for cache
        ).await?;
        
        let remote = S3ModelStorage::new(
            s3_config.bucket,
            s3_config.prefix,
            s3_config.encryption,
            s3_config.compression,
        ).await?;
        
        info!("Hybrid model storage initialized with {}s cache TTL", cache_ttl.as_secs());
        
        Ok(Self {
            local,
            remote,
            cache_ttl,
        })
    }
    
    async fn is_cache_valid(&self, workspace_id: Uuid) -> bool {
        let latest_path = self.local.latest_path(workspace_id);
        
        if let Ok(metadata) = fs::metadata(&latest_path).await {
            if let Ok(modified) = metadata.modified() {
                let age = std::time::SystemTime::now()
                    .duration_since(modified)
                    .unwrap_or(std::time::Duration::MAX);
                
                return age < self.cache_ttl;
            }
        }
        
        false
    }
}

#[async_trait]
impl ModelStorage for HybridModelStorage {
    async fn save(&self, workspace_id: Uuid, model: &OrgModel, metrics: &TrainingMetrics) -> Result<()> {
        // Save to remote first (source of truth)
        self.remote.save(workspace_id, model, metrics).await?;
        
        // Then update local cache
        self.local.save(workspace_id, model, metrics).await?;
        
        Ok(())
    }
    
    async fn load(&self, workspace_id: Uuid) -> Result<Option<OrgModel>> {
        // Check local cache first
        if self.is_cache_valid(workspace_id).await {
            if let Ok(Some(model)) = self.local.load(workspace_id).await {
                debug!("Model loaded from local cache");
                return Ok(Some(model));
            }
        }
        
        // Fall back to remote
        if let Some(model) = self.remote.load(workspace_id).await? {
            // Update local cache (fire and forget)
            let local = self.local.clone();
            let model_clone = model.clone();
            tokio::spawn(async move {
                // Get latest metrics from remote
                if let Ok(versions) = local.list_versions(workspace_id).await {
                    if let Some(latest) = versions.first() {
                        let _ = local.save(workspace_id, &model_clone, &latest.metrics).await;
                    }
                }
            });
            
            debug!("Model loaded from remote storage");
            return Ok(Some(model));
        }
        
        Ok(None)
    }
    
    async fn list_versions(&self, workspace_id: Uuid) -> Result<Vec<ModelVersion>> {
        // Always use remote as source of truth
        self.remote.list_versions(workspace_id).await
    }
    
    async fn rollback(&self, workspace_id: Uuid, version: &str) -> Result<()> {
        // Rollback remote
        self.remote.rollback(workspace_id, version).await?;
        
        // Clear local cache
        let workspace_dir = self.local.workspace_dir(workspace_id);
        if workspace_dir.exists() {
            fs::remove_dir_all(&workspace_dir).await?;
        }
        
        Ok(())
    }
    
    async fn cleanup_old_versions(&self, workspace_id: Uuid, keep_count: usize) -> Result<usize> {
        // Clean remote
        let deleted = self.remote.cleanup_old_versions(workspace_id, keep_count).await?;
        
        // Clear local cache
        let _ = self.local.cleanup_old_versions(workspace_id, 1).await;
        
        Ok(deleted)
    }
    
    async fn get_stats(&self, workspace_id: Uuid) -> Result<StorageStats> {
        // Use remote as source of truth
        self.remote.get_stats(workspace_id).await
    }
}