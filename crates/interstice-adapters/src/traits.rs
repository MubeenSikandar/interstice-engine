use async_trait::async_trait;
use interstice_core::{Platform, ProcessedArtifact, Result};
use serde_json::Value;

#[async_trait]
pub trait PlatformAdapter: Send + Sync {
    /// Which platform this adapter handles
    fn platform(&self) -> Platform;

    /// Process an incoming webhook/event
    async fn process_event(&self, event: Value) -> Result<ProcessedArtifact>;

    /// Send a response back to the platform
    async fn send_response(&self, response: PlatformResponse) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct PlatformResponse {
    pub channel: String,
    pub message: String,
    pub ephemeral: bool,
    pub user: Option<String>,
}
