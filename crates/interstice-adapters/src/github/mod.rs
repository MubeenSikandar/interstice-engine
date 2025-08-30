use crate::traits::{PlatformAdapter, PlatformResponse};
use async_trait::async_trait;
use interstice_core::{IntersticeEngine, Platform, ProcessedArtifact};
use std::sync::Arc;

pub struct GitHubAdapter {
    engine: Arc<IntersticeEngine>,
    webhook_secret: Option<String>,
}

impl GitHubAdapter {
    pub fn new(webhook_secret: Option<String>) -> Self {
        Self {
            engine: Arc::new(IntersticeEngine::new()),
            webhook_secret,
        }
    }

    pub fn verify_signature(&self, signature: &str, body: &str) -> bool {
        if let Some(secret) = &self.webhook_secret {
            // Implement GitHub signature verification
            use hmac::{Hmac, Mac};
            use sha2::Sha256;
            
            let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
            mac.update(body.as_bytes());
            let result = mac.finalize();
            let expected = format!("sha256={}", hex::encode(result.into_bytes()));
            
            expected == signature
        } else {
            true // No secret configured, accept all
        }
    }
}

#[async_trait]
impl PlatformAdapter for GitHubAdapter {
    fn platform(&self) -> Platform {
        Platform::GitHub
    }

    async fn process_event(
        &self,
        event: serde_json::Value,
    ) -> interstice_core::Result<ProcessedArtifact> {
        let action = event["action"].as_str().unwrap_or("");
        
        let text = match event["pull_request"].as_object() {
            Some(pr) => {
                format!("PR #{}: {}", 
                    pr["number"].as_u64().unwrap_or(0),
                    pr["title"].as_str().unwrap_or("")
                )
            }
            None => match event["issue"].as_object() {
                Some(issue) => {
                    format!("Issue #{}: {}",
                        issue["number"].as_u64().unwrap_or(0),
                        issue["title"].as_str().unwrap_or("")
                    )
                }
                None => String::new()
            }
        };

        if !text.is_empty() {
            self.engine.process(text, Platform::GitHub).await
        } else {
            Ok(ProcessedArtifact {
                artifacts: vec![],
                predictions: vec![],
                platform: Platform::GitHub,
            })
        }
    }

    async fn send_response(&self, response: PlatformResponse) -> interstice_core::Result<()> {
        // GitHub responses are typically comments on PRs/Issues
        // This would require GitHub API calls
        tracing::info!("GitHub response: {}", response.message);
        Ok(())
    }
}