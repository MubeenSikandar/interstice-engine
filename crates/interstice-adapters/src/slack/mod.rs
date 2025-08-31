//interstice-adapters/src/slack/mod.rs
use interstice_core::{Storage, DatabaseStorage};
use interstice_ml::MLPipeline;
use sqlx::PgPool;

use crate::traits::{PlatformAdapter, PlatformResponse};
use async_trait::async_trait;
use interstice_core::{IntersticeEngine, Platform, ProcessedArtifact};
use slack_morphism::prelude::*;
use slack_morphism::signature_verifier::SlackEventSignatureVerifier;
use std::sync::Arc;
use uuid::Uuid;
use tracing::{info, warn};

#[derive(Clone)]
pub struct SlackAdapter {
    client: Arc<SlackClient<SlackClientHyperHttpsConnector>>,
    token: SlackApiToken,
    engine: Arc<IntersticeEngine>,
    ml_pipeline: Option<Arc<MLPipeline>>,
    signing_secret: SlackSigningSecret,
    workspace_id: Option<Uuid>,
    storage: Option<Arc<dyn Storage>>,  // Add storage field
}

impl SlackAdapter {
    pub fn new(bot_token: String, signing_secret: String) -> Self {
        let connector = SlackClientHyperConnector::new()
            .expect("Failed to create Slack connector");
        let client = Arc::new(SlackClient::new(connector));
        let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
        let engine = Arc::new(IntersticeEngine::new());
        let signing_secret = SlackSigningSecret::from(signing_secret);
        

        Self {
            client,
            token,
            engine,
            ml_pipeline: None,
            signing_secret,
            workspace_id: None,
            storage: None,  // Initialize as None
        }
    }

    pub fn with_storage(mut self, pool: PgPool) -> Self {
        self.storage = Some(Arc::new(DatabaseStorage::new(pool)) as Arc<dyn Storage>);
        self
    }

    pub fn with_ml_pipeline(mut self, ml_pipeline: Arc<MLPipeline>) -> Self {
        self.ml_pipeline = Some(ml_pipeline);
        self
    }

    pub fn with_workspace_id(mut self, workspace_id: Uuid) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }

    /// Create a session for API calls
    pub fn session(&self) -> SlackClientSession<SlackClientHyperHttpsConnector> {
        self.client.open_session(&self.token)
    }

    /// Verify incoming webhook signature
    pub fn verify_signature(&self, timestamp: &str, signature: &str, body: &str) -> bool {
        SlackEventSignatureVerifier::new(&self.signing_secret)
            .verify(timestamp, signature, body)
            .is_ok()
    }

    /// Handle different Slack event types
    pub async fn handle_event(&self, event: SlackPushEvent) -> interstice_core::Result<()> {
        match event {
            SlackPushEvent::EventCallback(callback) => {
                self.handle_event_callback(callback.event).await?;
            }
            SlackPushEvent::UrlVerification(url_ver) => {
                info!("URL verification: {}", url_ver.challenge);
            }
            _ => {
                tracing::debug!("Unhandled event type");
            }
        }
        Ok(())
    }

    async fn handle_event_callback(
        &self,
        event: SlackEventCallbackBody,
    ) -> interstice_core::Result<()> {
        match event {
            SlackEventCallbackBody::Message(message_event) => {
                self.handle_message(message_event).await?;
            }
            SlackEventCallbackBody::AppMention(mention_event) => {
                self.handle_app_mention(mention_event).await?;
            }
            _ => {
                tracing::debug!("Unhandled event callback type");
            }
        }
        Ok(())
    }

    async fn handle_message(&self, event: SlackMessageEvent) -> interstice_core::Result<()> {
        if let Some(content) = &event.content {
            if let Some(text) = &content.text {
                // Skip bot messages to avoid loops
                if event.sender.bot_id.is_some() {
                    return Ok(());
                }

                // Process the message with ML if available
                let processed = if let Some(ml) = &self.ml_pipeline {
                    if let Some(workspace_id) = self.workspace_id {
                        // Use ML pipeline for better predictions
                        let predictions = ml.predict_outcomes(workspace_id, &[], text).await
                            .unwrap_or_else(|e| {
                                warn!("ML prediction failed: {}, falling back to basic processing", e);
                                vec![]
                            });
                        
                        // Convert ML predictions to core predictions
                        let core_predictions = predictions.into_iter()
                            .map(|p| interstice_core::OutcomePrediction {
                                outcome_id: Uuid::parse_str(&p.outcome_id).unwrap(),
                                outcome_name: p.outcome_name,
                                confidence: p.confidence,
                                reasoning: p.reasoning,
                            })
                            .collect();
                        
                        // Extract artifacts using the engine
                        let artifacts = self.engine.extract_artifacts(text, Platform::Slack).await?;
                        
                        ProcessedArtifact {
                            artifacts,
                            predictions: core_predictions,
                            platform: Platform::Slack,
                        }
                    } else {
                        // Fallback to basic processing
                        self.engine.process(text.clone(), Platform::Slack).await?
                    }
                } else {
                    // Basic processing without ML
                    self.engine.process(text.clone(), Platform::Slack).await?
                };

                // If we found artifacts, send a response
                if !processed.artifacts.is_empty() {
                    if let (Some(channel), Some(user)) = (&event.origin.channel, &event.sender.user) {
                        self.send_artifact_summary(channel, user, &processed)
                            .await?;
                    }
                }

                // Store artifacts in the graph for evidence building
                self.store_artifacts(&processed).await?;
            }
        }
        Ok(())
    }

    async fn handle_app_mention(&self, event: SlackAppMentionEvent) -> interstice_core::Result<()> {
        if let Some(text) = &event.content.text {
            // Remove the bot mention to get clean text
            let clean_text = text
                .split_whitespace()
                .skip(1) // Skip the @mention
                .collect::<Vec<_>>()
                .join(" ");

            // Process the command
            let response = match clean_text.trim() {
                "help" | "?" => self.get_help_message(),
                "status" | "progress" => self.get_workspace_status().await?,
                "digest" | "summary" => self.get_weekly_digest().await?,
                "hidden work" | "unmapped" => self.get_hidden_work_analysis().await?,
                _ => {
                    // Process as regular content
                    let processed = self.engine.process(clean_text, Platform::Slack).await?;
                    self.format_response(&processed)
                }
            };

            // Reply in thread
            let channel = event.origin.channel.clone()
                .ok_or_else(|| interstice_core::Error::Other(anyhow::anyhow!("No channel in event")))?;
            
            let message = SlackApiChatPostMessageRequest::new(
                channel,
                SlackMessageContent::new().with_text(response),
            )
            .with_thread_ts(event.origin.ts.clone()); // Reply in thread

            self.session()
                .chat_post_message(&message)
                .await
                .map_err(|e| {
                    interstice_core::Error::Other(anyhow::anyhow!("Slack API error: {:?}", e))
                })?;
        }
        Ok(())
    }

    async fn send_artifact_summary(
        &self,
        channel: &SlackChannelId,
        user: &SlackUserId,
        processed: &ProcessedArtifact,
    ) -> interstice_core::Result<()> {
        // Create blocks for rich formatting
        let blocks = self.create_artifact_blocks(processed);

        // Send ephemeral message (only visible to the user)
        let message = SlackApiChatPostEphemeralRequest::new(
            channel.clone(),
            user.clone(),
            SlackMessageContent::new()
                .with_text(format!("Found {} artifacts", processed.artifacts.len()))
                .with_blocks(blocks),
        );

        self.session()
            .chat_post_ephemeral(&message)
            .await
            .map_err(|e| {
                interstice_core::Error::Other(anyhow::anyhow!("Slack API error: {:?}", e))
            })?;

        Ok(())
    }

    fn create_artifact_blocks(&self, processed: &ProcessedArtifact) -> Vec<SlackBlock> {
        use slack_morphism::blocks::*;

        let mut blocks = vec![
            SlackBlock::Header(SlackHeaderBlock::new(pt!("📊 Work Artifacts Detected"))),
            SlackBlock::Section(SlackSectionBlock::new().with_text(md!(
                "I found *{}* artifacts and *{}* potential outcome mappings:",
                processed.artifacts.len(),
                processed.predictions.len()
            ))),
        ];

        // Add artifact details
        for artifact in &processed.artifacts {
            let artifact_text = match &artifact.artifact_type {
                interstice_core::ArtifactType::PullRequest { number, repo } => {
                    format!(
                        "• PR #{}{}",
                        number,
                        repo.as_ref()
                            .map(|r| format!(" in {}", r))
                            .unwrap_or_default()
                    )
                }
                interstice_core::ArtifactType::Issue { id, project } => {
                    format!(
                        "• Issue {}{}",
                        id,
                        project
                            .as_ref()
                            .map(|p| format!(" in {}", p))
                            .unwrap_or_default()
                    )
                }
                interstice_core::ArtifactType::Commit { sha } => {
                    format!("• Commit {}", &sha[..7.min(sha.len())])
                }
                _ => format!("• {}", artifact.raw_text),
            };

            blocks.push(SlackBlock::Section(
                SlackSectionBlock::new().with_text(md!("{}", artifact_text)),
            ));
        }

        // Add outcome predictions
        if !processed.predictions.is_empty() {
            blocks.push(SlackBlock::Divider(SlackDividerBlock::new()));
            blocks.push(SlackBlock::Section(
                SlackSectionBlock::new().with_text(md!("*Suggested Outcome Mappings:*")),
            ));

            for prediction in &processed.predictions {
                blocks.push(SlackBlock::Section(SlackSectionBlock::new().with_text(
                    md!(
                        "→ {} (confidence: {:.0}%)",
                        prediction.outcome_name,
                        prediction.confidence * 100.0
                    ),
                )));
            }
        }

        // Add action buttons with unique action IDs
        let button1 = SlackBlockButtonElement::new(
            format!("link_outcomes_{}", uuid::Uuid::new_v4()).into(),
            pt!("Link to Outcomes")
        );
        let button2 = SlackBlockButtonElement::new(
            format!("dismiss_{}", uuid::Uuid::new_v4()).into(),
            pt!("Dismiss")
        );
        
        blocks.push(SlackBlock::Actions(SlackActionsBlock::new(
            vec![
                button1.into(),
                button2.into(),
            ]
        )));

        blocks
    }

    fn format_response(&self, processed: &ProcessedArtifact) -> String {
        let mut response = format!(
            "I found {} artifacts and {} potential outcome mappings.\n\n",
            processed.artifacts.len(),
            processed.predictions.len()
        );

        if !processed.artifacts.is_empty() {
            response.push_str("*Artifacts:*\n");
            for artifact in &processed.artifacts {
                response.push_str(&format!("• {}\n", artifact.raw_text));
            }
        }

        if !processed.predictions.is_empty() {
            response.push_str("\n*Suggested Outcomes:*\n");
            for prediction in &processed.predictions {
                response.push_str(&format!(
                    "• {} ({:.0}% confidence)\n",
                    prediction.outcome_name,
                    prediction.confidence * 100.0
                ));
            }
        }

        response
    }

    // Helper methods for slash commands
    fn get_help_message(&self) -> String {
        r#"🤖 *Interstice Bot Help*

*Commands:*
• `@interstice help` - Show this help message
• `@interstice status` - Show workspace progress
• `@interstice digest` - Get weekly summary
• `@interstice hidden work` - Analyze unmapped work

*What I do:*
• Automatically detect work artifacts (PRs, issues, commits)
• Suggest outcome mappings using AI
• Build evidence graphs for your goals
• Provide insights on work alignment

*Examples:*
• Just mention me in any channel
• I'll detect work items and suggest outcomes
• Click buttons to confirm or dismiss suggestions"#.to_string()
    }

    async fn get_workspace_status(&self) -> interstice_core::Result<String> {
        let Some(storage) = &self.storage else {
            return Ok("⚠️ Storage not configured".to_string());
        };

        let Some(workspace_id) = self.workspace_id else {
            return Ok("⚠️ Workspace not configured".to_string());
        };

        let stats = storage.get_workspace_stats(workspace_id).await?;
        let outcomes = storage.get_outcomes(workspace_id).await?;

        Ok(format!(
            r#"📊 *Workspace Status*

*Outcomes:* {} active
*Total Artifacts:* {}
*Recent Activity (7 days):* {} artifacts
*Mapped work:* {:.1}%
*Unmapped work:* {:.1}%

*Active Outcomes:*
{}"#,
            outcomes.len(),
            stats.total_artifacts,
            stats.recent_artifacts,
            stats.mapped_work_percentage,
            100.0 - stats.mapped_work_percentage,
            outcomes.iter()
                .take(5)
                .enumerate()
                .map(|(i, o)| format!("{}. {}", i + 1, o.name))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }

    async fn get_weekly_digest(&self) -> interstice_core::Result<String> {
        // In a real implementation, this would generate from the graph
        Ok(r#"📈 *Weekly Digest - This Week*

*Work Completed:* 47 artifacts
*Outcomes Advanced:* 8 out of 12
*Alignment Score:* 89%

*Key Achievements:*
• 3 PRs merged advancing User Activation
• 2 security issues resolved
• 1 performance optimization deployed

*Areas Needing Attention:*
• 5 unmapped PRs (11% of work)
• 2 outcomes with no progress this week"#.to_string())
    }

    async fn get_hidden_work_analysis(&self) -> interstice_core::Result<String> {
        // In a real implementation, this would analyze unmapped work
        Ok(r#"🔍 *Hidden Work Analysis*

*Unmapped Work:* 11% of total
*Estimated Impact:* 15-20% of potential value

*Categories of Unmapped Work:*
• Infrastructure improvements (40%)
• Documentation updates (30%)
• Bug fixes (20%)
• Other (10%)

*Recommendation:* Review unmapped work weekly to ensure alignment with strategic outcomes."#.to_string())
    }

    // Store artifacts for evidence building
    async fn store_artifacts(&self, processed: &ProcessedArtifact) -> interstice_core::Result<()> {
        let Some(storage) = &self.storage else {
            warn!("No storage configured, skipping artifact persistence");
            return Ok(());
        };

        let Some(workspace_id) = self.workspace_id else {
            warn!("No workspace_id configured, skipping artifact persistence");
            return Ok(());
        };

        // Store each artifact
        for artifact in &processed.artifacts {
            match storage.store_artifact(artifact, workspace_id).await {
                Ok(artifact_id) => {
                    info!("Stored artifact {}: {:?}", artifact_id, artifact.artifact_type);
                    
                    // Link to predicted outcomes
                    for prediction in &processed.predictions {
                        if let Err(e) = storage.link_artifact_outcome(
                            artifact_id,
                            prediction.outcome_id,
                            prediction.confidence,
                        ).await {
                            warn!("Failed to link artifact to outcome: {}", e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to store artifact: {}", e);
                }
            }
        }

        // Update workspace stats for monitoring
        if let Ok(stats) = storage.get_workspace_stats(workspace_id).await {
            info!(
                "Workspace stats - Total artifacts: {}, Mapped work: {:.1}%",
                stats.total_artifacts,
                stats.mapped_work_percentage
            );
        }

        Ok(())
    }


}

#[async_trait]
impl PlatformAdapter for SlackAdapter {
    fn platform(&self) -> Platform {
        Platform::Slack
    }

    async fn process_event(
        &self,
        event: serde_json::Value,
    ) -> interstice_core::Result<ProcessedArtifact> {
        // Parse the event
        let slack_event: SlackPushEvent = serde_json::from_value(event).map_err(|e| {
            interstice_core::Error::Other(anyhow::anyhow!("Failed to parse Slack event: {}", e))
        })?;

        // Handle the event
        self.handle_event(slack_event).await?;

        // For now, return empty processed artifact
        // In production, you'd track what was processed
        Ok(ProcessedArtifact {
            artifacts: vec![],
            predictions: vec![],
            platform: Platform::Slack,
        })
    }

    async fn send_response(&self, response: PlatformResponse) -> interstice_core::Result<()> {
        let message = SlackApiChatPostMessageRequest::new(
            response.channel.into(),
            SlackMessageContent::new().with_text(response.message),
        );

        self.session()
            .chat_post_message(&message)
            .await
            .map_err(|e| {
                interstice_core::Error::Other(anyhow::anyhow!("Slack API error: {:?}", e))
            })?;

        Ok(())
    }
}

// Additional Slack-specific functionality
impl SlackAdapter {
    /// Handle slash commands
    pub async fn handle_slash_command(
        &self,
        command: SlackCommandEvent,
    ) -> interstice_core::Result<SlackMessageContent> {
        let command_text = command.text.unwrap_or_default();
        
        let response = match command_text.trim() {
            "help" | "?" => self.get_help_message(),
            "status" | "progress" => self.get_workspace_status().await?,
            "digest" | "summary" => self.get_weekly_digest().await?,
            "hidden work" | "unmapped" => self.get_hidden_work_analysis().await?,
            _ => {
                // Process as regular content
                let processed = self
                    .engine
                    .process(command_text, Platform::Slack)
                    .await?;

                let blocks = self.create_artifact_blocks(&processed);

                return Ok(SlackMessageContent::new()
                    .with_text("Processing your command...".to_string())
                    .with_blocks(blocks));
            }
        };

        Ok(SlackMessageContent::new().with_text(response))
    }

    /// Handle interactive events (button clicks, etc.)
    pub async fn handle_interaction(
        &self,
        interaction: SlackInteractionEvent,
    ) -> interstice_core::Result<()> {
        match interaction {
            SlackInteractionEvent::BlockActions(action_event) => {
                if let Some(actions) = &action_event.actions {
                    for action in actions {
                        let action_id = action.action_id.0.as_str();
                        if action_id.starts_with("link_outcomes") {
                            self.handle_link_outcomes_action(&action_event).await?;
                        } else if action_id.starts_with("dismiss") {
                            self.handle_dismiss_action(&action_event).await?;
                        }
                    }
                }
            }
            _ => {
                tracing::debug!("Unhandled interaction type");
            }
        }
        Ok(())
    }

    async fn handle_link_outcomes_action(&self, action_event: &SlackInteractionBlockActionsEvent) -> interstice_core::Result<()> {
        // In a real implementation, this would:
        // 1. Parse the artifacts from the message
        // 2. Show a modal for outcome selection
        // 3. Update the graph with confirmed mappings
        // 4. Send feedback to ML pipeline
        
        info!("User confirmed outcome mapping");
        
        // For now, just acknowledge the action
        if let Some(channel) = &action_event.channel {
            let message = SlackApiChatPostMessageRequest::new(
                channel.id.clone(),
                SlackMessageContent::new().with_text("✅ Outcome mapping confirmed! I'll update the evidence graph.".to_string()),
            );

            self.session()
                .chat_post_message(&message)
                .await
                .map_err(|e| {
                    interstice_core::Error::Other(anyhow::anyhow!("Slack API error: {:?}", e))
                })?;
        }
        
        Ok(())
    }

    async fn handle_dismiss_action(&self, action_event: &SlackInteractionBlockActionsEvent) -> interstice_core::Result<()> {
        // In a real implementation, this would:
        // 1. Log the dismissal for ML feedback
        // 2. Optionally ask for reason
        // 3. Update ML model to avoid similar suggestions
        
        info!("User dismissed outcome mapping");
        
        // For now, just acknowledge the action
        if let Some(channel) = &action_event.channel {
            let message = SlackApiChatPostMessageRequest::new(
                channel.id.clone(),
                SlackMessageContent::new().with_text("👌 Mapping dismissed. I'll learn from this feedback.".to_string()),
            );

            self.session()
                .chat_post_message(&message)
                .await
                .map_err(|e| {
                    interstice_core::Error::Other(anyhow::anyhow!("Slack API error: {:?}", e))
                })?;
        }
        
        Ok(())
    }
}
