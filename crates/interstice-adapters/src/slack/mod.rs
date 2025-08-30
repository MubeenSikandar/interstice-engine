use interstice_core::Storage;

use crate::traits::{PlatformAdapter, PlatformResponse};
use async_trait::async_trait;
use interstice_core::{IntersticeEngine, Platform, ProcessedArtifact};
use slack_morphism::prelude::*;
use slack_morphism::signature_verifier::SlackEventSignatureVerifier;
use std::sync::Arc;

#[derive(Clone)]
pub struct SlackAdapter {
    client: Arc<SlackClient<SlackClientHyperHttpsConnector>>,
    token: SlackApiToken,
    engine: Arc<IntersticeEngine>,
    signing_secret: SlackSigningSecret,
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
            signing_secret,
        }
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
                // This is handled at the API layer
                tracing::info!("URL verification: {}", url_ver.challenge);
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
                // Process the message
                let processed = self.engine.process(text.clone(), Platform::Slack).await?;

                // If we found artifacts, send a response
                if !processed.artifacts.is_empty() {
                    if let (Some(channel), Some(user)) = (&event.origin.channel, &event.sender.user) {
                        self.send_artifact_summary(channel, user, &processed)
                            .await?;
                    }
                }
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

            let processed = self.engine.process(clean_text, Platform::Slack).await?;

            // Reply in thread
            let response = self.format_response(&processed);

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

        // Add action buttons
        let button1 = SlackBlockButtonElement::new("link_outcomes".into(), pt!("Link to Outcomes"));
        let button2 = SlackBlockButtonElement::new("dismiss".into(), pt!("Dismiss"));
        
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
        let processed = self
            .engine
            .process(command.text.unwrap_or_default(), Platform::Slack)
            .await?;

        let blocks = self.create_artifact_blocks(&processed);

        Ok(SlackMessageContent::new()
            .with_text("Processing your command...".to_string())
            .with_blocks(blocks))
    }

    /// Handle interactive events (button clicks, etc.)
    pub async fn handle_interaction(
        &self,
        interaction: SlackInteractionEvent,
    ) -> interstice_core::Result<()> {
        match interaction {
            SlackInteractionEvent::BlockActions(_action_event) => {
                // For now, just log the interaction
                // In a real implementation, you would parse the action_event.actions
                // to determine which button was clicked
                tracing::info!("Received block action interaction");
            }
            _ => {
                tracing::debug!("Unhandled interaction type");
            }
        }
        Ok(())
    }
}
