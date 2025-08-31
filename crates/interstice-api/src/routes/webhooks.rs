use axum::{
    routing::{get, post},
    Router,
    extract::State,
};
use std::sync::Arc;
use crate::{AppState, handlers};

pub fn webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        // Slack endpoints
        .route("/slack/events", post(handlers::slack::handle_events))
        .route("/slack/commands", post(handlers::slack::handle_slash_commands))
        .route("/slack/interactions", post(handlers::slack::handle_interactions))
        .route("/slack/oauth/callback", post(handlers::slack::handle_oauth_callback))
        .route("/slack/health", get(handlers::slack::slack_health))
        
        // Other platform webhooks
        .route("/github", post(handlers::github::handle_webhook))
        .route("/jira", post(handlers::jira::handle_webhook))
}