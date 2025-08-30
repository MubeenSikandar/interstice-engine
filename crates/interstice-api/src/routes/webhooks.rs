use axum::{
    routing::post,
    Router,
    extract::State,
};
use std::sync::Arc;
use crate::{AppState, handlers};

pub fn webhook_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/slack/events", post(handlers::slack::handle_events))
        .route("/github", post(handlers::github::handle_webhook))
        .route("/jira", post(handlers::jira::handle_webhook))
}