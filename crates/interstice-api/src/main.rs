use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use interstice_adapters::{AdapterManager, SlackAdapter, PlatformAdapter};
use std::sync::Arc;
use tokio::net::TcpListener;

struct AppState {
    adapters: Arc<AdapterManager>,
    slack_adapter: Option<SlackAdapter>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Initialize adapters
    let mut adapters = AdapterManager::new();
    let mut slack_adapter = None;

    // Add Slack adapter if tokens exist
    if let (Ok(slack_token), Ok(signing_secret)) = (
        std::env::var("SLACK_BOT_TOKEN"),
        std::env::var("SLACK_SIGNING_SECRET"),
    ) {
        let adapter = SlackAdapter::new(slack_token, signing_secret);
        slack_adapter = Some(adapter.clone());
        adapters.register(Box::new(adapter) as Box<dyn PlatformAdapter>);
        tracing::info!("Slack adapter initialized");
    }

    let state = Arc::new(AppState {
        adapters: Arc::new(adapters),
        slack_adapter,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/webhooks/slack/events", post(handle_slack_events))
        .with_state(state);

    // Start server
    let addr = "0.0.0.0:3000";
    tracing::info!("Starting server on {}", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "OK"
}

async fn handle_slack_events(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(adapter) = &state.slack_adapter else {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    };

    // Convert body to string
    let body_str = String::from_utf8(body.to_vec()).map_err(|_| StatusCode::BAD_REQUEST)?;

    // Parse the JSON
    let event: serde_json::Value = serde_json::from_str(&body_str)
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    // Handle URL verification
    if let Some(challenge) = event.get("challenge") {
        return Ok(Json(serde_json::json!({ 
            "challenge": challenge.as_str().unwrap_or("") 
        })));
    }

    // Verify signature
    let timestamp = headers
        .get("x-slack-request-timestamp")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    
    let signature = headers
        .get("x-slack-signature")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");

    if !adapter.verify_signature(timestamp, &body_str, signature) {
        tracing::warn!("Invalid Slack signature");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Process the event
    adapter.process_event(event).await
        .map_err(|e| {
            tracing::error!("Error processing event: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(serde_json::json!({ "ok": true })))
}