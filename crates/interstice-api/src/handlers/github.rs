// use axum::{http::StatusCode, Json};
// use serde_json::Value;

// pub async fn handle_webhook(
//     Json(payload): Json<Value>
// ) -> Result<Json<Value>, StatusCode> {
//     // TODO: Implement GitHub webhook handling
//     tracing::info!("Received GitHub webhook");
//     Ok(Json(serde_json::json!({"status": "received"})))
// }