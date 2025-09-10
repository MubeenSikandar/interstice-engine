use axum::{
    routing::{get, post, delete},
    Router,
};
use std::sync::Arc;
use crate::{AppState, handlers::auth::*};

pub fn auth_public_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/login", post(login))
        .route("/register", post(register))
        .route("/refresh", post(refresh_token))
        .route("/password/reset", post(request_password_reset))
        .route("/password/confirm", post(confirm_password_reset))
        .route("/verify-email", get(verify_email))
}

pub fn auth_protected_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/logout", post(logout))
        .route("/me", get(get_current_user))
        .route("/api-keys", get(list_api_keys).post(create_api_key))
        .route("/api-keys/:id", delete(revoke_key))
}

pub fn admin_routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/users", get(list_users))
}