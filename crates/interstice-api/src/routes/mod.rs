mod api;
mod health;
mod webhooks;
mod auth;

pub use api::{
    workspace_routes,
    artifact_routes,
    outcome_routes,
    analytics_routes,
};

pub use health::health_routes;
pub use webhooks::webhook_routes;
pub use auth::{
    auth_public_routes,
    auth_protected_routes,
    admin_routes,
};
