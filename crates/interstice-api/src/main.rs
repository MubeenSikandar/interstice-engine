use axum::{
    middleware,
    Router,
};
use interstice_adapters::{AdapterManager, SlackAdapter, PlatformAdapter};
use interstice_core::IntersticeEngine;
use interstice_ml::MLPipeline;
use sqlx::{PgPool, Error as SqlxError};
use uuid::Uuid;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower_http::trace::TraceLayer;

use interstice_core::MLPredictor;
use interstice_ml::adapters::MLPredictorAdapter;

mod routes;
mod handlers;
mod middleware_layer;

pub struct AppState {
    adapters: Arc<AdapterManager>,
    slack_adapter: Option<SlackAdapter>,
    core: Arc<IntersticeEngine>,
    ml_pipeline: Arc<MLPipeline>,
    db: PgPool,
}

#[tokio::main]
async fn main() {

    rustls::crypto::ring::default_provider()
    .install_default()
    .expect("Failed to install rustls crypto provider");

    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load environment variables
    dotenv::dotenv().ok();

    // Database connection
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    let db = PgPool::connect(&database_url).await
        .expect("Failed to connect to database");

    // Run migrations
    match sqlx::migrate!("../../migrations")
        .run(&db)
        .await
    {
        Ok(_) => tracing::info!("Migrations applied successfully"),
        Err(e) => {
            match e {
                sqlx::migrate::MigrateError::VersionMismatch(version) => {
                    tracing::warn!("Migration version mismatch: expected {}, skipping", version);
                },
                sqlx::migrate::MigrateError::Execute(sqlx::Error::Database(db_err)) if db_err.code() == Some(std::borrow::Cow::Borrowed("42P07")) => {
                    tracing::warn!("Relation already exists: {}", db_err);
                }
                _ => {
                    panic!("Failed to run migrations: {:?}. Other MigrateError type.", e);
                }
            }
        }
    };

    // Initialize core components
    let core = Arc::new(IntersticeEngine::new());
    let ml_pipeline = Arc::new(
        MLPipeline::new(&database_url).await
            .expect("Failed to initialize ML pipeline")
    );

    // Initialize adapters
    let mut adapters = AdapterManager::new();
    let mut slack_adapter = None;

    dotenv::dotenv().ok();

    // Add Slack adapter if tokens exist
    if let (Ok(slack_token), Ok(signing_secret)) = (
        std::env::var("SLACK_BOT_TOKEN"),
        std::env::var("SLACK_SIGNING_SECRET"),
    ) {
        let adapter = SlackAdapter::new(slack_token, signing_secret)
            .with_storage(db.clone())  // Pass the database pool
            .with_ml_pipeline(ml_pipeline.clone())
            .with_workspace_id(Uuid::parse_str("92e07a23-a257-4353-a170-534a2019771a").unwrap());
        
        slack_adapter = Some(adapter.clone());
        adapters.register(Box::new(adapter) as Box<dyn PlatformAdapter>);
        tracing::info!("Slack adapter initialized with storage");
    }

    let state = Arc::new(AppState {
        adapters: Arc::new(adapters),
        slack_adapter,
        core,
        ml_pipeline,
        db,
    });

    // Build router with all routes
    let app = Router::<Arc<AppState>>::new()
        // Health check routes
        .nest("/health", routes::health_routes())
        
        // API routes (protected)
        .nest("/api/v1", 
            routes::api_routes()
                .layer(middleware::from_fn_with_state(
                    state.clone(), 
                    middleware_layer::auth_middleware
                ))
        )
        
        // Webhook routes (public but verified)
        .nest("/webhooks", routes::webhook_routes())
        
        // Add middleware
        .layer(middleware_layer::cors_layer())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr = std::env::var("SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    
    tracing::info!("Starting server on {}", addr);

    let listener = TcpListener::bind(&addr).await
        .expect("Failed to bind to address");
    
    axum::serve(listener, app).await
        .expect("Server failed");
}