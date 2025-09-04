// interstice-api/src/main.rs
use axum::{
    middleware,
    Router,
};
use interstice_adapters::{AdapterManager, SlackAdapter, PlatformAdapter};
use interstice_core::{IntersticeEngine, MLPredictor, DatabaseStorage, Storage};
use interstice_ml::{MLPipeline, adapters::MLPredictorAdapter};
use sqlx::PgPool;
use uuid::Uuid;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;
use tracing::{info, error, warn};

mod routes;
mod handlers;
mod middleware_layer;

// Import middleware functions from middleware_layer module
use middleware_layer::{
    cors_layer,
    request_id_middleware,
    logging_middleware,
    security_headers_middleware,
    create_middleware_stack,
};

pub struct AppState {
    /// Manages all platform adapters dynamically
    pub adapters: Arc<AdapterManager>,
    /// Slack adapter for direct access (most commonly used)
    pub slack_adapter: Option<SlackAdapter>,
    /// Core engine with ML predictor integrated
    pub core: Arc<IntersticeEngine>,
    /// ML pipeline for predictions and training
    pub ml_pipeline: Arc<MLPipeline>,
    /// Database connection pool
    pub db: PgPool,
}

impl AppState {
    /// Get the core engine for processing
    pub fn engine(&self) -> &Arc<IntersticeEngine> {
        &self.core
    }
    
    /// Get all registered adapters
    pub fn adapter_manager(&self) -> &Arc<AdapterManager> {
        &self.adapters
    }
    
    /// Get ML pipeline for predictions
    pub fn ml(&self) -> &Arc<MLPipeline> {
        &self.ml_pipeline
    }
}

#[tokio::main]
async fn main() {
    // Install crypto provider for rustls
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    // Initialize tracing with environment filter
    init_tracing();

    // Load environment variables
    dotenv::dotenv().ok();

    // Validate required environment variables
    validate_environment();

    // Database connection with retry logic
    let db = establish_database_connection().await;

    // Run migrations
    run_migrations(&db).await;

    // Initialize ML components
    let (ml_pipeline, ml_predictor) = initialize_ml_components(&db).await;

    // Initialize core engine with ML predictor and storage
    let core = initialize_core_engine(ml_predictor, db.clone()).await;

    // Initialize platform adapters
    let (adapters, slack_adapter) = initialize_adapters(
        &db,
        &ml_pipeline,
        &core,
    ).await;

    // Create application state
    let state = Arc::new(AppState {
        adapters: Arc::new(adapters),
        slack_adapter,
        core,
        ml_pipeline,
        db: db.clone(),
    });

    // Start background tasks
    start_background_tasks(state.clone());

    // Build and configure router
    let app = build_router(state.clone());

    // Start server
    start_server(app).await;
}

/// Initialize tracing with environment-based configuration
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};
    
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
    
    info!("Tracing initialized");
}

/// Validate that all required environment variables are set
fn validate_environment() {
    let required_vars = vec![
        "DATABASE_URL",
    ];
    
    let mut missing = Vec::new();
    for var in required_vars {
        if std::env::var(var).is_err() {
            missing.push(var);
        }
    }
    
    if !missing.is_empty() {
        error!("Missing required environment variables: {:?}", missing);
        std::process::exit(1);
    }
    
    info!("Environment validation successful");
}

/// Establish database connection with retry logic
async fn establish_database_connection() -> PgPool {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    
    let max_retries = 5;
    let mut retry_count = 0;
    
    loop {
        match PgPool::connect(&database_url).await {
            Ok(pool) => {
                info!("Database connection established");
                return pool;
            }
            Err(e) => {
                retry_count += 1;
                if retry_count >= max_retries {
                    error!("Failed to connect to database after {} attempts: {}", max_retries, e);
                    std::process::exit(1);
                }
                warn!("Database connection attempt {} failed: {}. Retrying in 5 seconds...", retry_count, e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

/// Run database migrations
async fn run_migrations(db: &PgPool) {
    match sqlx::migrate!("../../migrations").run(db).await {
        Ok(_) => info!("Migrations applied successfully"),
        Err(e) => {
            match e {
                sqlx::migrate::MigrateError::VersionMismatch(version) => {
                    warn!("Migration version mismatch: expected {}, skipping", version);
                }
                sqlx::migrate::MigrateError::Execute(sqlx::Error::Database(db_err)) => {
                    // Handle PostgreSQL-specific errors
                    match db_err.code().as_deref() {
                        Some("42P07") => {
                            // Table already exists - this is fine if migrations were already run
                            warn!("Some tables already exist, migrations may have been applied previously");
                        }
                        _ => {
                            error!("Database error during migration: {:?}", db_err);
                            std::process::exit(1);
                        }
                    }
                }
                _ => {
                    error!("Failed to run migrations: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
    }
}

/// Initialize ML components
async fn initialize_ml_components(
    _db: &PgPool,
) -> (Arc<MLPipeline>, Arc<dyn MLPredictor>) {
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    
    // Initialize ML pipeline
    let config = interstice_ml::PipelineConfig::production(&database_url);
    let ml_pipeline = Arc::new(
        MLPipeline::new(config).await
            .expect("Failed to initialize ML pipeline")
    );
    
    // Initialize ML predictor adapter
    let ml_predictor = match MLPredictorAdapter::with_defaults().await {
        Ok(predictor) => {
            info!("ML predictor initialized successfully");
            Arc::new(predictor) as Arc<dyn MLPredictor>
        }
        Err(e) => {
            warn!("Failed to initialize ML predictor: {}. Creating fallback predictor.", e);
            // Create a fallback predictor instead of trying to cast MLPipeline
            Arc::new(FallbackPredictor::new()) as Arc<dyn MLPredictor>
        }
    };
    
    (ml_pipeline, ml_predictor)
}

/// Initialize core engine with ML and storage
async fn initialize_core_engine(
    ml_predictor: Arc<dyn MLPredictor>,
    db: PgPool,
) -> Arc<IntersticeEngine> {
    // Create storage backend
    let storage = Arc::new(DatabaseStorage::new(db.clone())) as Arc<dyn Storage>;
    
    // Create engine with ML predictor and storage
    let engine = IntersticeEngine::new()
        .with_ml_predictor(ml_predictor)
        .with_storage(storage);
    
    info!("Core engine initialized with ML predictor and storage");
    
    Arc::new(engine)
}

/// Initialize platform adapters
async fn initialize_adapters(
    db: &PgPool,
    ml_pipeline: &Arc<MLPipeline>,
    _core: &Arc<IntersticeEngine>,
) -> (AdapterManager, Option<SlackAdapter>) {
    let mut adapters = AdapterManager::new();
    let mut slack_adapter = None;
    
    // Initialize Slack adapter if configured
    if let (Ok(slack_token), Ok(signing_secret)) = (
        std::env::var("SLACK_BOT_TOKEN"),
        std::env::var("SLACK_SIGNING_SECRET"),
    ) {
        // Get workspace ID from environment or use default
        let workspace_id = std::env::var("SLACK_WORKSPACE_ID")
            .ok()
            .and_then(|id| Uuid::parse_str(&id).ok())
            .unwrap_or_else(|| {
                warn!("SLACK_WORKSPACE_ID not set or invalid, using default");
                Uuid::parse_str("92e07a23-a257-4353-a170-534a2019771a").unwrap()
            });
        
        let adapter = SlackAdapter::new(slack_token, signing_secret)
            .with_storage(db.clone())
            .with_ml_pipeline(ml_pipeline.clone())
            .with_workspace_id(workspace_id);
        
        slack_adapter = Some(adapter.clone());
        adapters.register(Box::new(adapter) as Box<dyn PlatformAdapter>);
        info!("Slack adapter initialized with workspace {}", workspace_id);
    } else {
        warn!("Slack adapter not configured - missing SLACK_BOT_TOKEN or SLACK_SIGNING_SECRET");
    }
    
    // Initialize other adapters if configured
    // GitHub adapter
    if let Ok(_github_token) = std::env::var("GITHUB_TOKEN") {
        // adapters.register(Box::new(GitHubAdapter::new(github_token)));
        info!("GitHub adapter would be initialized here");
    }
    
    // Asana adapter
    if let Ok(_asana_token) = std::env::var("ASANA_TOKEN") {
        // adapters.register(Box::new(AsanaAdapter::new(asana_token)));
        info!("Asana adapter would be initialized here");
    }
    
    info!("Initialized {} platform adapters", adapters.len());
    
    (adapters, slack_adapter)
}

/// Start background tasks
fn start_background_tasks(state: Arc<AppState>) {
    // Clone state for each task
    let state_for_training = state.clone();
    let state_for_cleanup = state.clone();
    let state_for_monitoring = state.clone();
    
    // Periodic ML model retraining
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(24 * 60 * 60)); // 24 hours
        
        loop {
            interval.tick().await;
            info!("Starting scheduled ML model retraining");
            
            if let Err(e) = retrain_ml_models(&state_for_training).await {
                error!("ML retraining failed: {}", e);
            }
        }
    });
    
    // Periodic cleanup of old data
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(60 * 60)); // 1 hour
        
        loop {
            interval.tick().await;
            cleanup_old_data(&state_for_cleanup).await;
        }
    });
    
    // Health monitoring
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            monitor_system_health(&state_for_monitoring).await;
        }
    });
}

/// Retrain ML models with recent data
async fn retrain_ml_models(state: &Arc<AppState>) -> anyhow::Result<()> {
    // Get all workspaces
    let workspaces = sqlx::query!(
        "SELECT id FROM workspaces"
    )
    .fetch_all(&state.db)
    .await?;
    
    for workspace in workspaces {
        info!("Would retrain models for workspace {} here", workspace.id);
        // The actual retraining would be implemented in the ML module
        // For now, just log the intention
    }
    
    Ok(())
}

/// Clean up old data
async fn cleanup_old_data(state: &Arc<AppState>) {
    // Clean up old OAuth states
    let _ = sqlx::query!(
        "DELETE FROM oauth_states WHERE expires_at < NOW()"
    )
    .execute(&state.db)
    .await;
    
    // Clean up old event duplicates (older than 24 hours)
    let _ = sqlx::query!(
        "DELETE FROM slack_events WHERE processed_at < NOW() - INTERVAL '24 hours'"
    )
    .execute(&state.db)
    .await;
}

/// Monitor system health
async fn monitor_system_health(state: &Arc<AppState>) {
    // Check database connection
    if let Err(e) = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.db)
        .await 
    {
        error!("Database health check failed: {}", e);
    }
    
    // Check adapter health
    let adapter_count = state.adapters.len();
    if adapter_count == 0 {
        warn!("No platform adapters registered");
    }
    
    // Log memory usage (optional - requires sys-info crate)
    // if let Ok(mem_info) = sys_info::mem_info() {
    //     let used_percent = (mem_info.total - mem_info.free) * 100 / mem_info.total;
    //     if used_percent > 80 {
    //         warn!("High memory usage: {}%", used_percent);
    //     }
    // }
}

/// Build the application router with comprehensive middleware
fn build_router(state: Arc<AppState>) -> Router {
    // Create the base application with all middleware
    let app = Router::new()
        // Health check routes (public, minimal middleware)
        .nest("/health", routes::health_routes())
        
        // API routes (protected with full middleware stack)
        .nest("/api/v1", routes::api_routes())
        
        // Webhook routes (public but with signature verification)
        .nest("/webhooks", routes::webhook_routes())
        
        // Add global middleware that applies to all routes
        .layer(middleware::from_fn(security_headers_middleware))
        .layer(middleware::from_fn(logging_middleware))
        .layer(middleware::from_fn(request_id_middleware))
        
        // Add the comprehensive middleware stack for compression, tracing, etc.
        .layer(create_middleware_stack())
        
        // Add CORS as the outermost layer so it applies first
        .layer(cors_layer())
        
        // Attach the application state
        .with_state(state);
    
    app
}

/// Start the HTTP server
async fn start_server(app: Router) {
    let addr = std::env::var("SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    
    info!("Starting server on {}", addr);
    
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        }
    };
    
    // Add graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        info!("Shutdown signal received, starting graceful shutdown");
    };
    
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .expect("Server failed");
    
    info!("Server shut down successfully");
}

// Fallback predictor implementation
struct FallbackPredictor;

impl FallbackPredictor {
    fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl MLPredictor for FallbackPredictor {
    async fn predict_outcomes(
        &self,
        _artifacts: &[interstice_core::Artifact],
    ) -> anyhow::Result<Vec<interstice_core::traits::OutcomePrediction>> {
        // Return empty predictions as fallback
        Ok(vec![])
    }
}