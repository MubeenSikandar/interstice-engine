// interstice-api/src/main.rs
use axum::{
    extract::Request,
    middleware::{from_fn, from_fn_with_state, Next}, 
    Router,
    Extension,
};
use interstice_adapters::{slack::SlackConfig, AdapterManager, PlatformAdapter, slack::SlackAdapter};
use interstice_core::{analytics::AnalyticsEngine, storage::PostgresStorage, IntersticeEngine, MLPredictor, StorageBackend};
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
    timeout_middleware,
    TimeoutConfig,
    TimeoutManager,
};

use crate::{
    middleware_layer::{
        analytics_tracking, 
        auth::{
            auth_middleware,
            require_role,
            require_workspace,
            AuthConfig,
        },
        rate_limit::{
            RateLimiter,
            RateLimitConfig,
            RateLimitAlgorithm,
            auth_rate_limit_middleware,
            ip_rate_limit_middleware,
            slack_rate_limit_middleware,
            webhook_rate_limit_middleware,
            init_global_rate_limiter,
        }
    }, 
    routes::{
        auth_protected_routes, 
        auth_public_routes
    }
};

#[derive(Clone)]
pub struct AppState {
    /// Manages all platform adapters dynamically
    pub adapters: Arc<AdapterManager>,
    /// Slack adapter for direct access (most commonly used)
    pub slack_adapter: Option<Arc<SlackAdapter>>,
    /// Core engine with ML predictor integrated
    pub core: Arc<IntersticeEngine>,
    /// ML pipeline for predictions and training
    pub ml_pipeline: Arc<MLPipeline>,
    /// Database connection pool
    pub db: PgPool,
    /// Analytics engine for metrics and insights
    pub analytics: Option<Arc<AnalyticsEngine>>,
    /// Authentication configuration
    pub auth_config: AuthConfig,
    /// Rate limiter for API requests
    pub rate_limiter: Arc<RateLimiter>,
    /// Timeout manager for operation timeouts
    pub timeout_manager: Arc<TimeoutManager>,
}

impl AppState {
    /// Create a new app state
    pub async fn new(db: PgPool) -> Self {
        Self {
            adapters: Arc::new(AdapterManager::new()),
            slack_adapter: None,
            core: Arc::new(IntersticeEngine::new()),
            ml_pipeline: Arc::new(MLPipeline::new(interstice_ml::PipelineConfig::production(&std::env::var("DATABASE_URL").expect("DATABASE_URL must be set"))).await.expect("Failed to initialize ML pipeline")),
            db,
            analytics: None,
            auth_config: AuthConfig::from_env(),
            rate_limiter: Arc::new(RateLimiter::new(1000, Duration::from_secs(3600))),
            timeout_manager: Arc::new(TimeoutManager::new(TimeoutConfig::from_env())),
        }
    }
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

    // Initialize authentication system
    if let Err(e) = initialize_auth_system(&db).await {
        error!("Failed to initialize auth system: {}", e);
        std::process::exit(1);
    }

    if let Err(e) = handlers::slack::initialize_encryption_key() {
        error!("Failed to initialize encryption key: {}", e);
        std::process::exit(1);
    }

    // Initialize ML components
    let (ml_pipeline, ml_predictor) = initialize_ml_components(&db).await;

    // Initialize core engine with ML predictor and storage
    let core = initialize_core_engine(ml_predictor, db.clone()).await;

    // Initialize storage backend (share with core engine)
    let storage_config = interstice_core::storage::StorageConfig::default();
    let storage = Arc::new(PostgresStorage::new(storage_config).await.unwrap()) as Arc<dyn StorageBackend>;

    // Initialize analytics engine
    let analytics = initialize_analytics_engine(storage.clone()).await;

    // Initialize platform adapters
    let (adapters, slack_adapter) = initialize_adapters(
        &db,
        &ml_pipeline,
        &core,
    ).await;

    // Initialize distributed rate limiter
    let rate_limiter = initialize_rate_limiter(db.clone());
    
    // Initialize global rate limiter for the application
    init_global_rate_limiter(rate_limiter.clone());
    
    // Create application state
    let state = Arc::new(AppState {
        adapters: Arc::new(adapters),
        slack_adapter,
        core,
        ml_pipeline,
        analytics,  // Add this
        db: db.clone(),
        auth_config: AuthConfig::from_env(),
        rate_limiter: Arc::new(rate_limiter),
        timeout_manager: Arc::new(TimeoutManager::new(TimeoutConfig::from_env())),
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
    _db: PgPool,
) -> Arc<IntersticeEngine> {
    // Create storage backend
    let storage_config = interstice_core::storage::StorageConfig::default();
    let storage = Arc::new(PostgresStorage::new(storage_config).await.unwrap()) as Arc<dyn StorageBackend>;

    // Create engine with ML predictor and storage
    let engine = IntersticeEngine::new()
        .with_ml_predictor(ml_predictor)
        .with_storage(storage);
    
    info!("Core engine initialized with ML predictor and storage");
    
    Arc::new(engine)
}

/// Initialize production-ready distributed rate limiter
fn initialize_rate_limiter(db: PgPool) -> RateLimiter {
    let config = RateLimitConfig {
        limit: std::env::var("RATE_LIMIT_REQUESTS_PER_HOUR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10000), // 10k requests per hour default
        window_secs: 3600, // 1 hour window
        algorithm: RateLimitAlgorithm::SlidingWindowCounter,
        burst_capacity: Some(500), // Allow bursts up to 500 requests
        refill_rate: Some(2.8), // ~10k/hour = 2.8 requests/second
        distributed: true, // Use database for distributed rate limiting
        custom_headers: std::collections::HashMap::new(),
    };

    let limiter = RateLimiter::with_database(db, config);
    
    // Start background cleanup task
    let _cleanup_handle = limiter.start_cleanup_task();
    
    info!(
        "Rate limiter initialized with distributed storage ({}req/hr, {} burst)",
        std::env::var("RATE_LIMIT_REQUESTS_PER_HOUR").unwrap_or_else(|_| "10000".to_string()),
        500
    );
    
    limiter
}

/// Initialize analytics engine
async fn initialize_analytics_engine(
    storage: Arc<dyn StorageBackend>,
) -> Option<Arc<interstice_core::analytics::AnalyticsEngine>> {
    use interstice_core::analytics::{AnalyticsConfig, AnalyticsEngine};
    
    // Check if analytics is enabled via environment variable
    let analytics_enabled = std::env::var("ENABLE_ANALYTICS")
        .unwrap_or_else(|_| "true".to_string())
        .parse::<bool>()
        .unwrap_or(true);
    
    if !analytics_enabled {
        info!("Analytics disabled by configuration");
        return None;
    }
    
    // Configure analytics based on environment
    let config = AnalyticsConfig {
        buffer_size: std::env::var("ANALYTICS_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000),
        flush_interval: Duration::from_secs(
            std::env::var("ANALYTICS_FLUSH_INTERVAL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30)
        ),
        enable_anomaly_detection: std::env::var("ANALYTICS_ANOMALY_DETECTION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(true),
        retention_period: Duration::from_secs(
            std::env::var("ANALYTICS_RETENTION_DAYS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(90) * 24 * 3600
        ),
        rate_limit: std::env::var("ANALYTICS_RATE_LIMIT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1000),
        enable_compression: true,
        sampling_rate: std::env::var("ANALYTICS_SAMPLING_RATE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0),
    };
    
    match AnalyticsEngine::new(storage, config).await {
        Ok(engine) => {
            info!("Analytics engine initialized successfully");
            Some(Arc::new(engine))
        }
        Err(e) => {
            error!("Failed to initialize analytics engine: {}", e);
            None
        }
    }
}

/// Initialize platform adapters
async fn initialize_adapters(
    _db: &PgPool,
    ml_pipeline: &Arc<MLPipeline>,
    _core: &Arc<IntersticeEngine>,
    ) -> (AdapterManager, Option<Arc<SlackAdapter>>) {
    let adapters = AdapterManager::new();
    let mut slack_adapter = None;
    
    // Initialize Slack adapter if configured
    if let (Ok(slack_token), Ok(signing_secret)) = (
        std::env::var("SLACK_BOT_TOKEN"),
        std::env::var("SLACK_SIGNING_SECRET"),
    ) {
        // Get workspace ID from environment or use default
        let workspace_id = std::env::var("SLACK_WORKSPACE_ID")
            .ok()
            .and_then(|id| id.parse().ok())
            .unwrap_or_else(|| {
                warn!("SLACK_WORKSPACE_ID not set or invalid, using default");
                "92e07a23-a257-4353-a170-534a2019771a".parse().unwrap()
            });
        
        let config = SlackConfig {
            bot_token: slack_token.clone(),
            signing_secret: signing_secret.clone(),
            app_token: None,
            client_id: None,
            client_secret: None,
            workspace_id,
            enable_socket_mode: false,
            enable_events_api: true,
            retry_config: Default::default(),
            cache_config: Default::default(),
            feature_flags: Default::default(),
        };
        
        let adapter = SlackAdapter::new(config).await
            .expect("Failed to create Slack adapter")
            .with_ml_pipeline(ml_pipeline.clone());
        
        // Create a second instance for the adapter manager
        let config2 = SlackConfig {
            bot_token: slack_token,
            signing_secret,
            app_token: None,
            client_id: None,
            client_secret: None,
            workspace_id,
            enable_socket_mode: false,
            enable_events_api: true,
            retry_config: Default::default(),
            cache_config: Default::default(),
            feature_flags: Default::default(),
        };
        let adapter2 = SlackAdapter::new(config2).await
            .expect("Failed to create Slack adapter")
            .with_ml_pipeline(ml_pipeline.clone());
        
        slack_adapter = Some(Arc::new(adapter));
        if let Err(e) = adapters.register(Arc::new(adapter2) as Arc<dyn PlatformAdapter>) {
            error!("Failed to register Slack adapter: {}", e);
        }
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

    let oauth_cleanup_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(3600)); // Every hour
        
        loop {
            interval.tick().await;
            // Call the cleanup function from slack handler
            handlers::slack::cleanup_expired_oauth_states(&oauth_cleanup_state).await;
        }
    });
    
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
    // ========================================================================
    // Public Routes (No Authentication Required)
    // ========================================================================
    
    let public_routes = Router::new()
        // Health checks (no auth required)
        .nest("/health", routes::health_routes())
        // Authentication endpoints (public)
        .nest("/auth", auth_public_routes())
        // Apply strict rate limiting for auth endpoints
        .route_layer(from_fn_with_state(
            state.clone(),
            auth_rate_limit_middleware
        ))
        // Apply IP-based rate limiting to all public routes
        .route_layer(from_fn_with_state(
            state.clone(),
            ip_rate_limit_middleware
        ));

    // ========================================================================
    // Protected API Routes (Authentication Required)
    // ========================================================================
    
    let protected_api_routes = Router::new()
        // Auth management (logout, profile, api keys)
        .nest("/auth", auth_protected_routes())
        // Core business logic routes
        .nest("/workspaces", routes::workspace_routes())
        .nest("/artifacts", routes::artifact_routes())
        .nest("/outcomes", routes::outcome_routes())
        .nest("/analytics", routes::analytics_routes())
        // Note: General rate limiting handled through create_rate_limit_layer()
        // Authentication middleware (validates JWT/API keys)
        .layer(from_fn(|req: Request, next: Next| async move {
            // Get state from request extensions
            let state = req.extensions().get::<Arc<AppState>>().cloned()
                .expect("AppState not found in request extensions");
            auth_middleware(state, req, next).await
        }))
        // Workspace access validation
        .route_layer(from_fn_with_state(
            state.clone(),
            require_workspace
        ));

    // ========================================================================
    // Admin Routes (Admin Role Required)
    // ========================================================================
    
    let admin_routes = Router::new()
        .nest("/", routes::admin_routes())
        // Note: General rate limiting handled through create_rate_limit_layer()
        // Admin role requirement
        .route_layer(from_fn(|req, next| {
            Box::pin(async move {
                require_role("admin", req, next).await
            })
        }))
        .layer(from_fn(|req: Request, next: Next| async move {
            // Get state from request extensions
            let state = req.extensions().get::<Arc<AppState>>().cloned()
                .expect("AppState not found in request extensions");
            auth_middleware(state, req, next).await
        }));
    // ========================================================================
    // Webhook Routes (Signature Verification Instead of JWT)
    // ========================================================================
    
    let webhook_routes = Router::new()
        .nest("/webhooks", routes::webhook_routes())
        // Webhook-specific rate limiting
        .route_layer(from_fn_with_state(
            state.clone(),
            webhook_rate_limit_middleware
        ))
        // Slack-specific rate limiting for Slack webhooks
        .route_layer(from_fn_with_state(
            state.clone(),
            slack_rate_limit_middleware
        ))
        // Webhook-specific middleware (signature verification)
        .route_layer(from_fn_with_state(
            state.clone(), 
            middleware_layer::webhook_auth::verify_webhook_signature
        ));

    // ========================================================================
    // Combine All Routes with Global Middleware
    // ========================================================================
    
    Router::new()
        // Mount routes at their respective paths
        .merge(public_routes)
        .nest("/api/v1", protected_api_routes)
        .nest("/admin", admin_routes)
        .merge(webhook_routes)
        
        // Apply global middleware stack (order matters - applied bottom to top)
        .layer(Extension(state.clone()))
        .layer(from_fn(security_headers_middleware))
        .layer(from_fn(logging_middleware))
        .layer(from_fn(request_id_middleware))
        .layer(from_fn_with_state(state.clone(), timeout_middleware))
        .layer(from_fn_with_state(state.clone(), analytics_tracking))
        .layer(create_middleware_stack())
        .layer(cors_layer())
        
        // Attach application state
        .with_state(state)
}


async fn initialize_auth_system(db: &PgPool) -> Result<(), Box<dyn std::error::Error>> {
    // Verify critical tables exist
    let tables = sqlx::query_scalar::<_, String>(
        "SELECT table_name FROM information_schema.tables 
         WHERE table_schema = 'public' 
         AND table_name IN ('users', 'api_keys', 'refresh_tokens', 'sessions')
         ORDER BY table_name"
    )
    .fetch_all(db)
    .await?;

    if tables.len() < 4 {
        error!("Missing auth tables. Please run migrations.");
        return Err("Missing auth tables".into());
    }

    // Clean up expired data on startup
    sqlx::query!("DELETE FROM revoked_tokens WHERE expires_at < NOW()")
        .execute(db)
        .await?;
    
    sqlx::query!("DELETE FROM password_reset_tokens WHERE expires_at < NOW() AND used = false")
        .execute(db)
        .await?;
    
    sqlx::query!("DELETE FROM sessions WHERE last_activity < NOW() - INTERVAL '30 days'")
        .execute(db)
        .await?;

    // Create default admin user if none exists
    let admin_exists = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM users WHERE 'admin' = ANY(roles))"
    )
    .fetch_one(db)
    .await?;

    if !admin_exists {
        create_default_admin(db).await?;
    }

    info!("Auth system initialized successfully");
    Ok(())
}

async fn create_default_admin(db: &PgPool) -> Result<(), Box<dyn std::error::Error>> {
    use crate::middleware_layer::auth::hash_password;
    
    let admin_email = std::env::var("ADMIN_EMAIL")
        .unwrap_or_else(|_| "admin@interstice.com".to_string());
    let admin_password = std::env::var("ADMIN_PASSWORD")
        .unwrap_or_else(|_| "ChangeMe123!".to_string());
    
    let password_hash = hash_password(&admin_password)?;
    let admin_id = Uuid::new_v4();
    
    sqlx::query!(
        r#"
        INSERT INTO users (
            id, email, password_hash, roles, 
            email_verified, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, true, NOW(), NOW())
        ON CONFLICT (email) DO NOTHING
        "#,
        admin_id,
        admin_email,
        password_hash,
        &vec!["admin".to_string(), "user".to_string()]
    )
    .execute(db)
    .await?;
    
    warn!("Default admin user created: {} (CHANGE PASSWORD IMMEDIATELY)", admin_email);
    Ok(())
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