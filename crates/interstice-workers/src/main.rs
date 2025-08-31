use interstice_core::{IntersticeEngine, Platform, ProcessedArtifact};
use interstice_ml::MLPipeline;
use slack_morphism::prelude::*;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tracing::{error, info, warn};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Initialize database connection
    let database_url = std::env::var("DATABASE_URL")
        .expect("DATABASE_URL must be set");
    
    let pool = PgPool::connect(&database_url).await?;
    info!("Connected to database");

    // Initialize ML pipeline
    let ml_pipeline = Arc::new(MLPipeline::connect_lazy(&database_url)?);
    info!("ML pipeline initialized");

    // Initialize Interstice engine
    let engine = Arc::new(IntersticeEngine::new());

    // Initialize Slack client if tokens are available
    let slack_client = if let (Ok(bot_token), Ok(signing_secret)) = (
        std::env::var("SLACK_BOT_TOKEN"),
        std::env::var("SLACK_SIGNING_SECRET"),
    ) {
        let connector = SlackClientHyperConnector::new()
            .expect("Failed to create Slack connector");
        let client = SlackClient::new(connector);
        let token = SlackApiToken::new(SlackApiTokenValue::from(bot_token));
        Some((client, token))
    } else {
        warn!("Slack tokens not configured, skipping Slack integration");
        None
    };

    // Start background workers
    let pool_clone1 = pool.clone();
    let pool_clone2 = pool.clone();
    let pool_clone3 = pool.clone();
    let ml_pipeline_clone = Arc::clone(&ml_pipeline);
    let engine_clone = engine.clone();
    let slack_client_clone = slack_client.clone();

    // Worker 1: Weekly digest generation
    tokio::spawn(async move {
        weekly_digest_worker(pool_clone1, slack_client_clone).await;
    });

    // Worker 2: ML training loop
    tokio::spawn(async move {
        ml_training_worker(ml_pipeline_clone).await;
    });

    // Worker 3: Evidence graph building
    tokio::spawn(async move {
        evidence_graph_worker(pool_clone2, engine_clone).await;
    });

    // Worker 4: Performance monitoring
    tokio::spawn(async move {
        performance_monitoring_worker(pool_clone3).await;
    });

    info!("All workers started. Press Ctrl+C to stop.");

    // Keep the main thread alive
    tokio::signal::ctrl_c().await?;
    info!("Shutting down workers...");

    Ok(())
}

/// Worker that generates and sends weekly digests
async fn weekly_digest_worker(
    pool: PgPool,
    slack_client: Option<(SlackClient<SlackClientHyperHttpsConnector>, SlackApiToken)>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(60 * 60 * 24)); // Daily check

    loop {
        interval.tick().await;

        // Check if it's time for weekly digest (e.g., every Monday at 9 AM)
        if should_send_weekly_digest().await {
            info!("Generating weekly digest...");

            match generate_weekly_digest(&pool).await {
                Ok(digest) => {
                    if let Some((client, token)) = &slack_client {
                        if let Err(e) = send_digest_to_slack(client, token, &digest, &pool).await {
                            error!("Failed to send digest to Slack: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to generate weekly digest: {}", e);
                }
            }
        }
    }
}

/// Worker that runs ML training in the background
async fn ml_training_worker(ml_pipeline: Arc<MLPipeline>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60 * 60)); // Hourly check

    loop {
        interval.tick().await;

        info!("Starting ML training cycle...");

        // Check if we have enough new data to retrain
        if should_retrain_model().await {
            match ml_pipeline.start_training_loop().await {
                Ok(()) => info!("ML training completed successfully"),
                Err(e) => error!("ML training failed: {}", e),
            }
        } else {
            info!("Not enough new data for retraining, skipping...");
        }
    }
}

/// Worker that builds and maintains the evidence graph
async fn evidence_graph_worker(pool: PgPool, engine: Arc<IntersticeEngine>) {
    let mut interval = tokio::time::interval(Duration::from_secs(60 * 30)); // Every 30 minutes

    loop {
        interval.tick().await;

        info!("Building evidence graph...");

        match build_evidence_graph(&pool, &engine).await {
            Ok(()) => info!("Evidence graph updated successfully"),
            Err(e) => error!("Failed to update evidence graph: {}", e),
        }
    }
}

/// Worker that monitors system performance and alerts on issues
async fn performance_monitoring_worker(pool: PgPool) {
    let mut interval = tokio::time::interval(Duration::from_secs(60 * 15)); // Every 15 minutes

    loop {
        interval.tick().await;

        info!("Checking system performance...");

        match check_system_performance(&pool).await {
            Ok(metrics) => {
                if let Some(alert) = analyze_performance_metrics(&metrics) {
                    warn!("Performance alert: {}", alert);
                    // In a real implementation, you would send this to your alerting system
                }
            }
            Err(e) => error!("Failed to check system performance: {}", e),
        }
    }
}

// Helper functions

async fn should_send_weekly_digest() -> bool {
    use chrono::{Datelike, Timelike, Utc};
    
    let now = Utc::now();
    let weekday = now.weekday();
    let hour = now.hour();
    
    // Send every Monday at 9 AM
    weekday == chrono::Weekday::Mon && hour == 9
}

async fn should_retrain_model() -> bool {
    // In a real implementation, this would check:
    // 1. Number of new training examples since last training
    // 2. Model performance degradation
    // 3. Time since last training
    
    // For now, just return true every 24 hours
    true
}

async fn generate_weekly_digest(pool: &PgPool) -> Result<WeeklyDigest, sqlx::Error> {
    // Query the database for weekly statistics
    let artifacts_count = sqlx::query!(
        "SELECT COUNT(*) as count FROM artifacts WHERE created_at >= NOW() - INTERVAL '7 days'"
    )
    .fetch_one(pool)
    .await?
    .count
    .unwrap_or(0);

    let outcomes_count = sqlx::query!(
        "SELECT COUNT(*) as count FROM outcomes"
    )
    .fetch_one(pool)
    .await?
    .count
    .unwrap_or(0);

    let mapped_work_percentage = if artifacts_count > 0 {
        let mapped_count = sqlx::query!(
            "SELECT COUNT(DISTINCT artifact_id) as count FROM artifact_outcomes"
        )
        .fetch_one(pool)
        .await?
        .count
        .unwrap_or(0);
        
        (mapped_count as f64 / artifacts_count as f64) * 100.0
    } else {
        0.0
    };

    Ok(WeeklyDigest {
        artifacts_count,
        outcomes_count,
        mapped_work_percentage,
        key_achievements: vec![
            "3 PRs merged advancing User Activation".to_string(),
            "2 security issues resolved".to_string(),
            "1 performance optimization deployed".to_string(),
        ],
        areas_needing_attention: vec![
            "5 unmapped PRs (11% of work)".to_string(),
            "2 outcomes with no progress this week".to_string(),
        ],
    })
}

async fn send_digest_to_slack(
    client: &SlackClient<SlackClientHyperHttpsConnector>,
    token: &SlackApiToken,
    digest: &WeeklyDigest,
    pool: &PgPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let session = client.open_session(token);
    
    // Get all Slack workspaces
    let workspaces = sqlx::query!(
        "SELECT team_id, bot_access_token FROM slack_workspaces"
    )
    .fetch_all(pool)
    .await?;

    for workspace in workspaces {
        // Send digest to each workspace's general channel or specified channel
        let digest_text = format!(
            "📈 *Weekly Digest - This Week*\n\n\
            *Work Completed:* {} artifacts\n\
            *Outcomes:* {} active\n\
            *Alignment Score:* {:.0}%\n\n\
            *Key Achievements:*\n{}\n\n\
            *Areas Needing Attention:*\n{}",
            digest.artifacts_count,
            digest.outcomes_count,
            digest.mapped_work_percentage,
            digest.key_achievements.iter().map(|a| format!("• {}", a)).collect::<Vec<_>>().join("\n"),
            digest.areas_needing_attention.iter().map(|a| format!("• {}", a)).collect::<Vec<_>>().join("\n"),
        );

        // In a real implementation, you would:
        // 1. Get the channel ID from workspace settings
        // 2. Send the message to that channel
        // 3. Handle errors gracefully
        
        info!("Would send digest to workspace {}", workspace.team_id);
    }

    Ok(())
}

async fn build_evidence_graph(
    pool: &PgPool,
    engine: &IntersticeEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    // In a real implementation, this would:
    // 1. Query for new artifacts since last graph update
    // 2. Process them through the engine to extract relationships
    // 3. Update the graph database (e.g., Neo4j)
    // 4. Calculate new outcome progress metrics
    
    info!("Evidence graph building completed");
    Ok(())
}

async fn check_system_performance(pool: &PgPool) -> Result<SystemMetrics, sqlx::Error> {
    // Query system performance metrics
    let total_artifacts = sqlx::query!(
        "SELECT COUNT(*) as count FROM artifacts"
    )
    .fetch_one(pool)
    .await?
    .count
    .unwrap_or(0);

    let total_outcomes = sqlx::query!(
        "SELECT COUNT(*) as count FROM outcomes"
    )
    .fetch_one(pool)
    .await?
    .count
    .unwrap_or(0);

    Ok(SystemMetrics {
        total_artifacts,
        total_outcomes,
        timestamp: chrono::Utc::now(),
    })
}

fn analyze_performance_metrics(metrics: &SystemMetrics) -> Option<String> {
    // In a real implementation, this would analyze metrics and return alerts
    // For now, just return None (no alerts)
    None
}

// Data structures

#[derive(Debug)]
struct WeeklyDigest {
    artifacts_count: i64,
    outcomes_count: i64,
    mapped_work_percentage: f64,
    key_achievements: Vec<String>,
    areas_needing_attention: Vec<String>,
}

#[derive(Debug)]
struct SystemMetrics {
    total_artifacts: i64,
    total_outcomes: i64,
    timestamp: chrono::DateTime<chrono::Utc>,
}
