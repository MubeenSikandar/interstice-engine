-- Migration: 001_initial_schema
-- Description: Complete database schema initialization with ML training, webhooks, and workspace management
-- Version: 1.0.0
-- Date: 2025-01-01
-- Author: Database Team

-- ============================================================================
-- MIGRATION METADATA
-- ============================================================================

DO $$
BEGIN
    -- Ensure migrations table exists
    CREATE TABLE IF NOT EXISTS _sqlx_migrations (
        version BIGINT PRIMARY KEY,
        description TEXT NOT NULL,
        installed_on TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        success BOOLEAN NOT NULL,
        checksum BYTEA NOT NULL,
        execution_time BIGINT NOT NULL
    );
    
    -- Check if migration already applied
    IF EXISTS (SELECT 1 FROM _sqlx_migrations WHERE version = 1) THEN
        RAISE NOTICE 'Migration 001 already applied, skipping...';
        RETURN;
    END IF;
END $$;

-- ============================================================================
-- BEGIN TRANSACTION
-- ============================================================================

BEGIN;

-- Set transaction properties for safety
SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = ON;
SET check_function_bodies = FALSE;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = OFF;

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- UUID generation support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;
COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';

-- Vector similarity search support
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';

-- ============================================================================
-- CUSTOM TYPES
-- ============================================================================

-- Automation levels for ML systems
DO $$ BEGIN
    CREATE TYPE public.automation_level AS ENUM (
        'manual',
        'semi_automated',
        'automated',
        'autonomous'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Outcome states for workflow management
DO $$ BEGIN
    CREATE TYPE public.outcome_state AS ENUM (
        'draft',
        'planning',
        'ready',
        'in_progress',
        'review',
        'blocked',
        'completed',
        'cancelled',
        'archived'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Outcome types for task categorization
DO $$ BEGIN
    CREATE TYPE public.outcome_type AS ENUM (
        'strategic',
        'tactical',
        'operational',
        'project',
        'epic',
        'story',
        'task',
        'bug',
        'improvement',
        'research',
        'experiment'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Supported integration platforms
DO $$ BEGIN
    CREATE TYPE public.platform AS ENUM (
        'slack',
        'teams',
        'jira',
        'asana',
        'notion',
        'github',
        'vscode',
        'google_workspace',
        'monday',
        'trello',
        'zoom',
        'figma'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Priority levels
DO $$ BEGIN
    CREATE TYPE public.priority AS ENUM (
        'critical',
        'high',
        'medium',
        'low',
        'none'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Risk assessment levels
DO $$ BEGIN
    CREATE TYPE public.risk_level AS ENUM (
        'low',
        'medium',
        'high',
        'critical'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Validation methods for ML training
DO $$ BEGIN
    CREATE TYPE public.validation_method AS ENUM (
        'human',
        'automated',
        'heuristic'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Auto-update timestamp trigger function
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Duplicate function names for compatibility
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Workspaces: Central tenant/organization entity
CREATE TABLE IF NOT EXISTS public.workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_team_id VARCHAR(255) UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    slack_team_name VARCHAR(255),
    access_token_encrypted TEXT,
    bot_user_id VARCHAR(255),
    app_id VARCHAR(255),
    scopes TEXT,
    slack_enterprise_id VARCHAR(255),
    slack_enterprise_name VARCHAR(255),
    is_enterprise BOOLEAN DEFAULT FALSE,
    token_type VARCHAR(50) DEFAULT 'Bearer',
    is_enterprise_install BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    ml_enabled BOOLEAN DEFAULT FALSE
);

-- ============================================================================
-- ARTIFACT AND OUTCOME TABLES
-- ============================================================================

-- Artifacts: Content from various platforms
CREATE TABLE IF NOT EXISTS public.artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES public.workspaces(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    raw_text TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    channel_id VARCHAR(255),
    message_id VARCHAR(255)
);

-- Outcomes: Goals, tasks, and deliverables
CREATE TABLE IF NOT EXISTS public.outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES public.workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_value NUMERIC,
    current_value NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    state public.outcome_state NOT NULL DEFAULT 'draft',
    outcome_type public.outcome_type NOT NULL DEFAULT 'task',
    priority public.priority NOT NULL DEFAULT 'medium',
    progress DOUBLE PRECISION NOT NULL DEFAULT 0,
    parent_id UUID,
    children UUID[] DEFAULT '{}',
    dependencies UUID[] DEFAULT '{}',
    assignees TEXT[] DEFAULT '{}',
    owner_id TEXT NOT NULL DEFAULT '',
    artifacts UUID[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    platforms TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    due_date TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_hours DOUBLE PRECISION,
    actual_hours DOUBLE PRECISION,
    value_score DOUBLE PRECISION,
    risk_level public.risk_level NOT NULL DEFAULT 'low',
    automation_level public.automation_level NOT NULL DEFAULT 'manual'
);

-- Artifact-Outcome relationships
CREATE TABLE IF NOT EXISTS public.artifact_outcomes (
    artifact_id UUID NOT NULL REFERENCES public.artifacts(id) ON DELETE CASCADE,
    outcome_id UUID NOT NULL REFERENCES public.outcomes(id) ON DELETE CASCADE,
    confidence NUMERIC DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (artifact_id, outcome_id)
);

-- Outcome targets
CREATE TABLE IF NOT EXISTS public.outcome_targets (
    id UUID PRIMARY KEY,
    outcome_id UUID NOT NULL REFERENCES public.outcomes(id) ON DELETE CASCADE,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- ML TRAINING TABLES
-- ============================================================================

-- Training examples for ML models
CREATE TABLE IF NOT EXISTS public.training_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES public.workspaces(id) ON DELETE CASCADE,
    artifact_id UUID REFERENCES public.artifacts(id) ON DELETE CASCADE,
    input_text TEXT NOT NULL,
    input_embedding JSONB,
    suggested_outcome_id UUID REFERENCES public.outcomes(id) ON DELETE SET NULL,
    actual_outcome_id UUID REFERENCES public.outcomes(id) ON DELETE SET NULL,
    user_feedback VARCHAR(50),
    feedback_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    feedback_confidence DOUBLE PRECISION,
    is_validated BOOLEAN,
    feedback_score REAL,
    context JSONB,
    validation_method JSONB,
    validator_id UUID,
    validated_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Training history and metrics
CREATE TABLE IF NOT EXISTS public.training_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    accuracy DOUBLE PRECISION NOT NULL,
    precision DOUBLE PRECISION NOT NULL,
    recall DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    examples_used INTEGER NOT NULL,
    duration_ms BIGINT NOT NULL,
    model_version VARCHAR(100),
    training_config JSONB,
    model_saved BOOLEAN DEFAULT FALSE,
    improvement_delta DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE public.training_history IS 'Historical record of all training runs with metrics';
COMMENT ON COLUMN public.training_history.improvement_delta IS 'Accuracy improvement compared to previous version';

-- Training queue for async processing
CREATE TABLE IF NOT EXISTS public.training_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    processor_id VARCHAR(100),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    last_error TEXT,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE public.training_queue IS 'Queue for asynchronous training jobs';

-- Workspace ML models
CREATE TABLE IF NOT EXISTS public.workspace_models (
    workspace_id UUID PRIMARY KEY REFERENCES public.workspaces(id) ON DELETE CASCADE,
    best_accuracy DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    current_version VARCHAR(100),
    model_type VARCHAR(50),
    base_model VARCHAR(100),
    total_parameters BIGINT,
    trainable_parameters BIGINT,
    total_training_runs INTEGER DEFAULT 0,
    total_training_time_ms BIGINT DEFAULT 0,
    last_training_examples INTEGER,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE public.workspace_models IS 'Stores the current best model information for each workspace';
COMMENT ON COLUMN public.workspace_models.best_accuracy IS 'Highest accuracy achieved by any model version';

-- Model versions with storage info
CREATE TABLE IF NOT EXISTS public.model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    version VARCHAR(100) NOT NULL,
    storage_path TEXT NOT NULL,
    storage_backend VARCHAR(20) NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    compression_type VARCHAR(20),
    accuracy DOUBLE PRECISION NOT NULL,
    precision DOUBLE PRECISION NOT NULL,
    recall DOUBLE PRECISION NOT NULL,
    f1_score DOUBLE PRECISION NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    is_production BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, version)
);
COMMENT ON TABLE public.model_versions IS 'Version control for ML models with performance tracking';
COMMENT ON COLUMN public.model_versions.is_production IS 'Whether this version is currently serving production traffic';

-- Model performance tracking
CREATE TABLE IF NOT EXISTS public.model_performance (
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    predictions_made INTEGER DEFAULT 0,
    predictions_accepted INTEGER DEFAULT 0,
    predictions_rejected INTEGER DEFAULT 0,
    avg_confidence DOUBLE PRECISION,
    predictions_corrected BIGINT,
    accuracy DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (workspace_id, date)
);

-- Feature importance analysis
CREATE TABLE IF NOT EXISTS public.feature_importance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    model_version VARCHAR(100) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    importance_score DOUBLE PRECISION NOT NULL,
    importance_type VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Inference history
CREATE TABLE IF NOT EXISTS public.inference_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    artifact_id UUID,
    model_version VARCHAR(100) NOT NULL,
    input_text TEXT NOT NULL,
    predicted_outcome_id UUID,
    confidence DOUBLE PRECISION NOT NULL,
    latency_ms BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    prediction_data JSONB DEFAULT '{}'
);
COMMENT ON TABLE public.inference_history IS 'Log of all inference requests and results';

-- Prediction cache
CREATE TABLE IF NOT EXISTS public.prediction_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    input_hash VARCHAR(64) NOT NULL,
    prediction JSONB NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),
    UNIQUE(workspace_id, input_hash, model_version)
);
COMMENT ON TABLE public.prediction_cache IS 'Cache for model predictions to improve performance';

-- Training audit log
CREATE TABLE IF NOT EXISTS public.training_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    actor_id UUID,
    actor_type VARCHAR(20),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
COMMENT ON TABLE public.training_audit_log IS 'Audit trail for compliance and debugging';

-- Model update queue
CREATE TABLE IF NOT EXISTS public.model_update_queue (
    workspace_id UUID PRIMARY KEY,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Organization vocabulary
CREATE TABLE IF NOT EXISTS public.org_vocabulary (
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    term_type VARCHAR(50),
    frequency INTEGER DEFAULT 1,
    embedding vector(768),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace_id, term)
);

-- ============================================================================
-- FEEDBACK AND METRICS TABLES
-- ============================================================================

-- Feedback events
CREATE TABLE IF NOT EXISTS public.feedback_events (
    id UUID PRIMARY KEY,
    workspace_id UUID NOT NULL,
    artifact_id UUID NOT NULL,
    outcome_id UUID,
    feedback_type VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    user_id VARCHAR(255),
    metadata JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    update_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, artifact_id, outcome_id)
);

-- General metrics
CREATE TABLE IF NOT EXISTS public.metrics (
    id UUID PRIMARY KEY,
    metric_id VARCHAR(255) NOT NULL,
    workspace_id UUID NOT NULL,
    user_id VARCHAR(255),
    value JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    outcome_id UUID,
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- System events
CREATE TABLE IF NOT EXISTS public.system_events (
    id UUID PRIMARY KEY,
    event_type JSONB NOT NULL,
    workspace_id UUID,
    user_id VARCHAR(255),
    timestamp TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{}',
    correlation_id UUID,
    platform VARCHAR(50)
);

-- Event metrics
CREATE TABLE IF NOT EXISTS public.event_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    team_id VARCHAR(255),
    artifact_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Workspace analytics
CREATE TABLE IF NOT EXISTS public.workspace_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    artifact_count INTEGER DEFAULT 0,
    prediction_count INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    task_count INTEGER DEFAULT 0,
    document_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, date)
);

-- ============================================================================
-- SLACK INTEGRATION TABLES
-- ============================================================================

-- Slack workspaces
CREATE TABLE IF NOT EXISTS public.slack_workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id VARCHAR(255) NOT NULL UNIQUE,
    team_name VARCHAR(255),
    bot_user_id VARCHAR(255),
    access_token_encrypted TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    scopes TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Slack authenticated users
CREATE TABLE IF NOT EXISTS public.slack_authed_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID REFERENCES public.workspaces(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    scope TEXT,
    access_token_encrypted TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    token_type VARCHAR(50),
    updated_at TIMESTAMP,
    UNIQUE(workspace_id, user_id)
);

-- Slack conversations
CREATE TABLE IF NOT EXISTS public.slack_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES public.slack_workspaces(id) ON DELETE CASCADE,
    channel_id VARCHAR(255) NOT NULL,
    channel_name VARCHAR(255),
    is_private BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, channel_id)
);

-- Slack command usage
CREATE TABLE IF NOT EXISTS public.slack_command_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id VARCHAR(255) REFERENCES public.slack_workspaces(team_id),
    user_id VARCHAR(255) NOT NULL,
    command VARCHAR(255) NOT NULL,
    text TEXT,
    response_type VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Slack events
CREATE TABLE IF NOT EXISTS public.slack_events (
    event_id VARCHAR(255) PRIMARY KEY,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Slack event audit
CREATE TABLE IF NOT EXISTS public.slack_event_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255),
    event_type VARCHAR(100),
    team_id VARCHAR(255),
    event_data JSONB,
    artifacts_found INTEGER DEFAULT 0,
    predictions_made INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- WEBHOOK TABLES
-- ============================================================================

-- Webhooks configuration
CREATE TABLE IF NOT EXISTS public.webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    url TEXT NOT NULL,
    secret TEXT,
    active BOOLEAN DEFAULT TRUE,
    workspace_id UUID REFERENCES public.workspaces(id) ON DELETE CASCADE,
    events TEXT[] DEFAULT '{}',
    headers JSONB DEFAULT '{}',
    retry_config JSONB DEFAULT '{"max_retries": 3, "timeout": 30}',
    last_triggered_at TIMESTAMPTZ,
    trigger_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Webhook logs
CREATE TABLE IF NOT EXISTS public.webhook_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID NOT NULL REFERENCES public.webhooks(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    request_headers JSONB,
    request_body JSONB,
    response_headers JSONB,
    response_body TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Webhook delivery queue
CREATE TABLE IF NOT EXISTS public.webhook_delivery_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    webhook_id UUID NOT NULL REFERENCES public.webhooks(id) ON DELETE CASCADE,
    payload JSONB NOT NULL,
    headers JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 5,
    status VARCHAR(50) DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    next_retry_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Webhook IP whitelist
CREATE TABLE IF NOT EXISTS public.webhook_ip_whitelist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ip_address INET,
    ip_range CIDR,
    description TEXT,
    platform VARCHAR(50),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- AUTHENTICATION & SECURITY TABLES
-- ============================================================================

-- API keys
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    workspace_id UUID NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
    scopes TEXT[] DEFAULT '{}',
    rate_limit INTEGER,
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(255),
    revoked_by VARCHAR(255),
    revoked_at TIMESTAMPTZ
);

-- Auth tokens
CREATE TABLE IF NOT EXISTS public.auth_tokens (
    id UUID PRIMARY KEY,
    token_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(255),
    workspace_id UUID,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

-- OAuth states
CREATE TABLE IF NOT EXISTS public.oauth_states (
    state VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Revoked tokens
CREATE TABLE IF NOT EXISTS public.revoked_tokens (
    jti VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    revoked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    reason VARCHAR(255)
);

-- Failed auth attempts
CREATE TABLE IF NOT EXISTS public.failed_auth_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identifier VARCHAR(255) NOT NULL,
    attempt_type VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    error_reason VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Security audit log
CREATE TABLE IF NOT EXISTS public.security_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    workspace_id UUID,
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    severity VARCHAR(20) DEFAULT 'info',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS public.api_usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    workspace_id UUID,
    auth_method VARCHAR(50) NOT NULL,
    request_id VARCHAR(255),
    endpoint VARCHAR(255),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- RATE LIMITING TABLES
-- ============================================================================

-- Rate limit tracking
CREATE TABLE IF NOT EXISTS public.rate_limit_tracking (
    id BIGSERIAL PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL,
    timestamp BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Rate limit overrides
CREATE TABLE IF NOT EXISTS public.rate_limit_overrides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id VARCHAR(255) NOT NULL UNIQUE,
    rate_limit INTEGER NOT NULL,
    window_seconds INTEGER NOT NULL DEFAULT 60,
    reason TEXT,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- ============================================================================
-- STORED FUNCTIONS
-- ============================================================================

-- Webhook stats update function
CREATE OR REPLACE FUNCTION public.update_webhook_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status_code >= 200 AND NEW.status_code < 300 THEN
        UPDATE webhooks 
        SET last_triggered_at = NOW(),
            trigger_count = trigger_count + 1,
            consecutive_failures = 0
        WHERE id = NEW.webhook_id;
    ELSE
        UPDATE webhooks 
        SET last_triggered_at = NOW(),
            trigger_count = trigger_count + 1,
            consecutive_failures = consecutive_failures + 1,
            last_error = NEW.error_message
        WHERE id = NEW.webhook_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Auto-disable failing webhooks
CREATE OR REPLACE FUNCTION public.auto_disable_failing_webhooks()
RETURNS VOID AS $$
BEGIN
    UPDATE webhooks 
    SET active = FALSE 
    WHERE consecutive_failures >= 10 AND active = TRUE;
END;
$$ LANGUAGE plpgsql;

-- Cleanup expired tokens
CREATE OR REPLACE FUNCTION public.cleanup_expired_tokens()
RETURNS VOID AS $$
BEGIN
    DELETE FROM revoked_tokens WHERE expires_at < NOW();
    DELETE FROM rate_limit_tracking WHERE timestamp < EXTRACT(EPOCH FROM NOW() - INTERVAL '1 hour');
    DELETE FROM oauth_states WHERE expires_at < NOW();
    DELETE FROM failed_auth_attempts WHERE created_at < NOW() - INTERVAL '24 hours';
END;
$$ LANGUAGE plpgsql;

-- Cleanup old training data
CREATE OR REPLACE FUNCTION public.cleanup_old_training_data(p_days_to_keep INTEGER DEFAULT 90)
RETURNS TABLE(
    deleted_examples BIGINT,
    deleted_history BIGINT,
    deleted_audit BIGINT,
    deleted_cache BIGINT
) AS $$
DECLARE
    v_cutoff_date TIMESTAMPTZ;
    v_deleted_examples BIGINT;
    v_deleted_history BIGINT;
    v_deleted_audit BIGINT;
    v_deleted_cache BIGINT;
BEGIN
    v_cutoff_date := NOW() - (p_days_to_keep || ' days')::INTERVAL;
    
    -- Delete old training examples that are not validated
    DELETE FROM training_examples 
    WHERE created_at < v_cutoff_date 
        AND is_validated = false;
    GET DIAGNOSTICS v_deleted_examples = ROW_COUNT;
    
    -- Delete old training history (keep only recent N entries per workspace)
    WITH ranked_history AS (
        SELECT id,
               ROW_NUMBER() OVER (PARTITION BY workspace_id ORDER BY created_at DESC) as rn
        FROM training_history
    )
    DELETE FROM training_history
    WHERE id IN (
        SELECT id FROM ranked_history WHERE rn > 100
    );
    GET DIAGNOSTICS v_deleted_history = ROW_COUNT;
    
    -- Delete old audit logs
    DELETE FROM training_audit_log
    WHERE created_at < v_cutoff_date;
    GET DIAGNOSTICS v_deleted_audit = ROW_COUNT;
    
    -- Delete expired cache entries
    DELETE FROM prediction_cache
    WHERE expires_at < NOW();
    GET DIAGNOSTICS v_deleted_cache = ROW_COUNT;
    
    RETURN QUERY SELECT v_deleted_examples, v_deleted_history, v_deleted_audit, v_deleted_cache;
END;
$$ LANGUAGE plpgsql;

-- Get workspace training stats
CREATE OR REPLACE FUNCTION public.get_workspace_training_stats(p_workspace_id UUID)
RETURNS TABLE(
    total_examples BIGINT,
    validated_examples BIGINT,
    avg_feedback_score DOUBLE PRECISION,
    last_training TIMESTAMPTZ,
    current_accuracy DOUBLE PRECISION,
    training_runs INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(te.id) as total_examples,
        COUNT(te.id) FILTER (WHERE te.is_validated = true) as validated_examples,
        AVG(te.feedback_score) FILTER (WHERE te.feedback_score IS NOT NULL) as avg_feedback_score,
        MAX(th.created_at) as last_training,
        wm.best_accuracy as current_accuracy,
        wm.total_training_runs as training_runs
    FROM workspaces w
    LEFT JOIN training_examples te ON te.workspace_id = w.id
    LEFT JOIN training_history th ON th.workspace_id = w.id
    LEFT JOIN workspace_models wm ON wm.workspace_id = w.id
    WHERE w.id = p_workspace_id
    GROUP BY wm.best_accuracy, wm.total_training_runs;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Workspace indexes
CREATE INDEX IF NOT EXISTS idx_workspaces_active ON public.workspaces(active) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_workspaces_ml_enabled ON public.workspaces(ml_enabled) WHERE ml_enabled = TRUE;
CREATE INDEX IF NOT EXISTS idx_workspaces_enterprise ON public.workspaces(slack_enterprise_id) WHERE slack_enterprise_id IS NOT NULL;

-- Artifact indexes
CREATE INDEX IF NOT EXISTS idx_artifacts_workspace ON public.artifacts(workspace_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_platform ON public.artifacts(platform);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON public.artifacts(created_at DESC);

-- Outcome indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_workspace ON public.outcomes(workspace_id);

-- Training indexes
CREATE INDEX IF NOT EXISTS idx_training_examples_workspace ON public.training_examples(workspace_id);
CREATE INDEX IF NOT EXISTS idx_training_examples_created ON public.training_examples(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_examples_validation ON public.training_examples(workspace_id, is_validated, feedback_score) 
    WHERE is_validated = TRUE OR feedback_score > 0.7;
CREATE INDEX IF NOT EXISTS idx_training_examples_workspace_feedback ON public.training_examples(workspace_id, created_at DESC) 
    WHERE user_feedback IS NOT NULL;

-- Training history indexes
CREATE INDEX IF NOT EXISTS idx_history_workspace ON public.training_history(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_history_accuracy ON public.training_history(workspace_id, accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_history_created ON public.training_history(created_at DESC);

-- Training queue indexes
CREATE INDEX IF NOT EXISTS idx_queue_status ON public.training_queue(status, priority DESC, created_at) 
    WHERE status IN ('pending', 'processing');
CREATE INDEX IF NOT EXISTS idx_queue_workspace ON public.training_queue(workspace_id, status);

-- Model indexes
CREATE INDEX IF NOT EXISTS idx_models_updated ON public.workspace_models(last_updated DESC);
CREATE INDEX IF NOT EXISTS idx_versions_workspace ON public.model_versions(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_versions_active ON public.model_versions(workspace_id, is_active) WHERE is_active = TRUE;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_model_performance_workspace_date ON public.model_performance(workspace_id, date DESC);

-- Cache indexes
CREATE INDEX IF NOT EXISTS idx_cache_lookup ON public.prediction_cache(workspace_id, input_hash, model_version);
CREATE INDEX IF NOT EXISTS idx_cache_expiry ON public.prediction_cache(expires_at);

-- Inference indexes
CREATE INDEX IF NOT EXISTS idx_inference_workspace ON public.inference_history(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inference_artifact_id ON public.inference_history(artifact_id);

-- Feature importance indexes
CREATE INDEX IF NOT EXISTS idx_importance_workspace ON public.feature_importance(workspace_id, model_version);

-- Audit indexes
CREATE INDEX IF NOT EXISTS idx_audit_workspace ON public.training_audit_log(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action ON public.training_audit_log(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON public.training_audit_log(actor_id, created_at DESC);

-- Vocabulary indexes
CREATE INDEX IF NOT EXISTS idx_org_vocabulary_workspace ON public.org_vocabulary(workspace_id);

-- Feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_events_workspace_timestamp ON public.feedback_events(workspace_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_events_artifact ON public.feedback_events(artifact_id);

-- Webhook indexes
CREATE INDEX IF NOT EXISTS idx_webhooks_workspace ON public.webhooks(workspace_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_platform ON public.webhooks(platform) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_webhooks_active ON public.webhooks(active);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_webhook ON public.webhook_logs(webhook_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_created ON public.webhook_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_logs_status ON public.webhook_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_webhook_queue_webhook ON public.webhook_delivery_queue(webhook_id);
CREATE INDEX IF NOT EXISTS idx_webhook_queue_status ON public.webhook_delivery_queue(status, next_retry_at) 
    WHERE status IN ('pending', 'retry');
CREATE INDEX IF NOT EXISTS idx_webhook_ip_whitelist_ip ON public.webhook_ip_whitelist(ip_address) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_webhook_ip_whitelist_range ON public.webhook_ip_whitelist(ip_range) WHERE active = TRUE;

-- API key indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_workspace ON public.api_keys(workspace_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON public.api_keys(key_hash) WHERE revoked = FALSE;
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON public.api_keys(expires_at) WHERE expires_at IS NOT NULL;

-- API usage indexes
CREATE INDEX IF NOT EXISTS idx_api_usage_user ON public.api_usage(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_workspace ON public.api_usage(workspace_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_created ON public.api_usage(created_at DESC);

-- OAuth indexes
CREATE INDEX IF NOT EXISTS idx_oauth_states_created ON public.oauth_states(created_at);
CREATE INDEX IF NOT EXISTS idx_oauth_states_expires ON public.oauth_states(expires_at);

-- Token indexes
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires ON public.revoked_tokens(expires_at);

-- Failed auth indexes
CREATE INDEX IF NOT EXISTS idx_failed_auth_identifier ON public.failed_auth_attempts(identifier, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_failed_auth_ip ON public.failed_auth_attempts(ip_address, created_at DESC);

-- Security audit indexes
CREATE INDEX IF NOT EXISTS idx_security_audit_user ON public.security_audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_audit_type ON public.security_audit_log(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_audit_severity ON public.security_audit_log(severity, created_at DESC);

-- Rate limit indexes
CREATE INDEX IF NOT EXISTS idx_rate_limit_client ON public.rate_limit_tracking(client_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_rate_limit_timestamp ON public.rate_limit_tracking(timestamp);
CREATE INDEX IF NOT EXISTS idx_rate_limit_overrides_client ON public.rate_limit_overrides(client_id) WHERE active = TRUE;

-- Slack indexes
CREATE INDEX IF NOT EXISTS idx_slack_authed_users_workspace ON public.slack_authed_users(workspace_id);
CREATE INDEX IF NOT EXISTS idx_slack_command_usage_team_id ON public.slack_command_usage(team_id);

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_workspace_analytics_workspace_date ON public.workspace_analytics(workspace_id, date);
CREATE INDEX IF NOT EXISTS idx_workspace_analytics_date ON public.workspace_analytics(date);
CREATE INDEX IF NOT EXISTS idx_event_metrics_created ON public.event_metrics(created_at);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Updated_at triggers for all tables that need them
CREATE TRIGGER update_artifacts_updated_at BEFORE UPDATE ON public.artifacts 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_outcomes_updated_at BEFORE UPDATE ON public.outcomes 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER update_artifact_outcomes_updated_at BEFORE UPDATE ON public.artifact_outcomes 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER set_updated_at_artifact_outcomes BEFORE UPDATE ON public.artifact_outcomes 
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER update_training_examples_updated_at BEFORE UPDATE ON public.training_examples 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_training_queue_updated_at BEFORE UPDATE ON public.training_queue 
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Webhook stats trigger
CREATE TRIGGER webhook_stats_trigger AFTER INSERT ON public.webhook_logs 
    FOR EACH ROW EXECUTE FUNCTION public.update_webhook_stats();

-- ============================================================================
-- PERMISSIONS AND SECURITY
-- ============================================================================

-- Enable Row Level Security on sensitive tables
ALTER TABLE public.workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.auth_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.training_examples ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.webhooks ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- FINAL MIGRATION TRACKING
-- ============================================================================

-- Record successful migration
INSERT INTO _sqlx_migrations (
    version,
    description,
    success,
    checksum,
    execution_time
) VALUES (
    1,
    'Initial complete database schema with ML training, webhooks, and workspace management',
    TRUE,
    E'\\x01234567890ABCDEF01234567890ABCDEF01234567890ABCDEF01234567890ABCDEF',  -- Replace with actual checksum
    EXTRACT(EPOCH FROM (clock_timestamp() - transaction_timestamp()) * 1000)::BIGINT
);

-- ============================================================================
-- COMMIT TRANSACTION
-- ============================================================================

COMMIT;

-- ============================================================================
-- POST-MIGRATION VERIFICATION
-- ============================================================================

DO $$
DECLARE
    v_table_count INTEGER;
    v_index_count INTEGER;
    v_function_count INTEGER;
    v_trigger_count INTEGER;
BEGIN
    -- Count created objects
    SELECT COUNT(*) INTO v_table_count 
    FROM information_schema.tables 
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    
    SELECT COUNT(*) INTO v_index_count 
    FROM pg_indexes 
    WHERE schemaname = 'public';
    
    SELECT COUNT(*) INTO v_function_count 
    FROM information_schema.routines 
    WHERE routine_schema = 'public';
    
    SELECT COUNT(*) INTO v_trigger_count 
    FROM information_schema.triggers 
    WHERE trigger_schema = 'public';
    
    -- Log migration statistics
    RAISE NOTICE 'Migration completed successfully!';
    RAISE NOTICE 'Tables created: %', v_table_count;
    RAISE NOTICE 'Indexes created: %', v_index_count;
    RAISE NOTICE 'Functions created: %', v_function_count;
    RAISE NOTICE 'Triggers created: %', v_trigger_count;
END $$;
