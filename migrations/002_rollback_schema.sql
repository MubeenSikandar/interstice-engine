BEGIN;

-- Drop all triggers
DROP TRIGGER IF EXISTS webhook_stats_trigger ON public.webhook_logs;
DROP TRIGGER IF EXISTS update_training_queue_updated_at ON public.training_queue;
DROP TRIGGER IF EXISTS update_training_examples_updated_at ON public.training_examples;
DROP TRIGGER IF EXISTS set_updated_at_artifact_outcomes ON public.artifact_outcomes;
DROP TRIGGER IF EXISTS update_artifact_outcomes_updated_at ON public.artifact_outcomes;
DROP TRIGGER IF EXISTS update_outcomes_updated_at ON public.outcomes;
DROP TRIGGER IF EXISTS update_artifacts_updated_at ON public.artifacts;

-- Drop all functions
DROP FUNCTION IF EXISTS public.get_workspace_training_stats(UUID);
DROP FUNCTION IF EXISTS public.cleanup_old_training_data(INTEGER);
DROP FUNCTION IF EXISTS public.cleanup_expired_tokens();
DROP FUNCTION IF EXISTS public.auto_disable_failing_webhooks();
DROP FUNCTION IF EXISTS public.update_webhook_stats();
DROP FUNCTION IF EXISTS public.set_updated_at();
DROP FUNCTION IF EXISTS public.update_updated_at();
DROP FUNCTION IF EXISTS public.update_updated_at_column();

-- Drop all tables (in reverse dependency order)
DROP TABLE IF EXISTS public.webhook_ip_whitelist CASCADE;
DROP TABLE IF EXISTS public.webhook_delivery_queue CASCADE;
DROP TABLE IF EXISTS public.webhook_logs CASCADE;
DROP TABLE IF EXISTS public.webhooks CASCADE;
DROP TABLE IF EXISTS public.slack_event_audit CASCADE;
DROP TABLE IF EXISTS public.slack_events CASCADE;
DROP TABLE IF EXISTS public.slack_command_usage CASCADE;
DROP TABLE IF EXISTS public.slack_conversations CASCADE;
DROP TABLE IF EXISTS public.slack_authed_users CASCADE;
DROP TABLE IF EXISTS public.slack_workspaces CASCADE;
DROP TABLE IF EXISTS public.rate_limit_overrides CASCADE;
DROP TABLE IF EXISTS public.rate_limit_tracking CASCADE;
DROP TABLE IF EXISTS public.api_usage CASCADE;
DROP TABLE IF EXISTS public.security_audit_log CASCADE;
DROP TABLE IF EXISTS public.failed_auth_attempts CASCADE;
DROP TABLE IF EXISTS public.revoked_tokens CASCADE;
DROP TABLE IF EXISTS public.oauth_states CASCADE;
DROP TABLE IF EXISTS public.auth_tokens CASCADE;
DROP TABLE IF EXISTS public.api_keys CASCADE;
DROP TABLE IF EXISTS public.workspace_analytics CASCADE;
DROP TABLE IF EXISTS public.event_metrics CASCADE;
DROP TABLE IF EXISTS public.system_events CASCADE;
DROP TABLE IF EXISTS public.metrics CASCADE;
DROP TABLE IF EXISTS public.feedback_events CASCADE;
DROP TABLE IF EXISTS public.org_vocabulary CASCADE;
DROP TABLE IF EXISTS public.model_update_queue CASCADE;
DROP TABLE IF EXISTS public.training_audit_log CASCADE;
DROP TABLE IF EXISTS public.prediction_cache CASCADE;
DROP TABLE IF EXISTS public.inference_history CASCADE;
DROP TABLE IF EXISTS public.feature_importance CASCADE;
DROP TABLE IF EXISTS public.model_performance CASCADE;
DROP TABLE IF EXISTS public.model_versions CASCADE;
DROP TABLE IF EXISTS public.workspace_models CASCADE;
DROP TABLE IF EXISTS public.training_queue CASCADE;
DROP TABLE IF EXISTS public.training_history CASCADE;
DROP TABLE IF EXISTS public.training_examples CASCADE;
DROP TABLE IF EXISTS public.outcome_targets CASCADE;
DROP TABLE IF EXISTS public.artifact_outcomes CASCADE;
DROP TABLE IF EXISTS public.outcomes CASCADE;
DROP TABLE IF EXISTS public.artifacts CASCADE;
DROP TABLE IF EXISTS public.workspaces CASCADE;

-- Drop custom types
DROP TYPE IF EXISTS public.validation_method CASCADE;
DROP TYPE IF EXISTS public.risk_level CASCADE;
DROP TYPE IF EXISTS public.priority CASCADE;
DROP TYPE IF EXISTS public.platform CASCADE;
DROP TYPE IF EXISTS public.outcome_type CASCADE;
DROP TYPE IF EXISTS public.outcome_state CASCADE;
DROP TYPE IF EXISTS public.automation_level CASCADE;

-- Drop extensions
DROP EXTENSION IF EXISTS vector CASCADE;
DROP EXTENSION IF EXISTS "uuid-ossp" CASCADE;

-- Remove migration record
DELETE FROM _sqlx_migrations WHERE version = 1;
