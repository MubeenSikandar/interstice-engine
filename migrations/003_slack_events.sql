-- Add these tables to your migrations
CREATE TABLE IF NOT EXISTS slack_events (
    event_id VARCHAR(255) PRIMARY KEY,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS oauth_states (
    state VARCHAR(255) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_oauth_states_created ON oauth_states(created_at);

CREATE TABLE IF NOT EXISTS event_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform VARCHAR(50) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    team_id VARCHAR(255),
    artifact_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_event_metrics_created ON event_metrics(created_at);

CREATE TABLE IF NOT EXISTS slack_event_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id VARCHAR(255),
    event_type VARCHAR(100),
    team_id VARCHAR(255),
    event_data JSONB,
    artifacts_found INTEGER DEFAULT 0,
    predictions_made INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_slack_audit_created ON slack_event_audit(created_at);
CREATE INDEX idx_slack_audit_team ON slack_event_audit(team_id);