-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Workspaces table
CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_team_id VARCHAR(255) UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Artifacts table
CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    raw_text TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Outcomes table
CREATE TABLE outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    target_value DOUBLE PRECISION,
    current_value DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Junction table for artifact-outcome relationships
CREATE TABLE artifact_outcomes (
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    outcome_id UUID REFERENCES outcomes(id) ON DELETE CASCADE,
    confidence DOUBLE PRECISION DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (artifact_id, outcome_id)
);

-- Indexes for performance
CREATE INDEX idx_artifacts_workspace ON artifacts(workspace_id);
CREATE INDEX idx_artifacts_created ON artifacts(created_at DESC);
CREATE INDEX idx_outcomes_workspace ON outcomes(workspace_id);
