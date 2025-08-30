-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector"; -- For pgvector

-- Slack workspace information
CREATE TABLE slack_workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    team_id VARCHAR(255) UNIQUE NOT NULL,
    team_name VARCHAR(255),
    bot_user_id VARCHAR(255),
    bot_access_token TEXT, -- Will encrypt at application level
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE slack_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES slack_workspaces(id) ON DELETE CASCADE,
    channel_id VARCHAR(255) NOT NULL,
    channel_name VARCHAR(255),
    is_private BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(workspace_id, channel_id)
);

-- Training data collection
CREATE TABLE training_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    input_text TEXT NOT NULL,
    input_embedding vector(768), -- pgvector type
    suggested_outcome_id UUID REFERENCES outcomes(id) ON DELETE SET NULL,
    actual_outcome_id UUID REFERENCES outcomes(id) ON DELETE SET NULL,
    user_feedback VARCHAR(50), -- 'accepted', 'rejected', 'corrected', 'implicit'
    feedback_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Organization vocabulary learning
CREATE TABLE org_vocabulary (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    term TEXT NOT NULL,
    term_type VARCHAR(50), -- 'project', 'team', 'metric', 'tool'
    frequency INTEGER DEFAULT 1,
    embedding vector(768),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (workspace_id, term)
);

-- Model performance tracking
CREATE TABLE model_performance (
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    predictions_made INTEGER DEFAULT 0,
    predictions_accepted INTEGER DEFAULT 0,
    predictions_rejected INTEGER DEFAULT 0,
    avg_confidence FLOAT,
    PRIMARY KEY (workspace_id, date)
);

-- Indexes for performance
CREATE INDEX idx_training_examples_workspace ON training_examples(workspace_id);
CREATE INDEX idx_training_examples_created ON training_examples(created_at DESC);
CREATE INDEX idx_org_vocabulary_workspace ON org_vocabulary(workspace_id);
CREATE INDEX idx_model_performance_workspace_date ON model_performance(workspace_id, date DESC);