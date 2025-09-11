-- Migration 002: Add prediction-related tables
-- Created: 2024-01-01
-- Description: Adds tables for ML prediction storage, feedback, and training data

-- Prediction records table
CREATE TABLE IF NOT EXISTS prediction_records (
    id UUID PRIMARY KEY,
    predictions JSONB NOT NULL,
    artifact_ids JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    confidence_avg DOUBLE PRECISION,
    prediction_count INTEGER,
    highest_impact DOUBLE PRECISION,
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE INDEX idx_prediction_records_created_at ON prediction_records(created_at DESC);
CREATE INDEX idx_prediction_records_confidence ON prediction_records(confidence_avg DESC);
CREATE INDEX idx_prediction_records_workspace ON prediction_records(workspace_id);

-- Individual predictions for better querying
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY,
    record_id UUID REFERENCES prediction_records(id) ON DELETE CASCADE,
    outcome_id UUID NOT NULL,
    outcome_name TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    reasoning TEXT,
    suggested_targets JSONB,
    estimated_impact DOUBLE PRECISION,
    recommended_priority JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    feedback_count INTEGER DEFAULT 0,
    accuracy_rate DOUBLE PRECISION DEFAULT 0.5,
    UNIQUE(outcome_id, record_id)
);

CREATE INDEX idx_predictions_outcome ON predictions(outcome_id);
CREATE INDEX idx_predictions_confidence ON predictions(confidence DESC);
CREATE INDEX idx_predictions_record ON predictions(record_id);

-- Artifact-prediction associations
CREATE TABLE IF NOT EXISTS artifact_predictions (
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    prediction_id UUID REFERENCES prediction_records(id) ON DELETE CASCADE,
    outcome_id UUID NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    occurrence_count INTEGER DEFAULT 1,
    PRIMARY KEY (artifact_id, outcome_id)
);

CREATE INDEX idx_artifact_predictions_confidence ON artifact_predictions(confidence DESC);

-- Prediction feedback table
CREATE TABLE IF NOT EXISTS prediction_feedback (
    id UUID PRIMARY KEY,
    prediction_id UUID NOT NULL,
    actual_outcome TEXT NOT NULL,
    was_accurate BOOLEAN NOT NULL,
    feedback_at TIMESTAMPTZ NOT NULL,
    confidence_adjustment DOUBLE PRECISION,
    user_id TEXT,
    notes TEXT
);

CREATE INDEX idx_prediction_feedback_prediction ON prediction_feedback(prediction_id);
CREATE INDEX idx_prediction_feedback_time ON prediction_feedback(feedback_at DESC);

-- ML training data
CREATE TABLE IF NOT EXISTS ml_training_data (
    id UUID PRIMARY KEY,
    prediction_id UUID NOT NULL,
    actual_outcome TEXT NOT NULL,
    was_accurate BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    training_batch UUID,
    processed BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_ml_training_unprocessed ON ml_training_data(processed) WHERE NOT processed;
