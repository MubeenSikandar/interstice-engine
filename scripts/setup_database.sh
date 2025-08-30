#!/bin/bash

# Database setup script for Interstice Engine
# This script sets up the PostgreSQL database with required extensions and tables

set -e

# Configuration
DB_NAME="interstice_engine"
DB_USER="postgres"
DB_PASSWORD="root1234"
DB_HOST="localhost"
DB_PORT="5432"

echo "🚀 Setting up Interstice Engine database..."

# Check if PostgreSQL is running
if ! pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER > /dev/null 2>&1; then
    echo "❌ PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Create database if it doesn't exist
echo "📊 Creating database '$DB_NAME' if it doesn't exist..."
createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME 2>/dev/null || echo "Database already exists"

# Set environment variable for database connection
export DATABASE_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

# Install required extensions
echo "🔧 Installing required PostgreSQL extensions..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
    CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";
    CREATE EXTENSION IF NOT EXISTS \"vector\";
"

# Run migrations
echo "📝 Running database migrations..."
cd migrations

# Migration 001: Initial schema
echo "Running migration 001_initial.sql..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f 001_initial.sql

# Migration 002: ML tables
echo "Running migration 002_ml_tables.sql..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f 002_ml_tables.sql

cd ..

# Create additional indexes for performance
echo "⚡ Creating performance indexes..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
    -- Indexes for ML tables
    CREATE INDEX IF NOT EXISTS idx_training_examples_workspace_created 
    ON training_examples(workspace_id, created_at DESC);
    
    CREATE INDEX IF NOT EXISTS idx_org_vocabulary_term 
    ON org_vocabulary(term) USING gin(to_tsvector('english', term));
    
    CREATE INDEX IF NOT EXISTS idx_model_performance_workspace_date 
    ON model_performance(workspace_id, date DESC);
    
    -- Full-text search index for artifacts
    CREATE INDEX IF NOT EXISTS idx_artifacts_raw_text_search 
    ON artifacts USING gin(to_tsvector('english', raw_text));
"

# Insert sample data for testing
echo "🧪 Inserting sample data..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
    -- Insert sample workspace
    INSERT INTO workspaces (id, name, created_at) 
    VALUES (
        '550e8400-e29b-41d4-a716-446655440000',
        'Sample Workspace',
        NOW()
    ) ON CONFLICT DO NOTHING;
    
    -- Insert sample outcomes
    INSERT INTO outcomes (id, workspace_id, name, description, target_value, current_value) 
    VALUES 
    (
        '550e8400-e29b-41d4-a716-446655440001',
        '550e8400-e29b-41d4-a716-446655440000',
        'User Onboarding',
        'Improve user onboarding process',
        95.0,
        75.0
    ),
    (
        '550e8400-e29b-41d4-a716-446655440002',
        '550e8400-e29b-41d4-a716-446655440000',
        'Code Quality',
        'Maintain high code quality standards',
        90.0,
        85.0
    ) ON CONFLICT DO NOTHING;
"

echo "✅ Database setup completed successfully!"
echo ""
echo "📋 Database Information:"
echo "   Name: $DB_NAME"
echo "   Host: $DB_HOST:$DB_PORT"
echo "   User: $DB_USER"
echo "   Connection URL: $DATABASE_URL"
echo ""
echo "🔗 You can now connect to the database using:"
echo "   psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
echo ""
echo "🚀 Next steps:"
echo "   1. Update your .env file with the DATABASE_URL above"
echo "   2. Run 'cargo build' to compile the project"
echo "   3. Start the API server with 'cargo run -p interstice-api'"
