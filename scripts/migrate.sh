#!/bin/bash

# Interstice Engine Migration Script
# Usage: ./scripts/migrate.sh [up|down|status|reset]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default database URL
DEFAULT_DATABASE_URL="postgresql://postgres:root1234@localhost:5432/interstice_engine"

# Get database URL from environment or use default
DATABASE_URL=${DATABASE_URL:-$DEFAULT_DATABASE_URL}

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if database is accessible
check_database() {
    print_status "Checking database connection..."
    if ! psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null 2>&1; then
        print_error "Cannot connect to database. Please check your DATABASE_URL:"
        print_error "Current URL: $DATABASE_URL"
        print_error "Make sure PostgreSQL is running and the database exists."
        exit 1
    fi
    print_success "Database connection successful"
}

# Function to show migration status
show_status() {
    print_status "Migration Status:"
    echo ""
    if command -v sqlx > /dev/null 2>&1; then
        DATABASE_URL="$DATABASE_URL" sqlx migrate info
    else
        print_error "sqlx-cli not found. Please install it with: cargo install sqlx-cli"
        exit 1
    fi
}

# Function to migrate up
migrate_up() {
    print_status "Running migrations UP..."
    check_database
    
    if command -v sqlx > /dev/null 2>&1; then
        DATABASE_URL="$DATABASE_URL" sqlx migrate run
        print_success "Migrations completed successfully"
    else
        print_error "sqlx-cli not found. Please install it with: cargo install sqlx-cli"
        exit 1
    fi
}

# Function to migrate down
migrate_down() {
    print_status "Running migrations DOWN..."
    check_database
    
    if command -v sqlx > /dev/null 2>&1; then
        DATABASE_URL="$DATABASE_URL" sqlx migrate revert
        print_success "Migration reverted successfully"
    else
        print_error "sqlx-cli not found. Please install it with: cargo install sqlx-cli"
        exit 1
    fi
}

# Function to reset database (drop and recreate)
reset_database() {
    print_warning "This will DROP ALL DATA in the database!"
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_status "Reset cancelled"
        exit 0
    fi
    
    print_status "Resetting database..."
    check_database
    
    # Extract database name from URL
    DB_NAME=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    DB_URL_WITHOUT_DB=$(echo "$DATABASE_URL" | sed 's/\/[^?]*/\/postgres/')
    
    print_status "Dropping database: $DB_NAME"
    psql "$DB_URL_WITHOUT_DB" -c "DROP DATABASE IF EXISTS $DB_NAME;"
    
    print_status "Creating database: $DB_NAME"
    psql "$DB_URL_WITHOUT_DB" -c "CREATE DATABASE $DB_NAME;"
    
    print_status "Running migrations on fresh database..."
    migrate_up
}

# Function to show help
show_help() {
    echo "Interstice Engine Migration Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  up      Run all pending migrations"
    echo "  down    Revert the last migration"
    echo "  status  Show current migration status"
    echo "  reset   Drop and recreate database with all migrations"
    echo "  help    Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  DATABASE_URL    PostgreSQL connection string (default: $DEFAULT_DATABASE_URL)"
    echo ""
    echo "Examples:"
    echo "  $0 up                    # Run migrations"
    echo "  $0 down                  # Revert last migration"
    echo "  $0 status                # Show migration status"
    echo "  DATABASE_URL=postgresql://user:pass@host:port/db $0 up"
}

# Main script logic
case "${1:-help}" in
    up)
        migrate_up
        ;;
    down)
        migrate_down
        ;;
    status)
        show_status
        ;;
    reset)
        reset_database
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
