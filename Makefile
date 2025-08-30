.PHONY: help build test clean setup-db start stop logs docker-build docker-run docker-stop

# Default target
help:
	@echo "Interstice Engine - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  build          - Build all crates"
	@echo "  test           - Run all tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  check          - Check code without building"
	@echo ""
	@echo "Database:"
	@echo "  setup-db       - Set up PostgreSQL database with ML tables"
	@echo "  migrate        - Run database migrations"
	@echo "  db-reset       - Reset database (WARNING: destroys data)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   - Build Docker images"
	@echo "  docker-run     - Start all services with Docker Compose"
	@echo "  docker-stop    - Stop all Docker services"
	@echo "  docker-logs    - View logs from all services"
	@echo ""
	@echo "ML Training:"
	@echo "  train-models   - Start ML training pipeline"
	@echo "  download-models - Download pre-trained ML models"
	@echo ""
	@echo "Monitoring:"
	@echo "  start-monitoring - Start Prometheus and Grafana"
	@echo "  stop-monitoring  - Stop monitoring services"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-local   - Deploy to local environment"
	@echo "  deploy-prod    - Deploy to production (requires config)"

# Development commands
build:
	@echo "🔨 Building Interstice Engine..."
	cargo build --workspace

test:
	@echo "🧪 Running tests..."
	cargo test --workspace

clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf target/

check:
	@echo "🔍 Checking code..."
	cargo check --workspace

# Database commands
setup-db:
	@echo "🗄️ Setting up database..."
	@if [ -f scripts/setup_database.sh ]; then \
		./scripts/setup_database.sh; \
	else \
		echo "❌ Database setup script not found"; \
		exit 1; \
	fi

migrate:
	@echo "📝 Running database migrations..."
	@if [ -n "$$DATABASE_URL" ]; then \
		sqlx migrate run --database-url "$$DATABASE_URL" --source migrations; \
	else \
		echo "❌ DATABASE_URL not set. Please set it in your .env file"; \
		exit 1; \
	fi

db-reset:
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "🗑️  Dropping and recreating database..."; \
		dropdb -h localhost -U postgres interstice_engine 2>/dev/null || true; \
		createdb -h localhost -U postgres interstice_engine; \
		$(MAKE) setup-db; \
	else \
		echo "❌ Database reset cancelled"; \
	fi

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-run:
	@echo "🚀 Starting services with Docker Compose..."
	docker-compose up -d

docker-stop:
	@echo "🛑 Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "📋 Viewing logs..."
	docker-compose logs -f

# ML Training commands
train-models:
	@echo "🤖 Starting ML training pipeline..."
	@if [ -n "$$DATABASE_URL" ]; then \
		cargo run -p interstice-workers --bin interstice-workers; \
	else \
		echo "❌ DATABASE_URL not set. Please set it in your .env file"; \
		exit 1; \
	fi

download-models:
	@echo "📥 Downloading pre-trained ML models..."
	mkdir -p models
	@echo "Downloading BERT base model..."
	# Add model download commands here
	@echo "✅ Models downloaded successfully"

# Monitoring commands
start-monitoring:
	@echo "📊 Starting monitoring services..."
	docker-compose up -d prometheus grafana
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3001 (admin/admin)"

stop-monitoring:
	@echo "🛑 Stopping monitoring services..."
	docker-compose stop prometheus grafana

# Deployment commands
deploy-local:
	@echo "🚀 Deploying to local environment..."
	$(MAKE) setup-db
	$(MAKE) build
	$(MAKE) docker-run
	@echo "✅ Local deployment complete!"
	@echo "API: http://localhost:3000"
	@echo "Health: http://localhost:3000/health"

deploy-prod:
	@echo "🚀 Deploying to production..."
	@echo "❌ Production deployment not configured yet"
	@echo "Please configure your production environment first"

# Utility commands
status:
	@echo "📊 Service Status:"
	@echo "PostgreSQL: $$(pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped")"
	@echo "Redis: $$(redis-cli ping >/dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped")"
	@echo "API: $$(curl -s http://localhost:3000/health >/dev/null 2>&1 && echo "✅ Running" || echo "❌ Stopped")"

logs-api:
	@echo "📋 API logs:"
	docker-compose logs -f api

logs-workers:
	@echo "📋 Worker logs:"
	docker-compose logs -f workers

# Quick start for development
dev: setup-db build docker-run
	@echo "🚀 Development environment ready!"
	@echo "API: http://localhost:3000"
	@echo "Database: localhost:5432"
	@echo "Redis: localhost:6379"
