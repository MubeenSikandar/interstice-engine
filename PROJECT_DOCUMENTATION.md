# Interstice Engine - Comprehensive Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Crate Structure](#crate-structure)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Database Schema](#database-schema)
7. [Machine Learning Pipeline](#machine-learning-pipeline)
8. [Deployment](#deployment)
9. [Configuration](#configuration)
10. [Development Guide](#development-guide)
11. [Monitoring & Observability](#monitoring--observability)
12. [Security](#security)
13. [Testing](#testing)
14. [Contributing](#contributing)

## Project Overview

**Interstice Engine** is a production-ready, enterprise-grade platform that intelligently processes work artifacts across multiple platforms (Slack, GitHub, Jira, etc.) and uses machine learning to predict and map outcomes. The system provides continuous learning capabilities, real-time processing, and comprehensive analytics.

### Key Features

- **Intelligent Artifact Detection**: Automatically identifies work artifacts from various platforms
- **ML-Powered Outcome Mapping**: Uses machine learning to suggest relevant outcomes
- **Continuous Learning**: Improves predictions over time with user feedback
- **Multi-Platform Support**: Slack, GitHub, Jira, Teams, Asana, and more
- **Production Ready**: Docker, monitoring, and deployment configurations included
- **Real-time Processing**: Event-driven architecture with background workers
- **Comprehensive Analytics**: Detailed metrics and reporting capabilities

### Technology Stack

- **Language**: Rust (1.75+)
- **Web Framework**: Axum
- **Database**: PostgreSQL 15+ with pgvector extension
- **Cache**: Redis 7+
- **ML Framework**: Candle (Rust ML), ONNX Runtime
- **Containerization**: Docker & Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Message Queue**: Redis-based job processing

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Platform      │    │   ML Pipeline   │    │   Core Engine   │
│   Adapters      │◄──►│                 │◄──►│                 │
│                 │    │ • Embeddings    │    │ • Artifact      │
│ • Slack         │    │ • Predictions   │    │   Extraction    │
│ • GitHub        │    │ • Training      │    │ • Outcome       │
│ • Jira          │    │ • Feedback      │    │   Mapping      │
│ • Teams         │    │ • Vocabulary    │    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Storage       │
                    │   Layer         │
                    │                 │
                    │ • PostgreSQL    │
                    │ • Redis         │
                    │ • Vector DB     │
                    └─────────────────┘
```

### System Components

1. **API Server** (`interstice-api`): HTTP API and webhook handlers
2. **Background Workers** (`interstice-workers`): Asynchronous processing tasks
3. **Core Engine** (`interstice-core`): Business logic and domain models
4. **Platform Adapters** (`interstice-adapters`): Platform-specific integrations
5. **ML Pipeline** (`interstice-ml`): Machine learning and prediction engine

## Crate Structure

### 1. interstice-core

**Purpose**: Core domain models, business logic, and shared types

**Key Modules**:

- `artifact.rs`: Artifact extraction and processing logic
- `outcome.rs`: Outcome mapping and management
- `storage.rs`: Storage backend abstractions
- `analytics.rs`: Metrics and analytics collection
- `graph.rs`: Evidence graph construction
- `traits.rs`: Core trait definitions
- `types.rs`: Shared type definitions
- `error.rs`: Error handling and types

**Dependencies**:

- `tokio`: Async runtime
- `serde`: Serialization
- `uuid`: UUID generation
- `chrono`: Date/time handling
- `sqlx`: Database access
- `redis`: Caching
- `candle-core`: ML tensor operations

### 2. interstice-adapters

**Purpose**: Platform-specific integrations and adapters

**Key Modules**:

- `slack/`: Slack integration with event handling
- `github/`: GitHub API integration
- `jira/`: Jira API integration
- `teams/`: Microsoft Teams integration
- `asana/`: Asana project management integration
- `traits.rs`: Adapter trait definitions

**Dependencies**:

- `slack-morphism`: Slack API client
- `octocrab`: GitHub API client
- `reqwest`: HTTP client
- `interstice-core`: Core types and traits

### 3. interstice-api

**Purpose**: HTTP API server and webhook handlers

**Key Modules**:

- `main.rs`: Application entry point and initialization
- `handlers/`: Request handlers for different endpoints
- `routes/`: Route definitions and organization
- `middleware_layer/`: Custom middleware implementations

**Dependencies**:

- `axum`: Web framework
- `tower`: Middleware and service abstractions
- `tower-http`: HTTP-specific middleware
- `sqlx`: Database access with migrations
- `slack-morphism`: Slack integration

### 4. interstice-ml

**Purpose**: Machine learning pipeline and prediction engine

**Key Modules**:

- `embeddings/`: Text embedding generation
- `inference/`: Model inference and prediction
- `training/`: Model training and continuous learning
- `feedback/`: User feedback processing
- `models/`: Model definitions and management
- `adapters.rs`: ML predictor adapters

**Dependencies**:

- `candle-core`, `candle-nn`, `candle-transformers`: ML framework
- `ort`: ONNX Runtime for model inference
- `tokenizers`: Text tokenization
- `pgvector`: Vector database operations
- `prometheus`: Metrics collection

### 5. interstice-workers

**Purpose**: Background job processing and scheduled tasks

**Key Modules**:

- `main.rs`: Worker entry point
- `jobs/`: Background job implementations
- `schedulers/`: Task scheduling logic

**Dependencies**:

- `tokio`: Async runtime
- `sqlx`: Database access
- `slack-morphism`: Slack integration
- `interstice-core`: Core functionality
- `interstice-ml`: ML pipeline access

## Core Components

### IntersticeEngine

The main engine that orchestrates artifact processing:

```rust
pub struct IntersticeEngine {
    config: EngineConfig,
    extractor: Arc<ArtifactExtractor>,
    processor: Arc<ArtifactProcessor>,
    mapper: Arc<OutcomeMapper>,
    storage: Option<Arc<dyn StorageBackend>>,
    ml_predictor: Option<Arc<dyn MLPredictor>>,
    metrics: Arc<Metrics>,
}
```

**Key Methods**:

- `process()`: Main processing pipeline
- `extract_artifacts()`: Extract artifacts from content
- `store_processed_data()`: Persist processed data
- `health_check()`: System health monitoring

### Artifact Processing Pipeline

1. **Content Validation**: Size limits and format checking
2. **Artifact Extraction**: Platform-specific extraction logic
3. **ML Processing**: Embedding generation and prediction
4. **Outcome Mapping**: Map artifacts to relevant outcomes
5. **Storage**: Persist results to database
6. **Analytics**: Update metrics and monitoring

### ML Pipeline

The ML pipeline consists of several components:

1. **Text Embedder**: Converts text to vector embeddings
2. **Outcome Predictor**: Predicts relevant outcomes for artifacts
3. **Continuous Trainer**: Retrains models based on feedback
4. **Feedback Processor**: Processes user feedback for learning
5. **Vocabulary Learner**: Learns organization-specific terminology

## API Reference

### Health Endpoints

- `GET /health` - System health check
- `GET /health/detailed` - Detailed component health

### Webhook Endpoints

- `POST /webhooks/slack/events` - Slack event webhook
- `POST /webhooks/slack/interactions` - Slack interaction handling
- `POST /webhooks/github` - GitHub webhook
- `POST /webhooks/jira` - Jira webhook

### API Endpoints

- `GET /api/v1/workspaces/{id}/artifacts` - Get workspace artifacts
- `GET /api/v1/workspaces/{id}/outcomes` - Get workspace outcomes
- `POST /api/v1/workspaces/{id}/feedback` - Submit user feedback
- `GET /api/v1/workspaces/{id}/analytics` - Get analytics data

### Request/Response Examples

#### Health Check

```bash
curl http://localhost:3000/health
```

Response:

```json
{
  "healthy": true,
  "version": "0.1.0",
  "components": [
    {
      "name": "storage",
      "healthy": true,
      "message": "Storage backend is operational"
    },
    {
      "name": "ml_predictor",
      "healthy": true,
      "message": "ML predictor is configured"
    }
  ]
}
```

#### Process Artifacts

```bash
curl -X POST http://localhost:3000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Fixed bug in user authentication",
    "platform": "slack",
    "workspace_id": "123e4567-e89b-12d3-a456-426614174000"
  }'
```

## Database Schema

### Core Tables

#### workspaces

```sql
CREATE TABLE workspaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    slack_team_id VARCHAR(255) UNIQUE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### artifacts

```sql
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
```

#### outcomes

```sql
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
```

#### artifact_outcomes

```sql
CREATE TABLE artifact_outcomes (
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    outcome_id UUID REFERENCES outcomes(id) ON DELETE CASCADE,
    confidence DOUBLE PRECISION DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (artifact_id, outcome_id)
);
```

### ML Tables

#### training_examples

```sql
CREATE TABLE training_examples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
    input_text TEXT NOT NULL,
    suggested_outcome_id UUID REFERENCES outcomes(id),
    actual_outcome_id UUID REFERENCES outcomes(id),
    user_feedback JSONB,
    feedback_score DOUBLE PRECISION,
    context JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    is_validated BOOLEAN DEFAULT FALSE
);
```

#### model_metrics

```sql
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id UUID REFERENCES workspaces(id) ON DELETE CASCADE,
    model_version VARCHAR(50) NOT NULL,
    accuracy DOUBLE PRECISION,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Machine Learning Pipeline

### Architecture

The ML pipeline is designed for production use with the following components:

1. **Text Embedding**: BERT-based embeddings using Candle
2. **Outcome Prediction**: Organization-specific prediction models
3. **Continuous Training**: Retrains models based on user feedback
4. **Vocabulary Learning**: Learns organization-specific terminology
5. **Model Management**: Versioning and deployment of models

### Model Types

1. **Embedding Model**: Converts text to vector representations
2. **Classification Model**: Predicts outcome categories
3. **Confidence Model**: Estimates prediction confidence
4. **Vocabulary Model**: Learns organization-specific terms

### Training Process

1. **Data Collection**: Gathers artifacts and user feedback
2. **Preprocessing**: Cleans and prepares training data
3. **Feature Engineering**: Extracts relevant features
4. **Model Training**: Trains models using collected data
5. **Validation**: Evaluates model performance
6. **Deployment**: Deploys improved models to production

### Feedback Loop

1. **User Interaction**: Users interact with predictions
2. **Feedback Collection**: System collects explicit and implicit feedback
3. **Data Storage**: Feedback is stored for training
4. **Model Retraining**: Models are retrained with new data
5. **Performance Monitoring**: System monitors model performance

## Deployment

### Docker Deployment

The project includes comprehensive Docker configuration:

#### docker-compose.yml

- **PostgreSQL**: Database with pgvector extension
- **Redis**: Caching and job queues
- **API Server**: Main application server
- **Workers**: Background job processing
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

#### Dockerfile

Multi-stage build with:

- **Builder stage**: Compiles Rust application
- **Runtime stage**: Minimal Debian-based image

### Production Deployment

1. **Environment Setup**: Configure production environment variables
2. **Database Setup**: Set up PostgreSQL with pgvector
3. **SSL Configuration**: Configure SSL certificates
4. **Load Balancing**: Set up load balancer
5. **Monitoring**: Configure monitoring and alerting
6. **Backup**: Set up database backups

### Kubernetes Deployment

The project includes Kubernetes manifests in `deployments/kubernetes/`:

- **ConfigMaps**: Configuration management
- **Secrets**: Sensitive data management
- **Deployments**: Application deployments
- **Services**: Service definitions
- **Ingress**: External access configuration

## Configuration

### Environment Variables

#### Required Variables

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

#### Optional Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 3000)
- `RUST_LOG`: Log level (default: info)

#### Platform Integration

- `SLACK_BOT_TOKEN`: Slack bot token
- `SLACK_SIGNING_SECRET`: Slack signing secret
- `GITHUB_TOKEN`: GitHub API token
- `JIRA_API_TOKEN`: Jira API token

#### ML Configuration

- `MODEL_CACHE_DIR`: Model storage directory
- `EMBEDDING_MODEL_PATH`: Path to embedding model
- `MAX_SEQUENCE_LENGTH`: Maximum text sequence length
- `BATCH_SIZE`: ML batch size

### Configuration Profiles

#### Development

- Debug logging enabled
- Local database
- No SSL
- Reduced timeouts

#### Production

- Info logging
- Production database
- SSL enabled
- Optimized timeouts
- Monitoring enabled

## Development Guide

### Prerequisites

- Rust 1.75+
- PostgreSQL 15+ with pgvector
- Redis 7+
- Docker & Docker Compose

### Setup

1. **Clone Repository**

   ```bash
   git clone https://github.com/MubeenSikandar/interstice-engine.git
   cd interstice-engine
   ```

2. **Environment Setup**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Database Setup**

   ```bash
   make setup-db
   ```

4. **Build and Run**
   ```bash
   make build
   make docker-run
   ```

### Development Commands

```bash
# Build all crates
make build

# Run tests
make test

# Check code
make check

# Start development environment
make dev

# View logs
make logs-api
make logs-workers

# Stop services
make docker-stop
```

### Code Structure

The project follows Rust best practices:

- **Workspace**: Multi-crate workspace structure
- **Error Handling**: Comprehensive error types
- **Async/Await**: Modern async programming
- **Testing**: Unit and integration tests
- **Documentation**: Comprehensive documentation

### Adding New Platforms

1. **Create Adapter**: Implement `PlatformAdapter` trait
2. **Add Configuration**: Add platform-specific config
3. **Register Adapter**: Register in adapter manager
4. **Add Tests**: Write comprehensive tests
5. **Update Documentation**: Update API documentation

## Monitoring & Observability

### Metrics

The system exposes comprehensive metrics:

#### Application Metrics

- Request latency and throughput
- Error rates by endpoint
- Database connection pool status
- Cache hit/miss ratios

#### ML Metrics

- Prediction accuracy
- Model performance
- Training progress
- Feedback processing rates

#### System Metrics

- CPU and memory usage
- Disk I/O
- Network traffic
- Database performance

### Logging

Structured logging with:

- **Request IDs**: Track requests across services
- **Correlation IDs**: Link related events
- **Performance Metrics**: Track processing times
- **Error Context**: Detailed error information

### Health Checks

Comprehensive health checking:

- **Database**: Connection and query health
- **Redis**: Cache and queue health
- **ML Models**: Model availability and performance
- **External APIs**: Platform API health

### Dashboards

Pre-configured Grafana dashboards:

- **System Overview**: High-level system health
- **API Performance**: Request metrics and errors
- **ML Performance**: Model accuracy and training
- **Database Performance**: Query performance and connections

## Security

### Authentication & Authorization

- **JWT Tokens**: Secure API authentication
- **OAuth Integration**: Platform-specific OAuth flows
- **Role-Based Access**: Granular permission system

### Data Protection

- **Encryption at Rest**: Sensitive data encryption
- **Encryption in Transit**: TLS for all communications
- **Data Anonymization**: PII protection
- **Audit Logging**: Comprehensive activity logging

### API Security

- **Rate Limiting**: Prevent abuse
- **CORS Configuration**: Cross-origin request control
- **Input Validation**: Comprehensive input sanitization
- **Webhook Verification**: Signature verification

### Platform Security

- **Slack Signature Verification**: Verify Slack requests
- **GitHub Webhook Verification**: Verify GitHub webhooks
- **Secure Configuration**: Environment variable security

## Testing

### Test Structure

#### Unit Tests

- Individual component testing
- Mock external dependencies
- Fast execution
- High coverage

#### Integration Tests

- End-to-end testing
- Real database interactions
- API endpoint testing
- ML pipeline testing

#### Performance Tests

- Load testing
- Stress testing
- Memory profiling
- Database performance

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p interstice-core

# Run with coverage
cargo test --workspace -- --nocapture

# Run integration tests
cargo test --test integration
```

### Test Data

- **Fixtures**: Reusable test data
- **Factories**: Test data generation
- **Mocks**: External service mocks
- **Seeds**: Database seeding

## Contributing

### Development Workflow

1. **Fork Repository**: Create your fork
2. **Create Branch**: Create feature branch
3. **Make Changes**: Implement your changes
4. **Add Tests**: Write comprehensive tests
5. **Run Tests**: Ensure all tests pass
6. **Submit PR**: Create pull request

### Code Standards

- **Rustfmt**: Code formatting
- **Clippy**: Linting and best practices
- **Documentation**: Comprehensive documentation
- **Tests**: High test coverage

### Pull Request Process

1. **Description**: Clear description of changes
2. **Tests**: All tests must pass
3. **Documentation**: Update relevant documentation
4. **Review**: Code review by maintainers
5. **Merge**: Merge after approval

### Issue Reporting

- **Bug Reports**: Use bug report template
- **Feature Requests**: Use feature request template
- **Security Issues**: Report privately
- **Documentation**: Use documentation template

---

## Conclusion

The Interstice Engine is a comprehensive, production-ready platform for intelligent work artifact processing and outcome mapping. With its modular architecture, robust ML pipeline, and comprehensive monitoring, it provides a solid foundation for organizations looking to gain insights into their work processes and outcomes.

The system's design emphasizes:

- **Scalability**: Horizontal scaling capabilities
- **Reliability**: Comprehensive error handling and recovery
- **Maintainability**: Clean, well-documented code
- **Extensibility**: Easy addition of new platforms and features
- **Observability**: Comprehensive monitoring and analytics

For more information, please refer to the individual crate documentation and the API reference guide.
