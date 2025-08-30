# Interstice Engine

A production-ready Slackbot with integrated ML training capabilities for intelligent work artifact detection and outcome mapping.

## 🚀 Features

- **Intelligent Artifact Detection**: Automatically identifies work artifacts from Slack messages
- **ML-Powered Outcome Mapping**: Uses machine learning to suggest relevant outcomes
- **Continuous Learning**: Improves predictions over time with user feedback
- **Multi-Platform Support**: Slack, GitHub, Jira, and more
- **Production Ready**: Docker, monitoring, and deployment configurations included

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Slack Bot    │    │   ML Pipeline   │    │   Core Engine   │
│                 │◄──►│                 │◄──►│                 │
│ • Event Handler│    │ • Embeddings    │    │ • Artifact      │
│ • ML Response  │    │ • Predictions   │    │   Extraction    │
│ • User Feedback│    │ • Training      │    │ • Outcome       │
└─────────────────┘    └─────────────────┘    │   Mapping      │
                                              └─────────────────┘
                                                       │
                                              ┌─────────────────┐
                                              │   Storage       │
                                              │                 │
                                              │ • PostgreSQL    │
                                              │ • Redis         │
                                              │ • Vector DB     │
                                              └─────────────────┘
```

## 🛠️ Prerequisites

- **Rust 1.75+** - [Install Rust](https://rustup.rs/)
- **PostgreSQL 15+** with pgvector extension
- **Redis 7+** for caching and job queues
- **Docker & Docker Compose** (optional, for containerized deployment)

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MubeenSikandar/interstice-engine.git
cd interstice-engine
```

### 2. Set Up Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
nano .env
```

### 3. Set Up Database

```bash
# Option 1: Using the setup script (recommended)
./scripts/setup_database.sh

# Option 2: Manual setup
createdb interstice_engine
psql -d interstice_engine -c "CREATE EXTENSION IF NOT EXISTS vector;"
sqlx migrate run --database-url $DATABASE_URL --source migrations
```

### 4. Build the Project

```bash
cargo build --workspace
```

## 🚀 Quick Start

### Development Environment

```bash
# Start all services
make dev

# Or start step by step
make setup-db
make build
make docker-run
```

### Production Deployment

```bash
# Build and deploy
make docker-build
make docker-run

# Start monitoring
make start-monitoring
```

## 📊 Available Commands

```bash
# Development
make build          # Build all crates
make test           # Run tests
make check          # Check code

# Database
make setup-db       # Set up database
make migrate        # Run migrations
make db-reset       # Reset database

# Docker
make docker-build   # Build images
make docker-run     # Start services
make docker-stop    # Stop services

# ML Training
make train-models   # Start training pipeline
make download-models # Download pre-trained models

# Monitoring
make start-monitoring # Start Prometheus/Grafana
make status          # Check service status
```

## 🔧 Configuration

### Environment Variables

| Variable               | Description                  | Default                  |
| ---------------------- | ---------------------------- | ------------------------ |
| `DATABASE_URL`         | PostgreSQL connection string | Required                 |
| `REDIS_URL`            | Redis connection string      | `redis://localhost:6379` |
| `SLACK_BOT_TOKEN`      | Slack bot user OAuth token   | Required                 |
| `SLACK_SIGNING_SECRET` | Slack app signing secret     | Required                 |
| `HOST`                 | API server host              | `0.0.0.0`                |
| `PORT`                 | API server port              | `3000`                   |

### Slack App Configuration

1. Create a new Slack app at [api.slack.com](https://api.slack.com/apps)
2. Add the following OAuth scopes:
   - `channels:read`
   - `chat:write`
   - `chat:write.public`
   - `users:read`
3. Enable event subscriptions and add your webhook URL
4. Install the app to your workspace

## 🧠 ML Pipeline

### Components

- **Embeddings**: BERT-based text embeddings using Candle
- **Predictions**: Organization-specific outcome prediction models
- **Training**: Continuous learning with user feedback
- **Vocabulary**: Organization-specific term learning

### Training Data

The ML pipeline learns from:

- User feedback on predictions
- Implicit acceptance (no action taken)
- Organization vocabulary and context
- Historical artifact-outcome mappings

### Model Performance

Track model performance through:

- Prediction accuracy metrics
- User feedback analysis
- A/B testing capabilities
- Performance dashboards

## 📈 Monitoring

### Metrics Available

- **API Performance**: Request latency, throughput, error rates
- **ML Metrics**: Prediction accuracy, training progress, model versions
- **Database**: Connection pools, query performance, storage usage
- **System**: CPU, memory, disk usage

### Dashboards

- **Grafana**: Pre-configured dashboards for all metrics
- **Prometheus**: Time-series metrics storage
- **Custom Alerts**: Configurable alerting rules

## 🔒 Security

- **Request Verification**: Slack signature verification
- **Data Encryption**: Sensitive data encrypted at rest
- **Access Control**: Role-based access control
- **Audit Logging**: Comprehensive activity logging

## 🧪 Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p interstice-core
cargo test -p interstice-ml

# Run integration tests
cargo test --test integration
```

## 📚 API Documentation

### Endpoints

- `GET /health` - Health check
- `POST /api/v1/slack/events` - Slack event webhook
- `POST /api/v1/slack/interactions` - Slack interaction handling
- `GET /api/v1/workspaces/{id}/artifacts` - Get workspace artifacts
- `GET /api/v1/workspaces/{id}/outcomes` - Get workspace outcomes

### Webhook Events

- `message` - Process Slack messages
- `app_mention` - Handle bot mentions
- `interactive_message` - Process button clicks

## 🚀 Deployment

### Local Development

```bash
make dev
```

### Docker Deployment

```bash
make docker-build
make docker-run
```

### Production Deployment

1. Set production environment variables
2. Configure SSL certificates
3. Set up load balancing
4. Configure monitoring and alerting
5. Deploy with `make deploy-prod`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/MubeenSikandar/interstice-engine/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MubeenSikandar/interstice-engine/discussions)
- **Documentation**: [Wiki](https://github.com/MubeenSikandar/interstice-engine/wiki)

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Advanced ML models (GPT, BERT-large)
- [ ] Real-time collaboration features
- [ ] Mobile app
- [ ] Enterprise SSO integration
- [ ] Advanced analytics and reporting

---

Built with ❤️ by the Interstice Engine team
