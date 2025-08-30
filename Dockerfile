# Multi-stage build for Interstice Engine
FROM rust:1.75 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/*/Cargo.toml ./crates/

# Create dummy main.rs files to build dependencies
RUN mkdir -p crates/interstice-core/src crates/interstice-adapters/src crates/interstice-api/src crates/interstice-ml/src crates/interstice-workers/src
RUN echo "fn main() {}" > crates/interstice-core/src/main.rs
RUN echo "fn main() {}" > crates/interstice-adapters/src/main.rs
RUN echo "fn main() {}" > crates/interstice-api/src/main.rs
RUN echo "fn main() {}" > crates/interstice-ml/src/main.rs
RUN echo "fn main() {}" > crates/interstice-workers/src/main.rs

# Build dependencies
RUN cargo build --release

# Remove dummy files and copy source
RUN rm -rf crates/*/src
COPY . .

# Build the application
RUN cargo build --release --bin interstice-api --bin interstice-workers

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false app

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/target/release/interstice-api /usr/local/bin/
COPY --from=builder /app/target/release/interstice-workers /usr/local/bin/

# Copy models directory if it exists
COPY --from=builder /app/models /models

# Copy configuration files
COPY env.example /app/.env.example
COPY migrations /app/migrations

# Set ownership
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose ports
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["interstice-api"]
