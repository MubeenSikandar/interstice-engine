.PHONY: help build test run clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build all crates"
	@echo "  make test     - Run all tests"
	@echo "  make run-api  - Run the API server"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make check    - Run cargo check"
	@echo "  make fmt      - Format code"

build:
	cargo build --workspace

test:
	cargo test --workspace

run-api:
	cargo run --bin interstice-api

run-workers:
	cargo run --bin interstice-workers

clean:
	cargo clean

check:
	cargo check --workspace --all-targets

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

dev:
	cargo watch -x "run --bin interstice-api"
