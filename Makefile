# ============================================================
# Financial RAG Engine — Makefile
# All targets work on Linux/macOS/WSL.
# For Windows native: use run.ps1 for the 'run' target.
# ============================================================

.PHONY: install seed ingest train-sentiment setup run test test-unit \
        test-integration test-eval test-load lint format build up down \
        logs clean help

# Default target
.DEFAULT_GOAL := help

## install: Install all dependencies (production + dev)
install:
	pip install -r requirements.txt -r requirements-dev.txt

## seed: Generate 3 sample financial documents in data/raw/
seed:
	python scripts/seed_sample_data.py

## ingest: Ingest all documents in data/raw/ into the FAISS index
ingest:
	python scripts/ingest_documents.py --path data/raw

## train-sentiment: Download financial_phrasebank and train sentiment model
train-sentiment:
	python scripts/train_sentiment.py

## setup: Full one-command setup (install + seed + ingest + train)
setup: install seed ingest train-sentiment
	@echo ""
	@echo "✅ Project fully set up and ready!"
	@echo "   API server:   make run"
	@echo "   Tests:        make test"
	@echo "   Docker:       make up"

## run: Start the production server using Waitress
run:
	waitress-serve --host=0.0.0.0 --port=5000 --call "app:create_app"

## test-unit: Run unit tests with coverage enforcement (>= 80%)
test-unit:
	pytest tests/unit -v \
	  --cov=app \
	  --cov-report=term-missing \
	  --cov-report=html \
	  --cov-fail-under=80 \
	  -m "not slow"

## test-integration: Run integration tests
test-integration:
	pytest tests/integration -v -m "not slow"

## test: Run all tests (unit + integration) with coverage
test: test-unit test-integration
	@echo ""
	@echo "✅ All tests passed!"

## test-eval: Run full evaluation suite (retrieval + generation metrics)
test-eval:
	python scripts/run_evaluation.py

## test-load: Run Locust load test against running server (make run first)
test-load:
	@mkdir -p results
	locust -f tests/load/locustfile.py \
	  --headless -u 20 -r 2 \
	  --run-time 5m \
	  --html results/load_test_report.html \
	  --host http://localhost:5000
	@echo "Load test report: results/load_test_report.html"

## lint: Run all linters (flake8, black check, isort check)
lint:
	flake8 app/ tests/ scripts/ --max-line-length=100 --extend-ignore=E203,W503
	black --check app/ tests/ scripts/
	isort --check-only app/ tests/ scripts/

## format: Auto-format all Python code (black + isort)
format:
	black app/ tests/ scripts/
	isort app/ tests/ scripts/

## build: Build Docker image
build:
	docker build -f infra/Dockerfile -t financial-rag-engine:latest .
	@echo "✅ Docker image built: financial-rag-engine:latest"

## up: Start full Docker stack (API + Prometheus + Grafana)
up:
	docker-compose -f infra/docker-compose.yml up -d
	@echo ""
	@echo "✅ Stack is running:"
	@echo "   API + Swagger: http://localhost:5000/api/v1/docs"
	@echo "   Prometheus:    http://localhost:9090"
	@echo "   Grafana:       http://localhost:3000 (admin/admin)"

## down: Stop Docker stack
down:
	docker-compose -f infra/docker-compose.yml down

## logs: Follow API container logs
logs:
	docker-compose -f infra/docker-compose.yml logs -f rag-api

## clean: Remove generated artifacts (index, models, results, __pycache__)
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ .pytest_cache/
	rm -rf data/index/ data/processed/ models/sentiment/ results/
	@echo "Cleaned. Run 'make setup' to rebuild."

## help: Show this help message
help:
	@echo "Financial RAG Engine — Available Targets"
	@echo "==========================================="
	@grep -E '^## ' Makefile | sed 's/## //' | awk '{printf "  %-20s %s\n", $$1, substr($$0, index($$0,$$2))}'
