# ============================================================
# Financial RAG Engine — Multi-stage Dockerfile
# Stage 1: Build — install dependencies
# Stage 2: Runtime — non-root user, minimal image
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy source code
COPY --chown=appuser:appuser . .

# Create required directories with correct ownership
RUN mkdir -p data/raw data/processed data/index models/sentiment results \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Start production server using Waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "--call", "app:create_app"]
