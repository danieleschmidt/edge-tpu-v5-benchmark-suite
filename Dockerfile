# =============================================================================
# Multi-stage Dockerfile for Edge TPU v5 Benchmark Suite
# Optimized for security, size, and performance
# =============================================================================

# -----------------------------------------------------------------------------
# Build Stage: Compile dependencies and prepare application
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDX_VERSION

# Set build environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    curl \
    gnupg \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./

# Install Python dependencies in virtual environment
RUN pip install --no-deps -e .

# Install additional runtime dependencies
RUN pip install --no-deps \
    gunicorn \
    prometheus-client \
    structlog

# -----------------------------------------------------------------------------
# Runtime Stage: Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

# Runtime labels for metadata
LABEL org.opencontainers.image.title="Edge TPU v5 Benchmark Suite" \
      org.opencontainers.image.description="Comprehensive benchmarking framework for Google TPU v5 edge cards" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/edge-tpu-v5-benchmark-suite" \
      org.opencontainers.image.documentation="https://edge-tpu-v5-benchmark.readthedocs.io"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:$PATH" \
    EDGE_TPU_RUNTIME_DIR=/opt/edgetpu \
    DEBIAN_FRONTEND=noninteractive

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # TPU runtime dependencies
    curl \
    gnupg \
    ca-certificates \
    # System utilities
    tini \
    # Monitoring and debugging
    htop \
    strace \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Add Google Edge TPU repository with proper key handling
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list

# Install Edge TPU runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libedgetpu1-std \
    python3-pycoral \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create non-root user with specific UID/GID for security
RUN groupadd -r -g 1001 tpuuser && \
    useradd -r -g tpuuser -u 1001 -d /home/tpuuser -m -s /bin/bash tpuuser && \
    mkdir -p /home/tpuuser/.cache /home/tpuuser/.local && \
    chown -R tpuuser:tpuuser /home/tpuuser

# Copy application code with proper ownership
COPY --chown=tpuuser:tpuuser src/ src/
COPY --chown=tpuuser:tpuuser pyproject.toml README.md LICENSE ./

# Create directories for runtime data
RUN mkdir -p /app/data /app/results /app/models /app/logs && \
    chown -R tpuuser:tpuuser /app

# Setup proper device permissions for TPU access
RUN echo 'SUBSYSTEM=="apex", MODE="0664", GROUP="tpuuser"' > /etc/udev/rules.d/65-apex.rules

# Switch to non-root user
USER tpuuser

# Create cache directories
RUN mkdir -p /home/tpuuser/.cache/edge-tpu-v5-benchmark

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD edge-tpu-v5-benchmark detect --quiet || exit 1

# Expose ports for monitoring and web interface
EXPOSE 8000 9090

# Use tini for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["edge-tpu-v5-benchmark", "--help"]

# -----------------------------------------------------------------------------
# Development Stage: Full development environment
# -----------------------------------------------------------------------------
FROM runtime as development

# Switch back to root for development tool installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Development tools
    vim \
    nano \
    git \
    make \
    gcc \
    g++ \
    # Debugging tools
    gdb \
    valgrind \
    # Network tools
    netcat-openbsd \
    telnet \
    # Process tools
    procps \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-mock \
    black \
    ruff \
    mypy \
    pre-commit \
    jupyter \
    ipython \
    debugpy

# Copy development configuration
COPY --chown=tpuuser:tpuuser .devcontainer/ .devcontainer/
COPY --chown=tpuuser:tpuuser .vscode/ .vscode/
COPY --chown=tpuuser:tpuuser tests/ tests/

# Setup development environment
RUN chmod +x .devcontainer/setup.sh && \
    su - tpuuser -c "cd /app && ./.devcontainer/setup.sh"

# Switch back to non-root user
USER tpuuser

# Development-specific environment
ENV EDGE_TPU_DEV_MODE=1 \
    EDGE_TPU_LOG_LEVEL=DEBUG

# Development command
CMD ["bash"]

# -----------------------------------------------------------------------------
# Testing Stage: Optimized for CI/CD testing
# -----------------------------------------------------------------------------
FROM development as testing

# Switch to root for test setup
USER root

# Install additional testing tools
RUN /opt/venv/bin/pip install --no-cache-dir \
    pytest-xdist \
    pytest-benchmark \
    pytest-timeout \
    coverage \
    bandit \
    safety

# Copy test configuration
COPY --chown=tpuuser:tpuuser pytest.ini .coveragerc ./
COPY --chown=tpuuser:tpuuser .pre-commit-config.yaml ./

# Setup test permissions
RUN chown -R tpuuser:tpuuser /app

# Switch back to non-root user
USER tpuuser

# Test-specific environment
ENV PYTEST_CURRENT_TEST="" \
    COVERAGE_PROCESS_START=.coveragerc

# Test command
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=xml", "--cov-report=html"]