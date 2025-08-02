# Build Guide - Edge TPU v5 Benchmark Suite

This guide covers building, containerizing, and deploying the Edge TPU v5 Benchmark Suite.

## Quick Start

```bash
# Set up development environment
make setup-dev

# Run quality checks and tests
make quality test

# Build package
make build

# Build Docker images
make docker-build
```

## Table of Contents

- [Prerequisites](#prerequisites)
- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Docker Builds](#docker-builds)
- [Testing](#testing)
- [Quality Assurance](#quality-assurance)
- [Release Process](#release-process)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **Docker**: 20.10+ (for containerized builds)
- **Git**: 2.20+
- **Make**: 4.0+ (for build automation)

### Hardware Requirements

- **Minimum**: 4GB RAM, 10GB disk space
- **Recommended**: 8GB RAM, 50GB disk space
- **TPU Hardware**: Google Edge TPU v5 device (for hardware testing)

### Required Tools

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    make \
    curl \
    gnupg \
    software-properties-common

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Edge TPU Runtime (Optional)

```bash
# Add Google Edge TPU repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install Edge TPU runtime
sudo apt-get update
sudo apt-get install -y edgetpu-runtime-v5
```

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/danieleschmidt/edge-tpu-v5-benchmark-suite.git
cd edge-tpu-v5-benchmark-suite
```

### 2. Set Up Environment

```bash
# Automated setup (recommended)
make setup-dev

# Manual setup
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,test,docs]"
pre-commit install
```

### 3. Verify Installation

```bash
# Check tools
make check-tools

# Show project info
make info

# Run quick test
make test-quick
```

## Building from Source

### Python Package

```bash
# Clean previous builds
make clean

# Build wheel and source distribution
make build

# Validate packages
make validate

# Built packages will be in dist/
ls -la dist/
```

### Build Targets

- `make build-wheel` - Build wheel distribution only
- `make build-sdist` - Build source distribution only
- `make build` - Build both wheel and source distributions

### Installation from Source

```bash
# Install from wheel
pip install dist/edge_tpu_v5_benchmark-*.whl

# Install from source
pip install dist/edge-tpu-v5-benchmark-*.tar.gz

# Development installation
pip install -e .
```

## Docker Builds

### Multi-Stage Build Architecture

The project uses a multi-stage Dockerfile with optimized stages:

1. **Builder Stage**: Compiles dependencies and prepares application
2. **Runtime Stage**: Minimal production image
3. **Development Stage**: Full development environment
4. **Testing Stage**: Optimized for CI/CD testing

### Building Images

```bash
# Build all images
make docker-build

# Build specific stage
docker build --target runtime -t edge-tpu-v5-benchmark:latest .
docker build --target development -t edge-tpu-v5-benchmark:dev .
docker build --target testing -t edge-tpu-v5-benchmark:test .
```

### Image Variants

| Tag | Size | Purpose | Includes |
|-----|------|---------|----------|
| `latest` | ~200MB | Production | Runtime only |
| `dev` | ~500MB | Development | Dev tools, Jupyter |
| `test` | ~400MB | CI/CD | Testing tools |

### Running Containers

```bash
# Production container
make docker-run

# Development container
make docker-dev

# Testing container
make docker-test
```

### Docker Compose

```bash
# Start all services
make docker-compose-up

# Stop all services
make docker-compose-down

# View logs
make docker-compose-logs
```

Available services:
- `benchmark` - Production benchmarking
- `benchmark-dev` - Development environment
- `test` - Test execution
- `docs` - Documentation server
- `prometheus` - Metrics collection
- `grafana` - Monitoring dashboard

## Testing

### Test Categories

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# End-to-end tests
make test-e2e

# Performance tests
make test-performance

# Hardware tests (requires TPU)
make test-hardware
```

### Test Configuration

Test behavior is controlled by environment variables:

```bash
# Run slow tests
RUN_SLOW_TESTS=true make test

# Run hardware tests
RUN_HARDWARE_TESTS=true make test-hardware

# Run network tests
RUN_NETWORK_TESTS=true make test
```

### Coverage Reports

```bash
# Generate coverage report
make test

# View HTML coverage report
open htmlcov/index.html

# Coverage files
- htmlcov/          # HTML report
- coverage.xml      # XML report (for CI)
- .coverage         # Coverage database
```

## Quality Assurance

### Code Quality Tools

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security-check

# All quality checks
make quality
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
make pre-commit

# Update hook versions
pre-commit autoupdate
```

### Quality Metrics

| Tool | Purpose | Config File |
|------|---------|------------|
| Black | Code formatting | `pyproject.toml` |
| isort | Import sorting | `pyproject.toml` |
| Ruff | Fast linting | `pyproject.toml` |
| MyPy | Type checking | `pyproject.toml` |
| Bandit | Security linting | `.bandit` |
| Safety | Dependency security | N/A |

## Release Process

### Version Management

```bash
# Bump version (patch)
make bump-version

# Bump minor version
make bump-version PART=minor

# Bump major version
make bump-version PART=major
```

### Release Preparation

```bash
# Complete release preparation
make release

# This runs:
# - make clean
# - make quality
# - make test
# - make build
# - make validate
```

### Creating Releases

```bash
# Create git tag
make tag-version

# Push to PyPI (manual step)
twine upload dist/*

# Push Docker images
make docker-push
```

### Release Checklist

- [ ] Update CHANGELOG.md
- [ ] Bump version with `make bump-version`
- [ ] Run `make release` successfully
- [ ] Create and push git tag
- [ ] Upload to PyPI
- [ ] Push Docker images
- [ ] Create GitHub release
- [ ] Update documentation

## Deployment

### Docker Deployment

```bash
# Production deployment
docker run -d \
  --name tpu-benchmark \
  --device=/dev/apex_0:/dev/apex_0 \
  -v /path/to/models:/app/models:ro \
  -v /path/to/results:/app/results \
  edge-tpu-v5-benchmark:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tpu-benchmark
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tpu-benchmark
  template:
    metadata:
      labels:
        app: tpu-benchmark
    spec:
      containers:
      - name: benchmark
        image: edge-tpu-v5-benchmark:latest
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: tpu-device
          mountPath: /dev/apex_0
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: results
          mountPath: /app/results
      volumes:
      - name: tpu-device
        hostPath:
          path: /dev/apex_0
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: results
        persistentVolumeClaim:
          claimName: results-pvc
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_TPU_DEVICE_PATH` | `/dev/apex_0` | TPU device path |
| `BENCHMARK_DEFAULT_ITERATIONS` | `1000` | Default iterations |
| `POWER_MEASUREMENT_ENABLED` | `true` | Enable power monitoring |
| `EDGE_TPU_LOG_LEVEL` | `INFO` | Logging level |
| `TELEMETRY_ENABLED` | `false` | Anonymous telemetry |

## Performance Optimization

### Build Performance

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build with cache
docker build --cache-from edge-tpu-v5-benchmark:latest .

# Parallel testing
pytest -n auto tests/
```

### Runtime Performance

```bash
# Profile application
make profile

# Debug performance
make debug
```

### Memory Usage

- Production image: ~200MB
- Development image: ~500MB
- Runtime memory: ~50-100MB base + model size

## Security

### Container Security

- Non-root user (UID 1001)
- Minimal base image
- Security scanning with Trivy
- SBOM generation

### Security Scanning

```bash
# Generate SBOM
make sbom

# Run Trivy scan
make trivy

# Lint Dockerfile
make hadolint

# Security audit
make deps-audit
```

### Security Best Practices

1. **Secrets Management**: Use external secret management (never in images)
2. **Network Security**: Limit exposed ports and network access
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Updates**: Regularly update base images and dependencies
5. **Scanning**: Integrate security scanning in CI/CD

## Troubleshooting

### Common Build Issues

#### Python Version Issues

```bash
# Check Python version
python --version

# Use specific Python version
python3.8 -m venv venv
```

#### Permission Issues

```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix TPU device permissions
sudo chmod 666 /dev/apex_0
```

#### Memory Issues

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Reduce parallel processes
make test-unit  # Instead of make test-all
```

### Build Failures

#### Missing Dependencies

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade

# Reinstall Python dependencies
pip install --force-reinstall -e ".[dev,test]"
```

#### Docker Build Failures

```bash
# Clear Docker cache
docker system prune -a

# Build without cache
docker build --no-cache .

# Check Docker version
docker version
```

### Runtime Issues

#### TPU Not Detected

```bash
# Check TPU device
ls -la /dev/apex*

# Check TPU runtime
edge-tpu-v5-benchmark detect

# Install TPU runtime
sudo apt-get install -y edgetpu-runtime-v5
```

#### Import Errors

```bash
# Check installation
pip list | grep edge-tpu

# Reinstall package
pip uninstall edge-tpu-v5-benchmark
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Getting Help

1. **Documentation**: Check docs/ directory
2. **Issues**: GitHub issues for bug reports
3. **Discussions**: GitHub discussions for questions
4. **Logs**: Check logs/ directory for detailed error information

### Debug Commands

```bash
# Verbose build
make build V=1

# Debug Docker build
docker build --progress=plain .

# Debug tests
make test-quick -v

# System information
make info
```

## Contributing

### Development Workflow

1. Fork repository
2. Create feature branch
3. Make changes
4. Run quality checks: `make quality`
5. Run tests: `make test`
6. Submit pull request

### Build System

The build system is based on:
- Makefile for task automation
- pyproject.toml for Python packaging
- Docker multi-stage builds
- GitHub Actions for CI/CD

For more information, see [CONTRIBUTING.md](../CONTRIBUTING.md).