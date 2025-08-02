# =============================================================================
# Makefile for Edge TPU v5 Benchmark Suite
# Comprehensive build automation and development workflow
# =============================================================================

# Configuration
PYTHON := python3
PIP := pip
PROJECT_NAME := edge-tpu-v5-benchmark
DOCKER_IMAGE := $(PROJECT_NAME)
DOCKER_REGISTRY := ghcr.io
DOCKER_REPO := $(DOCKER_REGISTRY)/danieleschmidt/$(PROJECT_NAME)

# Version management
VERSION := $(shell python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
GIT_COMMIT := $(shell git rev-parse --short HEAD)
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

.PHONY: help install install-dev install-prod test test-unit test-integration test-e2e test-performance test-hardware test-quick test-all lint format type-check security-check clean clean-all build build-wheel build-sdist docs docs-serve docker docker-build docker-push docker-run docker-dev docker-test docker-compose-up docker-compose-down pre-commit setup setup-dev setup-prod quality ci release version bump-version tag-version monitoring benchmark profile debug validate deps deps-update deps-audit sbom trivy hadolint

# =============================================================================
# Help and Information
# =============================================================================
help: ## Show this help message
	@echo "$(GREEN)Edge TPU v5 Benchmark Suite - Build Automation$(NC)"
	@echo "$(BLUE)Version: $(VERSION) | Commit: $(GIT_COMMIT)$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  $(GREEN)make setup-dev$(NC)     - Set up development environment"
	@echo "  $(GREEN)make quality$(NC)       - Run all quality checks"
	@echo "  $(GREEN)make test$(NC)          - Run all tests"
	@echo "  $(GREEN)make build$(NC)         - Build package"
	@echo "  $(GREEN)make docker-build$(NC)  - Build Docker images"

version: ## Show version information
	@echo "$(GREEN)Project Information:$(NC)"
	@echo "  Name:        $(PROJECT_NAME)"
	@echo "  Version:     $(VERSION)"
	@echo "  Git Commit:  $(GIT_COMMIT)"
	@echo "  Build Date:  $(BUILD_DATE)"
	@echo "  Docker Repo: $(DOCKER_REPO)"

# =============================================================================
# Environment Setup
# =============================================================================
setup: setup-dev ## Alias for setup-dev

setup-dev: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev,test,docs]"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Run '$(GREEN)make quality$(NC)' to check code quality"
	@echo "  - Run '$(GREEN)make test$(NC)' to run tests"
	@echo "  - Run '$(GREEN)make docker-build$(NC)' to build Docker images"

setup-prod: ## Set up production environment
	@echo "$(GREEN)Setting up production environment...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install .
	@echo "$(GREEN)✓ Production environment ready!$(NC)"

install: install-dev ## Alias for install-dev

install-dev: ## Install package in development mode
	$(PIP) install -e ".[dev,test,docs]"

install-prod: ## Install package in production mode
	$(PIP) install .

# =============================================================================
# Testing
# =============================================================================
test: test-all ## Run all tests

test-all: ## Run comprehensive test suite
	@echo "$(GREEN)Running comprehensive test suite...$(NC)"
	pytest tests/ -v \
		--cov=src \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		--cov-report=term-missing \
		--junitxml=test-results.xml \
		--durations=10

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	pytest tests/unit/ -v --cov=src --cov-report=term

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running end-to-end tests...$(NC)"
	pytest tests/e2e/ -v -m e2e

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(NC)"
	pytest tests/performance/ -v -m performance --durations=0

test-hardware: ## Run hardware tests (requires TPU device)
	@echo "$(YELLOW)Running hardware tests (requires TPU device)...$(NC)"
	pytest tests/ -m hardware -v --runhardware

test-quick: ## Run quick tests for development
	pytest tests/unit/ -x -q --tb=short --disable-warnings

test-watch: ## Run tests in watch mode
	ptw tests/ -- --testmon

# =============================================================================
# Code Quality
# =============================================================================
format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ tests/ --line-length=88 --target-version=py38
	isort src/ tests/ --profile=black

lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	ruff check src/ tests/ --fix
	black --check src/ tests/
	isort --check-only src/ tests/

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(NC)"
	mypy src/ --strict --pretty

security-check: ## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json || true
	@echo "$(GREEN)✓ Security scan complete. Check *-report.json files.$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(GREEN)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files

quality: format lint type-check security-check ## Run all quality checks
	@echo "$(GREEN)✓ All quality checks passed!$(NC)"

# =============================================================================
# Build and Package
# =============================================================================
clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf test-results.xml
	rm -rf coverage.xml
	rm -rf bandit-report.json
	rm -rf safety-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean ## Clean all artifacts including Docker
	@echo "$(GREEN)Cleaning all artifacts...$(NC)"
	docker system prune -f
	docker volume prune -f

build: clean build-wheel build-sdist ## Build both wheel and source distribution

build-wheel: ## Build wheel distribution
	@echo "$(GREEN)Building wheel distribution...$(NC)"
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution
	@echo "$(GREEN)Building source distribution...$(NC)"
	$(PYTHON) -m build --sdist

validate: ## Validate built packages
	@echo "$(GREEN)Validating built packages...$(NC)"
	twine check dist/*

# =============================================================================
# Documentation
# =============================================================================
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	cd docs && make clean && make html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html/$(NC)"

docs-serve: docs ## Build and serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8000$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# =============================================================================
# Docker
# =============================================================================
docker-build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	docker build --target runtime -t $(DOCKER_IMAGE):latest -t $(DOCKER_IMAGE):$(VERSION) .
	docker build --target development -t $(DOCKER_IMAGE):dev .
	docker build --target testing -t $(DOCKER_IMAGE):test .
	@echo "$(GREEN)✓ Docker images built successfully$(NC)"

docker-build-prod: ## Build production Docker image only
	@echo "$(GREEN)Building production Docker image...$(NC)"
	docker build --target runtime -t $(DOCKER_IMAGE):latest -t $(DOCKER_IMAGE):$(VERSION) .

docker-push: ## Push Docker images to registry
	@echo "$(GREEN)Pushing Docker images to $(DOCKER_REGISTRY)...$(NC)"
	docker tag $(DOCKER_IMAGE):latest $(DOCKER_REPO):latest
	docker tag $(DOCKER_IMAGE):$(VERSION) $(DOCKER_REPO):$(VERSION)
	docker push $(DOCKER_REPO):latest
	docker push $(DOCKER_REPO):$(VERSION)

docker-run: ## Run Docker container
	docker run --rm -it \
		--device=/dev/apex_0:/dev/apex_0 \
		-v $(PWD)/models:/app/models:ro \
		-v $(PWD)/results:/app/results \
		$(DOCKER_IMAGE):latest

docker-dev: ## Run development Docker container
	docker run --rm -it \
		--device=/dev/apex_0:/dev/apex_0 \
		-v $(PWD):/app \
		-p 8000:8000 \
		$(DOCKER_IMAGE):dev

docker-test: ## Run tests in Docker
	docker run --rm \
		-v $(PWD):/app \
		-v $(PWD)/test-results:/app/test-results \
		$(DOCKER_IMAGE):test

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down -v

docker-compose-logs: ## Show docker-compose logs
	docker-compose logs -f

# =============================================================================
# Security and Compliance
# =============================================================================
sbom: ## Generate Software Bill of Materials
	@echo "$(GREEN)Generating SBOM...$(NC)"
	cyclonedx-py -o sbom.json

trivy: ## Run Trivy security scan on Docker image
	@echo "$(GREEN)Running Trivy security scan...$(NC)"
	trivy image $(DOCKER_IMAGE):latest

hadolint: ## Lint Dockerfile with hadolint
	@echo "$(GREEN)Linting Dockerfile...$(NC)"
	hadolint Dockerfile

# =============================================================================
# Dependencies
# =============================================================================
deps: ## Show dependency tree
	pipdeptree

deps-update: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(NC)"
	pip-compile --upgrade pyproject.toml
	pip-sync

deps-audit: ## Audit dependencies for security issues
	@echo "$(GREEN)Auditing dependencies...$(NC)"
	safety check
	pip-audit

# =============================================================================
# Development and Debugging
# =============================================================================
benchmark: ## Run quick benchmark
	@echo "$(GREEN)Running quick benchmark...$(NC)"
	edge-tpu-v5-benchmark detect
	edge-tpu-v5-benchmark run --model mobilenet_v3 --iterations 10

profile: ## Profile the application
	@echo "$(GREEN)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.prof -m edge_tpu_v5_benchmark.cli run --model mobilenet_v3 --iterations 10
	@echo "$(GREEN)Profile saved to profile.prof$(NC)"

debug: ## Run with debug logging
	@echo "$(GREEN)Running with debug logging...$(NC)"
	EDGE_TPU_LOG_LEVEL=DEBUG edge-tpu-v5-benchmark detect

# =============================================================================
# Monitoring and Observability
# =============================================================================
monitoring: ## Start monitoring stack
	@echo "$(GREEN)Starting monitoring stack...$(NC)"
	docker-compose up -d prometheus grafana
	@echo "$(GREEN)✓ Prometheus: http://localhost:9090$(NC)"
	@echo "$(GREEN)✓ Grafana: http://localhost:3000 (admin/admin)$(NC)"

# =============================================================================
# Release Management
# =============================================================================
bump-version: ## Bump version (patch by default, use PART=minor|major for other)
	@echo "$(GREEN)Bumping version...$(NC)"
	bumpversion $(or $(PART),patch)

tag-version: ## Create git tag for current version
	@echo "$(GREEN)Creating git tag v$(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)

release: clean quality test build validate ## Prepare release
	@echo "$(GREEN)✓ Release preparation complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  - Review dist/ contents"
	@echo "  - Run 'make tag-version' to create git tag"
	@echo "  - Upload to PyPI: twine upload dist/*"

# =============================================================================
# CI/CD Pipeline
# =============================================================================
ci: install-dev quality test-all build ## Full CI pipeline
	@echo "$(GREEN)✓ CI pipeline completed successfully!$(NC)"

ci-docker: ## CI pipeline with Docker
	$(MAKE) docker-build
	$(MAKE) docker-test
	@echo "$(GREEN)✓ Docker CI pipeline completed successfully!$(NC)"

# =============================================================================
# Utility targets
# =============================================================================
check-tools: ## Check if required tools are installed
	@echo "$(GREEN)Checking required tools...$(NC)"
	@command -v python3 >/dev/null 2>&1 || { echo "$(RED)python3 is required$(NC)"; exit 1; }
	@command -v pip >/dev/null 2>&1 || { echo "$(RED)pip is required$(NC)"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)docker is required$(NC)"; exit 1; }
	@command -v git >/dev/null 2>&1 || { echo "$(RED)git is required$(NC)"; exit 1; }
	@echo "$(GREEN)✓ All required tools are installed$(NC)"

info: version ## Show project information
	@echo ""
	@echo "$(YELLOW)Environment Information:$(NC)"
	@echo "  Python: $(shell python3 --version)"
	@echo "  Pip: $(shell pip --version)"
	@echo "  Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "  Git: $(shell git --version)"

# =============================================================================
# Watch and Development
# =============================================================================
watch-test: ## Watch for changes and run tests
	watchdog-cli --patterns="*.py" --ignore-patterns="*/.git/*" \
		--command="make test-quick" src/ tests/

watch-docs: ## Watch for changes and rebuild docs
	watchdog-cli --patterns="*.py;*.md;*.rst" --ignore-patterns="*/_build/*" \
		--command="make docs" src/ docs/