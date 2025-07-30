.PHONY: help install test lint format type-check clean build docs docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install package in development mode"
	@echo "  test         Run tests"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run type checking with mypy"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  docs         Build documentation"
	@echo "  docker       Build Docker image"
	@echo "  docker-test  Run tests in Docker"
	@echo "  pre-commit   Run pre-commit hooks"

install:
	pip install -e ".[dev,test,docs]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-quick:
	pytest tests/ -x -q --tb=short

test-hardware:
	pytest tests/ -m hardware -v

lint:
	ruff check src/ tests/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

docs:
	cd docs && make html

docker:
	docker build -t edge-tpu-v5-benchmark .

docker-test:
	docker-compose run --rm test

docker-dev:
	docker-compose run --rm benchmark-dev

pre-commit:
	pre-commit run --all-files

# Quality checks - run all quality tools
quality: format lint type-check test-quick

# Full CI pipeline
ci: install quality test build

# Development setup
dev-setup: install
	@echo "Development environment ready!"
	@echo "Run 'make help' to see available commands."