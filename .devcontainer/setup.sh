#!/bin/bash
set -e

echo "🚀 Setting up Edge TPU v5 Benchmark Suite development environment..."

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies for TPU development
echo "📦 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libopencv-dev \
    libhdf5-dev \
    libusb-1.0-0-dev \
    udev \
    curl \
    wget \
    git \
    vim \
    htop \
    tree

# Install Google Coral Edge TPU runtime (if available)
echo "🔧 Setting up Edge TPU runtime..."
if ! dpkg -l | grep -q "libedgetpu"; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - 2>/dev/null || echo "Warning: Could not add Google apt key"
    sudo apt-get update 2>/dev/null || echo "Warning: Could not update coral repositories"
    sudo apt-get install -y libedgetpu1-std python3-pycoral 2>/dev/null || echo "Warning: Could not install Edge TPU runtime"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development tools
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pre-commit \
    bandit \
    safety \
    ruff

# Install project dependencies
if [ -f "pyproject.toml" ]; then
    echo "📋 Installing project dependencies..."
    pip install -e ".[dev,test]" || pip install -e .
fi

# Install additional ML dependencies for development
echo "🤖 Installing ML development dependencies..."
pip install \
    numpy \
    opencv-python \
    pillow \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    tensorboard \
    onnx \
    onnxruntime

# Install TensorFlow Lite (for TPU development)
pip install tflite-runtime || pip install tensorflow

# Set up pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
fi

# Create useful aliases
echo "⚡ Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Edge TPU Benchmark Suite development aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Project-specific aliases
alias benchmark='python -m edge_tpu_v5_benchmark'
alias test='pytest'
alias testv='pytest -v'
alias testcov='pytest --cov=src --cov-report=html --cov-report=term'
alias lint='ruff check src tests'
alias format='black src tests && isort src tests'
alias typecheck='mypy src'
alias security='bandit -r src'

# Docker aliases
alias dps='docker ps'
alias dimg='docker images'
alias dlog='docker logs'

# Git aliases
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -10'
EOF

# Create .env.example if it doesn't exist
if [ ! -f ".env.example" ]; then
    echo "📝 Creating .env.example..."
    cat > .env.example << 'EOF'
# Edge TPU v5 Benchmark Suite Environment Variables

# Development settings
EDGE_TPU_DEV_MODE=1
EDGE_TPU_LOG_LEVEL=INFO
EDGE_TPU_CACHE_DIR=~/.edge-tpu-v5-benchmark/cache

# TPU device settings
EDGE_TPU_DEVICE_PATH=/dev/apex_0
EDGE_TPU_COMPILER_TIMEOUT=300

# Benchmark settings
BENCHMARK_DEFAULT_ITERATIONS=1000
BENCHMARK_WARMUP_ITERATIONS=100
BENCHMARK_OUTPUT_DIR=./results

# Power measurement settings
POWER_SAMPLING_RATE=1000
POWER_MEASUREMENT_ENABLED=true

# Leaderboard settings (optional)
LEADERBOARD_API_URL=https://api.edge-tpu-benchmark.org
LEADERBOARD_API_KEY=your_api_key_here
LEADERBOARD_SUBMISSION_ENABLED=false

# Testing settings
PYTEST_TIMEOUT=300
PYTEST_WORKERS=auto
COVERAGE_THRESHOLD=90

# Docker settings
DOCKER_REGISTRY=ghcr.io
DOCKER_IMAGE_NAME=edge-tpu-v5-benchmark
DOCKER_TAG=latest
EOF
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "  1. Copy .env.example to .env and customize settings"
echo "  2. Run 'benchmark detect' to verify TPU detection"
echo "  3. Run 'test' to execute the test suite"
echo "  4. Run 'benchmark run --help' to see available commands"
echo ""
echo "Happy benchmarking! 🚀"