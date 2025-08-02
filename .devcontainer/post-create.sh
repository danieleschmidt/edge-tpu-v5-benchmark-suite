#!/bin/bash

# Post-create script for Edge TPU v5 Benchmark Suite development environment

set -e

echo "🚀 Setting up Edge TPU v5 Benchmark Suite development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libusb-1.0-0-dev \
    libudev-dev \
    curl \
    wget \
    unzip \
    git-lfs

# Install Edge TPU runtime (mock for development)
echo "🔧 Installing Edge TPU runtime dependencies..."
# Note: Full TPU runtime requires actual hardware
# This installs development dependencies for testing
pip install --upgrade pip setuptools wheel

# Install project dependencies
echo "📚 Installing Python dependencies..."
pip install -e ".[dev,test,docs]"

# Setup pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create development directories
echo "📁 Creating development directories..."
mkdir -p {data,models,results,logs,notebooks}
mkdir -p .pytest_cache
mkdir -p .mypy_cache

# Download sample models for testing (mock files)
echo "🤖 Setting up sample models..."
mkdir -p models/samples
# Create mock model files for development
cat > models/samples/mobilenet_v3.onnx << 'EOF'
# This is a mock ONNX model file for development
# Replace with real models when testing with actual TPU hardware
EOF

cat > models/samples/efficientnet_lite.tflite << 'EOF'
# This is a mock TensorFlow Lite model file for development
# Replace with real models when testing with actual TPU hardware
EOF

# Create sample data
echo "📊 Creating sample data..."
python -c "
import numpy as np
import os

# Create sample input data
os.makedirs('data/samples', exist_ok=True)
sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
np.save('data/samples/sample_image.npy', sample_image)

# Create sample batch data
batch_data = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
np.save('data/samples/batch_images.npy', batch_data)

print('✅ Sample data created')
"

# Setup development configuration
echo "⚙️ Setting up development configuration..."
cat > .env.example << 'EOF'
# Edge TPU v5 Benchmark Suite - Development Environment Variables

# TPU Configuration
TPU_DEVICE_PATH=/dev/apex_0
TPU_COMPILER_PATH=/usr/local/bin/edgetpu_compiler
TPU_RUNTIME_PATH=/usr/local/lib/libedgetpu.so

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_PROFILING=true

# Model Storage
MODEL_CACHE_DIR=./models/cache
RESULTS_DIR=./results
DATA_DIR=./data

# Testing Configuration
MOCK_TPU_HARDWARE=true
ENABLE_HARDWARE_TESTS=false
TEST_TIMEOUT=300

# API Configuration (for leaderboard integration)
LEADERBOARD_API_URL=https://api.tpu-benchmark.org
LEADERBOARD_API_KEY=your_api_key_here

# Performance Settings
MAX_CONCURRENT_BENCHMARKS=1
POWER_SAMPLING_RATE_HZ=1000
THERMAL_MONITORING_ENABLED=true

# Security Settings
ENABLE_MODEL_VERIFICATION=true
ALLOW_UNSAFE_MODELS=false
SANDBOX_EXECUTION=true
EOF

# Setup Jupyter configuration
echo "📓 Setting up Jupyter configuration..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_notebook_config.py << 'EOF'
# Jupyter configuration for Edge TPU v5 Benchmark Suite

c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_root = True
c.NotebookApp.token = ''
c.NotebookApp.password = ''
c.NotebookApp.notebook_dir = '/workspaces/edge-tpu-v5-benchmark-suite'
EOF

# Install additional development tools
echo "🛠️ Installing additional development tools..."
pip install \
    jupyterlab \
    notebook \
    ipywidgets \
    ipykernel \
    plotly \
    seaborn \
    scikit-learn \
    onnxruntime \
    tensorboard

# Setup Git configuration helpers
echo "🔧 Setting up Git helpers..."
git config --global --add safe.directory /workspaces/edge-tpu-v5-benchmark-suite
git config --global pull.rebase false
git config --global init.defaultBranch main

# Create useful aliases
echo "🔗 Creating helpful aliases..."
cat >> ~/.bashrc << 'EOF'

# Edge TPU v5 Benchmark Suite aliases
alias benchmark='edge-tpu-v5-benchmark'
alias test-all='python -m pytest tests/ -v'
alias test-unit='python -m pytest tests/unit/ -v'
alias test-integration='python -m pytest tests/integration/ -v'
alias lint='ruff check src/ tests/'
alias format='black src/ tests/ && ruff --fix src/ tests/'
alias typecheck='mypy src/'
alias coverage='pytest --cov=edge_tpu_v5_benchmark --cov-report=html tests/'
alias docs='cd docs && make html && cd ..'
alias clean='find . -type f -name "*.pyc" -delete && find . -type d -name "__pycache__" -delete'

# Jupyter aliases
alias nb='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Development helpers
alias venv-info='python -c "import sys; print(f\"Python: {sys.version}\"); print(f\"Path: {sys.executable}\")"'
alias deps-tree='pip list | grep -E "(edge-tpu|onnx|tflite|numpy|pytest)"'
EOF

# Make scripts executable
chmod +x .devcontainer/post-create.sh

# Run initial tests to verify setup
echo "🧪 Running initial setup validation..."
python -c "
import sys
print(f'✅ Python version: {sys.version}')

try:
    import numpy
    print(f'✅ NumPy version: {numpy.__version__}')
except ImportError:
    print('❌ NumPy not available')

try:
    import onnx
    print(f'✅ ONNX version: {onnx.__version__}')
except ImportError:
    print('❌ ONNX not available')

try:
    import tflite_runtime
    print('✅ TensorFlow Lite runtime available')
except ImportError:
    print('❌ TensorFlow Lite runtime not available')

print('✅ Development environment setup complete!')
"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Copy .env.example to .env and customize as needed"
echo "  2. Run 'test-all' to verify everything works"
echo "  3. Start development with 'benchmark --help'"
echo "  4. Launch Jupyter with 'lab' or 'nb'"
echo ""
echo "🔧 Available commands:"
echo "  benchmark     - Run Edge TPU v5 benchmarks"
echo "  test-all      - Run all tests"
echo "  lint          - Check code quality"
echo "  format        - Format code"
echo "  typecheck     - Run type checking"
echo "  coverage      - Generate test coverage report"
echo "  lab/nb        - Start Jupyter Lab/Notebook"
echo ""
echo "Happy coding! 🚀"