# Development Guide

This guide covers development setup, workflows, and best practices for the Edge TPU v5 Benchmark Suite.

## 🚀 Quick Setup

### Prerequisites
- Python 3.8+ 
- Git
- TPU v5 edge device (for hardware tests)

### Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/edge-tpu-v5-benchmark-suite.git
cd edge-tpu-v5-benchmark-suite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Verify Installation
```bash
# Run basic tests
pytest tests/ -v

# Check CLI works
edge-tpu-v5-benchmark --help

# Verify TPU detection (requires hardware)
edge-tpu-v5-benchmark detect
```

## 🏗️ Project Structure

```
edge-tpu-v5-benchmark-suite/
├── src/edge_tpu_v5_benchmark/    # Main package
│   ├── __init__.py
│   ├── benchmark.py              # Core benchmark engine
│   ├── cli.py                    # Command-line interface
│   ├── models.py                 # Model loading/management
│   ├── power.py                  # Power profiling
│   ├── converter.py              # Model conversion utilities
│   ├── analysis.py               # Performance analysis
│   └── leaderboard.py           # Leaderboard integration
├── tests/                        # Test suite
│   ├── test_benchmark.py
│   ├── test_models.py
│   └── test_power.py
├── docs/                         # Documentation
├── examples/                     # Usage examples
└── scripts/                      # Development scripts
```

## 🧪 Testing Strategy

### Test Categories
```bash
# Unit tests (fast, no hardware)
pytest tests/ -m "not hardware"

# Integration tests 
pytest tests/ -m "integration"

# Hardware tests (requires TPU v5)
pytest tests/ -m "hardware"

# All tests with coverage
pytest --cov=src --cov-report=html
```

### Writing Tests
```python
# tests/test_your_feature.py
import pytest
from edge_tpu_v5_benchmark.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self):
        instance = YourClass()
        result = instance.method()
        assert result == expected_value
    
    @pytest.mark.hardware
    def test_tpu_required_feature(self):
        # Test requiring actual TPU hardware
        pass
```

### Test Data
```bash
# Generate test models
python scripts/generate_test_models.py

# Download reference datasets  
python scripts/download_test_data.py
```

## 🔧 Code Quality

### Automated Formatting
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks
Pre-commit hooks run automatically on `git commit`:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Ruff)
- Type checking (mypy)
- Test execution

### Code Style Guidelines
- **Line length**: 88 characters
- **Type hints**: Required for public APIs
- **Docstrings**: Google format
- **Variable naming**: `snake_case`
- **Class naming**: `PascalCase`
- **Constants**: `UPPER_CASE`

## 📊 Performance Profiling

### Benchmark Development
```python
# Create custom benchmark
from edge_tpu_v5_benchmark.base import BaseBenchmark

class CustomVisionBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(
            name="custom_vision",
            category="vision",
            metrics=["fps", "latency_p99", "accuracy"]
        )
    
    def prepare_model(self, model_path):
        # Model loading logic
        return compiled_model
    
    def run_iteration(self, model, input_data):
        # Single inference
        return {"latency": 0.001, "output": result}
```

### Power Measurement
```python
# Profile power consumption
from edge_tpu_v5_benchmark.power import PowerProfiler

profiler = PowerProfiler(sample_rate=1000)
with profiler.measure() as measurement:
    # Run benchmark
    results = benchmark.run(model, iterations=1000)

stats = measurement.get_statistics()
print(f"Average power: {stats.mean:.3f} W")
```

## 🚀 Release Process

### Version Management
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md

# Create release branch
git checkout -b release/v0.2.0

# Run full test suite
pytest tests/ --cov=src

# Build package
python -m build

# Test installation
pip install dist/edge_tpu_v5_benchmark-0.2.0-py3-none-any.whl
```

### Publishing
```bash
# Upload to PyPI (maintainers only)
twine upload dist/*

# Create GitHub release
gh release create v0.2.0 --title "v0.2.0" --notes "Release notes"
```

## 📈 Performance Optimization

### Profiling Code
```bash
# Profile Python code
python -m cProfile -o profile.stats src/edge_tpu_v5_benchmark/benchmark.py

# Analyze results
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

### Memory Usage
```python
# Monitor memory usage
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

## 🐛 Debugging

### Common Issues

**TPU Not Detected**
```bash
# Check device permissions
ls -la /dev/apex_*
sudo chmod 666 /dev/apex_0

# Verify TPU runtime
edge-tpu-v5-benchmark detect --verbose
```

**Model Compilation Errors**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model compatibility
from edge_tpu_v5_benchmark.analyzer import ModelAnalyzer
analyzer = ModelAnalyzer()
analysis = analyzer.check_compatibility("model.onnx")
```

**Performance Issues**
```bash
# Profile benchmark execution
edge-tpu-v5-benchmark run --model mobilenet_v3 --profile --verbose

# Check system resources
htop
nvidia-smi  # If using GPU fallback
```

## 📚 Documentation

### Building Docs
```bash
cd docs/
make html
open _build/html/index.html
```

### Adding Documentation
- **API docs**: Add docstrings to code
- **Tutorials**: Add to `docs/tutorials/`
- **Guides**: Add to `docs/guides/`
- **Examples**: Add Jupyter notebooks to `examples/`

### Documentation Style
```python
def benchmark_function(param1: str, param2: int = 100) -> BenchmarkResults:
    """Benchmark a specific model configuration.
    
    This function runs a comprehensive benchmark including throughput,
    latency, and power measurements.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2. Defaults to 100.
        
    Returns:
        BenchmarkResults containing performance metrics
        
    Raises:
        TPUError: If TPU device is not available
        ModelError: If model compilation fails
        
    Example:
        >>> benchmark = TPUv5Benchmark()
        >>> results = benchmark_function("mobilenet_v3", iterations=1000)
        >>> print(f"Throughput: {results.throughput:.1f} FPS")
    """
```

## 🔒 Security

### Dependency Management
```bash
# Check for vulnerabilities
pip-audit

# Update dependencies
pip-compile --upgrade requirements.in
```

### Code Security
```bash
# Security linting
bandit -r src/

# Check for secrets
detect-secrets scan --all-files
```

## 🤝 Contributing Workflow

1. **Create issue** for feature/bug
2. **Fork repository** 
3. **Create feature branch**: `git checkout -b feature/awesome-feature`
4. **Make changes** with tests
5. **Run quality checks**: `pre-commit run --all-files`
6. **Push changes**: `git push origin feature/awesome-feature`
7. **Create pull request**

### Pull Request Guidelines
- Clear description of changes
- Reference related issues
- Include tests for new features
- Update documentation if needed
- Ensure CI passes

## 📞 Getting Help

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bug reports and feature requests
- **Email**: daniel@terragonlabs.com (maintainer)
- **Documentation**: https://edge-tpu-v5-benchmark.readthedocs.io