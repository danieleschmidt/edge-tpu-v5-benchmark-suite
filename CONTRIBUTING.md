# Contributing to Edge TPU v5 Benchmark Suite

We welcome contributions to make this the definitive benchmark suite for TPU v5 edge devices! 

## 🎯 Priority Areas

- **Additional model benchmarks** - New vision, NLP, and custom models
- **Compiler optimization discoveries** - TPU v5 compiler quirks and optimizations  
- **Power measurement improvements** - Enhanced energy profiling accuracy
- **Documentation of undocumented features** - TPU v5 hidden capabilities

## 🚀 Quick Start

1. **Fork and clone**
   ```bash
   git clone https://github.com/yourusername/edge-tpu-v5-benchmark-suite.git
   cd edge-tpu-v5-benchmark-suite
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Run tests**
   ```bash
   pytest
   ```

## 📋 Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes** following our code style
3. **Add tests** for new functionality  
4. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   
   # Lint code  
   ruff check src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest --cov=src
   ```

5. **Submit pull request**

## 🧪 Testing Guidelines

### Test Categories
- **Unit tests** - Fast, isolated component tests
- **Integration tests** - Multi-component interaction tests  
- **Hardware tests** - Require actual TPU v5 hardware (mark with `@pytest.mark.hardware`)

### Test Structure
```python
class TestYourFeature:
    def test_specific_behavior(self):
        # Arrange
        setup_code()
        
        # Act  
        result = function_under_test()
        
        # Assert
        assert result.expected_property == expected_value
```

### Hardware Tests
Mark tests requiring TPU hardware:
```python
@pytest.mark.hardware
def test_tpu_inference():
    # Test that requires actual TPU v5 device
    pass
```

Run tests without hardware: `pytest -m "not hardware"`

## 💻 Code Style

### Python Code Standards
- **Line length**: 88 characters (Black default)
- **Import sorting**: isort compatible
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for public functions

### Example Function
```python
def benchmark_model(
    model: CompiledTPUModel,
    iterations: int = 1000,
    warmup: int = 100
) -> BenchmarkResults:
    """Run benchmark on TPU v5 model.
    
    Args:
        model: Compiled TPU model to benchmark
        iterations: Number of inference iterations
        warmup: Warmup iterations before measurement
        
    Returns:
        BenchmarkResults with performance metrics
        
    Raises:
        TPUError: If TPU device is unavailable
    """
    # Implementation
```

## 🔧 Adding New Benchmarks

### Custom Benchmark Class
```python
from edge_tpu_v5_benchmark.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(
            name="my_custom_benchmark",
            category="vision",  # or "nlp", "multimodal" 
            metrics=["throughput", "latency", "accuracy"]
        )
    
    def prepare_model(self, model_path: str) -> Any:
        """Load and prepare model for benchmarking."""
        pass
        
    def run_iteration(self, model: Any, input_data: Any) -> Dict[str, float]:
        """Run single benchmark iteration."""
        pass
        
    def calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate final benchmark metrics.""" 
        pass

# Register benchmark
from edge_tpu_v5_benchmark import register_benchmark
register_benchmark(MyBenchmark)
```

### Model Conversion Helpers
```python
from edge_tpu_v5_benchmark.converter import ONNXToTPUv5

converter = ONNXToTPUv5()
tpu_model = converter.convert(
    "model.onnx",
    optimization_profile="balanced"
)
```

## 📖 Documentation

### Adding Documentation
- **API docs**: Docstrings in Google format
- **Guides**: Markdown files in `docs/guides/`  
- **Examples**: Jupyter notebooks in `examples/`

### Building Docs Locally
```bash
cd docs/
make html
open _build/html/index.html
```

## 🐛 Bug Reports

### Bug Report Template
```markdown
**Bug Description**
Clear description of the bug

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10.2  
- TPU: v5 edge
- Package version: 0.1.0

**Reproduction Steps**
1. Step one
2. Step two
3. See error

**Expected vs Actual**
- Expected: X should happen
- Actual: Y happened instead

**Additional Context**
Any other relevant information
```

## 🚀 Feature Requests

### Feature Request Template  
```markdown
**Feature Description**
Clear description of proposed feature

**Use Case**
Why this feature would be valuable

**Proposed Implementation**
Technical approach (optional)

**Alternatives Considered**
Other approaches you've considered
```

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## 🏆 Recognition

Contributors are recognized in:
- Repository contributors list
- Release notes for significant contributions
- Optional co-authorship for major features

## ❓ Questions

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue  
- **Security issues**: See [SECURITY.md](SECURITY.md)

Thank you for contributing to the Edge TPU v5 Benchmark Suite! 🚀