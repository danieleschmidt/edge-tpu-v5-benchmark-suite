# 📚 Edge TPU v5 Benchmark Suite - API Reference

## Overview

The Edge TPU v5 Benchmark Suite provides a comprehensive Python API for benchmarking Google's TPU v5 edge cards. This reference covers all public classes, methods, and configuration options.

## Table of Contents

- [Core Benchmarking](#core-benchmarking)
- [Model Management](#model-management)
- [Power Profiling](#power-profiling)
- [Quantum Task Planning](#quantum-task-planning)
- [Configuration Management](#configuration-management)
- [Validation & Health](#validation--health)
- [Caching System](#caching-system)
- [Research Framework](#research-framework)
- [CLI Interface](#cli-interface)

---

## Core Benchmarking

### TPUv5Benchmark

Main benchmark class for TPU v5 edge devices.

```python
from edge_tpu_v5_benchmark import TPUv5Benchmark

benchmark = TPUv5Benchmark(
    device_path="/dev/apex_0",
    enable_power_monitoring=True
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device_path` | `str` | `"/dev/apex_0"` | Path to TPU device |
| `enable_power_monitoring` | `bool` | `True` | Enable real-time power monitoring |

#### Methods

##### `run(model, input_shape, iterations=1000, warmup=100, **kwargs)`

Run comprehensive benchmark on a model.

**Parameters:**
- `model` (`CompiledTPUModel`): Compiled TPU model
- `input_shape` (`tuple`): Input tensor shape
- `iterations` (`int`): Number of benchmark iterations
- `warmup` (`int`): Number of warmup iterations
- `batch_size` (`int`, optional): Batch size for inference
- `measure_power` (`bool`, optional): Override power measurement setting
- `confidence_level` (`float`, optional): Statistical confidence level (0.95)

**Returns:** `BenchmarkResults`

**Example:**
```python
results = benchmark.run(
    model=compiled_model,
    input_shape=(1, 3, 224, 224),
    iterations=1000,
    warmup=100,
    measure_power=True
)

print(f"Throughput: {results.throughput:.1f} FPS")
print(f"Latency p99: {results.latency_p99:.2f} ms")
print(f"Power: {results.avg_power:.2f} W")
```

##### `get_system_info()`

Get comprehensive system information.

**Returns:** `Dict[str, Any]` - System information including TPU details

---

### BenchmarkResults

Results container for benchmark execution.

```python
@dataclass
class BenchmarkResults:
    throughput: float
    latency_p99: float
    latency_p95: float
    latency_p50: float
    latency_mean: float
    latency_std: float
    avg_power: float
    peak_power: float
    energy_consumed: float
    inferences_per_watt: float
    total_iterations: int
    success_rate: float
    # ... additional metrics
```

#### Methods

##### `to_dict()`

Convert results to dictionary for serialization.

**Returns:** `Dict[str, Any]`

##### `save_json(filepath)`

Save results to JSON file.

**Parameters:**
- `filepath` (`Union[str, Path]`): Output file path

---

## Model Management

### ModelLoader

Factory class for loading models in various formats.

#### Class Methods

##### `from_onnx(model_path, optimization_level=3, target="tpu_v5_edge")`

Load model from ONNX format.

**Parameters:**
- `model_path` (`str`): Path to ONNX model file
- `optimization_level` (`int`): Optimization level (1-3)
- `target` (`str`): Target device architecture

**Returns:** `CompiledTPUModel`

**Example:**
```python
from edge_tpu_v5_benchmark import ModelLoader

model = ModelLoader.from_onnx(
    "mobilenet_v3.onnx",
    optimization_level=3,
    target="tpu_v5_edge"
)
```

##### `from_tflite(model_path)`

Load model from TensorFlow Lite format.

**Parameters:**
- `model_path` (`str`): Path to TFLite model file

**Returns:** `CompiledTPUModel`

---

### CompiledTPUModel

Represents a compiled TPU v5 model.

#### Methods

##### `run(input_data)`

Run inference on the model.

**Parameters:**
- `input_data`: Input tensor data

**Returns:** Model output

##### `get_info()`

Get detailed model information.

**Returns:** `Dict[str, Any]` - Model metadata and statistics

##### `get_performance_stats()`

Get performance statistics.

**Returns:** `Dict[str, float]` - Performance metrics

---

### ModelRegistry

Registry for managing available models.

```python
from edge_tpu_v5_benchmark import ModelRegistry

registry = ModelRegistry()
models = registry.list_models()
```

#### Methods

##### `list_models(category=None)`

List available models.

**Parameters:**
- `category` (`str`, optional): Filter by category

**Returns:** `Dict[str, Dict[str, Any]]` - Model registry

##### `get_model_info(model_id)`

Get information about a specific model.

**Parameters:**
- `model_id` (`str`): Model identifier

**Returns:** `Optional[Dict[str, Any]]` - Model information

##### `register_model(model_id, info)`

Register a new model.

**Parameters:**
- `model_id` (`str`): Unique model identifier
- `info` (`Dict[str, Any]`): Model metadata

---

## Power Profiling

### PowerProfiler

Real-time power consumption profiler.

```python
from edge_tpu_v5_benchmark import PowerProfiler

profiler = PowerProfiler(
    device="/dev/apex_0",
    sample_rate=1000
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"/dev/apex_0"` | TPU device path |
| `sample_rate` | `int` | `1000` | Sampling rate in Hz |

#### Context Manager Usage

```python
with profiler.measure() as measurement:
    # Run benchmarks
    for _ in range(100):
        model.run(input_data)

# Analyze results
stats = measurement.get_statistics()
print(f"Average power: {stats.mean:.3f} W")
print(f"Peak power: {stats.max:.3f} W")
```

### PowerMeasurement

Power measurement container with statistical analysis.

#### Methods

##### `get_statistics()`

Get power consumption statistics.

**Returns:** `PowerStatistics` - Statistical summary

##### `plot_timeline(save_path, show_events=True)`

Generate power consumption timeline plot.

**Parameters:**
- `save_path` (`str`): Output file path
- `show_events` (`bool`): Show benchmark events

---

## Quantum Task Planning

### QuantumTaskPlanner

Quantum-inspired task scheduler for optimal resource allocation.

```python
from edge_tpu_v5_benchmark import QuantumTaskPlanner

planner = QuantumTaskPlanner(
    max_concurrent_tasks=5,
    optimization_strategy="balanced"
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent_tasks` | `int` | `4` | Maximum concurrent tasks |
| `optimization_strategy` | `str` | `"balanced"` | Optimization strategy |
| `resource_constraints` | `Dict`, optional | `None` | Resource limits |

#### Methods

##### `plan_task_execution(tasks)`

Create optimized execution plan for tasks.

**Parameters:**
- `tasks` (`List[Dict[str, Any]]`): Task definitions

**Returns:** `List[QuantumTask]` - Optimized task schedule

**Example:**
```python
tasks = [
    {"model": "mobilenet_v3", "priority": "high", "complexity": 0.3},
    {"model": "yolov8n", "priority": "medium", "complexity": 0.8}
]

plan = planner.plan_task_execution(tasks)
```

##### `execute_plan(plan)`

Execute the optimized task plan.

**Parameters:**
- `plan` (`List[QuantumTask]`): Task execution plan

**Returns:** `Dict[str, Any]` - Execution results

---

### QuantumAutoScaler

Intelligent auto-scaling system.

```python
from edge_tpu_v5_benchmark import QuantumAutoScaler

scaler = QuantumAutoScaler(
    min_nodes=1,
    max_nodes=10,
    target_cpu_utilization=70
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_nodes` | `int` | `1` | Minimum node count |
| `max_nodes` | `int` | `10` | Maximum node count |
| `target_cpu_utilization` | `float` | `70.0` | Target CPU utilization % |
| `scale_up_threshold` | `float` | `80.0` | Scale up threshold % |
| `scale_down_threshold` | `float` | `30.0` | Scale down threshold % |

---

## Configuration Management

### get_config()

Get current benchmark configuration.

**Returns:** `BenchmarkSuiteConfig`

### initialize_config(config_path=None, **overrides)

Initialize configuration system.

**Parameters:**
- `config_path` (`Optional[str]`): Path to configuration file
- `**overrides`: Configuration overrides

**Example:**
```python
from edge_tpu_v5_benchmark import initialize_config, get_config

initialize_config(
    device_path="/dev/apex_1",
    log_level="DEBUG",
    power_monitoring=True
)

config = get_config()
print(f"Device: {config.device_path}")
```

---

## Validation & Health

### BenchmarkValidator

Comprehensive input validation system.

```python
from edge_tpu_v5_benchmark.validation import BenchmarkValidator

validator = BenchmarkValidator()
```

#### Methods

##### `validate_benchmark_config(**config)`

Validate benchmark configuration parameters.

**Parameters:**
- `iterations` (`int`): Number of iterations
- `warmup` (`int`): Warmup iterations
- `batch_size` (`int`): Batch size
- `input_shape` (`tuple`): Input tensor shape
- `confidence_level` (`float`): Statistical confidence level

**Returns:** `ValidationResult`

**Example:**
```python
result = validator.validate_benchmark_config(
    iterations=1000,
    warmup=100,
    batch_size=1,
    input_shape=(1, 3, 224, 224),
    confidence_level=0.95
)

if result.is_valid:
    print("Configuration is valid")
else:
    for issue in result.issues:
        print(f"Issue: {issue.message}")
```

##### `validate_model_path(model_path)`

Validate model file path and format.

**Parameters:**
- `model_path` (`Union[str, Path]`): Path to model file

**Returns:** `ValidationResult`

---

### Health Monitoring

#### check_system_health()

Perform comprehensive system health check.

**Returns:** `SystemHealth`

**Example:**
```python
from edge_tpu_v5_benchmark.health import check_system_health

health = check_system_health()
print(f"Overall status: {health.overall_status.value}")

for check in health.checks:
    print(f"{check.name}: {check.status.value}")
```

---

## Caching System

### get_cache(name)

Get named cache instance.

**Parameters:**
- `name` (`str`): Cache name ("models", "results", "analysis", "conversions")

**Returns:** `Optional[SmartCache]`

**Example:**
```python
from edge_tpu_v5_benchmark import get_cache

models_cache = get_cache('models')
if models_cache:
    # Cache compiled model
    models_cache.set("mobilenet_v3_opt3", compiled_model, ttl=3600)
    
    # Retrieve cached model
    cached_model = models_cache.get("mobilenet_v3_opt3")
```

### SmartCache

Intelligent multi-tier caching system.

#### Methods

##### `get(key, default=None)`

Get value from cache.

**Parameters:**
- `key` (`str`): Cache key
- `default` (`Any`): Default value if key not found

**Returns:** Cached value or default

##### `set(key, value, ttl=None, force_disk=False)`

Set value in cache.

**Parameters:**
- `key` (`str`): Cache key
- `value` (`Any`): Value to cache
- `ttl` (`Optional[int]`): Time to live in seconds
- `force_disk` (`bool`): Force disk storage

**Returns:** `bool` - Success status

##### `get_statistics()`

Get cache performance statistics.

**Returns:** `Dict[str, Any]` - Cache statistics

---

## Research Framework

### QuantumTPUResearchFramework

Academic-grade research platform for TPU benchmarking studies.

```python
from edge_tpu_v5_benchmark.research_framework import QuantumTPUResearchFramework

framework = QuantumTPUResearchFramework()
```

#### Methods

##### `create_study(name, models, **config)`

Create a new research study.

**Parameters:**
- `name` (`str`): Study name
- `models` (`List[str]`): Models to benchmark
- `statistical_significance` (`float`): Significance level
- `min_samples` (`int`): Minimum sample size
- Additional configuration parameters

**Returns:** `ResearchStudy`

##### `execute_study(study)`

Execute research study with statistical validation.

**Parameters:**
- `study` (`ResearchStudy`): Study configuration

**Returns:** `StudyResults` - Publication-ready results

**Example:**
```python
study = framework.create_study(
    name="TPU v5 Performance Analysis",
    models=["mobilenet_v3", "efficientnet_lite"],
    batch_sizes=[1, 2, 4, 8],
    statistical_significance=0.05,
    min_samples=30
)

results = framework.execute_study(study)
print(f"Significant findings: {results.significant_findings}")
```

---

## CLI Interface

### Command Line Usage

The benchmark suite provides a comprehensive CLI interface.

#### Main Commands

```bash
# Detect TPU devices
edge-tpu-v5-benchmark detect

# Run benchmark suite
edge-tpu-v5-benchmark run --workload all --iterations 1000

# Run specific model
edge-tpu-v5-benchmark run --model mobilenet_v3 --profile-power

# Quick benchmark on single model
edge-tpu-v5-benchmark quick-benchmark --model-path model.onnx --iterations 100

# View leaderboard
edge-tpu-v5-benchmark leaderboard --category vision --metric throughput

# Quantum planning commands
edge-tpu-v5-benchmark quantum plan --tasks task_config.yaml
edge-tpu-v5-benchmark quantum execute --plan execution_plan.json
edge-tpu-v5-benchmark quantum monitor --duration 60
```

#### Global Options

| Option | Description |
|--------|-------------|
| `--device` | TPU device path |
| `--config` | Configuration file path |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `--output` | Output file for results |
| `--format` | Output format (json, csv, yaml) |

---

## Error Handling

### Exception Hierarchy

```python
from edge_tpu_v5_benchmark.exceptions import (
    TPUBenchmarkError,        # Base exception
    DeviceError,              # TPU device issues
    ModelError,               # Model loading/compilation
    BenchmarkError,           # Benchmark execution
    ValidationError,          # Input validation
    ResourceError,            # System resources
    TimeoutError,             # Operation timeouts
    SecurityError             # Security violations
)
```

### Error Context

```python
from edge_tpu_v5_benchmark.exceptions import ErrorContext, handle_error

context = ErrorContext(
    component="benchmark",
    operation="model_inference",
    model_name="mobilenet_v3"
)

try:
    # Benchmark operation
    results = benchmark.run(model, input_shape)
except TPUBenchmarkError as e:
    error_report = handle_error(e, context, reraise=False)
    print(f"Error: {error_report['error_message']}")
```

---

## Type Hints and Annotations

The API extensively uses Python type hints for better IDE support and validation:

```python
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

def run_benchmark(
    model: CompiledTPUModel,
    input_shape: Tuple[int, ...],
    iterations: int = 1000,
    config: Optional[Dict[str, Any]] = None
) -> BenchmarkResults:
    # Implementation
    pass
```

---

## Environment Variables

### Configuration Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_TPU_DEVICE_PATH` | `/dev/apex_0` | TPU device path |
| `EDGE_TPU_LOG_LEVEL` | `INFO` | Logging level |
| `EDGE_TPU_CACHE_DIR` | `~/.cache/edge-tpu-v5-benchmark` | Cache directory |
| `BENCHMARK_OUTPUT_DIR` | `./results` | Output directory |
| `POWER_MEASUREMENT_ENABLED` | `true` | Enable power monitoring |
| `QUANTUM_PLANNING_ENABLED` | `true` | Enable quantum planning |
| `TELEMETRY_ENABLED` | `false` | Enable telemetry |

### Development Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGE_TPU_DEV_MODE` | `false` | Development mode |
| `PYTEST_CURRENT_TEST` | - | Current test identifier |
| `COVERAGE_PROCESS_START` | - | Coverage configuration |

---

## Best Practices

### 1. Model Loading
```python
# Good: Use context manager for resource cleanup
with ModelLoader.from_onnx("model.onnx") as model:
    results = benchmark.run(model, input_shape)

# Good: Specify optimization level explicitly
model = ModelLoader.from_onnx(
    "model.onnx", 
    optimization_level=3,  # Maximum optimization
    target="tpu_v5_edge"
)
```

### 2. Error Handling
```python
# Good: Handle specific exceptions
try:
    results = benchmark.run(model, input_shape)
except DeviceNotFoundError:
    logger.warning("No TPU device found, using simulation")
    benchmark = TPUv5Benchmark(simulation_mode=True)
except ValidationError as e:
    logger.error(f"Invalid configuration: {e.message}")
    return
```

### 3. Configuration Management
```python
# Good: Initialize configuration early
initialize_config(
    device_path="/dev/apex_0",
    log_level="INFO",
    power_monitoring=True
)

config = get_config()
benchmark = TPUv5Benchmark(**config.benchmark_settings)
```

### 4. Resource Management
```python
# Good: Use appropriate cache for data type
models_cache = get_cache('models')
results_cache = get_cache('results')

# Cache large objects to disk
models_cache.set("large_model", model, force_disk=True)

# Cache small results in memory
results_cache.set("quick_benchmark", results, ttl=3600)
```

---

## Examples and Tutorials

For complete examples and tutorials, see:
- `/examples/basic_benchmark.py` - Basic benchmarking workflow
- `/examples/quantum_planning_demo.py` - Quantum task planning
- `/examples/quantum_research_demo.py` - Research framework usage
- `/docs/guides/` - Detailed usage guides

---

## Version Information

**API Version**: 0.1.0  
**Compatibility**: Python 3.8+  
**TPU Support**: v5 Edge (primary), v4 (legacy)  
**Last Updated**: August 2025

For the latest API documentation and updates, visit: https://edge-tpu-v5-benchmark.readthedocs.io