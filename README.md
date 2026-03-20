# Edge TPU v5 Benchmark Suite

A Python benchmarking toolkit for estimating the performance of neural network models on the **Google Edge TPU v5**.

## Features

- **Parameter counting** — exact param counts for conv, depthwise conv, linear, batchnorm layers
- **Operation counting** — MACs (multiply-accumulate) estimation per layer
- **Inference simulation** — latency, throughput (QPS), and memory utilisation estimates
- **Power estimation** — power draw and **50 TOPS/W efficiency** target tracking
- **Multi-model suite** — batch-benchmark multiple models, export to CSV or JSON
- **CLI** — `edge-tpu-v5-bench` command for quick one-off analysis

## Hardware Specs (Edge TPU v5)

| Spec                | Value          |
|---------------------|----------------|
| Peak performance    | 4 TOPS (INT8)  |
| Memory bandwidth    | 8 GB/s         |
| Power budget        | ~2 W           |
| **Efficiency**      | **50 TOPS/W**  |
| Quantization        | INT8, INT4     |

## Installation

```bash
pip install -e .
```

## Quick Start

### Python API

```python
from edge_tpu_v5_bench import (
    ModelBenchmark,
    InferenceSimulator,
    PowerEstimator,
    BenchmarkSuite,
)

# Describe your model
model = ModelBenchmark("MobileNet-tiny", [
    {"type": "conv", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "output_size": [112, 112]},
    {"type": "depthwise_conv", "channels": 32, "kernel_size": 3, "output_size": [56, 56]},
    {"type": "linear", "in_features": 512, "out_features": 1000},
])

simulator = InferenceSimulator(tops=4.0, memory_bandwidth_gbps=8.0)
estimator = PowerEstimator(power_budget_w=2.0, efficiency_tops_per_w=50.0)

suite = BenchmarkSuite()
suite.add_model(model)
results = suite.run_all(simulator, estimator, quantization="INT8")

for r in results:
    print(r)

suite.save_json(results, "results.json")
suite.save_csv(results, "results.csv")
```

### CLI

```bash
edge-tpu-v5-bench \
  --model-name "MyModel" \
  --layers '[{"type":"conv","in_channels":3,"out_channels":32,"kernel_size":3}]' \
  --batch-size 1 \
  --quantization INT8 \
  --output-json results.json
```

### Demo

```bash
python demo.py
```

## Layer Spec Format

Each layer is a dict with a `"type"` key and type-specific fields:

| type             | required fields                                              |
|------------------|--------------------------------------------------------------|
| `conv`           | `in_channels`, `out_channels`, `kernel_size`                 |
| `depthwise_conv` | `channels`, `kernel_size`                                    |
| `linear`         | `in_features`, `out_features`                                |
| `batchnorm`      | `num_features`                                               |

All spatial layers accept an optional `"output_size": [H, W]` for accurate MAC counting (defaults to `[1, 1]`).

## Running Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

## Project Structure

```
edge_tpu_v5_bench/
  __init__.py           — package exports
  model_benchmark.py    — ModelBenchmark class
  inference_simulator.py — InferenceSimulator class
  power_estimator.py    — PowerEstimator class
  suite.py              — BenchmarkSuite class
  cli.py                — CLI entry point
tests/
  test_model_benchmark.py
  test_inference_simulator.py
  test_power_estimator.py
  test_suite.py
demo.py
setup.py
```

## License

MIT
