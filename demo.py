#!/usr/bin/env python3
"""
demo.py — Demonstrate the edge_tpu_v5_bench suite with example models.

Models included:
  - MobileNet-like (depthwise separable convolutions)
  - ResNet-like (standard convolutions + FC head)
  - Tiny MLP (linear layers only)
"""
import json
from edge_tpu_v5_bench import (
    ModelBenchmark,
    InferenceSimulator,
    PowerEstimator,
    BenchmarkSuite,
)

# ---------------------------------------------------------------------------
# Define models via layer specs
# ---------------------------------------------------------------------------

MOBILENET_LIKE = [
    # Initial conv
    {"type": "conv", "in_channels": 3, "out_channels": 32, "kernel_size": 3, "output_size": [112, 112]},
    # Depthwise separable blocks (simplified)
    {"type": "depthwise_conv", "channels": 32, "kernel_size": 3, "output_size": [112, 112]},
    {"type": "conv", "in_channels": 32, "out_channels": 64, "kernel_size": 1, "output_size": [112, 112]},
    {"type": "depthwise_conv", "channels": 64, "kernel_size": 3, "output_size": [56, 56]},
    {"type": "conv", "in_channels": 64, "out_channels": 128, "kernel_size": 1, "output_size": [56, 56]},
    {"type": "depthwise_conv", "channels": 128, "kernel_size": 3, "output_size": [28, 28]},
    {"type": "conv", "in_channels": 128, "out_channels": 256, "kernel_size": 1, "output_size": [28, 28]},
    {"type": "depthwise_conv", "channels": 256, "kernel_size": 3, "output_size": [14, 14]},
    {"type": "conv", "in_channels": 256, "out_channels": 512, "kernel_size": 1, "output_size": [14, 14]},
    # Classifier head
    {"type": "linear", "in_features": 512, "out_features": 1000},
]

RESNET_LIKE = [
    # Stem
    {"type": "conv", "in_channels": 3, "out_channels": 64, "kernel_size": 7, "output_size": [56, 56]},
    {"type": "batchnorm", "num_features": 64},
    # Stage 1
    {"type": "conv", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "output_size": [56, 56]},
    {"type": "conv", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "output_size": [56, 56]},
    # Stage 2
    {"type": "conv", "in_channels": 64, "out_channels": 128, "kernel_size": 3, "output_size": [28, 28]},
    {"type": "conv", "in_channels": 128, "out_channels": 128, "kernel_size": 3, "output_size": [28, 28]},
    # Stage 3
    {"type": "conv", "in_channels": 128, "out_channels": 256, "kernel_size": 3, "output_size": [14, 14]},
    {"type": "conv", "in_channels": 256, "out_channels": 256, "kernel_size": 3, "output_size": [14, 14]},
    # Classifier
    {"type": "linear", "in_features": 256, "out_features": 1000},
]

TINY_MLP = [
    {"type": "linear", "in_features": 784, "out_features": 256},
    {"type": "linear", "in_features": 256, "out_features": 128},
    {"type": "linear", "in_features": 128, "out_features": 64},
    {"type": "linear", "in_features": 64, "out_features": 10},
]


def main():
    # Setup
    simulator = InferenceSimulator(tops=4.0, memory_bandwidth_gbps=8.0)
    estimator = PowerEstimator(power_budget_w=2.0, efficiency_tops_per_w=50.0)
    suite = BenchmarkSuite()

    suite.add_model(ModelBenchmark("MobileNet-like", MOBILENET_LIKE))
    suite.add_model(ModelBenchmark("ResNet-like", RESNET_LIKE))
    suite.add_model(ModelBenchmark("Tiny-MLP", TINY_MLP))

    print("=" * 70)
    print("  Edge TPU v5 Benchmark Suite — Demo")
    print("  Target efficiency: 50 TOPS/W | Power budget: 2.0 W")
    print("=" * 70)

    for quantization in ("INT8", "INT4"):
        print(f"\n[Quantization: {quantization}]")
        results = suite.run_all(simulator, estimator, batch_size=1, quantization=quantization)
        for r in results:
            budget_str = "✓" if r["within_budget"] else "✗"
            print(
                f"  {r['model_name']:<20} "
                f"params={r['param_count']:>10,}  "
                f"latency={r['latency_ms']:>10.4f}ms  "
                f"qps={r['throughput_qps']:>12.2f}  "
                f"mem={r['memory_mb_int8']:>7.3f}MB  "
                f"power={r['power_w']:.4f}W {budget_str}"
            )

    # Save final INT8 results
    results = suite.run_all(simulator, estimator, batch_size=1, quantization="INT8")
    suite.save_json(results, "/tmp/demo_results.json")
    print(f"\nResults saved to /tmp/demo_results.json")
    print("Done.")


if __name__ == "__main__":
    main()
