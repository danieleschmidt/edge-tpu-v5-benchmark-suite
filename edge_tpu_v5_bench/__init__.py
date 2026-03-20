"""
edge_tpu_v5_bench — Benchmark suite for Google Edge TPU v5.

Exports:
  ModelBenchmark      — compute params, ops, memory for a model
  InferenceSimulator  — simulate inference latency/throughput on Edge TPU v5
  PowerEstimator      — estimate power draw and TOPS/W efficiency
  BenchmarkSuite      — orchestrate multiple models, save CSV/JSON
"""

from .model_benchmark import ModelBenchmark
from .inference_simulator import InferenceSimulator
from .power_estimator import PowerEstimator
from .suite import BenchmarkSuite

__all__ = [
    "ModelBenchmark",
    "InferenceSimulator",
    "PowerEstimator",
    "BenchmarkSuite",
]

__version__ = "0.1.0"
