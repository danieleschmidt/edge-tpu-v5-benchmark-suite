"""
BenchmarkSuite: orchestrate multiple model benchmarks and persist results.
"""
from __future__ import annotations
import csv
import json
from typing import List

from .model_benchmark import ModelBenchmark
from .inference_simulator import InferenceSimulator
from .power_estimator import PowerEstimator


class BenchmarkSuite:
    """Manage a collection of models and run them through the benchmark pipeline."""

    def __init__(self) -> None:
        self._models: List[ModelBenchmark] = []

    def add_model(self, model_benchmark: ModelBenchmark) -> None:
        """Register a model for benchmarking."""
        self._models.append(model_benchmark)

    def run_all(
        self,
        simulator: InferenceSimulator,
        estimator: PowerEstimator,
        batch_size: int = 1,
        quantization: str = "INT8",
    ) -> List[dict]:
        """Run all registered models and return a list of result dicts.

        Each result dict contains model metadata plus inference and power metrics.
        """
        results = []
        for model in self._models:
            infer = simulator.simulate(model, batch_size=batch_size, quantization=quantization)
            power = estimator.estimate(model, batch_size=batch_size)
            result = {
                "model_name": model.model_name,
                "param_count": model.param_count(),
                "op_count": model.op_count(),
                "memory_mb_int8": round(model.memory_mb(dtype_bytes=1), 4),
                "latency_ms": round(infer["latency_ms"], 6),
                "throughput_qps": round(infer["throughput_qps"], 4),
                "memory_util": round(infer["memory_util"], 6),
                "power_w": round(power["power_w"], 6),
                "tops_utilized": round(power["tops_utilized"], 6),
                "tops_per_watt": round(power["tops_per_watt"], 4),
                "within_budget": power["within_budget"],
                "batch_size": batch_size,
                "quantization": quantization,
            }
            results.append(result)
        return results

    def save_csv(self, results: List[dict], path: str) -> None:
        """Write results to a CSV file."""
        if not results:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    def save_json(self, results: List[dict], path: str) -> None:
        """Write results to a JSON file."""
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"BenchmarkSuite(models={len(self._models)})"
