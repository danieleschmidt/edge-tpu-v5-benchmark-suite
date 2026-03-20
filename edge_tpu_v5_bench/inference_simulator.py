"""
InferenceSimulator: simulate inference performance on the Google Edge TPU v5.

Specs reference:
  - Performance: 4 TOPS (INT8)
  - Memory bandwidth: 8 GB/s
  - Quantization support: INT8, INT4 (2× faster due to halved compute)
"""
from __future__ import annotations
from .model_benchmark import ModelBenchmark


# Quantization adjustment factors (relative to INT8 baseline)
_QUANT_FACTORS: dict = {
    "INT8": 1.0,
    "INT4": 0.5,   # 2× throughput (half the bits → half the compute)
    "FP16": 2.0,   # Half the throughput vs INT8
    "FP32": 4.0,   # Quarter throughput
}


class InferenceSimulator:
    """Simulate inference latency/throughput on the Edge TPU v5.

    Args:
        tops: Tera Operations Per Second (default 4.0 for Edge TPU v5).
        memory_bandwidth_gbps: Memory bandwidth in GB/s (default 8.0).
    """

    def __init__(self, tops: float = 4.0, memory_bandwidth_gbps: float = 8.0) -> None:
        self.tops = tops
        self.memory_bandwidth_gbps = memory_bandwidth_gbps

    def simulate(
        self,
        model_benchmark: ModelBenchmark,
        batch_size: int = 1,
        quantization: str = "INT8",
    ) -> dict:
        """Simulate inference and return performance metrics.

        Returns:
            dict with keys:
              - latency_ms (float): time per inference batch in milliseconds
              - throughput_qps (float): queries per second
              - memory_util (float): fraction of memory bandwidth utilised [0, 1]
        """
        quant_upper = quantization.upper()
        quant_factor = _QUANT_FACTORS.get(quant_upper, 1.0)

        ops = model_benchmark.op_count() * batch_size  # total MACs for this batch
        effective_tops = self.tops / quant_factor  # adjusted for quantization
        ops_per_second = effective_tops * 1e12

        if ops_per_second > 0:
            latency_ms = (ops / ops_per_second) * 1000.0
        else:
            latency_ms = float("inf")

        throughput_qps = (1000.0 / latency_ms) if latency_ms > 0 else 0.0

        # Memory utilisation: bytes moved / available bandwidth
        dtype_bytes = {"INT8": 1, "INT4": 0.5, "FP16": 2, "FP32": 4}.get(quant_upper, 1)
        bytes_moved = model_benchmark.param_count() * dtype_bytes * batch_size
        bandwidth_bytes_per_ms = self.memory_bandwidth_gbps * 1e9 / 1000.0
        if bandwidth_bytes_per_ms > 0 and latency_ms > 0:
            memory_util = min(bytes_moved / (bandwidth_bytes_per_ms * latency_ms), 1.0)
        else:
            memory_util = 0.0

        return {
            "latency_ms": latency_ms,
            "throughput_qps": throughput_qps,
            "memory_util": memory_util,
        }

    def __repr__(self) -> str:
        return (
            f"InferenceSimulator(tops={self.tops}, "
            f"memory_bandwidth_gbps={self.memory_bandwidth_gbps})"
        )
