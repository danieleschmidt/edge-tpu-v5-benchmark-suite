"""Tests for benchmark module."""

import pytest
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark, BenchmarkResults


class TestTPUv5Benchmark:
    """Test cases for TPUv5Benchmark class."""
    
    def test_init_default_device(self):
        """Test benchmark initialization with default device."""
        benchmark = TPUv5Benchmark()
        assert benchmark.device_path == "/dev/apex_0"
    
    def test_init_custom_device(self):
        """Test benchmark initialization with custom device."""
        custom_path = "/dev/apex_1"
        benchmark = TPUv5Benchmark(device_path=custom_path)
        assert benchmark.device_path == custom_path
    
    def test_run_benchmark(self):
        """Test running a benchmark."""
        benchmark = TPUv5Benchmark()
        mock_model = Mock()
        
        results = benchmark.run(
            model=mock_model,
            input_shape=(1, 3, 224, 224),
            iterations=100,
            warmup=10
        )
        
        assert isinstance(results, BenchmarkResults)
        assert results.total_iterations == 100
        assert results.throughput > 0
        assert results.latency_p99 > 0
        assert results.avg_power > 0
        assert results.inferences_per_watt > 0
    
    def test_get_system_info(self):
        """Test system information retrieval."""
        benchmark = TPUv5Benchmark()
        info = benchmark.get_system_info()
        
        assert "device_path" in info
        assert "tpu_version" in info
        assert "compiler_version" in info
        assert "runtime_version" in info
        assert info["tpu_version"] == "v5_edge"


class TestBenchmarkResults:
    """Test cases for BenchmarkResults dataclass."""
    
    def test_benchmark_results_creation(self):
        """Test BenchmarkResults creation and attributes."""
        results = BenchmarkResults(
            throughput=1000.0,
            latency_p99=1.2,
            avg_power=0.85,
            inferences_per_watt=1176,
            total_iterations=1000,
            duration_seconds=1.0
        )
        
        assert results.throughput == 1000.0
        assert results.latency_p99 == 1.2
        assert results.avg_power == 0.85
        assert results.inferences_per_watt == 1176
        assert results.total_iterations == 1000
        assert results.duration_seconds == 1.0