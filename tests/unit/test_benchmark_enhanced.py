"""Enhanced unit tests for benchmark module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from pathlib import Path

from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark, BenchmarkResults


class TestTPUv5BenchmarkEnhanced:
    """Enhanced test cases for TPUv5Benchmark class."""
    
    @pytest.fixture
    def benchmark(self):
        """Create a benchmark instance for testing."""
        return TPUv5Benchmark(device_path="/dev/mock_tpu")
    
    def test_init_default_device(self):
        """Test benchmark initialization with default device."""
        benchmark = TPUv5Benchmark()
        assert benchmark.device_path == "/dev/apex_0"
    
    def test_init_custom_device(self):
        """Test benchmark initialization with custom device."""
        custom_path = "/dev/apex_1"
        benchmark = TPUv5Benchmark(device_path=custom_path)
        assert benchmark.device_path == custom_path
    
    def test_init_device_not_available(self):
        """Test benchmark initialization when device is not available."""
        # Skip this test as TPUDevice class doesn't exist in current implementation
        pytest.skip("TPUDevice class not implemented yet")
    
    def test_run_benchmark_basic(self, benchmark, mock_model, sample_input_data):
        """Test running a basic benchmark."""
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=10,
            warmup_iterations=2
        )
        
        assert isinstance(results, BenchmarkResults)
        assert results.iterations == 10
        assert results.total_time > 0
        assert results.avg_latency > 0
        assert results.throughput > 0
    
    def test_run_benchmark_with_power_monitoring(self, benchmark, mock_model, 
                                                sample_input_data, mock_power_profiler):
        """Test running benchmark with power monitoring."""
        benchmark.power_profiler = mock_power_profiler
        
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=10,
            monitor_power=True
        )
        
        assert results.power_consumption > 0
        assert results.energy_per_inference > 0
        mock_power_profiler.start_measurement.assert_called_once()
        mock_power_profiler.stop_measurement.assert_called_once()
    
    def test_run_benchmark_with_thermal_monitoring(self, benchmark, mock_model,
                                                  sample_input_data, mock_thermal_monitor):
        """Test running benchmark with thermal monitoring."""
        benchmark.thermal_monitor = mock_thermal_monitor
        
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=10,
            monitor_thermal=True
        )
        
        assert results.temperature_avg > 0
        assert results.temperature_max >= results.temperature_avg
        mock_thermal_monitor.get_temperature.assert_called()
    
    def test_run_benchmark_accuracy_validation(self, benchmark, mock_model, sample_input_data):
        """Test running benchmark with accuracy validation."""
        # Mock expected outputs for accuracy calculation
        expected_outputs = [np.random.random((1, 1000)) for _ in range(5)]
        
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=10,
            validate_accuracy=True,
            expected_outputs=expected_outputs
        )
        
        assert 'accuracy' in results.accuracy_metrics
        assert 0 <= results.accuracy_metrics['accuracy'] <= 1
    
    def test_run_benchmark_timeout(self, benchmark, mock_model, sample_input_data):
        """Test benchmark timeout handling."""
        # Mock a slow model that exceeds timeout
        def slow_inference(input_data):
            import time
            time.sleep(2)  # Simulate slow inference
            return np.random.random((1, 1000))
        
        mock_model.run_inference = slow_inference
        
        with pytest.raises(TimeoutError, match="Benchmark exceeded timeout"):
            benchmark.run(
                model=mock_model,
                input_data=sample_input_data,
                iterations=10,
                timeout=1  # 1 second timeout
            )
    
    @pytest.mark.parametrize("iterations,warmup", [
        (1, 0),
        (10, 2),
        (100, 10),
        (1000, 100),
    ])
    def test_run_benchmark_different_iterations(self, benchmark, mock_model, 
                                              sample_input_data, iterations, warmup):
        """Test benchmark with different iteration counts."""
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=iterations,
            warmup_iterations=warmup
        )
        
        assert results.iterations == iterations
        assert results.total_time > 0
        assert results.avg_latency > 0
    
    def test_calculate_latency_percentiles(self, benchmark):
        """Test latency percentile calculations."""
        latencies = [0.1, 0.2, 0.15, 0.3, 0.12, 0.18, 0.25, 0.11, 0.22, 0.16]
        
        percentiles = benchmark._calculate_latency_percentiles(latencies)
        
        assert 'p50' in percentiles
        assert 'p90' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
        assert percentiles['p50'] <= percentiles['p90']
        assert percentiles['p90'] <= percentiles['p95']
        assert percentiles['p95'] <= percentiles['p99']
    
    def test_validate_input_data(self, benchmark, mock_model):
        """Test input data validation."""
        # Valid input data
        valid_input = np.random.random((1, 3, 224, 224)).astype(np.float32)
        assert benchmark._validate_input_data(valid_input, mock_model) is True
        
        # Invalid shape
        invalid_input = np.random.random((1, 3, 128, 128)).astype(np.float32)
        assert benchmark._validate_input_data(invalid_input, mock_model) is False
        
        # Invalid dtype
        invalid_dtype = np.random.random((1, 3, 224, 224)).astype(np.int32)
        assert benchmark._validate_input_data(invalid_dtype, mock_model) is False
    
    def test_warmup_execution(self, benchmark, mock_model, sample_input_data):
        """Test warmup execution before benchmark."""
        with patch.object(benchmark, '_run_single_inference') as mock_inference:
            mock_inference.return_value = 0.1  # Mock latency
            
            benchmark.run(
                model=mock_model,
                input_data=sample_input_data,
                iterations=10,
                warmup_iterations=5
            )
            
            # Verify warmup + actual iterations were called
            assert mock_inference.call_count == 15  # 5 warmup + 10 actual
    
    def test_memory_usage_tracking(self, benchmark, mock_model, sample_input_data):
        """Test memory usage tracking during benchmark."""
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 128_000_000  # 128 MB
            mock_process.return_value.memory_info.return_value = mock_memory_info
            
            results = benchmark.run(
                model=mock_model,
                input_data=sample_input_data,
                iterations=10,
                track_memory=True
            )
            
            assert results.memory_usage > 0
            mock_process.assert_called()
    
    def test_error_handling_model_failure(self, benchmark, sample_input_data):
        """Test error handling when model inference fails."""
        failing_model = Mock()
        failing_model.run_inference.side_effect = RuntimeError("Model inference failed")
        failing_model.input_shape = (1, 3, 224, 224)
        
        with pytest.raises(RuntimeError, match="Model inference failed"):
            benchmark.run(
                model=failing_model,
                input_data=sample_input_data,
                iterations=10
            )
    
    def test_benchmark_results_serialization(self, sample_benchmark_results):
        """Test benchmark results serialization to dict."""
        result_dict = sample_benchmark_results.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'model_name' in result_dict
        assert 'iterations' in result_dict
        assert 'avg_latency' in result_dict
        assert 'throughput' in result_dict
        assert 'power_consumption' in result_dict
    
    def test_benchmark_results_json_serialization(self, sample_benchmark_results):
        """Test benchmark results JSON serialization."""
        import json
        
        json_str = sample_benchmark_results.to_json()
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict)
        assert parsed['model_name'] == 'test_model'
        assert parsed['iterations'] == 100
    
    @pytest.mark.slow
    def test_stress_benchmark(self, benchmark, mock_model, sample_input_data):
        """Test benchmark under stress conditions (many iterations)."""
        results = benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=1000,
            warmup_iterations=100
        )
        
        assert results.iterations == 1000
        assert results.avg_latency > 0
        assert results.throughput > 0
        # Verify statistical measures are reasonable with large sample
        assert results.latency_p99 > results.latency_p95
        assert results.latency_p95 > results.latency_p90