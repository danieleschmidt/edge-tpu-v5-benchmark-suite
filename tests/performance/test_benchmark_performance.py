"""Performance tests for benchmark operations."""

import pytest
import time
from unittest.mock import Mock, patch
import numpy as np


class TestBenchmarkPerformance:
    """Performance tests for benchmark operations."""
    
    @pytest.fixture
    def performance_benchmark(self, mock_tpu_device):
        """Create benchmark instance optimized for performance testing."""
        from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark
        
        with patch('edge_tpu_v5_benchmark.benchmark.TPUDevice') as mock_device_class:
            mock_device_class.return_value = mock_tpu_device
            return TPUv5Benchmark(device_path="/dev/mock_tpu")
    
    @pytest.mark.performance
    def test_benchmark_startup_time(self, mock_tpu_device):
        """Test benchmark initialization performance."""
        from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark
        
        with patch('edge_tpu_v5_benchmark.benchmark.TPUDevice') as mock_device_class:
            mock_device_class.return_value = mock_tpu_device
            
            start_time = time.time()
            benchmark = TPUv5Benchmark(device_path="/dev/mock_tpu")
            initialization_time = time.time() - start_time
            
            # Benchmark initialization should be fast
            assert initialization_time < 1.0  # Less than 1 second
            assert benchmark is not None
    
    @pytest.mark.performance
    def test_single_inference_overhead(self, performance_benchmark, mock_model, sample_input_data):
        """Test overhead of single inference measurement."""
        # Warm up
        for _ in range(10):
            performance_benchmark._run_single_inference(mock_model, sample_input_data)
        
        # Measure overhead
        start_time = time.time()
        for _ in range(100):
            latency = performance_benchmark._run_single_inference(mock_model, sample_input_data)
            assert latency > 0
        measurement_time = time.time() - start_time
        
        # Overhead should be minimal
        overhead_per_inference = measurement_time / 100
        assert overhead_per_inference < 0.001  # Less than 1ms overhead per measurement
    
    @pytest.mark.performance
    def test_memory_efficiency_during_benchmark(self, performance_benchmark, mock_model, sample_input_data):
        """Test memory efficiency during benchmarking."""
        import psutil
        import gc
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Run benchmark
        results = performance_benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=1000,
            warmup_iterations=100
        )
        
        # Check memory usage after benchmark
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        assert results is not None
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_batch_performance(self, performance_benchmark, mock_model):
        """Test performance with large batch processing."""
        # Create large input batch
        large_input = np.random.random((32, 3, 224, 224)).astype(np.float32)
        
        # Mock model to handle batch input
        def batch_inference(input_data):
            batch_size = input_data.shape[0]
            return np.random.random((batch_size, 1000)).astype(np.float32)
        
        mock_model.run_inference = batch_inference
        mock_model.input_shape = (32, 3, 224, 224)
        
        start_time = time.time()
        results = performance_benchmark.run(
            model=mock_model,
            input_data=large_input,
            iterations=100,
            warmup_iterations=10
        )
        total_time = time.time() - start_time
        
        # Large batch processing should be efficient
        assert total_time < 60  # Should complete within 1 minute
        assert results.throughput > 0
        assert results.avg_latency > 0
    
    @pytest.mark.performance
    def test_concurrent_benchmark_performance(self, mock_tpu_device):
        """Test performance with concurrent benchmark instances."""
        from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark
        import threading
        
        def run_benchmark_thread(device_path, results_list):
            with patch('edge_tpu_v5_benchmark.benchmark.TPUDevice') as mock_device_class:
                mock_device_class.return_value = mock_tpu_device
                benchmark = TPUv5Benchmark(device_path=device_path)
                
                mock_model = Mock()
                mock_model.run_inference.return_value = np.random.random((1, 1000))
                mock_model.input_shape = (1, 3, 224, 224)
                
                input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
                
                start_time = time.time()
                result = benchmark.run(
                    model=mock_model,
                    input_data=input_data,
                    iterations=100
                )
                end_time = time.time()
                
                results_list.append({
                    'duration': end_time - start_time,
                    'throughput': result.throughput
                })
        
        # Run multiple benchmarks concurrently
        threads = []
        results_list = []
        
        for i in range(3):
            thread = threading.Thread(
                target=run_benchmark_thread,
                args=(f"/dev/mock_tpu_{i}", results_list)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Verify all benchmarks completed successfully
        assert len(results_list) == 3
        assert all(result['throughput'] > 0 for result in results_list)
        assert all(result['duration'] < 20 for result in results_list)  # Each should complete within 20 seconds
    
    @pytest.mark.performance
    def test_statistical_calculation_performance(self, performance_benchmark):
        """Test performance of statistical calculations."""
        # Generate large dataset for statistical calculations
        latencies = np.random.exponential(0.1, 10000)  # 10k latency measurements
        
        start_time = time.time()
        percentiles = performance_benchmark._calculate_latency_percentiles(latencies)
        calculation_time = time.time() - start_time
        
        # Statistical calculations should be fast
        assert calculation_time < 0.1  # Less than 100ms
        assert 'p50' in percentiles
        assert 'p90' in percentiles
        assert 'p95' in percentiles
        assert 'p99' in percentiles
    
    @pytest.mark.performance
    def test_result_serialization_performance(self, sample_benchmark_results):
        """Test performance of result serialization."""
        # Test JSON serialization performance
        start_time = time.time()
        for _ in range(1000):
            json_str = sample_benchmark_results.to_json()
            assert len(json_str) > 0
        json_time = time.time() - start_time
        
        # Test dict serialization performance  
        start_time = time.time()
        for _ in range(1000):
            result_dict = sample_benchmark_results.to_dict()
            assert isinstance(result_dict, dict)
        dict_time = time.time() - start_time
        
        # Serialization should be fast
        assert json_time < 1.0  # Less than 1 second for 1000 serializations
        assert dict_time < 0.5  # Dict should be faster than JSON
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_benchmark_accuracy_vs_speed_tradeoff(self, performance_benchmark, mock_model, sample_input_data):
        """Test accuracy vs speed tradeoff in benchmarking."""
        # Test different iteration counts
        iteration_counts = [10, 100, 1000, 10000]
        results = []
        
        for iterations in iteration_counts:
            start_time = time.time()
            result = performance_benchmark.run(
                model=mock_model,
                input_data=sample_input_data,
                iterations=iterations,
                warmup_iterations=min(10, iterations // 10)
            )
            duration = time.time() - start_time
            
            results.append({
                'iterations': iterations,
                'duration': duration,
                'latency_std': result.latency_p99 - result.latency_p50  # Measure of precision
            })
        
        # Verify that more iterations provide better precision but take longer
        for i in range(1, len(results)):
            prev_result = results[i-1]
            curr_result = results[i]
            
            # Duration should increase with more iterations
            assert curr_result['duration'] > prev_result['duration']
            
            # Precision should generally improve (lower std deviation)
            # Allow some variance due to mocking
            precision_improvement = prev_result['latency_std'] / curr_result['latency_std']
            assert precision_improvement > 0.5  # At least some improvement
    
    @pytest.mark.performance
    def test_power_monitoring_overhead(self, performance_benchmark, mock_model, 
                                     sample_input_data, mock_power_profiler):
        """Test overhead of power monitoring during benchmarks."""
        performance_benchmark.power_profiler = mock_power_profiler
        
        # Benchmark without power monitoring
        start_time = time.time()
        result_no_power = performance_benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=100,
            monitor_power=False
        )
        time_no_power = time.time() - start_time
        
        # Benchmark with power monitoring
        start_time = time.time()
        result_with_power = performance_benchmark.run(
            model=mock_model,
            input_data=sample_input_data,
            iterations=100,
            monitor_power=True
        )
        time_with_power = time.time() - start_time
        
        # Power monitoring overhead should be minimal
        overhead = time_with_power - time_no_power
        overhead_percentage = (overhead / time_no_power) * 100
        
        assert overhead_percentage < 20  # Less than 20% overhead
        assert result_with_power.power_consumption > 0
        assert result_no_power.power_consumption == 0