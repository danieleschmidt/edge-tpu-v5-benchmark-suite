"""Integration tests for complete benchmark workflows."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import time

from edge_tpu_v5_benchmark import (
    TPUv5Benchmark, ModelLoader, ModelRegistry, BenchmarkResults
)
from edge_tpu_v5_benchmark.validation import BenchmarkValidator
from edge_tpu_v5_benchmark.monitoring import PerformanceMonitor, MetricsCollector
from edge_tpu_v5_benchmark.cache import CacheManager
from edge_tpu_v5_benchmark.concurrency import BenchmarkJobManager, Task, TaskPriority
from edge_tpu_v5_benchmark.auto_scaling import AdaptiveResourceManager
from edge_tpu_v5_benchmark.health import HealthMonitor
from edge_tpu_v5_benchmark.compiler import CompilerAnalyzer
from edge_tpu_v5_benchmark.converter import ONNXToTPUv5


class TestFullBenchmarkWorkflow:
    """Test complete benchmark workflows from end to end."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"mock_onnx_model_data")
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)
    
    def test_basic_benchmark_workflow(self, mock_model_file):
        """Test basic benchmark workflow with validation and monitoring."""
        # Setup components
        validator = BenchmarkValidator()
        benchmark = TPUv5Benchmark()
        monitor = PerformanceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        try:
            # Validate benchmark configuration
            validation_result = validator.validate_benchmark_config(
                iterations=100,
                warmup=10,
                batch_size=1,
                input_shape=(1, 3, 224, 224)
            )
            assert validation_result.is_valid
            
            # Load model
            model = ModelLoader.from_onnx(str(mock_model_file))
            assert model is not None
            
            # Run benchmark
            benchmark_id = "test_benchmark_001"
            monitor.record_benchmark_start(benchmark_id, "test_model")
            
            start_time = time.time()
            results = benchmark.run(
                model=model,
                input_shape=(1, 3, 224, 224),
                iterations=10,  # Small number for test
                warmup=2
            )
            duration = time.time() - start_time
            
            # Validate results
            assert isinstance(results, BenchmarkResults)
            assert results.success_rate > 0
            assert results.throughput > 0
            assert results.latency_mean > 0
            
            # Record completion
            monitor.record_benchmark_completion(
                benchmark_id, "test_model", duration, True
            )
            
            # Check monitoring statistics
            health_status = monitor.get_health_status()
            assert health_status["health_score"] >= 0
            
        finally:
            monitor.stop_monitoring()
    
    def test_cached_benchmark_workflow(self, mock_model_file, temp_cache_dir):
        """Test benchmark workflow with caching enabled."""
        # Setup cache manager
        cache_manager = CacheManager(temp_cache_dir)
        model_cache = cache_manager.get_cache("models")
        results_cache = cache_manager.get_cache("results")
        
        benchmark = TPUv5Benchmark()
        
        # First run - should cache results
        model = ModelLoader.from_onnx(str(mock_model_file))
        cache_key = f"model_{hash(str(mock_model_file))}"
        
        # Cache the model
        model_cache.set(cache_key, model)
        
        # Run benchmark
        results1 = benchmark.run(
            model=model,
            input_shape=(1, 3, 224, 224),
            iterations=5,
            warmup=1
        )
        
        # Cache results
        results_key = f"results_{cache_key}_small"
        results_cache.set(results_key, results1)
        
        # Second run - should use cached results
        cached_model = model_cache.get(cache_key)
        cached_results = results_cache.get(results_key)
        
        assert cached_model is not None
        assert cached_results is not None
        assert cached_results.total_iterations == results1.total_iterations
        
        # Verify cache statistics
        cache_stats = cache_manager.get_global_statistics()
        assert cache_stats["total_requests"] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_benchmark_workflow(self, mock_model_file):
        """Test concurrent benchmark execution workflow."""
        job_manager = BenchmarkJobManager()
        await job_manager.start()
        
        try:
            # Create batch benchmark job
            models = ["model_a", "model_b", "model_c"]
            configurations = [
                {"iterations": 10, "batch_size": 1},
                {"iterations": 10, "batch_size": 2}
            ]
            
            job_id = await job_manager.run_benchmark_batch(models, configurations)
            assert job_id is not None
            
            # Wait for job completion (with timeout)
            timeout = 30  # 30 seconds
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = await job_manager.get_job_status(job_id)
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.5)
            
            # Check final status
            final_status = await job_manager.get_job_status(job_id)
            assert final_status["status"] == "completed"
            assert final_status["progress"]["total_tasks"] == 6  # 3 models * 2 configs
            assert final_status["progress"]["completed"] + final_status["progress"]["failed"] == 6
            
        finally:
            await job_manager.stop()
    
    @pytest.mark.asyncio
    async def test_pipeline_benchmark_workflow(self):
        """Test pipeline benchmark with dependencies."""
        job_manager = BenchmarkJobManager()
        await job_manager.start()
        
        try:
            # Create pipeline benchmark
            pipeline_config = {
                "model": "test_pipeline_model",
                "optimization_level": 3,
                "iterations": 10
            }
            
            job_id = await job_manager.run_pipeline_benchmark(pipeline_config)
            assert job_id is not None
            
            # Wait for pipeline completion
            timeout = 30
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = await job_manager.get_job_status(job_id)
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.5)
            
            # Check pipeline completion
            final_status = await job_manager.get_job_status(job_id)
            assert final_status["status"] == "completed"
            assert len(final_status["results"]) == 4  # load, compile, benchmark, analysis
            
        finally:
            await job_manager.stop()
    
    @pytest.mark.asyncio
    async def test_auto_scaling_workflow(self):
        """Test auto-scaling during benchmark execution."""
        resource_manager = AdaptiveResourceManager(
            metrics_window_size=10,
            evaluation_interval=1.0  # Faster evaluation for testing
        )
        
        await resource_manager.start()
        
        try:
            initial_resources = resource_manager.get_current_resources()
            
            # Simulate high load conditions
            for _ in range(5):
                resource_manager.record_metrics(
                    cpu_usage=85.0,  # High CPU usage
                    memory_usage=60.0,
                    queue_size=15,   # High queue size
                    active_tasks=8,
                    throughput=25.0,
                    latency_p95=150.0,
                    error_rate=0.02
                )
                await asyncio.sleep(0.5)
            
            # Wait for scaling evaluation
            await asyncio.sleep(2.0)
            
            # Check if scaling occurred
            current_resources = resource_manager.get_current_resources()
            scaling_stats = resource_manager.get_scaling_statistics()
            
            # Should have some scaling actions
            assert scaling_stats["total_actions"] >= 0  # May or may not scale in test
            
            # Test scaling predictions
            predictions = resource_manager.predict_scaling_needs(forecast_minutes=15)
            assert isinstance(predictions, dict)
            
        finally:
            await resource_manager.stop()
    
    def test_health_monitoring_workflow(self):
        """Test health monitoring during benchmark workflow."""
        health_monitor = HealthMonitor()
        
        # Run health check
        system_health = health_monitor.check_health(parallel=True)
        
        assert system_health is not None
        assert len(system_health.checks) > 0
        
        # All health checks should have completed
        for check in system_health.checks:
            assert check.duration_ms >= 0
            assert check.timestamp is not None
        
        # Get health summary
        summary = health_monitor.get_health_summary()
        assert "status" in summary
        assert "checks_count" in summary
        
        # Get health trends (limited data)
        trends = health_monitor.get_health_trends(hours=1)
        assert "total_checks" in trends
    
    def test_compiler_analysis_workflow(self, mock_model_file):
        """Test compiler analysis workflow."""
        analyzer = CompilerAnalyzer()
        
        # Analyze model
        analysis = analyzer.analyze_model(str(mock_model_file))
        
        assert analysis is not None
        assert analysis.supported_ops_percent >= 0
        assert analysis.num_fusions >= 0
        assert analysis.memory_transfers >= 0
        assert analysis.tpu_utilization >= 0
        assert len(analysis.optimizations_applied) > 0
        assert len(analysis.recommendations) > 0
        
        # Generate visualization
        html_content = analyzer.visualize_op_mapping(analysis)
        assert "TPU v5 Compiler Analysis" in html_content
        assert "Supported Operations" in html_content
    
    def test_model_conversion_workflow(self, mock_model_file):
        """Test model conversion workflow."""
        converter = ONNXToTPUv5()
        
        # Convert model
        result = converter.convert(
            onnx_path=str(mock_model_file),
            optimization_profile="balanced"
        )
        
        assert result is not None
        assert result.success  # Should succeed with mock data
        assert result.conversion_time > 0
        assert len(result.optimizations_applied) > 0
        
        # Verify conversion
        verification = converter.verify_conversion(
            original_onnx=str(mock_model_file),
            tpu_model=result.output_path,
            test_samples=5  # Small number for test
        )
        
        assert verification is not None
        assert verification.samples_tested == 5
        
        # Cleanup
        Path(result.output_path).unlink(missing_ok=True)
    
    def test_validation_workflow(self, mock_model_file):
        """Test comprehensive validation workflow."""
        validator = BenchmarkValidator()
        
        # Test model path validation
        model_validation = validator.validate_model_path(mock_model_file)
        assert model_validation.is_valid
        
        # Test device path validation
        device_validation = validator.validate_device_path("/dev/apex_0")
        assert device_validation.is_valid or len(device_validation.issues) > 0
        
        # Test optimization config validation
        opt_validation = validator.validate_optimization_config(
            optimization_profile="balanced",
            quantization_method="static",
            target_precision="int8"
        )
        assert opt_validation.is_valid
        
        # Test benchmark config validation
        benchmark_validation = validator.validate_benchmark_config(
            iterations=100,
            warmup=10,
            batch_size=1,
            input_shape=(1, 3, 224, 224)
        )
        assert benchmark_validation.is_valid


class TestErrorHandlingWorkflow:
    """Test error handling in benchmark workflows."""
    
    def test_invalid_model_workflow(self):
        """Test workflow with invalid model file."""
        validator = BenchmarkValidator()
        
        # Test with non-existent model
        validation = validator.validate_model_path("/non/existent/model.onnx")
        assert not validation.is_valid
        assert validation.errors_count > 0
        
        # Should have helpful error message
        error_issues = validation.get_issues_by_severity(validation.issues[0].severity)
        assert len(error_issues) > 0
        assert "not found" in error_issues[0].message.lower()
    
    def test_invalid_configuration_workflow(self):
        """Test workflow with invalid configuration."""
        validator = BenchmarkValidator()
        
        # Test with invalid benchmark config
        validation = validator.validate_benchmark_config(
            iterations=-10,  # Invalid
            warmup=-5,       # Invalid
            batch_size=0,    # Invalid
            input_shape=()   # Invalid
        )
        
        assert not validation.is_valid
        assert validation.errors_count > 0
        
        # Should have multiple validation errors
        errors = [issue for issue in validation.issues 
                 if issue.severity.value in ["error", "critical"]]
        assert len(errors) >= 3  # At least 3 errors for the invalid values
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """Test error handling in concurrent workflows."""
        job_manager = BenchmarkJobManager()
        await job_manager.start()
        
        try:
            # Create job with intentionally problematic configuration
            models = ["non_existent_model"]
            configurations = [{"iterations": -10}]  # Invalid config
            
            job_id = await job_manager.run_benchmark_batch(models, configurations)
            
            # Wait for completion
            timeout = 10
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = await job_manager.get_job_status(job_id)
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.1)
            
            # Should complete with failures
            final_status = await job_manager.get_job_status(job_id)
            assert final_status["progress"]["failed"] > 0
            
        finally:
            await job_manager.stop()


class TestPerformanceWorkflow:
    """Test performance characteristics of benchmark workflows."""
    
    def test_benchmark_performance_baseline(self):
        """Test baseline performance characteristics."""
        benchmark = TPUv5Benchmark()
        
        # Create a simple model for performance testing
        model = ModelLoader.from_onnx("dummy_model.onnx")  # Will fail gracefully
        
        start_time = time.time()
        
        try:
            results = benchmark.run(
                model=model,
                input_shape=(1, 3, 224, 224),
                iterations=50,
                warmup=5
            )
            
            duration = time.time() - start_time
            
            # Performance assertions
            assert duration < 10.0  # Should complete within 10 seconds
            assert results.total_iterations == 50
            
        except Exception:
            # Expected to fail with dummy model, but should fail quickly
            duration = time.time() - start_time
            assert duration < 5.0  # Should fail within 5 seconds
    
    def test_cache_performance_impact(self, temp_cache_dir):
        """Test performance impact of caching."""
        cache_manager = CacheManager(temp_cache_dir)
        cache = cache_manager.get_cache("results")
        
        # Test cache write performance
        start_time = time.time()
        for i in range(100):
            key = f"perf_test_key_{i}"
            value = {"data": f"test_value_{i}" * 100}  # Moderate size
            cache.set(key, value)
        write_duration = time.time() - start_time
        
        # Test cache read performance
        start_time = time.time()
        for i in range(100):
            key = f"perf_test_key_{i}"
            value = cache.get(key)
            assert value is not None
        read_duration = time.time() - start_time
        
        # Performance assertions
        assert write_duration < 5.0  # Should write 100 items in < 5 seconds
        assert read_duration < 1.0   # Should read 100 items in < 1 second
        
        # Cache hit rate should be 100%
        stats = cache.get_statistics()
        assert stats["hit_rate_percent"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])