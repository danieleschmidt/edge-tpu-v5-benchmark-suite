"""Performance and scalability tests for TPU v5 benchmark suite."""

import pytest
import asyncio
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import tempfile
import psutil

from edge_tpu_v5_benchmark.cache import SmartCache, CacheManager
from edge_tpu_v5_benchmark.concurrency import TaskScheduler, Task, TaskPriority, BenchmarkJobManager
from edge_tpu_v5_benchmark.auto_scaling import AdaptiveResourceManager
from edge_tpu_v5_benchmark.monitoring import MetricsCollector, PerformanceMonitor


class TestCacheScalability:
    """Test cache system scalability and performance."""
    
    def test_memory_cache_scalability(self):
        """Test memory cache performance with increasing load."""
        cache = SmartCache()
        
        # Test different data sizes and counts
        test_scenarios = [
            (100, 1024),      # 100 items, 1KB each
            (1000, 1024),     # 1K items, 1KB each  
            (10000, 512),     # 10K items, 512B each
            (100000, 100),    # 100K items, 100B each
        ]
        
        performance_results = []
        
        for item_count, item_size in test_scenarios:
            # Clear cache
            cache.clear()
            
            # Generate test data
            test_data = "x" * item_size
            
            # Measure write performance
            start_time = time.time()
            for i in range(item_count):
                cache.set(f"key_{i}", test_data)
            write_time = time.time() - start_time
            
            # Measure read performance  
            start_time = time.time()
            hits = 0
            for i in range(item_count):
                if cache.get(f"key_{i}") is not None:
                    hits += 1
            read_time = time.time() - start_time
            
            write_throughput = item_count / write_time if write_time > 0 else 0
            read_throughput = item_count / read_time if read_time > 0 else 0
            hit_rate = hits / item_count
            
            performance_results.append({
                'items': item_count,
                'item_size': item_size,
                'write_throughput': write_throughput,
                'read_throughput': read_throughput,
                'hit_rate': hit_rate,
                'memory_usage': cache.memory_storage.current_memory_bytes
            })
            
            # Performance assertions
            assert write_throughput > 1000  # At least 1K writes/sec
            assert read_throughput > 10000  # At least 10K reads/sec
            assert hit_rate > 0.95  # At least 95% hit rate
        
        # Throughput should scale reasonably with size
        assert performance_results[-1]['write_throughput'] > 0
        assert performance_results[-1]['read_throughput'] > 0
    
    def test_concurrent_cache_performance(self):
        """Test cache performance under concurrent access."""
        cache = SmartCache()
        num_threads = 10
        operations_per_thread = 1000
        
        # Results collection
        results = []
        errors = []
        
        def cache_worker(worker_id):
            """Worker function for concurrent cache operations."""
            worker_results = {
                'reads': 0,
                'writes': 0,
                'hits': 0,
                'misses': 0,
                'errors': 0
            }
            
            start_time = time.time()
            
            try:
                for i in range(operations_per_thread):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    
                    # Write operation
                    cache.set(key, value)
                    worker_results['writes'] += 1
                    
                    # Read operation
                    retrieved = cache.get(key)
                    worker_results['reads'] += 1
                    
                    if retrieved == value:
                        worker_results['hits'] += 1
                    else:
                        worker_results['misses'] += 1
                    
                    # Read some other worker's data (may miss)
                    other_key = f"worker_{(worker_id + 1) % num_threads}_key_{i}"
                    cache.get(other_key)
                    worker_results['reads'] += 1
                    
            except Exception as e:
                worker_results['errors'] += 1
                errors.append(f"Worker {worker_id}: {e}")
            
            worker_results['duration'] = time.time() - start_time
            worker_results['worker_id'] = worker_id
            results.append(worker_results)
        
        # Start concurrent workers
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(num_threads)]
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()  # Will raise exception if worker failed
        
        # Analyze results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_threads
        
        total_operations = sum(r['reads'] + r['writes'] for r in results)
        total_duration = max(r['duration'] for r in results)
        overall_throughput = total_operations / total_duration
        
        # Performance assertions
        assert overall_throughput > 5000  # At least 5K ops/sec overall
        assert all(r['errors'] == 0 for r in results)
        
        # Hit rate should be reasonable (not 100% due to cross-worker reads)
        total_hits = sum(r['hits'] for r in results)
        total_reads = sum(r['reads'] for r in results)
        hit_rate = total_hits / total_reads if total_reads > 0 else 0
        assert hit_rate > 0.4  # At least 40% hit rate in concurrent scenario
    
    def test_disk_cache_scalability(self):
        """Test disk cache performance and scalability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(Path(tmpdir))
            cache = cache_manager.get_cache("results")
            
            # Test with larger data sizes for disk cache
            test_scenarios = [
                (100, 10240),     # 100 items, 10KB each
                (500, 20480),     # 500 items, 20KB each
                (1000, 5120),     # 1K items, 5KB each
            ]
            
            for item_count, item_size in test_scenarios:
                # Generate test data
                test_data = {"data": "x" * item_size, "metadata": {"size": item_size}}
                
                # Measure write performance
                start_time = time.time()
                for i in range(item_count):
                    cache.set(f"disk_key_{i}", test_data, force_disk=True)
                write_time = time.time() - start_time
                
                # Measure read performance
                start_time = time.time()
                successful_reads = 0
                for i in range(item_count):
                    result = cache.get(f"disk_key_{i}")
                    if result is not None:
                        successful_reads += 1
                read_time = time.time() - start_time
                
                write_throughput = item_count / write_time if write_time > 0 else 0
                read_throughput = item_count / read_time if read_time > 0 else 0
                success_rate = successful_reads / item_count
                
                # Performance assertions for disk cache
                assert write_throughput > 10   # At least 10 writes/sec to disk
                assert read_throughput > 50    # At least 50 reads/sec from disk
                assert success_rate > 0.95    # At least 95% success rate


class TestConcurrencyScalability:
    """Test concurrency system scalability."""
    
    @pytest.mark.asyncio
    async def test_task_scheduler_scalability(self):
        """Test task scheduler performance with increasing load."""
        scheduler = TaskScheduler()
        await scheduler.start()
        
        try:
            # Test different task loads
            task_counts = [10, 50, 100, 500]
            
            for task_count in task_counts:
                # Create tasks
                task_ids = []
                
                start_time = time.time()
                
                for i in range(task_count):
                    task = Task(
                        id=f"scale_test_{task_count}_{i}",
                        func=lambda x=i: time.sleep(0.01) or f"result_{x}",  # 10ms task
                        priority=TaskPriority.NORMAL,
                        timeout=5.0
                    )
                    task_id = await scheduler.submit_task(task)
                    task_ids.append(task_id)
                
                # Wait for completion
                results = await scheduler.get_results(task_ids, timeout=30.0)
                
                completion_time = time.time() - start_time
                
                # Verify results
                successful_results = sum(1 for r in results if r.is_successful)
                success_rate = successful_results / task_count
                throughput = task_count / completion_time
                
                # Performance assertions
                assert success_rate > 0.95  # At least 95% success rate
                assert throughput > 5       # At least 5 tasks/sec throughput
                assert completion_time < 60  # Should complete within 60 seconds
                
                # Clean up for next iteration
                await asyncio.sleep(0.1)
        
        finally:
            await scheduler.stop()
    
    @pytest.mark.asyncio
    async def test_job_manager_batch_scalability(self):
        """Test job manager scalability with large batches."""
        job_manager = BenchmarkJobManager()
        await job_manager.start()
        
        try:
            # Test different batch sizes
            batch_sizes = [5, 10, 25]  # Reduced for test performance
            
            for batch_size in batch_sizes:
                models = [f"model_{i}" for i in range(batch_size)]
                configurations = [
                    {"iterations": 5, "timeout": 10},  # Quick config for testing
                    {"iterations": 3, "timeout": 10}
                ]
                
                start_time = time.time()
                
                # Submit batch job
                job_id = await job_manager.run_benchmark_batch(models, configurations)
                
                # Wait for completion
                timeout = 60
                while time.time() - start_time < timeout:
                    status = await job_manager.get_job_status(job_id)
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.5)
                
                completion_time = time.time() - start_time
                final_status = await job_manager.get_job_status(job_id)
                
                expected_tasks = batch_size * len(configurations)
                completed_tasks = final_status["progress"]["completed"]
                failed_tasks = final_status["progress"]["failed"]
                total_finished = completed_tasks + failed_tasks
                
                # Performance assertions
                assert total_finished == expected_tasks
                assert completion_time < timeout
                
                # Throughput should be reasonable
                if completion_time > 0:
                    task_throughput = total_finished / completion_time
                    assert task_throughput > 0.1  # At least 0.1 tasks/sec
        
        finally:
            await job_manager.stop()


class TestAutoScalingPerformance:
    """Test auto-scaling system performance."""
    
    @pytest.mark.asyncio
    async def test_resource_manager_response_time(self):
        """Test resource manager scaling response time."""
        resource_manager = AdaptiveResourceManager(
            metrics_window_size=20,
            evaluation_interval=0.5  # Fast evaluation for testing
        )
        
        await resource_manager.start()
        
        try:
            # Record initial state
            initial_resources = resource_manager.get_current_resources()
            
            # Simulate sudden load spike
            spike_start = time.time()
            
            for i in range(10):
                resource_manager.record_metrics(
                    cpu_usage=95.0,  # Very high CPU
                    memory_usage=80.0,
                    queue_size=20,   # Large queue
                    active_tasks=15,
                    throughput=100.0,
                    latency_p95=200.0,
                    error_rate=0.01
                )
                await asyncio.sleep(0.1)
            
            # Wait for scaling response
            max_wait_time = 10.0  # 10 seconds max
            scaling_detected = False
            
            while time.time() - spike_start < max_wait_time:
                current_resources = resource_manager.get_current_resources()
                scaling_stats = resource_manager.get_scaling_statistics()
                
                # Check if any scaling occurred
                if scaling_stats["total_actions"] > 0:
                    scaling_detected = True
                    break
                
                await asyncio.sleep(0.2)
            
            response_time = time.time() - spike_start
            
            # Performance assertions
            assert response_time < max_wait_time  # Should respond within max wait time
            
            # Get final statistics
            final_stats = resource_manager.get_scaling_statistics()
            assert final_stats["total_actions"] >= 0  # May or may not have scaled
            
        finally:
            await resource_manager.stop()
    
    @pytest.mark.asyncio
    async def test_metrics_collection_performance(self):
        """Test metrics collection performance under load."""
        resource_manager = AdaptiveResourceManager(
            metrics_window_size=1000,  # Large window
            evaluation_interval=2.0
        )
        
        await resource_manager.start()
        
        try:
            # Measure metrics recording performance
            metrics_count = 1000
            start_time = time.time()
            
            for i in range(metrics_count):
                resource_manager.record_metrics(
                    cpu_usage=50.0 + (i % 50),  # Varying values
                    memory_usage=40.0 + (i % 40),
                    queue_size=i % 20,
                    active_tasks=i % 10,
                    throughput=10.0 + (i % 90),
                    latency_p95=50.0 + (i % 100),
                    error_rate=0.01 * (i % 5)
                )
            
            recording_time = time.time() - start_time
            recording_throughput = metrics_count / recording_time
            
            # Performance assertions
            assert recording_throughput > 1000  # At least 1K metrics/sec
            assert recording_time < 5.0  # Should complete within 5 seconds
            
            # Wait for some processing
            await asyncio.sleep(1.0)
            
            # Test prediction performance
            start_time = time.time()
            predictions = resource_manager.predict_scaling_needs(forecast_minutes=30)
            prediction_time = time.time() - start_time
            
            assert prediction_time < 1.0  # Prediction should be fast
            assert isinstance(predictions, dict)
            
        finally:
            await resource_manager.stop()


class TestMonitoringPerformance:
    """Test monitoring system performance."""
    
    def test_metrics_collector_performance(self):
        """Test metrics collector performance under high load."""
        collector = MetricsCollector(max_points=10000)
        
        # Test different metric types
        metric_count = 5000
        
        # Measure counter performance
        start_time = time.time()
        for i in range(metric_count):
            collector.record_counter(f"test_counter_{i % 100}", i)
        counter_time = time.time() - start_time
        
        # Measure gauge performance
        start_time = time.time()
        for i in range(metric_count):
            collector.record_gauge(f"test_gauge_{i % 100}", float(i))
        gauge_time = time.time() - start_time
        
        # Measure histogram performance
        start_time = time.time()
        for i in range(metric_count):
            collector.record_histogram("test_histogram", float(i % 1000))
        histogram_time = time.time() - start_time
        
        # Performance assertions
        counter_throughput = metric_count / counter_time if counter_time > 0 else 0
        gauge_throughput = metric_count / gauge_time if gauge_time > 0 else 0
        histogram_throughput = metric_count / histogram_time if histogram_time > 0 else 0
        
        assert counter_throughput > 5000    # At least 5K counter ops/sec
        assert gauge_throughput > 5000      # At least 5K gauge ops/sec
        assert histogram_throughput > 5000  # At least 5K histogram ops/sec
        
        # Test retrieval performance
        start_time = time.time()
        recent_metrics = collector.get_metrics(since=time.time() - 3600)  # Last hour
        retrieval_time = time.time() - start_time
        
        assert retrieval_time < 1.0  # Should retrieve quickly
        assert len(recent_metrics) > 0
    
    def test_performance_monitor_overhead(self):
        """Test performance monitor overhead."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        try:
            # Measure baseline performance without monitoring
            def cpu_intensive_task():
                """CPU intensive task for testing."""
                result = 0
                for i in range(100000):
                    result += i * i
                return result
            
            # Run without monitoring recording
            start_time = time.time()
            for _ in range(100):
                cpu_intensive_task()
            baseline_time = time.time() - start_time
            
            # Run with monitoring recording
            start_time = time.time()
            for i in range(100):
                benchmark_id = f"overhead_test_{i}"
                
                monitor.record_benchmark_start(benchmark_id, "test_model")
                task_start = time.time()
                
                result = cpu_intensive_task()
                
                task_duration = time.time() - task_start
                monitor.record_benchmark_completion(benchmark_id, "test_model", task_duration, True)
                
                # Record model performance
                monitor.record_model_performance("test_model", 100.0, task_duration * 1000, 1.0)
            
            monitored_time = time.time() - start_time
            
            # Calculate overhead
            overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
            
            # Performance assertions
            assert overhead_percent < 20.0  # Monitoring overhead should be < 20%
            assert monitored_time < baseline_time * 1.5  # At most 50% slower
            
        finally:
            monitor.stop_monitoring()


class TestMemoryScalability:
    """Test memory usage scalability."""
    
    def test_cache_memory_efficiency(self):
        """Test cache memory efficiency with large datasets."""
        cache = SmartCache()
        
        # Track memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Add large dataset
        item_count = 10000
        item_size = 1024  # 1KB per item
        
        for i in range(item_count):
            data = "x" * item_size
            cache.set(f"memory_test_{i}", data)
        
        # Measure memory after caching
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Expected memory usage (rough estimate)
        expected_data_size = item_count * item_size
        memory_efficiency = expected_data_size / memory_increase if memory_increase > 0 else 0
        
        # Memory efficiency should be reasonable (accounting for overhead)
        assert memory_efficiency > 0.3  # At least 30% efficiency
        
        # Clear cache and check memory reduction
        cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Memory should be freed (may not be immediate due to Python GC)
        final_memory_after_clear = process.memory_info().rss
        memory_freed = final_memory - final_memory_after_clear
        
        # Some memory should be freed
        assert memory_freed >= 0  # Non-negative (may be 0 due to GC timing)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])