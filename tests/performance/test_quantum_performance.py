"""Performance tests for quantum task execution system."""

import asyncio
import pytest
import time
import statistics
from concurrent.futures import as_completed
import psutil
import gc

from edge_tpu_v5_benchmark.quantum_planner import (
    QuantumTaskPlanner, QuantumTask, QuantumResource, QuantumState
)
from edge_tpu_v5_benchmark.quantum_performance import (
    OptimizedQuantumTaskPlanner, PerformanceProfile, OptimizationStrategy,
    AdaptiveCache, ResourcePoolManager
)
from edge_tpu_v5_benchmark.quantum_auto_scaling import (
    QuantumAutoScaler, QuantumNode, ScalingPolicy, LoadBalancer
)


class TestQuantumPerformanceBenchmarks:
    """Performance benchmark tests for quantum system."""
    
    @pytest.mark.asyncio
    async def test_task_execution_throughput(self):
        """Benchmark task execution throughput."""
        # Test different planner configurations
        configurations = [
            ("Basic", QuantumTaskPlanner()),
            ("Optimized Latency", OptimizedQuantumTaskPlanner(
                PerformanceProfile(strategy=OptimizationStrategy.LATENCY_FIRST)
            )),
            ("Optimized Throughput", OptimizedQuantumTaskPlanner(
                PerformanceProfile(strategy=OptimizationStrategy.THROUGHPUT_FIRST, max_concurrent_tasks=4)
            ))
        ]
        
        num_tasks = 50
        results = {}
        
        for config_name, planner in configurations:
            # Add tasks
            tasks = []
            for i in range(num_tasks):
                task = QuantumTask(
                    id=f"throughput_task_{i}",
                    name=f"Throughput Task {i}",
                    estimated_duration=0.01,  # Very fast tasks
                    priority=float(i % 5 + 1)
                )
                tasks.append(task)
                planner.add_task(task)
            
            # Measure execution time
            start_time = time.time()
            
            # Execute all tasks
            max_cycles = 100
            cycle_count = 0
            
            while planner.get_ready_tasks() and cycle_count < max_cycles:
                cycle_count += 1
                await planner.run_quantum_execution_cycle()
                
                # Minimal delay
                await asyncio.sleep(0.001)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Calculate metrics
            completed_tasks = len(planner.completed_tasks)
            throughput = completed_tasks / execution_time if execution_time > 0 else 0
            
            results[config_name] = {
                "execution_time": execution_time,
                "completed_tasks": completed_tasks,
                "throughput": throughput,
                "cycles": cycle_count
            }
            
            # Cleanup for next test
            if hasattr(planner, 'shutdown'):
                await planner.shutdown()
        
        # Print results for analysis
        print(f"\nThroughput Benchmark Results ({num_tasks} tasks):")
        for config, metrics in results.items():
            print(f"{config:20s}: {metrics['throughput']:8.2f} tasks/sec "
                  f"({metrics['completed_tasks']}/{num_tasks} in {metrics['execution_time']:.3f}s)")
        
        # Verify performance expectations
        assert results["Basic"]["throughput"] > 0
        
        # Optimized versions should generally perform better for larger loads
        if num_tasks >= 20:
            throughput_optimized = results["Optimized Throughput"]["throughput"]
            basic_throughput = results["Basic"]["throughput"]
            
            # Allow some variance, but optimized should be competitive
            assert throughput_optimized >= basic_throughput * 0.5
    
    @pytest.mark.asyncio
    async def test_execution_latency_distribution(self):
        """Test task execution latency distribution."""
        planner = OptimizedQuantumTaskPlanner(
            PerformanceProfile(strategy=OptimizationStrategy.LATENCY_FIRST)
        )
        
        num_samples = 100
        latencies = []
        
        for i in range(num_samples):
            task = QuantumTask(
                id=f"latency_task_{i}",
                name=f"Latency Task {i}",
                estimated_duration=0.01
            )
            planner.add_task(task)
            
            # Measure individual task latency
            start_time = time.time()
            
            results = await planner.run_quantum_execution_cycle()
            
            end_time = time.time()
            cycle_latency = end_time - start_time
            
            if results["tasks_executed"]:
                latencies.append(cycle_latency)
        
        # Analyze latency distribution
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            print(f"\nLatency Distribution Analysis ({len(latencies)} samples):")
            print(f"Average:  {avg_latency*1000:8.2f} ms")
            print(f"Median:   {median_latency*1000:8.2f} ms")
            print(f"P95:      {p95_latency*1000:8.2f} ms")
            print(f"Max:      {max_latency*1000:8.2f} ms")
            print(f"Min:      {min_latency*1000:8.2f} ms")
            
            # Performance assertions
            assert avg_latency < 1.0  # Should be under 1 second on average
            assert p95_latency < 2.0  # 95% should be under 2 seconds
            assert max_latency < 5.0  # No task should take more than 5 seconds
        
        await planner.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self):
        """Test memory usage scaling with task count."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Test memory scaling with different task counts
        task_counts = [10, 50, 100, 200]
        memory_usage = {}
        
        for task_count in task_counts:
            planner = OptimizedQuantumTaskPlanner()
            gc.collect()  # Force garbage collection
            
            start_memory = psutil.Process().memory_info().rss
            
            # Add tasks
            for i in range(task_count):
                task = QuantumTask(
                    id=f"memory_task_{task_count}_{i}",
                    name=f"Memory Task {i}",
                    estimated_duration=0.005,
                    complexity=1.0 + (i % 3),
                    resource_requirements={"cpu_cores": 1.0}
                )
                planner.add_task(task)
            
            peak_memory = psutil.Process().memory_info().rss
            memory_per_task = (peak_memory - start_memory) / task_count
            
            memory_usage[task_count] = {
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "memory_per_task_kb": memory_per_task / 1024
            }
            
            # Execute a few tasks to test runtime memory
            for _ in range(min(5, task_count)):
                await planner.run_quantum_execution_cycle()
                await asyncio.sleep(0.001)
            
            runtime_memory = psutil.Process().memory_info().rss
            memory_usage[task_count]["runtime_memory_mb"] = runtime_memory / (1024 * 1024)
            
            await planner.shutdown()
            del planner
            gc.collect()
        
        # Analyze memory scaling
        print(f"\nMemory Usage Scaling Analysis:")
        print(f"{'Tasks':>6s} {'Peak MB':>10s} {'KB/Task':>10s} {'Runtime MB':>12s}")
        
        for task_count, metrics in memory_usage.items():
            print(f"{task_count:6d} {metrics['peak_memory_mb']:10.2f} "
                  f"{metrics['memory_per_task_kb']:10.2f} {metrics['runtime_memory_mb']:12.2f}")
        
        # Verify reasonable memory usage
        for task_count, metrics in memory_usage.items():
            # Memory per task should be reasonable (less than 100KB per task)
            assert metrics['memory_per_task_kb'] < 100, f"Memory per task too high for {task_count} tasks"
            
            # Peak memory should not exceed reasonable limits
            assert metrics['peak_memory_mb'] < 500, f"Peak memory too high for {task_count} tasks"
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_scaling(self):
        """Test performance scaling with concurrent execution."""
        concurrency_levels = [1, 2, 4, 8]
        results = {}
        
        for max_concurrent in concurrency_levels:
            profile = PerformanceProfile(
                strategy=OptimizationStrategy.THROUGHPUT_FIRST,
                max_concurrent_tasks=max_concurrent,
                batch_size=max_concurrent
            )
            planner = OptimizedQuantumTaskPlanner(performance_profile=profile)
            
            # Add CPU-bound tasks that benefit from concurrency
            num_tasks = 20
            for i in range(num_tasks):
                task = QuantumTask(
                    id=f"concurrent_task_{max_concurrent}_{i}",
                    name=f"Concurrent Task {i}",
                    estimated_duration=0.02,  # Slightly longer tasks
                    complexity=2.0,
                    resource_requirements={"cpu_cores": 1.0}
                )
                planner.add_task(task)
            
            # Measure execution time
            start_time = time.time()
            
            while planner.get_ready_tasks():
                await planner.run_quantum_execution_cycle()
                await asyncio.sleep(0.001)
            
            execution_time = time.time() - start_time
            throughput = len(planner.completed_tasks) / execution_time
            
            results[max_concurrent] = {
                "execution_time": execution_time,
                "throughput": throughput,
                "completed_tasks": len(planner.completed_tasks)
            }
            
            await planner.shutdown()
        
        # Analyze concurrency scaling
        print(f"\nConcurrency Scaling Analysis:")
        print(f"{'Concurrency':>11s} {'Time (s)':>10s} {'Throughput':>12s} {'Speedup':>10s}")
        
        baseline_time = results[1]["execution_time"]
        
        for concurrency, metrics in results.items():
            speedup = baseline_time / metrics["execution_time"]
            print(f"{concurrency:11d} {metrics['execution_time']:10.3f} "
                  f"{metrics['throughput']:12.2f} {speedup:10.2f}x")
        
        # Verify scaling benefits
        assert results[2]["throughput"] > results[1]["throughput"] * 0.8  # Some improvement expected
        assert results[4]["execution_time"] <= results[1]["execution_time"]  # Should be faster or equal
    
    def test_cache_performance(self):
        """Test adaptive cache performance characteristics."""
        cache = AdaptiveCache(max_size_mb=10, ttl_seconds=60)
        
        # Test cache insertion performance
        num_items = 1000
        insertion_times = []
        
        for i in range(num_items):
            test_data = f"test_data_{i}" * 100  # Create some data
            
            start_time = time.time()
            cache.put(f"key_{i}", test_data, len(test_data))
            insertion_time = time.time() - start_time
            
            insertion_times.append(insertion_time)
        
        # Test cache retrieval performance
        retrieval_times = []
        hit_count = 0
        
        for i in range(num_items):
            start_time = time.time()
            result = cache.get(f"key_{i}")
            retrieval_time = time.time() - start_time
            
            retrieval_times.append(retrieval_time)
            if result is not None:
                hit_count += 1
        
        # Analyze cache performance
        avg_insertion = statistics.mean(insertion_times) * 1000  # Convert to ms
        avg_retrieval = statistics.mean(retrieval_times) * 1000
        hit_rate = hit_count / num_items
        
        cache_stats = cache.get_statistics()
        
        print(f"\nCache Performance Analysis ({num_items} items):")
        print(f"Average insertion time: {avg_insertion:.4f} ms")
        print(f"Average retrieval time:  {avg_retrieval:.4f} ms")
        print(f"Hit rate: {hit_rate:.2%}")
        print(f"Cache size: {cache_stats['total_size_mb']:.2f} MB")
        print(f"Entry count: {cache_stats['entry_count']}")
        
        # Performance assertions
        assert avg_insertion < 1.0, "Cache insertion should be under 1ms"
        assert avg_retrieval < 0.1, "Cache retrieval should be under 0.1ms"
        assert hit_rate > 0.5, "Hit rate should be reasonable after insertions"
    
    @pytest.mark.asyncio
    async def test_resource_allocation_performance(self):
        """Test resource allocation and management performance."""
        pool_manager = ResourcePoolManager()
        
        # Create resource pools
        cpu_resources = []
        for i in range(8):
            resource = QuantumResource(
                name=f"cpu_pool_{i}",
                total_capacity=4.0,
                allocation_quantum=1.0
            )
            cpu_resources.append(resource)
        
        pool_manager.add_resource_pool("cpu_cores", cpu_resources)
        
        # Test allocation performance
        num_tasks = 200
        allocation_times = []
        successful_allocations = 0
        
        tasks = []
        for i in range(num_tasks):
            task = QuantumTask(
                id=f"resource_task_{i}",
                name=f"Resource Task {i}",
                resource_requirements={"cpu_cores": float(i % 3 + 1)}
            )
            tasks.append(task)
        
        # Measure allocation performance
        start_time = time.time()
        
        allocated_resources = []
        for task in tasks:
            alloc_start = time.time()
            resources = pool_manager.allocate_optimal_resources(task)
            alloc_time = time.time() - alloc_start
            
            allocation_times.append(alloc_time)
            
            if resources:
                successful_allocations += 1
                allocated_resources.append((task, resources))
        
        total_allocation_time = time.time() - start_time
        
        # Test release performance
        release_start_time = time.time()
        
        for task, resources in allocated_resources:
            pool_manager.release_task_resources(task, resources)
        
        total_release_time = time.time() - release_start_time
        
        # Analyze performance
        avg_allocation_time = statistics.mean(allocation_times) * 1000
        allocation_rate = successful_allocations / total_allocation_time
        
        print(f"\nResource Pool Performance Analysis:")
        print(f"Tasks processed: {num_tasks}")
        print(f"Successful allocations: {successful_allocations}")
        print(f"Average allocation time: {avg_allocation_time:.4f} ms")
        print(f"Allocation rate: {allocation_rate:.2f} allocations/sec")
        print(f"Total release time: {total_release_time*1000:.2f} ms")
        
        # Performance assertions
        assert avg_allocation_time < 10.0, "Resource allocation should be under 10ms"
        assert allocation_rate > 10.0, "Should handle at least 10 allocations per second"
        assert successful_allocations > num_tasks * 0.3, "Should successfully allocate for some tasks"
    
    @pytest.mark.asyncio
    async def test_auto_scaling_performance(self):
        """Test auto-scaling system performance."""
        # Create auto-scaler with aggressive scaling policy
        scaling_policy = ScalingPolicy(
            scale_up_cooldown=0.1,
            scale_down_cooldown=0.1,
            scale_up_queue_threshold=5,
            scale_down_queue_threshold=2
        )
        
        auto_scaler = QuantumAutoScaler(scaling_policy)
        
        # Add initial nodes
        for i in range(2):
            node = QuantumNode(
                node_id=f"perf_node_{i}",
                planner=OptimizedQuantumTaskPlanner(),
                max_capacity=10
            )
            auto_scaler.add_node(node)
        
        # Simulate load and measure scaling response time
        scaling_decision_times = []
        
        for load_cycle in range(10):
            # Generate varying load
            num_tasks = (load_cycle % 4 + 1) * 10
            
            tasks = []
            for i in range(num_tasks):
                task = QuantumTask(
                    id=f"scale_task_{load_cycle}_{i}",
                    name=f"Scale Task {i}",
                    estimated_duration=0.01
                )
                tasks.append(task)
            
            # Assign tasks to cluster
            assignment_start = time.time()
            
            assigned_count = 0
            for task in tasks:
                node = await auto_scaler.assign_task_to_cluster(task)
                if node:
                    assigned_count += 1
            
            assignment_time = time.time() - assignment_start
            
            # Measure scaling decision time
            decision_start = time.time()
            
            metrics = auto_scaler.collect_scaling_metrics()
            scaling_decision = auto_scaler.make_scaling_decision(metrics)
            
            decision_time = time.time() - decision_start
            scaling_decision_times.append(decision_time)
            
            # Execute scaling if needed
            if scaling_decision != auto_scaler.policy.enabled:  # Placeholder comparison
                await auto_scaler.execute_scaling_action(scaling_decision)
            
            await asyncio.sleep(0.01)  # Brief pause between cycles
        
        # Analyze scaling performance
        avg_decision_time = statistics.mean(scaling_decision_times) * 1000
        
        cluster_status = auto_scaler.get_cluster_status()
        
        print(f"\nAuto-scaling Performance Analysis:")
        print(f"Average scaling decision time: {avg_decision_time:.4f} ms")
        print(f"Final cluster size: {cluster_status['cluster_info']['total_nodes']}")
        print(f"Total capacity: {cluster_status['cluster_info']['total_capacity']}")
        print(f"Load factor: {cluster_status['cluster_info']['load_factor']:.2%}")
        
        # Performance assertions
        assert avg_decision_time < 50.0, "Scaling decisions should be under 50ms"
        assert cluster_status['cluster_info']['total_nodes'] >= 2, "Should maintain minimum nodes"
    
    @pytest.mark.asyncio
    async def test_quantum_optimization_performance(self):
        """Test quantum annealing optimization performance."""
        planner = QuantumTaskPlanner()
        
        # Create complex optimization scenario
        num_tasks = 50
        tasks = []
        
        # Create tasks with complex dependency chains
        for i in range(num_tasks):
            dependencies = set()
            if i > 0:
                # Add random dependencies to create optimization challenges
                for j in range(min(3, i)):
                    if (i + j) % 7 == 0:  # Pseudo-random pattern
                        dependencies.add(f"opt_task_{j}")
            
            task = QuantumTask(
                id=f"opt_task_{i}",
                name=f"Optimization Task {i}",
                priority=float((i * 7) % 10 + 1),  # Varied priorities
                complexity=float((i * 3) % 5 + 1),  # Varied complexity
                estimated_duration=float((i % 3 + 1) * 0.01),
                dependencies=dependencies,
                resource_requirements={"cpu_cores": float(i % 3 + 1)}
            )
            tasks.append(task)
            planner.add_task(task)
        
        # Measure optimization performance
        optimization_times = []
        
        for iteration in range(5):  # Multiple runs for averaging
            opt_start = time.time()
            
            optimized_schedule = planner.optimize_schedule()
            
            opt_time = time.time() - opt_start
            optimization_times.append(opt_time)
            
            # Reset task states for next iteration
            for task in planner.tasks.values():
                if task.state == QuantumState.COLLAPSED:
                    task.state = QuantumState.SUPERPOSITION
                    task.probability_amplitude = 1.0 + 0j
        
        # Analyze optimization performance
        avg_opt_time = statistics.mean(optimization_times)
        tasks_per_second = num_tasks / avg_opt_time
        
        print(f"\nQuantum Optimization Performance Analysis:")
        print(f"Tasks optimized: {num_tasks}")
        print(f"Average optimization time: {avg_opt_time:.4f} s")
        print(f"Optimization rate: {tasks_per_second:.2f} tasks/sec")
        print(f"Optimization consistency: {statistics.stdev(optimization_times):.4f} s std dev")
        
        # Performance assertions
        assert avg_opt_time < 5.0, f"Optimization should complete within 5 seconds for {num_tasks} tasks"
        assert tasks_per_second > 5.0, "Should optimize at least 5 tasks per second"
        
        # Verify optimization quality
        final_schedule = planner.optimize_schedule()
        assert len(final_schedule) > 0, "Optimization should produce a schedule"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])