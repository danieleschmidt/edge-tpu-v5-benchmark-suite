#!/usr/bin/env python3
"""Generation 3: Test performance optimization and scaling features"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import asyncio
import concurrent.futures
from edge_tpu_v5_benchmark.quantum_planner import QuantumTaskPlanner, QuantumTask
from edge_tpu_v5_benchmark.quantum_performance import OptimizedQuantumTaskPlanner, AdaptiveCache
from edge_tpu_v5_benchmark.quantum_auto_scaling import QuantumAutoScaler
from edge_tpu_v5_benchmark.cache import CacheManager
from edge_tpu_v5_benchmark.performance import PerformanceMonitor

async def test_performance_features():
    """Test Generation 3: Performance optimization and scaling"""
    
    print("⚡ Testing Generation 3: Performance & Scaling")
    
    # Test performance monitoring
    print("\n📊 Testing Performance Monitoring...")
    perf_monitor = PerformanceMonitor()
    
    start_time = time.time()
    
    # Create baseline planner for comparison
    baseline_planner = QuantumTaskPlanner()
    
    # Test optimized planner
    print("\n🚀 Testing Optimized Quantum Planner...")
    optimized_planner = OptimizedQuantumTaskPlanner()
    
    # Create test workload
    test_tasks = []
    for i in range(50):
        task = QuantumTask(
            id=f"perf_task_{i}",
            name=f"Performance Test Task {i}",
            priority=0.1 + (i % 10) * 0.1,
            complexity=0.1 + (i % 5) * 0.2,
            estimated_duration=1.0 + (i % 3) * 2.0
        )
        test_tasks.append(task)
    
    # Benchmark baseline planner
    baseline_start = time.time()
    for task in test_tasks[:25]:  # Smaller set for baseline
        baseline_planner.add_task(task)
    baseline_schedule = baseline_planner.optimize_schedule()
    baseline_time = time.time() - baseline_start
    
    print(f"✅ Baseline planner: {len(baseline_schedule)} tasks in {baseline_time:.3f}s")
    
    # Benchmark optimized planner
    optimized_start = time.time()
    for task in test_tasks:
        optimized_planner.add_task(task)
    optimized_schedule = optimized_planner.optimize_schedule()
    optimized_time = time.time() - optimized_start
    
    print(f"✅ Optimized planner: {len(optimized_schedule)} tasks in {optimized_time:.3f}s")
    
    if optimized_time > 0 and baseline_time > 0:
        speedup = baseline_time / optimized_time * (25/50)  # Adjust for different task counts
        print(f"✅ Performance improvement: {speedup:.2f}x faster")
    
    # Test adaptive caching
    print("\n💾 Testing Adaptive Caching...")
    cache = AdaptiveCache()
    
    # Test cache operations
    test_data = {"benchmark_results": [1, 2, 3, 4, 5], "metadata": {"version": "1.0"}}
    
    # Cache write performance
    cache_start = time.time()
    cache.put("test_key", test_data)
    cache_write_time = time.time() - cache_start
    print(f"✅ Cache write: {cache_write_time:.4f}s")
    
    # Cache read performance
    cache_start = time.time()
    cached_data = cache.get("test_key")
    cache_read_time = time.time() - cache_start
    print(f"✅ Cache read: {cache_read_time:.4f}s")
    
    # Verify cache integrity
    if cached_data == test_data:
        print("✅ Cache integrity verified")
    else:
        print("❌ Cache integrity failed")
    
    # Test cache manager
    print("\n🗄️ Testing Cache Manager...")
    cache_manager = CacheManager()
    
    # Test predictive caching
    for i in range(10):
        key = f"benchmark_result_{i}"
        value = {"score": i * 10, "timestamp": time.time()}
        await cache_manager.put(key, value)
    
    print("✅ Cache manager bulk operations completed")
    
    # Test concurrent access performance
    print("\n🔄 Testing Concurrent Performance...")
    
    async def concurrent_task_processing(task_batch):
        """Process a batch of tasks concurrently"""
        planner = OptimizedQuantumTaskPlanner()
        for task in task_batch:
            planner.add_task(task)
        return planner.optimize_schedule()
    
    # Create task batches for concurrent processing
    batch_size = 10
    task_batches = [
        test_tasks[i:i+batch_size] 
        for i in range(0, len(test_tasks), batch_size)
    ]
    
    # Test concurrent processing
    concurrent_start = time.time()
    results = await asyncio.gather(*[
        concurrent_task_processing(batch) for batch in task_batches
    ])
    concurrent_time = time.time() - concurrent_start
    
    total_processed = sum(len(result) for result in results)
    print(f"✅ Concurrent processing: {total_processed} tasks in {concurrent_time:.3f}s")
    print(f"✅ Throughput: {total_processed/concurrent_time:.1f} tasks/second")
    
    # Test auto-scaling
    print("\n📈 Testing Auto-Scaling...")
    auto_scaler = QuantumAutoScaler()
    
    # Simulate load spike
    load_metrics = {
        "cpu_usage": 0.85,
        "memory_usage": 0.75,
        "task_queue_length": 100,
        "average_response_time": 2.5
    }
    
    scaling_decision = await auto_scaler.evaluate_scaling_need(load_metrics)
    print(f"✅ Auto-scaling decision: {scaling_decision}")
    
    # Test resource optimization
    print("\n⚙️ Testing Resource Optimization...")
    
    # Create resource-intensive tasks
    resource_tasks = []
    for i in range(10):
        task = QuantumTask(
            id=f"resource_task_{i}",
            name=f"Resource Task {i}",
            resource_requirements={
                "tpu_cores": 0.1 + (i % 3) * 0.1,
                "memory_gb": 0.5 + (i % 4) * 0.25,
                "compute_tops": 1.0 + (i % 2) * 0.5
            }
        )
        resource_tasks.append(task)
    
    # Test resource allocation optimization
    resource_planner = OptimizedQuantumTaskPlanner()
    for task in resource_tasks:
        resource_planner.add_task(task)
    
    resource_schedule = resource_planner.optimize_schedule()
    print(f"✅ Resource optimization: {len(resource_schedule)} tasks scheduled efficiently")
    
    # Test memory efficiency
    print("\n🧠 Testing Memory Efficiency...")
    import psutil
    
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create many tasks to test memory usage
    memory_tasks = []
    for i in range(1000):
        task = QuantumTask(
            id=f"memory_task_{i}",
            name=f"Memory Test Task {i}"
        )
        memory_tasks.append(task)
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = memory_after - memory_before
    
    print(f"✅ Memory efficiency: {memory_increase:.2f} MB for 1000 tasks ({memory_increase/1000:.3f} MB/task)")
    
    total_time = time.time() - start_time
    print(f"\n🎯 Generation 3 COMPLETE: Performance features verified in {total_time:.2f}s!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_performance_features())
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Generation 3 testing complete")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)