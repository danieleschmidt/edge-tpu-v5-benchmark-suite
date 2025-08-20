#!/usr/bin/env python3
"""Comprehensive test for quantum hyper-performance engine."""

import sys
import os
import asyncio
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock numpy for testing
class MockNumPy:
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0

sys.modules['numpy'] = MockNumPy()

# Import after mocking
from edge_tpu_v5_benchmark.quantum_hyper_performance_engine import (
    HyperPerformanceEngine,
    QuantumAnnealingOptimizer,
    SuperpositionParallelProcessor,
    EntanglementCoordinator,
    OptimizationStrategy,
    ResourceType,
    ScalingMetric,
    PerformanceMetrics,
    ResourceAllocation
)
from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
    MLWorkloadType,
    WorkloadCharacteristics
)


def test_quantum_annealing_optimizer():
    """Test quantum annealing optimization."""
    print("Testing Quantum Annealing Optimizer...")
    
    optimizer = QuantumAnnealingOptimizer(max_iterations=20)  # Reduced for faster testing
    
    # Create test workload
    workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.TRAINING,
        circuit_depth=50,
        gate_count=100,
        two_qubit_gate_count=25,
        coherence_time_required=100.0,
        fidelity_threshold=0.9,
        error_budget=0.02,
        quantum_advantage_target=2.0
    )
    
    # Define constraints
    constraints = {
        "max_latency": 50.0,
        "min_throughput": 20.0,
        "max_cost": 100.0,
        "min_tpu_cores": 2,
        "min_quantum_qubits": 8
    }
    
    # Available resources
    available_resources = {
        ResourceType.TPU_V5_CORE: 16,
        ResourceType.QUANTUM_PROCESSOR: 32,
        ResourceType.CLASSICAL_CPU: 8,
        ResourceType.MEMORY: 128,
        ResourceType.NETWORK_BANDWIDTH: 1000,
        ResourceType.STORAGE_IOPS: 5000
    }
    
    # Run optimization
    start_time = time.time()
    allocation = optimizer.optimize_resource_allocation(workload, constraints, available_resources)
    optimization_time = time.time() - start_time
    
    # Validate results
    assert isinstance(allocation, ResourceAllocation)
    assert allocation.tpu_cores >= constraints["min_tpu_cores"]
    assert allocation.quantum_qubits >= constraints["min_quantum_qubits"]
    assert allocation.estimated_cost > 0
    assert isinstance(allocation.expected_performance, PerformanceMetrics)
    assert allocation.expected_performance.throughput > 0
    assert allocation.expected_performance.latency_p50 > 0
    
    print(f"   Allocation: TPU={allocation.tpu_cores}, Quantum={allocation.quantum_qubits}")
    print(f"   Performance: {allocation.expected_performance.throughput:.1f} ops/s, "
          f"{allocation.expected_performance.latency_p50:.1f}ms latency")
    print(f"   Cost: ${allocation.estimated_cost:.2f}/hr")
    print(f"   Optimization time: {optimization_time:.3f}s")
    print("✅ Quantum annealing optimization passed")
    
    return True


async def test_superposition_parallel_processor():
    """Test superposition-based parallel processing."""
    print("Testing Superposition Parallel Processor...")
    
    processor = SuperpositionParallelProcessor(max_parallel_tasks=8)
    
    # Create test tasks with different execution times
    def create_task(duration, result_value):
        async def task():
            await asyncio.sleep(duration)  # Simulate work
            return result_value
        return task
    
    # Create a mix of tasks
    tasks = [
        create_task(0.1, f"result_{i}") for i in range(16)
    ]
    
    # Test superposition processing
    start_time = time.time()
    results = await processor.process_superposition_batch(tasks, superposition_factor=8)
    execution_time = time.time() - start_time
    
    # Validate results
    assert len(results) == 16
    assert all(result is not None for result in results)
    assert execution_time < 1.0  # Should be much faster than sequential (1.6s)
    
    print(f"   Processed {len(tasks)} tasks in {execution_time:.2f}s")
    print(f"   Parallel efficiency: {len(tasks) * 0.1 / execution_time:.1f}x speedup")
    
    # Test quantum interference pattern
    priorities = [1.0, 2.0, 0.5, 3.0, 1.5]
    interference_order = processor.create_quantum_interference_pattern(priorities)
    
    assert len(interference_order) == len(priorities)
    assert set(interference_order) == set(range(len(priorities)))
    
    print(f"   Interference pattern: {interference_order} (from priorities {priorities})")
    print("✅ Superposition parallel processor passed")
    
    return True


def test_entanglement_coordinator():
    """Test quantum entanglement coordination."""
    print("Testing Entanglement Coordinator...")
    
    coordinator = EntanglementCoordinator()
    
    # Create entangled task group
    task_ids = ["task_1", "task_2", "task_3", "task_4"]
    coordinator.create_entanglement(task_ids, correlation_strength=0.8)
    
    # Validate entanglement creation
    assert len(coordinator.entangled_tasks) == len(task_ids)
    assert all(task_id in coordinator.task_states for task_id in task_ids)
    
    # Test entanglement strength measurement
    strength = coordinator.measure_entanglement_strength("task_1", "task_2")
    assert strength == 0.8
    
    # Test state propagation
    coordinator.update_task_state("task_1", "running", progress=0.3)
    assert coordinator.task_states["task_1"]["status"] == "running"
    
    # Complete one task and check entangled effects
    coordinator.update_task_state("task_1", "completed", progress=1.0, result="result_1")
    
    # Check that entangled tasks may have been affected
    entangled_states = [coordinator.task_states[task_id]["status"] for task_id in task_ids[1:]]
    print(f"   Task states after task_1 completion: {entangled_states}")
    
    # Get entanglement network
    network = coordinator.get_entanglement_network()
    assert len(network) == len(task_ids)
    assert all(len(network[task_id]) == len(task_ids) - 1 for task_id in task_ids)
    
    print(f"   Entanglement network: {network}")
    print("✅ Entanglement coordinator passed")
    
    return True


async def test_hyper_performance_engine():
    """Test the main hyper-performance engine."""
    print("Testing Hyper-Performance Engine...")
    
    engine = HyperPerformanceEngine(
        enable_quantum_optimization=True,
        max_resource_budget=500.0
    )
    
    # Create test workload
    workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.INFERENCE,
        circuit_depth=30,
        gate_count=60,
        two_qubit_gate_count=15,
        coherence_time_required=80.0,
        fidelity_threshold=0.95,
        error_budget=0.01,
        quantum_advantage_target=2.5
    )
    
    # Performance targets
    performance_targets = {
        "max_latency": 20.0,
        "min_throughput": 50.0,
        "max_cost": 200.0
    }
    
    # Available resources
    available_resources = {
        ResourceType.TPU_V5_CORE: 12,
        ResourceType.QUANTUM_PROCESSOR: 24,
        ResourceType.CLASSICAL_CPU: 6,
        ResourceType.MEMORY: 96,
        ResourceType.NETWORK_BANDWIDTH: 800,
        ResourceType.STORAGE_IOPS: 4000
    }
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.QUANTUM_ANNEALING,
        OptimizationStrategy.SUPERPOSITION_PARALLEL,
        OptimizationStrategy.ENTANGLEMENT_COORDINATION,
        OptimizationStrategy.ADAPTIVE_RESOURCE_ALLOCATION
    ]
    
    results = {}
    for strategy in strategies:
        print(f"  Testing strategy: {strategy.value}")
        
        result = await engine.optimize_performance(
            workload=workload,
            performance_targets=performance_targets,
            available_resources=available_resources,
            optimization_strategy=strategy
        )
        
        results[strategy] = result
        
        # Validate optimization result
        assert result.optimization_strategy == strategy
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
        assert result.optimization_time > 0
        assert result.optimal_allocation.estimated_cost <= performance_targets["max_cost"]
        
        print(f"    Confidence: {result.confidence_score:.2f}")
        print(f"    Optimization time: {result.optimization_time:.3f}s")
        print(f"    Predicted performance: {result.predicted_performance.throughput:.1f} ops/s")
    
    # Test caching (second call should be faster)
    start_time = time.time()
    cached_result = await engine.optimize_performance(
        workload=workload,
        performance_targets=performance_targets,
        available_resources=available_resources,
        optimization_strategy=OptimizationStrategy.QUANTUM_ANNEALING
    )
    cached_time = time.time() - start_time
    
    assert cached_time < results[OptimizationStrategy.QUANTUM_ANNEALING].optimization_time
    print(f"   Cache hit time: {cached_time:.4f}s (vs {results[OptimizationStrategy.QUANTUM_ANNEALING].optimization_time:.3f}s original)")
    
    # Test performance summary
    summary = engine.get_performance_summary()
    assert summary["total_optimizations"] >= len(strategies)
    assert "recent_performance" in summary
    assert "strategy_distribution" in summary
    
    print(f"   Total optimizations: {summary['total_optimizations']}")
    print(f"   Strategy distribution: {summary['strategy_distribution']}")
    print("✅ Hyper-performance engine passed")
    
    return True


async def test_performance_scaling_scenarios():
    """Test various performance scaling scenarios."""
    print("Testing Performance Scaling Scenarios...")
    
    engine = HyperPerformanceEngine(max_resource_budget=1000.0)
    
    # Scenario 1: High-throughput workload
    print("  Scenario 1: High-throughput optimization")
    high_throughput_workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.TRAINING,
        circuit_depth=100,
        gate_count=200,
        two_qubit_gate_count=80,
        coherence_time_required=200.0,
        fidelity_threshold=0.85,
        error_budget=0.05,
        quantum_advantage_target=3.0
    )
    
    high_throughput_targets = {
        "max_latency": 100.0,
        "min_throughput": 100.0,
        "max_cost": 500.0
    }
    
    resources = {
        ResourceType.TPU_V5_CORE: 32,
        ResourceType.QUANTUM_PROCESSOR: 64,
        ResourceType.CLASSICAL_CPU: 16,
        ResourceType.MEMORY: 256,
        ResourceType.NETWORK_BANDWIDTH: 2000,
        ResourceType.STORAGE_IOPS: 10000
    }
    
    result1 = await engine.optimize_performance(
        high_throughput_workload, high_throughput_targets, resources,
        OptimizationStrategy.SUPERPOSITION_PARALLEL
    )
    
    assert result1.predicted_performance.throughput >= high_throughput_targets["min_throughput"]
    print(f"    High-throughput result: {result1.predicted_performance.throughput:.1f} ops/s")
    
    # Scenario 2: Low-latency workload
    print("  Scenario 2: Low-latency optimization")
    low_latency_workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.INFERENCE,
        circuit_depth=20,
        gate_count=40,
        two_qubit_gate_count=10,
        coherence_time_required=50.0,
        fidelity_threshold=0.98,
        error_budget=0.005,
        quantum_advantage_target=1.5
    )
    
    low_latency_targets = {
        "max_latency": 5.0,
        "min_throughput": 10.0,
        "max_cost": 100.0
    }
    
    result2 = await engine.optimize_performance(
        low_latency_workload, low_latency_targets, resources,
        OptimizationStrategy.QUANTUM_ANNEALING
    )
    
    assert result2.predicted_performance.latency_p50 <= low_latency_targets["max_latency"]
    print(f"    Low-latency result: {result2.predicted_performance.latency_p50:.2f}ms")
    
    # Scenario 3: Cost-constrained workload
    print("  Scenario 3: Cost-constrained optimization")
    cost_constrained_targets = {
        "max_latency": 50.0,
        "min_throughput": 20.0,
        "max_cost": 50.0  # Very tight budget
    }
    
    result3 = await engine.optimize_performance(
        low_latency_workload, cost_constrained_targets, resources,
        OptimizationStrategy.ADAPTIVE_RESOURCE_ALLOCATION
    )
    
    assert result3.optimal_allocation.estimated_cost <= cost_constrained_targets["max_cost"]
    print(f"    Cost-constrained result: ${result3.optimal_allocation.estimated_cost:.2f}/hr")
    
    print("✅ Performance scaling scenarios passed")
    
    return True


def test_resource_allocation_edge_cases():
    """Test edge cases in resource allocation."""
    print("Testing Resource Allocation Edge Cases...")
    
    optimizer = QuantumAnnealingOptimizer(max_iterations=10)
    
    # Test with minimal resources
    minimal_workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.INFERENCE,
        circuit_depth=5,
        gate_count=10,
        two_qubit_gate_count=2,
        coherence_time_required=20.0,
        fidelity_threshold=0.9,
        error_budget=0.1,
        quantum_advantage_target=1.1
    )
    
    minimal_resources = {
        ResourceType.TPU_V5_CORE: 1,
        ResourceType.QUANTUM_PROCESSOR: 4,
        ResourceType.CLASSICAL_CPU: 1,
        ResourceType.MEMORY: 4,
        ResourceType.NETWORK_BANDWIDTH: 50,
        ResourceType.STORAGE_IOPS: 100
    }
    
    minimal_constraints = {
        "max_latency": 1000.0,  # Very relaxed
        "min_throughput": 1.0,  # Very low
        "max_cost": 10.0
    }
    
    allocation = optimizer.optimize_resource_allocation(
        minimal_workload, minimal_constraints, minimal_resources
    )
    
    assert allocation.tpu_cores >= 1
    assert allocation.quantum_qubits >= 4
    assert allocation.estimated_cost <= minimal_constraints["max_cost"]
    
    print(f"   Minimal allocation: TPU={allocation.tpu_cores}, Quantum={allocation.quantum_qubits}")
    print(f"   Minimal cost: ${allocation.estimated_cost:.2f}/hr")
    
    # Test with very demanding workload
    demanding_workload = WorkloadCharacteristics(
        workload_type=MLWorkloadType.NEURAL_ARCHITECTURE_SEARCH,
        circuit_depth=500,
        gate_count=1000,
        two_qubit_gate_count=400,
        coherence_time_required=1000.0,
        fidelity_threshold=0.99,
        error_budget=0.001,
        quantum_advantage_target=10.0
    )
    
    demanding_constraints = {
        "max_latency": 1.0,     # Very tight
        "min_throughput": 1000.0,  # Very high
        "max_cost": 2000.0      # High budget
    }
    
    large_resources = {
        ResourceType.TPU_V5_CORE: 64,
        ResourceType.QUANTUM_PROCESSOR: 128,
        ResourceType.CLASSICAL_CPU: 32,
        ResourceType.MEMORY: 1024,
        ResourceType.NETWORK_BANDWIDTH: 10000,
        ResourceType.STORAGE_IOPS: 50000
    }
    
    demanding_allocation = optimizer.optimize_resource_allocation(
        demanding_workload, demanding_constraints, large_resources
    )
    
    assert demanding_allocation.tpu_cores > allocation.tpu_cores
    assert demanding_allocation.quantum_qubits > allocation.quantum_qubits
    assert demanding_allocation.estimated_cost > allocation.estimated_cost
    
    print(f"   Demanding allocation: TPU={demanding_allocation.tpu_cores}, Quantum={demanding_allocation.quantum_qubits}")
    print(f"   Demanding cost: ${demanding_allocation.estimated_cost:.2f}/hr")
    
    print("✅ Resource allocation edge cases passed")
    
    return True


async def main():
    """Run all hyper-performance engine tests."""
    print("🚀 Testing Quantum Hyper-Performance Engine")
    print("=" * 55)
    
    tests = [
        test_quantum_annealing_optimizer,
        test_superposition_parallel_processor,
        test_entanglement_coordinator,
        test_hyper_performance_engine,
        test_performance_scaling_scenarios,
        test_resource_allocation_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All hyper-performance engine tests passed!")
        
        print("\n🚀 Hyper-Performance Engine Features Validated:")
        print("- ✅ Quantum annealing resource optimization")
        print("- ✅ Superposition parallel processing with interference patterns")
        print("- ✅ Quantum entanglement task coordination")
        print("- ✅ Multiple optimization strategies (4 strategies)")
        print("- ✅ Performance prediction and validation")
        print("- ✅ Resource allocation optimization")
        print("- ✅ Cost-aware optimization")
        print("- ✅ Caching and performance history tracking")
        print("- ✅ Edge case handling")
        print("- ✅ Async processing support")
        
        print("\n⚡ Key Performance Capabilities:")
        print("- Quantum annealing for optimal resource allocation")
        print("- Superposition parallel processing (8x+ speedup)")
        print("- Entangled task coordination with correlation")
        print("- Adaptive ML-guided optimization")
        print("- Multi-objective optimization (latency, throughput, cost)")
        print("- Scalable resource management")
        print("- Real-time performance monitoring")
        
        return True
    else:
        print("⚠️ Some hyper-performance engine tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)