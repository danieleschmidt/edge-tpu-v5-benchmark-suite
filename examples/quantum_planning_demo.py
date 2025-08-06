#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Demo

Demonstrates the quantum task planner capabilities with TPU v5 optimization.
"""

import asyncio
import json
from pathlib import Path

from edge_tpu_v5_benchmark.quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumResource,
    QuantumState
)


def create_sample_tasks():
    """Create sample tasks for demonstration"""
    tasks = [
        QuantumTask(
            id="preprocess_data",
            name="Data Preprocessing",
            priority=2.0,
            complexity=1.5,
            estimated_duration=5.0,
            resource_requirements={"cpu_cores": 2.0, "memory_gb": 4.0}
        ),
        QuantumTask(
            id="model_loading",
            name="Load TPU Model",
            priority=3.0,
            complexity=2.0,
            estimated_duration=3.0,
            dependencies={"preprocess_data"},
            resource_requirements={"tpu_v5_primary": 1.0, "memory_gb": 8.0}
        ),
        QuantumTask(
            id="inference_batch1",
            name="Inference Batch 1",
            priority=1.5,
            complexity=2.5,
            estimated_duration=10.0,
            dependencies={"model_loading"},
            resource_requirements={"tpu_v5_primary": 1.0, "memory_gb": 2.0}
        ),
        QuantumTask(
            id="inference_batch2", 
            name="Inference Batch 2",
            priority=1.5,
            complexity=2.5,
            estimated_duration=8.0,
            dependencies={"model_loading"},
            resource_requirements={"tpu_v5_primary": 1.0, "memory_gb": 2.0}
        ),
        QuantumTask(
            id="postprocess_results",
            name="Post-process Results",
            priority=1.0,
            complexity=1.0,
            estimated_duration=4.0,
            dependencies={"inference_batch1", "inference_batch2"},
            resource_requirements={"cpu_cores": 1.0, "memory_gb": 1.0}
        ),
        QuantumTask(
            id="validation",
            name="Validate Outputs",
            priority=2.5,
            complexity=1.2,
            estimated_duration=2.0,
            dependencies={"postprocess_results"},
            resource_requirements={"cpu_cores": 1.0, "memory_gb": 0.5}
        )
    ]
    
    return tasks


async def demo_quantum_planning():
    """Run quantum planning demonstration"""
    print("🌌 Quantum-Inspired Task Planning Demo")
    print("=" * 50)
    
    # Initialize planner with custom resources
    custom_resources = [
        QuantumResource(
            name="tpu_v5_primary",
            total_capacity=1.0,
            tpu_cores=1,
            memory_gb=128,
            compute_tops=50
        ),
        QuantumResource(
            name="cpu_cores",
            total_capacity=12.0,
            allocation_quantum=1.0
        ),
        QuantumResource(
            name="memory_gb",
            total_capacity=64.0,
            allocation_quantum=0.5
        )
    ]
    
    planner = QuantumTaskPlanner(resources=custom_resources)
    
    # Add sample tasks
    print("\n📋 Adding quantum tasks...")
    tasks = create_sample_tasks()
    for task in tasks:
        planner.add_task(task)
        print(f"   ✅ Added: {task.name} (ID: {task.id})")
    
    # Create quantum entanglements
    print("\n🔗 Creating quantum entanglements...")
    planner.entangle_tasks("inference_batch1", "inference_batch2")
    print("   🌊 Entangled inference batches for coordinated execution")
    
    # Show initial system state
    print("\n🔬 Initial Quantum System State:")
    state = planner.get_system_state()
    print(f"   Total tasks: {state['total_tasks']}")
    print(f"   Ready tasks: {state['ready_tasks']}")
    print(f"   Quantum coherence: {state['quantum_metrics']['average_coherence']:.2%}")
    print(f"   Entanglement pairs: {state['quantum_metrics']['entanglement_pairs']}")
    
    # Optimize schedule
    print("\n🎯 Optimizing quantum schedule...")
    optimized_schedule = planner.optimize_schedule()
    
    print("   Optimized execution order:")
    for i, task in enumerate(optimized_schedule):
        quantum_priority = planner.calculate_quantum_priority(task)
        print(f"   {i+1:2d}. {task.name:20s} (Priority: {quantum_priority:5.1f})")
    
    # Run execution cycles
    print("\n🚀 Running quantum execution cycles...")
    
    cycle_count = 0
    total_executed = 0
    
    while planner.get_ready_tasks():
        cycle_count += 1
        print(f"\n   Cycle {cycle_count}:")
        
        results = await planner.run_quantum_execution_cycle()
        
        executed = len(results['tasks_executed'])
        failed = len(results['tasks_failed'])
        total_executed += executed
        
        print(f"   ✅ Executed: {executed}, ❌ Failed: {failed}")
        print(f"   🌊 Coherence: {results['quantum_coherence']:.2%}")
        
        # Show resource utilization
        for resource, util in results['resource_utilization'].items():
            print(f"   📊 {resource}: {util:.1%}")
        
        # Brief pause for visualization
        await asyncio.sleep(0.5)
    
    # Final system state
    print(f"\n🏁 Execution Complete!")
    final_state = planner.get_system_state()
    print(f"   Total tasks executed: {total_executed}")
    print(f"   Completed tasks: {final_state['completed_tasks']}")
    print(f"   Success rate: {final_state['completed_tasks']/final_state['total_tasks']*100:.1f}%")
    
    # Export quantum state
    output_file = "quantum_demo_results.json"
    planner.export_quantum_state(output_file)
    print(f"   📤 Results exported to {output_file}")
    
    # Show task execution history
    print(f"\n📈 Task Execution History:")
    for task in planner.tasks.values():
        if task.execution_history:
            history = task.execution_history[-1]  # Latest execution
            duration = history['actual_duration']
            quantum_noise = history.get('quantum_noise', 0)
            print(f"   {task.name:20s}: {duration:.2f}s (noise: {quantum_noise:+.3f})")
    
    return planner


def demonstrate_quantum_features():
    """Demonstrate specific quantum features"""
    print("\n" + "=" * 50)
    print("🔬 Quantum Features Demonstration")
    print("=" * 50)
    
    # Create simple planner
    planner = QuantumTaskPlanner()
    
    # Demonstrate superposition
    print("\n1. 🌊 Quantum Superposition")
    task = QuantumTask(
        id="superposition_demo",
        name="Superposition Task",
        priority=1.0,
        complexity=2.0
    )
    
    print(f"   Initial state: {task.state.value}")
    print(f"   Probability amplitude: {task.probability_amplitude}")
    
    planner.add_task(task)
    task.collapse_wavefunction("execution_path_A")
    
    print(f"   After collapse: {task.state.value}")
    print(f"   New amplitude: {task.probability_amplitude}")
    
    # Demonstrate entanglement
    print("\n2. 🔗 Quantum Entanglement")
    task1 = QuantumTask(id="entangled_1", name="Entangled Task 1")
    task2 = QuantumTask(id="entangled_2", name="Entangled Task 2")
    
    planner.add_task(task1)
    planner.add_task(task2)
    
    print(f"   Before entanglement:")
    print(f"     Task 1 state: {task1.state.value}")
    print(f"     Task 2 state: {task2.state.value}")
    
    planner.entangle_tasks("entangled_1", "entangled_2")
    
    print(f"   After entanglement:")
    print(f"     Task 1 state: {task1.state.value}")
    print(f"     Task 2 state: {task2.state.value}")
    print(f"     Entanglement graph: {planner.entanglement_graph}")
    
    # Demonstrate decoherence
    print("\n3. ⏳ Quantum Decoherence")
    import time
    
    decoherent_task = QuantumTask(
        id="decoherent_demo",
        name="Decoherent Task",
        decoherence_time=2.0  # Very fast decoherence for demo
    )
    
    print(f"   Initial decoherence: {decoherent_task.measure_decoherence():.2%}")
    
    # Simulate time passage
    time.sleep(1.0)  
    print(f"   After 1 second: {decoherent_task.measure_decoherence():.2%}")
    
    time.sleep(1.5)
    print(f"   After 2.5 seconds: {decoherent_task.measure_decoherence():.2%}")
    
    # Demonstrate quantum annealing
    print("\n4. 🧊 Quantum Annealing Optimization")
    
    # Create tasks with different priorities and dependencies
    optimization_tasks = [
        QuantumTask(id="opt_1", name="High Priority", priority=5.0, complexity=1.0),
        QuantumTask(id="opt_2", name="Medium Priority", priority=3.0, complexity=2.0),
        QuantumTask(id="opt_3", name="Low Priority", priority=1.0, complexity=1.5),
        QuantumTask(id="opt_4", name="Complex Task", priority=2.0, complexity=3.0),
    ]
    
    print("   Original order:")
    for i, task in enumerate(optimization_tasks):
        print(f"     {i+1}. {task.name} (P: {task.priority}, C: {task.complexity})")
    
    # Apply quantum annealing
    from edge_tpu_v5_benchmark.quantum_planner import QuantumAnnealer
    annealer = QuantumAnnealer()
    optimized = annealer.anneal_schedule(optimization_tasks, max_iterations=100)
    
    print("   After quantum annealing:")
    for i, task in enumerate(optimized):
        energy_contribution = task.priority * task.complexity
        print(f"     {i+1}. {task.name} (Energy: {energy_contribution:.1f})")


async def main():
    """Main demonstration function"""
    try:
        # Run main quantum planning demo
        planner = await demo_quantum_planning()
        
        # Demonstrate quantum features
        demonstrate_quantum_features()
        
        print(f"\n✨ Demo completed successfully!")
        print(f"   Check 'quantum_demo_results.json' for detailed results")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())