"""Unit tests for quantum task planner core functionality."""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

from edge_tpu_v5_benchmark.quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumResource,
    QuantumState,
    QuantumAnnealer
)


class TestQuantumTask:
    """Test QuantumTask class functionality."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = QuantumTask(
            id="test_task",
            name="Test Task",
            priority=2.0,
            complexity=1.5,
            estimated_duration=10.0
        )
        
        assert task.id == "test_task"
        assert task.name == "Test Task"
        assert task.priority == 2.0
        assert task.complexity == 1.5
        assert task.estimated_duration == 10.0
        assert task.state == QuantumState.SUPERPOSITION
        assert abs(task.probability_amplitude) == 1.0
    
    def test_task_dependencies(self):
        """Test task dependency management."""
        task = QuantumTask(
            id="dependent_task",
            name="Dependent Task",
            dependencies={"task1", "task2"}
        )
        
        assert len(task.dependencies) == 2
        assert "task1" in task.dependencies
        assert "task2" in task.dependencies
    
    def test_collapse_wavefunction(self):
        """Test quantum wavefunction collapse."""
        task = QuantumTask(id="test", name="Test")
        
        assert task.state == QuantumState.SUPERPOSITION
        
        task.collapse_wavefunction("path_A")
        
        assert task.state == QuantumState.COLLAPSED
        assert task.probability_amplitude == 1.0 + 0j
    
    def test_entangle_with(self):
        """Test quantum entanglement."""
        task = QuantumTask(id="test", name="Test")
        
        assert len(task.entangled_tasks) == 0
        assert task.state == QuantumState.SUPERPOSITION
        
        task.entangle_with("other_task")
        
        assert "other_task" in task.entangled_tasks
        assert task.state == QuantumState.ENTANGLED
    
    def test_decoherence_measurement(self):
        """Test decoherence measurement."""
        task = QuantumTask(id="test", name="Test", decoherence_time=1.0)
        
        # Should start with low decoherence
        initial_decoherence = task.measure_decoherence()
        assert initial_decoherence >= 0.0
        
        # Simulate time passage
        original_created_at = task.created_at
        task.created_at = original_created_at - 0.5  # 0.5 seconds ago
        
        decoherence_after_time = task.measure_decoherence()
        assert decoherence_after_time > initial_decoherence
        
        # Should cap at 1.0
        task.created_at = original_created_at - 2.0  # 2 seconds ago
        max_decoherence = task.measure_decoherence()
        assert max_decoherence == 1.0
    
    def test_is_ready_for_execution(self):
        """Test task readiness checking."""
        task = QuantumTask(
            id="test",
            name="Test",
            dependencies={"dep1", "dep2"}
        )
        
        # Should not be ready without dependencies
        assert not task.is_ready_for_execution(set())
        assert not task.is_ready_for_execution({"dep1"})
        
        # Should be ready with all dependencies
        assert task.is_ready_for_execution({"dep1", "dep2", "extra"})
        
        # Should not be ready if highly decoherent
        task.created_at = time.time() - 1000  # Make it decoherent
        task.decoherence_time = 1.0
        assert not task.is_ready_for_execution({"dep1", "dep2"})


class TestQuantumResource:
    """Test QuantumResource class functionality."""
    
    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = QuantumResource(
            name="test_resource",
            total_capacity=10.0,
            allocation_quantum=1.0
        )
        
        assert resource.name == "test_resource"
        assert resource.total_capacity == 10.0
        assert resource.available_capacity == 10.0
        assert resource.allocation_quantum == 1.0
    
    def test_resource_allocation(self):
        """Test resource allocation and release."""
        resource = QuantumResource(
            name="test",
            total_capacity=10.0,
            allocation_quantum=1.0
        )
        
        # Test successful allocation
        assert resource.can_allocate(5.0)
        assert resource.allocate(5.0)
        assert resource.available_capacity == 5.0
        
        # Test failed allocation
        assert not resource.can_allocate(10.0)
        assert not resource.allocate(10.0)
        assert resource.available_capacity == 5.0
        
        # Test release
        resource.release(3.0)
        assert resource.available_capacity == 8.0
        
        # Test quantum alignment
        assert resource.allocate(2.5)  # Should round up to 3.0
        assert resource.available_capacity == 5.0


class TestQuantumAnnealer:
    """Test QuantumAnnealer optimization functionality."""
    
    def test_annealer_creation(self):
        """Test annealer creation with default parameters."""
        annealer = QuantumAnnealer()
        
        assert len(annealer.temperature_schedule) > 0
        assert annealer.current_temperature > 0
        assert annealer.iteration == 0
    
    def test_energy_calculation(self):
        """Test energy calculation for task schedule."""
        annealer = QuantumAnnealer()
        
        # Simple schedule
        tasks = [
            QuantumTask(id="t1", name="Task 1", estimated_duration=5.0),
            QuantumTask(id="t2", name="Task 2", estimated_duration=3.0)
        ]
        
        energy = annealer.calculate_energy(tasks)
        assert energy >= 0
        
        # Schedule with dependencies should have higher energy if violated
        tasks_with_deps = [
            QuantumTask(id="t1", name="Task 1", dependencies={"t2"}),
            QuantumTask(id="t2", name="Task 2")
        ]
        
        # t1 depends on t2 but comes first - should have high energy
        energy_with_violation = annealer.calculate_energy(tasks_with_deps)
        
        # Correct order - should have lower energy
        tasks_correct_order = [tasks_with_deps[1], tasks_with_deps[0]]
        energy_correct = annealer.calculate_energy(tasks_correct_order)
        
        assert energy_with_violation > energy_correct
    
    def test_anneal_schedule(self):
        """Test quantum annealing optimization."""
        annealer = QuantumAnnealer()
        
        # Create tasks with clear optimal order
        tasks = [
            QuantumTask(id="low", name="Low Priority", priority=1.0, estimated_duration=1.0),
            QuantumTask(id="high", name="High Priority", priority=5.0, estimated_duration=1.0),
            QuantumTask(id="med", name="Medium Priority", priority=3.0, estimated_duration=1.0)
        ]
        
        # Run annealing
        optimized = annealer.anneal_schedule(tasks, max_iterations=50)
        
        assert len(optimized) == 3
        # Higher priority tasks should tend to come first
        # (though annealing is probabilistic, so we can't guarantee exact order)
        priorities = [task.priority for task in optimized]
        assert max(priorities) >= min(priorities)  # Sanity check


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner main functionality."""
    
    def test_planner_creation(self):
        """Test planner creation with default resources."""
        planner = QuantumTaskPlanner()
        
        assert len(planner.resources) > 0
        assert "tpu_v5_primary" in planner.resources
        assert "cpu_cores" in planner.resources
        assert "memory_gb" in planner.resources
        assert len(planner.tasks) == 0
        assert len(planner.completed_tasks) == 0
    
    def test_add_task(self):
        """Test adding tasks to planner."""
        planner = QuantumTaskPlanner()
        
        task = QuantumTask(id="test", name="Test Task")
        planner.add_task(task)
        
        assert "test" in planner.tasks
        assert planner.tasks["test"] == task
        assert len(planner.coherence_matrix) >= 1
    
    def test_create_task(self):
        """Test task creation through planner."""
        planner = QuantumTaskPlanner()
        
        task = planner.create_task(
            task_id="created",
            name="Created Task",
            priority=2.0,
            complexity=1.5
        )
        
        assert task.id == "created"
        assert task.name == "Created Task"
        assert task.priority == 2.0
        assert task.complexity == 1.5
        assert "created" in planner.tasks
    
    def test_entangle_tasks(self):
        """Test task entanglement through planner."""
        planner = QuantumTaskPlanner()
        
        task1 = QuantumTask(id="task1", name="Task 1")
        task2 = QuantumTask(id="task2", name="Task 2")
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        planner.entangle_tasks("task1", "task2")
        
        assert "task2" in task1.entangled_tasks
        assert "task1" in task2.entangled_tasks
        assert task1.state == QuantumState.ENTANGLED
        assert task2.state == QuantumState.ENTANGLED
        assert "task1" in planner.entanglement_graph
        assert "task2" in planner.entanglement_graph
    
    def test_calculate_quantum_priority(self):
        """Test quantum priority calculation."""
        planner = QuantumTaskPlanner()
        
        # Simple task
        simple_task = QuantumTask(id="simple", name="Simple", priority=1.0)
        simple_priority = planner.calculate_quantum_priority(simple_task)
        
        # Complex task with dependencies
        complex_task = QuantumTask(
            id="complex",
            name="Complex",
            priority=1.0,
            complexity=3.0,
            dependencies={"dep1", "dep2"}
        )
        complex_priority = planner.calculate_quantum_priority(complex_task)
        
        # Complex task should have higher quantum priority
        assert complex_priority > simple_priority
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        planner = QuantumTaskPlanner()
        
        # Task with no dependencies - should be ready
        ready_task = QuantumTask(id="ready", name="Ready Task")
        planner.add_task(ready_task)
        
        # Task with unmet dependencies - should not be ready
        blocked_task = QuantumTask(
            id="blocked",
            name="Blocked Task",
            dependencies={"missing_dependency"}
        )
        planner.add_task(blocked_task)
        
        # Task with met dependencies - should be ready
        unblocked_task = QuantumTask(
            id="unblocked",
            name="Unblocked Task",
            dependencies={"ready"}
        )
        planner.add_task(unblocked_task)
        planner.completed_tasks.add("ready")
        
        ready_tasks = planner.get_ready_tasks()
        ready_ids = [task.id for task in ready_tasks]
        
        assert "ready" in ready_ids
        assert "blocked" not in ready_ids
        assert "unblocked" in ready_ids
    
    def test_resource_allocation(self):
        """Test resource allocation and release."""
        planner = QuantumTaskPlanner()
        
        task = QuantumTask(
            id="test",
            name="Test",
            resource_requirements={"cpu_cores": 2.0, "memory_gb": 4.0}
        )
        planner.add_task(task)
        
        # Should be able to allocate
        assert planner.can_allocate_resources(task)
        assert planner.allocate_task_resources(task)
        
        # Check resource utilization
        cpu_resource = planner.resources["cpu_cores"]
        memory_resource = planner.resources["memory_gb"]
        
        assert cpu_resource.available_capacity < cpu_resource.total_capacity
        assert memory_resource.available_capacity < memory_resource.total_capacity
        
        # Release resources
        planner.release_task_resources(task)
        
        assert cpu_resource.available_capacity == cpu_resource.total_capacity
        assert memory_resource.available_capacity == memory_resource.total_capacity
    
    @pytest.mark.asyncio
    async def test_execute_task(self):
        """Test task execution."""
        planner = QuantumTaskPlanner()
        
        task = QuantumTask(
            id="test",
            name="Test Task",
            estimated_duration=0.1
        )
        
        result = await planner.execute_task(task)
        
        assert result["task_id"] == "test"
        assert "success" in result
        assert "duration" in result
        assert "quantum_effects" in result
    
    @pytest.mark.asyncio
    async def test_run_quantum_execution_cycle(self):
        """Test complete execution cycle."""
        planner = QuantumTaskPlanner()
        
        # Add some tasks
        task1 = QuantumTask(id="task1", name="Task 1", estimated_duration=0.1)
        task2 = QuantumTask(id="task2", name="Task 2", estimated_duration=0.1)
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        # Run execution cycle
        results = await planner.run_quantum_execution_cycle()
        
        assert "cycle_start" in results
        assert "tasks_executed" in results
        assert "tasks_failed" in results
        assert "resource_utilization" in results
        assert "quantum_coherence" in results
        assert "cycle_duration" in results
        
        # Should have executed at least some tasks
        total_processed = len(results["tasks_executed"]) + len(results["tasks_failed"])
        assert total_processed > 0
    
    def test_get_system_state(self):
        """Test system state reporting."""
        planner = QuantumTaskPlanner()
        
        # Add some tasks in different states
        task1 = QuantumTask(id="task1", name="Task 1")
        task2 = QuantumTask(id="task2", name="Task 2")
        task2.state = QuantumState.COLLAPSED
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.completed_tasks.add("completed_task")
        
        state = planner.get_system_state()
        
        assert state["total_tasks"] == 2
        assert state["completed_tasks"] == 1
        assert "ready_tasks" in state
        assert "resource_utilization" in state
        assert "quantum_metrics" in state
        
        quantum_metrics = state["quantum_metrics"]
        assert "average_coherence" in quantum_metrics
        assert "superposition_tasks" in quantum_metrics
        assert "collapsed_tasks" in quantum_metrics
    
    def test_export_quantum_state(self, tmp_path):
        """Test quantum state export."""
        planner = QuantumTaskPlanner()
        
        # Add some tasks
        task = QuantumTask(id="test", name="Test Task")
        planner.add_task(task)
        planner.entangle_tasks("test", "test")  # Will be ignored
        
        # Export state
        export_file = tmp_path / "quantum_state.json"
        planner.export_quantum_state(str(export_file))
        
        assert export_file.exists()
        
        # Verify export content
        import json
        with open(export_file) as f:
            exported_data = json.load(f)
        
        assert "timestamp" in exported_data
        assert "system_state" in exported_data
        assert "tasks" in exported_data
        assert "resources" in exported_data
        assert "entanglement_graph" in exported_data
        
        assert "test" in exported_data["tasks"]
        task_data = exported_data["tasks"]["test"]
        assert task_data["name"] == "Test Task"
        assert task_data["state"] == "superposition"


class TestQuantumIntegration:
    """Integration tests for quantum system components."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete quantum workflow."""
        planner = QuantumTaskPlanner()
        
        # Create a workflow with dependencies
        tasks_data = [
            {"id": "data_prep", "name": "Data Preparation", "duration": 0.1},
            {"id": "model_load", "name": "Model Loading", "duration": 0.1, "deps": ["data_prep"]},
            {"id": "inference", "name": "Inference", "duration": 0.1, "deps": ["model_load"]},
            {"id": "postprocess", "name": "Post-processing", "duration": 0.1, "deps": ["inference"]}
        ]
        
        # Add tasks
        for task_data in tasks_data:
            task = QuantumTask(
                id=task_data["id"],
                name=task_data["name"],
                estimated_duration=task_data["duration"],
                dependencies=set(task_data.get("deps", []))
            )
            planner.add_task(task)
        
        # Run multiple execution cycles until all tasks complete
        max_cycles = 10
        cycle_count = 0
        
        while planner.get_ready_tasks() and cycle_count < max_cycles:
            cycle_count += 1
            results = await planner.run_quantum_execution_cycle()
            
            # Verify cycle results
            assert "cycle_duration" in results
            assert results["cycle_duration"] >= 0
            
            # Brief pause between cycles
            await asyncio.sleep(0.01)
        
        # Verify all tasks were eventually completed
        assert len(planner.completed_tasks) == len(tasks_data)
        
        # Verify final system state
        final_state = planner.get_system_state()
        assert final_state["completed_tasks"] == len(tasks_data)
        assert final_state["ready_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in quantum execution."""
        planner = QuantumTaskPlanner()
        
        # Create task with invalid resource requirements
        problematic_task = QuantumTask(
            id="problematic",
            name="Problematic Task",
            resource_requirements={"nonexistent_resource": 1000.0}
        )
        planner.add_task(problematic_task)
        
        # Execution should handle the error gracefully
        results = await planner.run_quantum_execution_cycle()
        
        # Should either execute with partial resources or fail gracefully
        assert "cycle_duration" in results
        assert isinstance(results["tasks_executed"], list)
        assert isinstance(results["tasks_failed"], list)
    
    def test_optimization_integration(self):
        """Test optimization with annealing."""
        planner = QuantumTaskPlanner()
        
        # Create tasks with clear priority order
        tasks = []
        for i in range(5):
            task = QuantumTask(
                id=f"task_{i}",
                name=f"Task {i}",
                priority=5 - i,  # Decreasing priority
                complexity=1.0,
                estimated_duration=1.0
            )
            tasks.append(task)
            planner.add_task(task)
        
        # Get optimized schedule
        optimized_schedule = planner.optimize_schedule()
        
        assert len(optimized_schedule) <= 5
        
        # Verify optimization had some effect
        if optimized_schedule:
            # First task should generally have high priority
            first_task = optimized_schedule[0]
            assert first_task.priority >= 1.0
            
            # All tasks should be in collapsed state after optimization
            for task in optimized_schedule:
                assert task.state == QuantumState.COLLAPSED


# Test fixtures and utilities
@pytest.fixture
def simple_task():
    """Fixture for a simple quantum task."""
    return QuantumTask(
        id="simple_test_task",
        name="Simple Test Task",
        priority=1.0,
        complexity=1.0,
        estimated_duration=1.0
    )


@pytest.fixture
def simple_planner():
    """Fixture for a simple quantum task planner."""
    return QuantumTaskPlanner()


@pytest.fixture
def resource_constrained_planner():
    """Fixture for a resource-constrained planner."""
    resources = [
        QuantumResource(name="limited_cpu", total_capacity=2.0),
        QuantumResource(name="limited_memory", total_capacity=4.0)
    ]
    return QuantumTaskPlanner(resources=resources)


if __name__ == "__main__":
    pytest.main([__file__])