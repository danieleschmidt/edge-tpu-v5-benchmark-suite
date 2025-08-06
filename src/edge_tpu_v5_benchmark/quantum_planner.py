"""Quantum-Inspired Task Planner

Implements quantum computing principles for intelligent task scheduling and optimization
on TPU v5 hardware, using superposition, entanglement, and quantum annealing concepts.
"""

import numpy as np
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class QuantumState(Enum):
    """Quantum states for task representation"""
    SUPERPOSITION = "superposition"  # Task exploring multiple execution paths
    COLLAPSED = "collapsed"          # Task committed to single execution path
    ENTANGLED = "entangled"         # Task dependent on other tasks
    DECOHERENT = "decoherent"       # Task failed or cancelled


@dataclass
class QuantumTask:
    """Quantum-inspired task representation with superposition capabilities"""
    
    id: str
    name: str
    priority: float = 1.0
    complexity: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    
    # Quantum properties
    state: QuantumState = QuantumState.SUPERPOSITION
    probability_amplitude: complex = 1.0 + 0j
    entangled_tasks: Set[str] = field(default_factory=set)
    decoherence_time: float = 300.0  # seconds
    
    # Execution properties
    estimated_duration: float = 1.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    execution_history: List[Dict] = field(default_factory=list)
    
    # TPU-specific properties
    tpu_affinity: Optional[str] = None
    model_requirements: List[str] = field(default_factory=list)
    memory_footprint: float = 0.0
    
    created_at: float = field(default_factory=time.time)
    
    def collapse_wavefunction(self, chosen_path: str) -> None:
        """Collapse superposition to specific execution path"""
        self.state = QuantumState.COLLAPSED
        self.probability_amplitude = 1.0 + 0j
        logger.info(f"Task {self.id} wavefunction collapsed to path: {chosen_path}")
    
    def entangle_with(self, other_task_id: str) -> None:
        """Create quantum entanglement with another task"""
        self.entangled_tasks.add(other_task_id)
        self.state = QuantumState.ENTANGLED
        logger.debug(f"Task {self.id} entangled with {other_task_id}")
    
    def measure_decoherence(self) -> float:
        """Calculate current decoherence level based on time"""
        elapsed = time.time() - self.created_at
        return min(1.0, elapsed / self.decoherence_time)
    
    def is_ready_for_execution(self, completed_tasks: Set[str]) -> bool:
        """Check if task dependencies are satisfied"""
        return (
            self.state in [QuantumState.SUPERPOSITION, QuantumState.COLLAPSED] and
            self.dependencies.issubset(completed_tasks) and
            self.measure_decoherence() < 0.8
        )


@dataclass
class QuantumResource:
    """Resource representation with quantum allocation capabilities"""
    
    name: str
    total_capacity: float
    available_capacity: float = 0.0
    allocation_quantum: float = 0.1  # minimum allocation unit
    
    # TPU-specific properties
    tpu_cores: int = 0
    memory_gb: float = 0.0
    compute_tops: float = 0.0
    
    def __post_init__(self):
        if self.available_capacity == 0.0:
            self.available_capacity = self.total_capacity
    
    def can_allocate(self, required: float) -> bool:
        """Check if resource can satisfy quantum allocation"""
        return self.available_capacity >= required
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource with quantum constraints"""
        quantum_aligned = np.ceil(amount / self.allocation_quantum) * self.allocation_quantum
        if self.can_allocate(quantum_aligned):
            self.available_capacity -= quantum_aligned
            return True
        return False
    
    def release(self, amount: float) -> None:
        """Release allocated resource"""
        quantum_aligned = np.ceil(amount / self.allocation_quantum) * self.allocation_quantum
        self.available_capacity = min(
            self.total_capacity,
            self.available_capacity + quantum_aligned
        )


class QuantumAnnealer:
    """Quantum annealing simulator for task optimization"""
    
    def __init__(self, temperature_schedule: Optional[List[float]] = None):
        self.temperature_schedule = temperature_schedule or [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        self.current_temperature = self.temperature_schedule[0]
        self.iteration = 0
    
    def calculate_energy(self, task_schedule: List[QuantumTask]) -> float:
        """Calculate system energy for given task schedule"""
        energy = 0.0
        
        # Minimize total completion time
        total_time = sum(task.estimated_duration for task in task_schedule)
        energy += total_time * 0.5
        
        # Penalize dependency violations
        completed = set()
        for task in task_schedule:
            if not task.dependencies.issubset(completed):
                energy += 100.0  # Heavy penalty
            completed.add(task.id)
        
        # Minimize resource contention
        resource_usage = {}
        for task in task_schedule:
            for resource, amount in task.resource_requirements.items():
                resource_usage[resource] = resource_usage.get(resource, 0) + amount
                if resource_usage[resource] > 1.0:  # Over-allocation
                    energy += (resource_usage[resource] - 1.0) * 50.0
        
        # Minimize decoherence
        for task in task_schedule:
            energy += task.measure_decoherence() * 10.0
        
        return energy
    
    def anneal_schedule(self, tasks: List[QuantumTask], max_iterations: int = 1000) -> List[QuantumTask]:
        """Use quantum annealing to optimize task schedule"""
        current_schedule = tasks.copy()
        current_energy = self.calculate_energy(current_schedule)
        best_schedule = current_schedule.copy()
        best_energy = current_energy
        
        for iteration in range(max_iterations):
            # Update temperature
            temp_idx = min(iteration // (max_iterations // len(self.temperature_schedule)), 
                          len(self.temperature_schedule) - 1)
            self.current_temperature = self.temperature_schedule[temp_idx]
            
            # Generate neighbor solution (swap two tasks)
            new_schedule = current_schedule.copy()
            if len(new_schedule) > 1:
                i, j = np.random.choice(len(new_schedule), 2, replace=False)
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
            
            # Calculate energy difference
            new_energy = self.calculate_energy(new_schedule)
            delta_energy = new_energy - current_energy
            
            # Accept or reject based on quantum annealing probability
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / self.current_temperature):
                current_schedule = new_schedule
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_schedule = new_schedule.copy()
                    best_energy = new_energy
        
        logger.info(f"Quantum annealing completed. Best energy: {best_energy:.2f}")
        return best_schedule


class QuantumTaskPlanner:
    """Main quantum-inspired task planner with TPU optimization"""
    
    def __init__(self, resources: Optional[List[QuantumResource]] = None):
        self.tasks: Dict[str, QuantumTask] = {}
        self.resources: Dict[str, QuantumResource] = {}
        self.completed_tasks: Set[str] = set()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.annealer = QuantumAnnealer()
        
        # Initialize default TPU resources
        if resources:
            for resource in resources:
                self.resources[resource.name] = resource
        else:
            self._init_default_resources()
        
        # Quantum coherence tracking
        self.coherence_matrix = np.eye(0)  # Will expand as tasks are added
        self.entanglement_graph: Dict[str, Set[str]] = {}
    
    def _init_default_resources(self) -> None:
        """Initialize default TPU v5 resources"""
        self.resources = {
            "tpu_v5_primary": QuantumResource(
                name="tpu_v5_primary",
                total_capacity=1.0,
                tpu_cores=1,
                memory_gb=128,
                compute_tops=50
            ),
            "cpu_cores": QuantumResource(
                name="cpu_cores",
                total_capacity=8.0,
                allocation_quantum=1.0
            ),
            "memory_gb": QuantumResource(
                name="memory_gb", 
                total_capacity=32.0,
                allocation_quantum=0.5
            )
        }
    
    def add_task(self, task: QuantumTask) -> None:
        """Add task to quantum superposition state"""
        self.tasks[task.id] = task
        self._expand_coherence_matrix()
        logger.info(f"Added quantum task {task.id} in superposition state")
    
    def create_task(
        self,
        task_id: str,
        name: str,
        priority: float = 1.0,
        complexity: float = 1.0,
        dependencies: Optional[Set[str]] = None,
        **kwargs
    ) -> QuantumTask:
        """Create and add a new quantum task"""
        task = QuantumTask(
            id=task_id,
            name=name,
            priority=priority,
            complexity=complexity,
            dependencies=dependencies or set(),
            **kwargs
        )
        self.add_task(task)
        return task
    
    def entangle_tasks(self, task1_id: str, task2_id: str) -> None:
        """Create quantum entanglement between two tasks"""
        if task1_id in self.tasks and task2_id in self.tasks:
            self.tasks[task1_id].entangle_with(task2_id)
            self.tasks[task2_id].entangle_with(task1_id)
            
            # Update entanglement graph
            self.entanglement_graph.setdefault(task1_id, set()).add(task2_id)
            self.entanglement_graph.setdefault(task2_id, set()).add(task1_id)
            
            logger.info(f"Quantum entanglement created between {task1_id} and {task2_id}")
    
    def _expand_coherence_matrix(self) -> None:
        """Expand coherence matrix for new tasks"""
        n_tasks = len(self.tasks)
        if self.coherence_matrix.shape[0] < n_tasks:
            new_matrix = np.eye(n_tasks)
            old_size = self.coherence_matrix.shape[0]
            if old_size > 0:
                new_matrix[:old_size, :old_size] = self.coherence_matrix
            self.coherence_matrix = new_matrix
    
    def calculate_quantum_priority(self, task: QuantumTask) -> float:
        """Calculate quantum-enhanced priority using superposition principles"""
        base_priority = task.priority
        
        # Complexity factor (more complex = higher priority for early scheduling)
        complexity_factor = 1.0 + (task.complexity - 1.0) * 0.5
        
        # Dependency factor (more dependencies = higher priority)
        dependency_factor = 1.0 + len(task.dependencies) * 0.2
        
        # Entanglement factor (entangled tasks get priority boost)
        entanglement_factor = 1.0 + len(task.entangled_tasks) * 0.3
        
        # Decoherence penalty (tasks near decoherence get priority)
        decoherence_penalty = 1.0 + task.measure_decoherence() * 2.0
        
        # Quantum amplitude influence
        amplitude_factor = abs(task.probability_amplitude) ** 2
        
        quantum_priority = (
            base_priority * 
            complexity_factor * 
            dependency_factor * 
            entanglement_factor * 
            decoherence_penalty * 
            amplitude_factor
        )
        
        return quantum_priority
    
    def get_ready_tasks(self) -> List[QuantumTask]:
        """Get tasks ready for execution using quantum selection"""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.is_ready_for_execution(self.completed_tasks):
                ready_tasks.append(task)
        
        # Sort by quantum priority
        ready_tasks.sort(key=self.calculate_quantum_priority, reverse=True)
        
        return ready_tasks
    
    def optimize_schedule(self) -> List[QuantumTask]:
        """Use quantum annealing to optimize task execution schedule"""
        ready_tasks = self.get_ready_tasks()
        if not ready_tasks:
            return []
        
        optimized_schedule = self.annealer.anneal_schedule(ready_tasks)
        
        # Collapse wavefunctions for scheduled tasks
        for i, task in enumerate(optimized_schedule):
            task.collapse_wavefunction(f"schedule_position_{i}")
        
        return optimized_schedule
    
    def can_allocate_resources(self, task: QuantumTask) -> bool:
        """Check if resources can be allocated for task execution"""
        for resource_name, required_amount in task.resource_requirements.items():
            if resource_name in self.resources:
                resource = self.resources[resource_name]
                if not resource.can_allocate(required_amount):
                    return False
        return True
    
    def allocate_task_resources(self, task: QuantumTask) -> bool:
        """Allocate resources for task execution"""
        allocated_resources = []
        
        try:
            for resource_name, required_amount in task.resource_requirements.items():
                if resource_name in self.resources:
                    resource = self.resources[resource_name]
                    if resource.allocate(required_amount):
                        allocated_resources.append((resource_name, required_amount))
                    else:
                        # Rollback allocations
                        for res_name, amount in allocated_resources:
                            self.resources[res_name].release(amount)
                        return False
            return True
        except Exception as e:
            # Rollback on error
            for res_name, amount in allocated_resources:
                self.resources[res_name].release(amount)
            logger.error(f"Resource allocation failed for task {task.id}: {e}")
            return False
    
    def release_task_resources(self, task: QuantumTask) -> None:
        """Release resources after task completion"""
        for resource_name, amount in task.resource_requirements.items():
            if resource_name in self.resources:
                self.resources[resource_name].release(amount)
    
    async def execute_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute a quantum task with TPU optimization"""
        start_time = time.time()
        
        try:
            # Simulate task execution with quantum decoherence
            execution_time = task.estimated_duration
            
            # Add quantum noise based on complexity
            quantum_noise = np.random.normal(0, task.complexity * 0.1)
            actual_execution_time = max(0.1, execution_time + quantum_noise)
            
            # Simulate execution delay
            await asyncio.sleep(min(actual_execution_time, 0.1))  # Cap for demo
            
            # Update task state
            task.state = QuantumState.COLLAPSED
            
            # Record execution history
            execution_record = {
                "start_time": start_time,
                "end_time": time.time(),
                "actual_duration": time.time() - start_time,
                "estimated_duration": execution_time,
                "quantum_noise": quantum_noise,
                "success": True
            }
            task.execution_history.append(execution_record)
            
            logger.info(f"Quantum task {task.id} executed successfully")
            
            return {
                "task_id": task.id,
                "success": True,
                "duration": time.time() - start_time,
                "quantum_effects": {
                    "noise": quantum_noise,
                    "decoherence": task.measure_decoherence()
                }
            }
            
        except Exception as e:
            task.state = QuantumState.DECOHERENT
            logger.error(f"Quantum task {task.id} execution failed: {e}")
            
            return {
                "task_id": task.id,
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def run_quantum_execution_cycle(self) -> Dict[str, Any]:
        """Run one quantum execution cycle"""
        cycle_start = time.time()
        results = {
            "cycle_start": cycle_start,
            "tasks_executed": [],
            "tasks_failed": [],
            "resource_utilization": {},
            "quantum_coherence": 0.0
        }
        
        # Get optimized schedule
        optimized_schedule = self.optimize_schedule()
        
        if not optimized_schedule:
            logger.info("No tasks ready for execution")
            return results
        
        # Execute tasks with resource constraints
        executing_tasks = []
        
        for task in optimized_schedule:
            if self.can_allocate_resources(task) and self.allocate_task_resources(task):
                # Start task execution
                task_coroutine = self.execute_task(task)
                executing_tasks.append((task, task_coroutine))
                
                logger.info(f"Started execution of quantum task {task.id}")
                
                # Limit concurrent executions
                if len(executing_tasks) >= 3:
                    break
        
        # Wait for task completions
        if executing_tasks:
            task_results = await asyncio.gather(
                *[task_coro for _, task_coro in executing_tasks],
                return_exceptions=True
            )
            
            # Process results
            for (task, _), result in zip(executing_tasks, task_results):
                if isinstance(result, dict) and result.get("success"):
                    results["tasks_executed"].append(result)
                    self.completed_tasks.add(task.id)
                    
                    # Handle entangled tasks
                    for entangled_id in task.entangled_tasks:
                        if entangled_id in self.tasks:
                            entangled_task = self.tasks[entangled_id]
                            # Quantum measurement affects entangled tasks
                            entangled_task.probability_amplitude *= 0.9
                else:
                    results["tasks_failed"].append({
                        "task_id": task.id,
                        "error": str(result) if isinstance(result, Exception) else "Unknown error"
                    })
                
                # Release resources
                self.release_task_resources(task)
        
        # Calculate resource utilization
        for name, resource in self.resources.items():
            utilization = 1.0 - (resource.available_capacity / resource.total_capacity)
            results["resource_utilization"][name] = utilization
        
        # Calculate quantum coherence
        coherent_tasks = [t for t in self.tasks.values() 
                         if t.state != QuantumState.DECOHERENT]
        total_coherence = sum(abs(t.probability_amplitude)**2 for t in coherent_tasks)
        results["quantum_coherence"] = total_coherence / max(len(coherent_tasks), 1)
        
        results["cycle_duration"] = time.time() - cycle_start
        
        logger.info(f"Quantum execution cycle completed: "
                   f"{len(results['tasks_executed'])} executed, "
                   f"{len(results['tasks_failed'])} failed")
        
        return results
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current quantum system state"""
        state = {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(self.completed_tasks),
            "ready_tasks": len(self.get_ready_tasks()),
            "resource_utilization": {},
            "quantum_metrics": {
                "average_coherence": 0.0,
                "entanglement_pairs": len(self.entanglement_graph),
                "superposition_tasks": 0,
                "collapsed_tasks": 0,
                "decoherent_tasks": 0
            }
        }
        
        # Resource utilization
        for name, resource in self.resources.items():
            utilization = 1.0 - (resource.available_capacity / resource.total_capacity)
            state["resource_utilization"][name] = utilization
        
        # Quantum metrics
        state_counts = {
            QuantumState.SUPERPOSITION: 0,
            QuantumState.COLLAPSED: 0,
            QuantumState.ENTANGLED: 0,
            QuantumState.DECOHERENT: 0
        }
        
        total_coherence = 0.0
        for task in self.tasks.values():
            state_counts[task.state] += 1
            total_coherence += abs(task.probability_amplitude)**2
        
        state["quantum_metrics"].update({
            "average_coherence": total_coherence / max(len(self.tasks), 1),
            "superposition_tasks": state_counts[QuantumState.SUPERPOSITION],
            "collapsed_tasks": state_counts[QuantumState.COLLAPSED],
            "entangled_tasks": state_counts[QuantumState.ENTANGLED],
            "decoherent_tasks": state_counts[QuantumState.DECOHERENT]
        })
        
        return state
    
    def export_quantum_state(self, filename: str) -> None:
        """Export current quantum state to JSON file"""
        export_data = {
            "timestamp": time.time(),
            "system_state": self.get_system_state(),
            "tasks": {},
            "resources": {},
            "entanglement_graph": {k: list(v) for k, v in self.entanglement_graph.items()}
        }
        
        # Export task details
        for task_id, task in self.tasks.items():
            export_data["tasks"][task_id] = {
                "name": task.name,
                "state": task.state.value,
                "priority": task.priority,
                "complexity": task.complexity,
                "dependencies": list(task.dependencies),
                "entangled_tasks": list(task.entangled_tasks),
                "probability_amplitude": {
                    "real": task.probability_amplitude.real,
                    "imag": task.probability_amplitude.imag
                },
                "decoherence_level": task.measure_decoherence(),
                "execution_history": task.execution_history
            }
        
        # Export resource status
        for name, resource in self.resources.items():
            export_data["resources"][name] = {
                "total_capacity": resource.total_capacity,
                "available_capacity": resource.available_capacity,
                "utilization": 1.0 - (resource.available_capacity / resource.total_capacity)
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Quantum state exported to {filename}")