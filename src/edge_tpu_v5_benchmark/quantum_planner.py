"""Quantum-Inspired Task Planner

Implements quantum computing principles for intelligent task scheduling and optimization
on TPU v5 hardware, using superposition, entanglement, and quantum annealing concepts.
Enhanced with comprehensive error handling, input validation, and resilience patterns.
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .exceptions import (
    AsyncErrorHandlingContext,
    ErrorContext,
    QuantumDecoherenceError,
    QuantumEntanglementError,
    QuantumStateError,
    QuantumValidationError,
    ResourceManager,
    handle_quantum_error,
    quantum_operation,
    sanitize_input,
    validate_input,
    validate_quantum_state,
)
from .security import DataSanitizer, InputValidator

# Configure structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter for structured logging
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(component)s:%(operation)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

    @validate_input(
        lambda self, chosen_path: InputValidator.validate_string(chosen_path, min_length=1),
        "Invalid chosen_path for wavefunction collapse"
    )
    @validate_quantum_state
    def collapse_wavefunction(self, chosen_path: str) -> None:
        """Collapse superposition to specific execution path with validation"""
        try:
            if self.state == QuantumState.DECOHERENT:
                raise QuantumStateError(f"Cannot collapse decoherent task {self.id}")

            self.state = QuantumState.COLLAPSED
            self.probability_amplitude = 1.0 + 0j

            logger.info(
                f"Task {self.id} wavefunction collapsed to path: {chosen_path}",
                extra={"component": "quantum_task", "operation": "collapse_wavefunction",
                      "task_id": self.id, "chosen_path": chosen_path}
            )
        except Exception as e:
            context = ErrorContext(
                component="quantum_task",
                operation="collapse_wavefunction",
                task_id=self.id,
                quantum_state={"state": self.state.value, "amplitude": abs(self.probability_amplitude)}
            )
            handle_quantum_error(e, context)

    @validate_input(
        lambda self, other_task_id: InputValidator.validate_quantum_task_id(other_task_id),
        "Invalid task ID for entanglement"
    )
    @validate_quantum_state
    def entangle_with(self, other_task_id: str) -> None:
        """Create quantum entanglement with another task with validation"""
        try:
            if other_task_id == self.id:
                raise QuantumEntanglementError(f"Task {self.id} cannot entangle with itself")

            if self.state == QuantumState.DECOHERENT:
                raise QuantumStateError(f"Cannot entangle decoherent task {self.id}")

            if len(self.entangled_tasks) >= 5:  # Limit entanglements
                raise QuantumEntanglementError(
                    f"Task {self.id} has too many entanglements ({len(self.entangled_tasks)})"
                )

            self.entangled_tasks.add(other_task_id)
            self.state = QuantumState.ENTANGLED

            logger.debug(
                f"Task {self.id} entangled with {other_task_id}",
                extra={"component": "quantum_task", "operation": "entangle_with",
                      "task_id": self.id, "other_task_id": other_task_id}
            )
        except Exception as e:
            context = ErrorContext(
                component="quantum_task",
                operation="entangle_with",
                task_id=self.id,
                quantum_state={"entangled_count": len(self.entangled_tasks)}
            )
            handle_quantum_error(e, context)

    def measure_decoherence(self) -> float:
        """Calculate current decoherence level based on time with bounds checking"""
        try:
            elapsed = time.time() - self.created_at
            if elapsed < 0:
                logger.warning(
                    f"Negative elapsed time detected for task {self.id}",
                    extra={"component": "quantum_task", "operation": "measure_decoherence"}
                )
                return 0.0

            decoherence = min(1.0, elapsed / max(self.decoherence_time, 0.1))

            if decoherence > 0.9:
                logger.warning(
                    f"Task {self.id} is highly decoherent: {decoherence:.1%}",
                    extra={"component": "quantum_task", "operation": "measure_decoherence",
                          "task_id": self.id, "decoherence": decoherence}
                )

            return decoherence
        except Exception as e:
            context = ErrorContext(
                component="quantum_task",
                operation="measure_decoherence",
                task_id=self.id
            )
            handle_quantum_error(e, context)
            return 1.0  # Safe default - assume fully decoherent

    def is_ready_for_execution(self, completed_tasks: Set[str]) -> bool:
        """Check if task dependencies are satisfied with comprehensive validation"""
        try:
            # Validate inputs
            if not isinstance(completed_tasks, set):
                logger.error(
                    f"Invalid completed_tasks type for task {self.id}: {type(completed_tasks)}",
                    extra={"component": "quantum_task", "operation": "is_ready_for_execution"}
                )
                return False

            # Check state validity
            if self.state == QuantumState.DECOHERENT:
                return False

            # Check valid execution states
            valid_states = [QuantumState.SUPERPOSITION, QuantumState.COLLAPSED, QuantumState.ENTANGLED]
            if self.state not in valid_states:
                return False

            # Check dependencies
            if not self.dependencies.issubset(completed_tasks):
                missing_deps = self.dependencies - completed_tasks
                logger.debug(
                    f"Task {self.id} waiting for dependencies: {missing_deps}",
                    extra={"component": "quantum_task", "operation": "is_ready_for_execution",
                          "task_id": self.id, "missing_dependencies": list(missing_deps)}
                )
                return False

            # Check decoherence
            decoherence = self.measure_decoherence()
            if decoherence >= 0.8:
                logger.debug(
                    f"Task {self.id} too decoherent for execution: {decoherence:.1%}",
                    extra={"component": "quantum_task", "operation": "is_ready_for_execution",
                          "task_id": self.id, "decoherence": decoherence}
                )
                return False

            return True

        except Exception as e:
            context = ErrorContext(
                component="quantum_task",
                operation="is_ready_for_execution",
                task_id=self.id
            )
            handle_quantum_error(e, context, reraise=False)
            return False  # Safe default


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

    @validate_input(
        lambda self, required: InputValidator.validate_numeric(required, min_value=0.0),
        "Invalid resource requirement amount"
    )
    def can_allocate(self, required: float) -> bool:
        """Check if resource can satisfy quantum allocation with validation"""
        try:
            return self.available_capacity >= required
        except Exception as e:
            context = ErrorContext(
                component="quantum_resource",
                operation="can_allocate",
                resource=self.name
            )
            handle_quantum_error(e, context, reraise=False)
            return False

    @validate_input(
        lambda self, amount: InputValidator.validate_numeric(amount, min_value=0.0),
        "Invalid allocation amount"
    )
    def allocate(self, amount: float) -> bool:
        """Allocate resource with quantum constraints and validation"""
        try:
            if amount <= 0:
                return True  # Nothing to allocate

            quantum_aligned = np.ceil(amount / max(self.allocation_quantum, 0.001)) * self.allocation_quantum

            if self.can_allocate(quantum_aligned):
                self.available_capacity -= quantum_aligned
                logger.debug(
                    f"Allocated {quantum_aligned} units of {self.name}",
                    extra={"component": "quantum_resource", "operation": "allocate",
                          "resource": self.name, "amount": quantum_aligned}
                )
                return True
            else:
                logger.warning(
                    f"Cannot allocate {quantum_aligned} units of {self.name} (available: {self.available_capacity})",
                    extra={"component": "quantum_resource", "operation": "allocate",
                          "resource": self.name, "requested": quantum_aligned,
                          "available": self.available_capacity}
                )
                return False
        except Exception as e:
            context = ErrorContext(
                component="quantum_resource",
                operation="allocate",
                resource=self.name
            )
            handle_quantum_error(e, context, reraise=False)
            return False

    @validate_input(
        lambda self, amount: InputValidator.validate_numeric(amount, min_value=0.0),
        "Invalid release amount"
    )
    def release(self, amount: float) -> None:
        """Release allocated resource with validation and bounds checking"""
        try:
            if amount <= 0:
                return  # Nothing to release

            quantum_aligned = np.ceil(amount / max(self.allocation_quantum, 0.001)) * self.allocation_quantum
            old_capacity = self.available_capacity

            self.available_capacity = min(
                self.total_capacity,
                self.available_capacity + quantum_aligned
            )

            released = self.available_capacity - old_capacity
            logger.debug(
                f"Released {released} units of {self.name}",
                extra={"component": "quantum_resource", "operation": "release",
                      "resource": self.name, "amount": released}
            )
        except Exception as e:
            context = ErrorContext(
                component="quantum_resource",
                operation="release",
                resource=self.name
            )
            handle_quantum_error(e, context, reraise=False)


class QuantumAnnealer:
    """Quantum annealing simulator for task optimization with enhanced error handling"""

    def __init__(self, temperature_schedule: Optional[List[float]] = None):
        # Validate and sanitize temperature schedule
        if temperature_schedule is None:
            self.temperature_schedule = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
        else:
            # Validate temperature schedule
            self.temperature_schedule = []
            for temp in temperature_schedule:
                if not isinstance(temp, (int, float)) or temp <= 0:
                    logger.warning(
                        f"Invalid temperature {temp}, using default",
                        extra={"component": "quantum_annealer", "operation": "__init__"}
                    )
                    continue
                self.temperature_schedule.append(float(temp))

            # Ensure we have at least one temperature
            if not self.temperature_schedule:
                self.temperature_schedule = [1.0]

        self.current_temperature = self.temperature_schedule[0]
        self.iteration = 0
        self._lock = threading.RLock()

    @validate_input(
        lambda self, task_schedule: isinstance(task_schedule, list) and all(hasattr(t, 'estimated_duration') for t in task_schedule),
        "Invalid task schedule for energy calculation"
    )
    def calculate_energy(self, task_schedule: List[QuantumTask]) -> float:
        """Calculate system energy for given task schedule with comprehensive validation"""
        try:
            with self._lock:
                energy = 0.0

                if not task_schedule:
                    return energy

                # Minimize total completion time with bounds checking
                try:
                    total_time = sum(
                        max(0.0, task.estimated_duration) for task in task_schedule
                        if hasattr(task, 'estimated_duration') and
                           isinstance(task.estimated_duration, (int, float))
                    )
                    energy += total_time * 0.5
                except (TypeError, AttributeError) as e:
                    logger.warning(
                        f"Error calculating total time: {e}",
                        extra={"component": "quantum_annealer", "operation": "calculate_energy"}
                    )
                    energy += 1000.0  # High penalty for invalid tasks

                # Penalize dependency violations with validation
                completed = set()
                for task in task_schedule:
                    try:
                        if hasattr(task, 'dependencies') and hasattr(task, 'id'):
                            if not task.dependencies.issubset(completed):
                                missing_deps = len(task.dependencies - completed)
                                energy += missing_deps * 100.0  # Heavy penalty per missing dependency
                            completed.add(task.id)
                    except (AttributeError, TypeError) as e:
                        logger.warning(
                            f"Error processing task dependencies: {e}",
                            extra={"component": "quantum_annealer", "operation": "calculate_energy"}
                        )
                        energy += 100.0  # Penalty for invalid task

                # Minimize resource contention with validation
                resource_usage = defaultdict(float)
                for task in task_schedule:
                    try:
                        if hasattr(task, 'resource_requirements') and isinstance(task.resource_requirements, dict):
                            for resource, amount in task.resource_requirements.items():
                                if isinstance(amount, (int, float)) and amount >= 0:
                                    resource_usage[resource] += amount
                                    if resource_usage[resource] > 1.0:
                                        energy += (resource_usage[resource] - 1.0) * 50.0
                    except (AttributeError, TypeError) as e:
                        logger.warning(
                            f"Error processing task resources: {e}",
                            extra={"component": "quantum_annealer", "operation": "calculate_energy"}
                        )
                        energy += 50.0  # Penalty for invalid resource requirements

                # Minimize decoherence with validation
                for task in task_schedule:
                    try:
                        if hasattr(task, 'measure_decoherence'):
                            decoherence = task.measure_decoherence()
                            if isinstance(decoherence, (int, float)) and 0 <= decoherence <= 1:
                                energy += decoherence * 10.0
                            else:
                                energy += 10.0  # Assume worst case
                    except Exception as e:
                        logger.warning(
                            f"Error measuring task decoherence: {e}",
                            extra={"component": "quantum_annealer", "operation": "calculate_energy"}
                        )
                        energy += 10.0  # Penalty for unmeasurable decoherence

                return max(0.0, energy)  # Ensure non-negative energy

        except Exception as e:
            context = ErrorContext(
                component="quantum_annealer",
                operation="calculate_energy"
            )
            handle_quantum_error(e, context, reraise=False)
            return float('inf')  # Return high energy for error cases

    @validate_input(
        lambda self, tasks, max_iterations=1000: (
            isinstance(tasks, list) and
            isinstance(max_iterations, int) and max_iterations > 0
        ),
        "Invalid parameters for annealing schedule"
    )
    @quantum_operation("anneal_schedule", timeout_seconds=300.0)
    def anneal_schedule(self, tasks: List[QuantumTask], max_iterations: int = 1000) -> List[QuantumTask]:
        """Use quantum annealing to optimize task schedule with enhanced error handling"""
        try:
            with self._lock:
                if not tasks:
                    logger.info(
                        "No tasks to anneal",
                        extra={"component": "quantum_annealer", "operation": "anneal_schedule"}
                    )
                    return []

                # Validate and sanitize max_iterations
                max_iterations = max(1, min(max_iterations, 10000))  # Reasonable bounds

                current_schedule = tasks.copy()
                current_energy = self.calculate_energy(current_schedule)
                best_schedule = current_schedule.copy()
                best_energy = current_energy

                # Track convergence
                improvement_count = 0
                stagnation_count = 0

                logger.info(
                    f"Starting quantum annealing with {len(tasks)} tasks, {max_iterations} iterations",
                    extra={"component": "quantum_annealer", "operation": "anneal_schedule",
                          "task_count": len(tasks), "initial_energy": current_energy}
                )

                for iteration in range(max_iterations):
                    try:
                        # Update temperature with bounds checking
                        temp_idx = min(
                            iteration // max(1, (max_iterations // len(self.temperature_schedule))),
                            len(self.temperature_schedule) - 1
                        )
                        self.current_temperature = self.temperature_schedule[temp_idx]
                        self.iteration = iteration

                        # Generate neighbor solution (swap two tasks)
                        new_schedule = current_schedule.copy()
                        if len(new_schedule) > 1:
                            try:
                                i, j = np.random.choice(len(new_schedule), 2, replace=False)
                                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
                            except (ValueError, IndexError) as e:
                                logger.warning(
                                    f"Error generating neighbor solution: {e}",
                                    extra={"component": "quantum_annealer", "operation": "anneal_schedule"}
                                )
                                continue

                        # Calculate energy difference
                        new_energy = self.calculate_energy(new_schedule)
                        if new_energy == float('inf'):
                            continue  # Skip invalid schedules

                        delta_energy = new_energy - current_energy

                        # Accept or reject based on quantum annealing probability
                        acceptance_probability = 0.0
                        if delta_energy < 0:
                            acceptance_probability = 1.0
                        elif self.current_temperature > 0:
                            try:
                                acceptance_probability = np.exp(-delta_energy / self.current_temperature)
                            except (OverflowError, ZeroDivisionError):
                                acceptance_probability = 0.0

                        if acceptance_probability > np.random.random():
                            current_schedule = new_schedule
                            current_energy = new_energy

                            if new_energy < best_energy:
                                best_schedule = new_schedule.copy()
                                best_energy = new_energy
                                improvement_count += 1
                                stagnation_count = 0

                                logger.debug(
                                    f"New best energy: {best_energy:.2f} at iteration {iteration}",
                                    extra={"component": "quantum_annealer", "operation": "anneal_schedule",
                                          "iteration": iteration, "energy": best_energy}
                                )
                            else:
                                stagnation_count += 1
                        else:
                            stagnation_count += 1

                        # Early termination for convergence
                        if stagnation_count > max_iterations // 10:  # 10% of iterations without improvement
                            logger.info(
                                f"Early termination due to convergence at iteration {iteration}",
                                extra={"component": "quantum_annealer", "operation": "anneal_schedule",
                                      "iteration": iteration}
                            )
                            break

                    except Exception as e:
                        logger.warning(
                            f"Error in annealing iteration {iteration}: {e}",
                            extra={"component": "quantum_annealer", "operation": "anneal_schedule",
                                  "iteration": iteration}
                        )
                        continue  # Skip this iteration

                logger.info(
                    f"Quantum annealing completed. Best energy: {best_energy:.2f}, Improvements: {improvement_count}",
                    extra={"component": "quantum_annealer", "operation": "anneal_schedule",
                          "final_energy": best_energy, "improvements": improvement_count}
                )

                return best_schedule

        except Exception as e:
            context = ErrorContext(
                component="quantum_annealer",
                operation="anneal_schedule"
            )
            handle_quantum_error(e, context, reraise=False)
            return tasks  # Return original schedule on error


class QuantumTaskPlanner:
    """Main quantum-inspired task planner with TPU optimization and enhanced resilience"""

    def __init__(self, resources: Optional[List[QuantumResource]] = None,
                 max_tasks: int = 10000, enable_monitoring: bool = True):
        # Core data structures with thread safety
        self._lock = threading.RLock()
        self.tasks: Dict[str, QuantumTask] = {}
        self.resources: Dict[str, QuantumResource] = {}
        self.completed_tasks: Set[str] = set()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.failed_tasks: Set[str] = set()
        self.max_tasks = max(1, min(max_tasks, 100000))  # Reasonable bounds

        # Enhanced monitoring
        self.enable_monitoring = enable_monitoring
        self.metrics = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'resource_allocations': 0,
            'resource_failures': 0
        }

        # Resource management
        self.resource_manager = ResourceManager()

        # Quantum annealing with error handling
        try:
            self.annealer = QuantumAnnealer()
        except Exception as e:
            logger.warning(
                f"Error initializing quantum annealer, using default: {e}",
                extra={"component": "quantum_planner", "operation": "__init__"}
            )
            self.annealer = QuantumAnnealer([1.0])

        # Initialize resources with validation
        if resources:
            for resource in resources:
                try:
                    if hasattr(resource, 'name') and isinstance(resource.name, str):
                        self.resources[resource.name] = resource
                        self.resource_manager.register_resource(resource)
                    else:
                        logger.warning(
                            f"Invalid resource object: {resource}",
                            extra={"component": "quantum_planner", "operation": "__init__"}
                        )
                except Exception as e:
                    logger.warning(
                        f"Error registering resource {resource}: {e}",
                        extra={"component": "quantum_planner", "operation": "__init__"}
                    )
        else:
            self._init_default_resources()

        # Quantum coherence tracking with bounds checking
        try:
            self.coherence_matrix = np.eye(0)  # Will expand as tasks are added
        except Exception as e:
            logger.warning(
                f"Error initializing coherence matrix: {e}",
                extra={"component": "quantum_planner", "operation": "__init__"}
            )
            self.coherence_matrix = np.array([[]])

        self.entanglement_graph: Dict[str, Set[str]] = {}

        # Cleanup tracking
        self._cleanup_callbacks: List[Callable] = []

        logger.info(
            f"Quantum task planner initialized with {len(self.resources)} resources",
            extra={"component": "quantum_planner", "operation": "__init__",
                  "resource_count": len(self.resources), "max_tasks": self.max_tasks}
        )

    def _init_default_resources(self) -> None:
        """Initialize default TPU v5 resources with error handling"""
        try:
            default_resources = {
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

            for name, resource in default_resources.items():
                self.resources[name] = resource
                self.resource_manager.register_resource(
                    resource,
                    lambda r=resource: setattr(r, 'available_capacity', r.total_capacity)
                )

            logger.info(
                f"Initialized {len(default_resources)} default resources",
                extra={"component": "quantum_planner", "operation": "_init_default_resources"}
            )

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="_init_default_resources"
            )
            handle_quantum_error(e, context, reraise=False)

            # Fallback minimal resources
            self.resources = {
                "cpu_cores": QuantumResource("cpu_cores", 1.0),
                "memory_gb": QuantumResource("memory_gb", 1.0)
            }

    @validate_input(
        lambda self, task: hasattr(task, 'id') and hasattr(task, 'state'),
        "Invalid QuantumTask object"
    )
    def add_task(self, task: QuantumTask) -> None:
        """Add task to quantum superposition state with comprehensive validation"""
        try:
            with self._lock:
                # Check task limit
                if len(self.tasks) >= self.max_tasks:
                    raise QuantumValidationError(
                        f"Maximum task limit reached ({self.max_tasks})"
                    )

                # Validate task ID
                if not InputValidator.validate_quantum_task_id(task.id):
                    raise QuantumValidationError(f"Invalid task ID format: {task.id}")

                # Check for duplicate task ID
                if task.id in self.tasks:
                    raise QuantumValidationError(f"Task {task.id} already exists")

                # Sanitize task data
                task.name = DataSanitizer.sanitize_string(task.name, max_length=200)
                task.priority = DataSanitizer.sanitize_numeric(task.priority, min_value=0.0, max_value=100.0)
                task.complexity = DataSanitizer.sanitize_numeric(task.complexity, min_value=0.1, max_value=100.0)

                # Validate resource requirements
                if hasattr(task, 'resource_requirements') and task.resource_requirements:
                    task.resource_requirements = DataSanitizer.sanitize_dict(
                        task.resource_requirements, max_keys=20
                    )

                # Add task to system
                self.tasks[task.id] = task
                self._expand_coherence_matrix()

                # Update metrics
                self.metrics['tasks_created'] += 1

                logger.info(
                    f"Added quantum task {task.id} in superposition state",
                    extra={"component": "quantum_planner", "operation": "add_task",
                          "task_id": task.id, "task_name": task.name,
                          "total_tasks": len(self.tasks)}
                )

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="add_task",
                task_id=getattr(task, 'id', 'unknown')
            )
            handle_quantum_error(e, context)

    @sanitize_input(lambda *args, **kwargs: (
        (DataSanitizer.sanitize_string(args[1]) if len(args) > 1 else args[0],
         DataSanitizer.sanitize_string(args[2]) if len(args) > 2 else args[1]) + args[3:],
        {k: (DataSanitizer.sanitize_numeric(v) if isinstance(v, (int, float)) else
             DataSanitizer.sanitize_string(v) if isinstance(v, str) else v)
         for k, v in kwargs.items()}
    ))
    def create_task(
        self,
        task_id: str,
        name: str,
        priority: float = 1.0,
        complexity: float = 1.0,
        dependencies: Optional[Set[str]] = None,
        **kwargs
    ) -> QuantumTask:
        """Create and add a new quantum task with comprehensive validation"""
        try:
            # Input validation
            if not InputValidator.validate_quantum_task_id(task_id):
                raise QuantumValidationError(f"Invalid task ID: {task_id}")

            if not InputValidator.validate_string(name, min_length=1, max_length=200):
                raise QuantumValidationError(f"Invalid task name: {name}")

            if not InputValidator.validate_priority(priority):
                raise QuantumValidationError(f"Invalid priority: {priority}")

            if not InputValidator.validate_complexity(complexity):
                raise QuantumValidationError(f"Invalid complexity: {complexity}")

            # Validate dependencies
            validated_dependencies = set()
            if dependencies:
                for dep_id in dependencies:
                    if InputValidator.validate_quantum_task_id(dep_id):
                        validated_dependencies.add(dep_id)
                    else:
                        logger.warning(
                            f"Invalid dependency ID ignored: {dep_id}",
                            extra={"component": "quantum_planner", "operation": "create_task",
                                  "task_id": task_id}
                        )

            # Validate kwargs
            validated_kwargs = {}
            for key, value in kwargs.items():
                clean_key = DataSanitizer.sanitize_string(key, max_length=50)
                if key == 'estimated_duration':
                    validated_kwargs[clean_key] = DataSanitizer.sanitize_numeric(value, min_value=0.1)
                elif key == 'decoherence_time':
                    validated_kwargs[clean_key] = DataSanitizer.sanitize_numeric(value, min_value=1.0)
                elif isinstance(value, (int, float)):
                    validated_kwargs[clean_key] = DataSanitizer.sanitize_numeric(value)
                elif isinstance(value, str):
                    validated_kwargs[clean_key] = DataSanitizer.sanitize_string(value)
                elif isinstance(value, dict):
                    validated_kwargs[clean_key] = DataSanitizer.sanitize_dict(value)
                else:
                    validated_kwargs[clean_key] = value

            task = QuantumTask(
                id=task_id,
                name=name,
                priority=priority,
                complexity=complexity,
                dependencies=validated_dependencies,
                **validated_kwargs
            )

            self.add_task(task)
            return task

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="create_task",
                task_id=task_id
            )
            handle_quantum_error(e, context)

    @validate_input(
        lambda self, task1_id, task2_id: (
            InputValidator.validate_quantum_task_id(task1_id) and
            InputValidator.validate_quantum_task_id(task2_id) and
            task1_id != task2_id
        ),
        "Invalid task IDs for entanglement"
    )
    def entangle_tasks(self, task1_id: str, task2_id: str) -> None:
        """Create quantum entanglement between two tasks with comprehensive validation"""
        try:
            with self._lock:
                # Validate tasks exist
                if task1_id not in self.tasks:
                    raise QuantumEntanglementError(f"Task {task1_id} not found")
                if task2_id not in self.tasks:
                    raise QuantumEntanglementError(f"Task {task2_id} not found")

                task1 = self.tasks[task1_id]
                task2 = self.tasks[task2_id]

                # Check entanglement limits
                if len(task1.entangled_tasks) >= 5:
                    raise QuantumEntanglementError(
                        f"Task {task1_id} has too many entanglements ({len(task1.entangled_tasks)})"
                    )
                if len(task2.entangled_tasks) >= 5:
                    raise QuantumEntanglementError(
                        f"Task {task2_id} has too many entanglements ({len(task2.entangled_tasks)})"
                    )

                # Check for existing entanglement
                if task2_id in task1.entangled_tasks:
                    logger.warning(
                        f"Tasks {task1_id} and {task2_id} are already entangled",
                        extra={"component": "quantum_planner", "operation": "entangle_tasks"}
                    )
                    return

                # Create entanglement
                task1.entangle_with(task2_id)
                task2.entangle_with(task1_id)

                # Update entanglement graph
                self.entanglement_graph.setdefault(task1_id, set()).add(task2_id)
                self.entanglement_graph.setdefault(task2_id, set()).add(task1_id)

                logger.info(
                    f"Quantum entanglement created between {task1_id} and {task2_id}",
                    extra={"component": "quantum_planner", "operation": "entangle_tasks",
                          "task1_id": task1_id, "task2_id": task2_id}
                )

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="entangle_tasks"
            )
            handle_quantum_error(e, context)

    def _expand_coherence_matrix(self) -> None:
        """Expand coherence matrix for new tasks with error handling"""
        try:
            n_tasks = len(self.tasks)
            if n_tasks > 1000:  # Prevent excessive memory usage
                logger.warning(
                    f"Large number of tasks ({n_tasks}), coherence matrix may consume significant memory",
                    extra={"component": "quantum_planner", "operation": "_expand_coherence_matrix"}
                )

            if self.coherence_matrix.shape[0] < n_tasks:
                try:
                    new_matrix = np.eye(n_tasks, dtype=np.float32)  # Use float32 to save memory
                    old_size = self.coherence_matrix.shape[0]
                    if old_size > 0:
                        new_matrix[:old_size, :old_size] = self.coherence_matrix
                    self.coherence_matrix = new_matrix
                except MemoryError as e:
                    logger.error(
                        f"Memory error expanding coherence matrix to {n_tasks}x{n_tasks}: {e}",
                        extra={"component": "quantum_planner", "operation": "_expand_coherence_matrix"}
                    )
                    # Fallback to smaller matrix or disable coherence tracking
                    self.coherence_matrix = np.eye(min(100, n_tasks), dtype=np.float32)

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="_expand_coherence_matrix"
            )
            handle_quantum_error(e, context, reraise=False)

    @validate_input(
        lambda self, task: hasattr(task, 'priority') and hasattr(task, 'complexity'),
        "Invalid task for priority calculation"
    )
    def calculate_quantum_priority(self, task: QuantumTask) -> float:
        """Calculate quantum-enhanced priority using superposition principles with bounds checking"""
        try:
            # Validate base priority
            base_priority = max(0.0, min(100.0, task.priority))

            # Complexity factor (more complex = higher priority for early scheduling)
            complexity_value = max(0.1, min(100.0, task.complexity))
            complexity_factor = 1.0 + (complexity_value - 1.0) * 0.5

            # Dependency factor (more dependencies = higher priority)
            dependency_count = len(task.dependencies) if hasattr(task, 'dependencies') else 0
            dependency_factor = 1.0 + min(10, dependency_count) * 0.2  # Cap at 10 dependencies

            # Entanglement factor (entangled tasks get priority boost)
            entanglement_count = len(task.entangled_tasks) if hasattr(task, 'entangled_tasks') else 0
            entanglement_factor = 1.0 + min(5, entanglement_count) * 0.3  # Cap at 5 entanglements

            # Decoherence penalty (tasks near decoherence get priority)
            try:
                decoherence = task.measure_decoherence()
                decoherence_penalty = 1.0 + max(0.0, min(1.0, decoherence)) * 2.0
            except Exception:
                decoherence_penalty = 1.0  # Default if decoherence measurement fails

            # Quantum amplitude influence with bounds checking
            try:
                amplitude = abs(task.probability_amplitude)
                amplitude_factor = max(0.001, min(10.0, amplitude ** 2))  # Reasonable bounds
            except (AttributeError, TypeError, ArithmeticError):
                amplitude_factor = 1.0  # Default amplitude factor

            quantum_priority = (
                base_priority *
                complexity_factor *
                dependency_factor *
                entanglement_factor *
                decoherence_penalty *
                amplitude_factor
            )

            # Final bounds check
            quantum_priority = max(0.0, min(10000.0, quantum_priority))

            logger.debug(
                f"Calculated quantum priority for task {task.id}: {quantum_priority:.2f}",
                extra={"component": "quantum_planner", "operation": "calculate_quantum_priority",
                      "task_id": task.id, "priority": quantum_priority,
                      "factors": {
                          "base": base_priority, "complexity": complexity_factor,
                          "dependency": dependency_factor, "entanglement": entanglement_factor,
                          "decoherence": decoherence_penalty, "amplitude": amplitude_factor
                      }}
            )

            return quantum_priority

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="calculate_quantum_priority",
                task_id=getattr(task, 'id', 'unknown')
            )
            handle_quantum_error(e, context, reraise=False)
            return 1.0  # Safe default priority

    @quantum_operation("get_ready_tasks", circuit_breaker=False, retry_attempts=1)
    def get_ready_tasks(self) -> List[QuantumTask]:
        """Get tasks ready for execution using quantum selection with error handling"""
        try:
            with self._lock:
                ready_tasks = []

                for task in self.tasks.values():
                    try:
                        if task.is_ready_for_execution(self.completed_tasks):
                            ready_tasks.append(task)
                    except Exception as e:
                        logger.warning(
                            f"Error checking readiness for task {task.id}: {e}",
                            extra={"component": "quantum_planner", "operation": "get_ready_tasks",
                                  "task_id": task.id}
                        )
                        continue  # Skip problematic tasks

                # Sort by quantum priority with error handling
                try:
                    ready_tasks.sort(key=self.calculate_quantum_priority, reverse=True)
                except Exception as e:
                    logger.warning(
                        f"Error sorting tasks by priority: {e}",
                        extra={"component": "quantum_planner", "operation": "get_ready_tasks"}
                    )
                    # Keep original order if sorting fails

                logger.debug(
                    f"Found {len(ready_tasks)} ready tasks",
                    extra={"component": "quantum_planner", "operation": "get_ready_tasks",
                          "ready_count": len(ready_tasks), "total_tasks": len(self.tasks)}
                )

                return ready_tasks

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="get_ready_tasks"
            )
            handle_quantum_error(e, context, reraise=False)
            return []  # Safe default

    @quantum_operation("optimize_schedule", retry_attempts=2, timeout_seconds=60.0)
    def optimize_schedule(self) -> List[QuantumTask]:
        """Use quantum annealing to optimize task execution schedule with resilience"""
        try:
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                logger.debug(
                    "No ready tasks for optimization",
                    extra={"component": "quantum_planner", "operation": "optimize_schedule"}
                )
                return []

            logger.info(
                f"Optimizing schedule for {len(ready_tasks)} ready tasks",
                extra={"component": "quantum_planner", "operation": "optimize_schedule",
                      "task_count": len(ready_tasks)}
            )

            # Use quantum annealer with error handling
            try:
                optimized_schedule = self.annealer.anneal_schedule(ready_tasks)
            except Exception as e:
                logger.error(
                    f"Quantum annealing failed, using priority order: {e}",
                    extra={"component": "quantum_planner", "operation": "optimize_schedule"}
                )
                # Fallback to simple priority ordering
                optimized_schedule = ready_tasks.copy()

            # Collapse wavefunctions for scheduled tasks with error handling
            successfully_collapsed = 0
            for i, task in enumerate(optimized_schedule):
                try:
                    task.collapse_wavefunction(f"schedule_position_{i}")
                    successfully_collapsed += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to collapse wavefunction for task {task.id}: {e}",
                        extra={"component": "quantum_planner", "operation": "optimize_schedule",
                              "task_id": task.id}
                    )
                    # Continue with other tasks

            logger.info(
                f"Schedule optimization completed: {len(optimized_schedule)} tasks, "
                f"{successfully_collapsed} wavefunctions collapsed",
                extra={"component": "quantum_planner", "operation": "optimize_schedule",
                      "scheduled_tasks": len(optimized_schedule),
                      "collapsed_tasks": successfully_collapsed}
            )

            return optimized_schedule

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="optimize_schedule"
            )
            handle_quantum_error(e, context, reraise=False)
            return []  # Safe default

    @validate_input(
        lambda self, task: hasattr(task, 'resource_requirements'),
        "Task missing resource requirements"
    )
    @quantum_operation("can_allocate_resources", circuit_breaker=False)
    def can_allocate_resources(self, task: QuantumTask) -> bool:
        """Check if resources can be allocated for task execution with validation"""
        try:
            if not hasattr(task, 'resource_requirements') or not task.resource_requirements:
                return True  # No resources required

            for resource_name, required_amount in task.resource_requirements.items():
                try:
                    # Validate resource name and amount
                    if not isinstance(resource_name, str) or not resource_name:
                        logger.warning(
                            f"Invalid resource name for task {task.id}: {resource_name}",
                            extra={"component": "quantum_planner", "operation": "can_allocate_resources",
                                  "task_id": task.id}
                        )
                        continue

                    if not isinstance(required_amount, (int, float)) or required_amount < 0:
                        logger.warning(
                            f"Invalid resource amount for task {task.id}: {required_amount}",
                            extra={"component": "quantum_planner", "operation": "can_allocate_resources",
                                  "task_id": task.id, "resource": resource_name}
                        )
                        continue

                    if resource_name in self.resources:
                        resource = self.resources[resource_name]
                        if not resource.can_allocate(required_amount):
                            logger.debug(
                                f"Insufficient {resource_name} for task {task.id}: "
                                f"need {required_amount}, available {resource.available_capacity}",
                                extra={"component": "quantum_planner", "operation": "can_allocate_resources",
                                      "task_id": task.id, "resource": resource_name,
                                      "required": required_amount, "available": resource.available_capacity}
                            )
                            return False
                    else:
                        logger.warning(
                            f"Unknown resource {resource_name} required by task {task.id}",
                            extra={"component": "quantum_planner", "operation": "can_allocate_resources",
                                  "task_id": task.id, "resource": resource_name}
                        )
                        return False

                except Exception as e:
                    logger.error(
                        f"Error checking resource {resource_name} for task {task.id}: {e}",
                        extra={"component": "quantum_planner", "operation": "can_allocate_resources",
                              "task_id": task.id, "resource": resource_name}
                    )
                    return False

            return True

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="can_allocate_resources",
                task_id=getattr(task, 'id', 'unknown')
            )
            handle_quantum_error(e, context, reraise=False)
            return False  # Safe default

    @validate_input(
        lambda self, task: hasattr(task, 'resource_requirements'),
        "Task missing resource requirements"
    )
    @quantum_operation("allocate_task_resources", retry_attempts=2)
    def allocate_task_resources(self, task: QuantumTask) -> bool:
        """Allocate resources for task execution with comprehensive error handling and rollback"""
        allocated_resources = []

        try:
            with self._lock:
                if not hasattr(task, 'resource_requirements') or not task.resource_requirements:
                    return True  # No resources to allocate

                # Pre-check all resources to avoid partial allocation failures
                for resource_name, required_amount in task.resource_requirements.items():
                    if resource_name not in self.resources:
                        logger.error(
                            f"Resource {resource_name} not available for task {task.id}",
                            extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                                  "task_id": task.id, "resource": resource_name}
                        )
                        return False

                    resource = self.resources[resource_name]
                    if not resource.can_allocate(required_amount):
                        logger.debug(
                            f"Cannot allocate {required_amount} of {resource_name} for task {task.id}",
                            extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                                  "task_id": task.id, "resource": resource_name,
                                  "required": required_amount, "available": resource.available_capacity}
                        )
                        return False

                # Perform actual allocations
                for resource_name, required_amount in task.resource_requirements.items():
                    try:
                        resource = self.resources[resource_name]
                        if resource.allocate(required_amount):
                            allocated_resources.append((resource_name, required_amount))
                            logger.debug(
                                f"Allocated {required_amount} of {resource_name} for task {task.id}",
                                extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                                      "task_id": task.id, "resource": resource_name, "amount": required_amount}
                            )
                        else:
                            # This should not happen due to pre-check, but handle it anyway
                            logger.error(
                                f"Allocation failed for {resource_name} despite pre-check",
                                extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                                      "task_id": task.id, "resource": resource_name}
                            )
                            # Rollback all previous allocations
                            self._rollback_allocations(allocated_resources)
                            return False
                    except Exception as e:
                        logger.error(
                            f"Exception during allocation of {resource_name} for task {task.id}: {e}",
                            extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                                  "task_id": task.id, "resource": resource_name}
                        )
                        # Rollback all previous allocations
                        self._rollback_allocations(allocated_resources)
                        return False

                # Update metrics
                self.metrics['resource_allocations'] += len(allocated_resources)

                logger.info(
                    f"Successfully allocated {len(allocated_resources)} resources for task {task.id}",
                    extra={"component": "quantum_planner", "operation": "allocate_task_resources",
                          "task_id": task.id, "resource_count": len(allocated_resources)}
                )

                return True

        except Exception as e:
            # Rollback all allocations on any error
            self._rollback_allocations(allocated_resources)
            self.metrics['resource_failures'] += 1

            context = ErrorContext(
                component="quantum_planner",
                operation="allocate_task_resources",
                task_id=getattr(task, 'id', 'unknown')
            )
            handle_quantum_error(e, context, reraise=False)
            return False

    def _rollback_allocations(self, allocated_resources: List[Tuple[str, float]]) -> None:
        """Rollback resource allocations safely"""
        for res_name, amount in allocated_resources:
            try:
                if res_name in self.resources:
                    self.resources[res_name].release(amount)
                    logger.debug(
                        f"Rolled back allocation of {amount} {res_name}",
                        extra={"component": "quantum_planner", "operation": "_rollback_allocations",
                              "resource": res_name, "amount": amount}
                    )
            except Exception as e:
                logger.error(
                    f"Error rolling back allocation of {amount} {res_name}: {e}",
                    extra={"component": "quantum_planner", "operation": "_rollback_allocations",
                          "resource": res_name}
                )

    @validate_input(
        lambda self, task: hasattr(task, 'resource_requirements'),
        "Task missing resource requirements"
    )
    def release_task_resources(self, task: QuantumTask) -> None:
        """Release resources after task completion with comprehensive error handling"""
        try:
            with self._lock:
                if not hasattr(task, 'resource_requirements') or not task.resource_requirements:
                    return  # No resources to release

                released_count = 0
                for resource_name, amount in task.resource_requirements.items():
                    try:
                        if resource_name in self.resources and isinstance(amount, (int, float)) and amount > 0:
                            self.resources[resource_name].release(amount)
                            released_count += 1
                            logger.debug(
                                f"Released {amount} of {resource_name} from task {task.id}",
                                extra={"component": "quantum_planner", "operation": "release_task_resources",
                                      "task_id": task.id, "resource": resource_name, "amount": amount}
                            )
                        else:
                            logger.warning(
                                f"Cannot release resource {resource_name} (amount: {amount}) for task {task.id}",
                                extra={"component": "quantum_planner", "operation": "release_task_resources",
                                      "task_id": task.id, "resource": resource_name}
                            )
                    except Exception as e:
                        logger.error(
                            f"Error releasing resource {resource_name} for task {task.id}: {e}",
                            extra={"component": "quantum_planner", "operation": "release_task_resources",
                                  "task_id": task.id, "resource": resource_name}
                        )
                        continue  # Continue with other resources

                logger.debug(
                    f"Released {released_count} resources for task {task.id}",
                    extra={"component": "quantum_planner", "operation": "release_task_resources",
                          "task_id": task.id, "released_count": released_count}
                )

        except Exception as e:
            context = ErrorContext(
                component="quantum_planner",
                operation="release_task_resources",
                task_id=getattr(task, 'id', 'unknown')
            )
            handle_quantum_error(e, context, reraise=False)

    @validate_input(
        lambda self, task: hasattr(task, 'id') and hasattr(task, 'estimated_duration'),
        "Invalid task for execution"
    )
    async def execute_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute a quantum task with TPU optimization and comprehensive error handling"""
        start_time = time.time()
        task_id = getattr(task, 'id', 'unknown')
        execution_record = None

        async with AsyncErrorHandlingContext(
            component="quantum_planner",
            operation="execute_task",
            task_id=task_id,
            suppress_exceptions=True
        ) as error_ctx:

            try:
                # Validate task state before execution
                if not hasattr(task, 'state') or task.state == QuantumState.DECOHERENT:
                    raise QuantumStateError(f"Task {task_id} is in invalid state for execution")

                # Check decoherence level
                try:
                    decoherence = task.measure_decoherence()
                    if decoherence >= 0.9:
                        raise QuantumDecoherenceError(
                            f"Task {task_id} is too decoherent for execution: {decoherence:.1%}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Cannot measure decoherence for task {task_id}: {e}",
                        extra={"component": "quantum_planner", "operation": "execute_task",
                              "task_id": task_id}
                    )
                    decoherence = 0.5  # Assume moderate decoherence

                # Validate and sanitize execution parameters
                execution_time = getattr(task, 'estimated_duration', 1.0)
                if not isinstance(execution_time, (int, float)) or execution_time <= 0:
                    execution_time = 1.0
                    logger.warning(
                        f"Invalid estimated_duration for task {task_id}, using default: 1.0s",
                        extra={"component": "quantum_planner", "operation": "execute_task",
                              "task_id": task_id}
                    )

                complexity = getattr(task, 'complexity', 1.0)
                if not isinstance(complexity, (int, float)) or complexity <= 0:
                    complexity = 1.0

                logger.info(
                    f"Starting execution of quantum task {task_id}",
                    extra={"component": "quantum_planner", "operation": "execute_task",
                          "task_id": task_id, "estimated_duration": execution_time,
                          "complexity": complexity, "decoherence": decoherence}
                )

                # Add quantum noise based on complexity with bounds checking
                try:
                    quantum_noise = np.random.normal(0, min(complexity * 0.1, 1.0))
                    quantum_noise = max(-execution_time * 0.5, min(execution_time * 0.5, quantum_noise))
                except Exception as e:
                    logger.warning(
                        f"Error generating quantum noise for task {task_id}: {e}",
                        extra={"component": "quantum_planner", "operation": "execute_task",
                              "task_id": task_id}
                    )
                    quantum_noise = 0.0

                actual_execution_time = max(0.1, min(execution_time + quantum_noise, 300.0))  # Cap at 5 minutes

                # Simulate execution delay with cancellation support
                simulation_time = min(actual_execution_time, 0.5)  # Cap simulation time
                try:
                    await asyncio.sleep(simulation_time)
                except asyncio.CancelledError:
                    logger.info(
                        f"Task {task_id} execution was cancelled",
                        extra={"component": "quantum_planner", "operation": "execute_task",
                              "task_id": task_id}
                    )
                    raise

                # Update task state safely
                if hasattr(task, 'state'):
                    old_state = task.state
                    task.state = QuantumState.COLLAPSED
                    logger.debug(
                        f"Task {task_id} state changed from {old_state.value} to {QuantumState.COLLAPSED.value}",
                        extra={"component": "quantum_planner", "operation": "execute_task",
                              "task_id": task_id}
                    )

                # Record execution history
                end_time = time.time()
                execution_record = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "actual_duration": end_time - start_time,
                    "estimated_duration": execution_time,
                    "quantum_noise": quantum_noise,
                    "success": True,
                    "decoherence_at_start": decoherence
                }

                if hasattr(task, 'execution_history'):
                    if not isinstance(task.execution_history, list):
                        task.execution_history = []
                    task.execution_history.append(execution_record)
                    # Limit history to prevent memory bloat
                    if len(task.execution_history) > 100:
                        task.execution_history = task.execution_history[-50:]

                # Update metrics
                self.metrics['tasks_completed'] += 1

                logger.info(
                    f"Quantum task {task_id} executed successfully in {end_time - start_time:.2f}s",
                    extra={"component": "quantum_planner", "operation": "execute_task",
                          "task_id": task_id, "duration": end_time - start_time,
                          "quantum_noise": quantum_noise}
                )

                return {
                    "task_id": task_id,
                    "success": True,
                    "duration": end_time - start_time,
                    "quantum_effects": {
                        "noise": quantum_noise,
                        "decoherence": decoherence,
                        "final_decoherence": task.measure_decoherence() if hasattr(task, 'measure_decoherence') else 0.0
                    },
                    "execution_record": execution_record
                }

            except asyncio.CancelledError:
                # Handle cancellation gracefully
                if hasattr(task, 'state'):
                    task.state = QuantumState.DECOHERENT
                self.metrics['tasks_failed'] += 1

                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "Task execution cancelled",
                    "duration": time.time() - start_time,
                    "cancelled": True
                }

            except Exception as e:
                # Handle all other exceptions
                if hasattr(task, 'state'):
                    task.state = QuantumState.DECOHERENT

                # Record failed execution
                execution_record = {
                    "start_time": start_time,
                    "end_time": time.time(),
                    "actual_duration": time.time() - start_time,
                    "estimated_duration": getattr(task, 'estimated_duration', 1.0),
                    "success": False,
                    "error": str(e),
                    "exception_type": type(e).__name__
                }

                if hasattr(task, 'execution_history'):
                    if not isinstance(task.execution_history, list):
                        task.execution_history = []
                    task.execution_history.append(execution_record)

                self.metrics['tasks_failed'] += 1
                self.failed_tasks.add(task_id)

                logger.error(
                    f"Quantum task {task_id} execution failed: {e}",
                    extra={"component": "quantum_planner", "operation": "execute_task",
                          "task_id": task_id, "error": str(e), "exception_type": type(e).__name__}
                )

                return {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - start_time,
                    "exception_type": type(e).__name__,
                    "execution_record": execution_record
                }

        # This should not be reached, but included for safety
        return {
            "task_id": task_id,
            "success": False,
            "error": "Unexpected execution path",
            "duration": time.time() - start_time
        }

    @quantum_operation("run_quantum_execution_cycle", retry_attempts=1, timeout_seconds=120.0)
    async def run_quantum_execution_cycle(self, max_concurrent_tasks: int = 3) -> Dict[str, Any]:
        """Run one quantum execution cycle with enhanced error handling and resource management"""
        cycle_start = time.time()

        # Initialize results with comprehensive structure
        results = {
            "cycle_start": cycle_start,
            "cycle_id": f"cycle_{int(cycle_start)}",
            "tasks_executed": [],
            "tasks_failed": [],
            "tasks_cancelled": [],
            "resource_utilization": {},
            "quantum_coherence": 0.0,
            "metrics": {},
            "errors": []
        }

        # Resource manager for cleanup
        async with AsyncErrorHandlingContext(
            component="quantum_planner",
            operation="run_quantum_execution_cycle",
            suppress_exceptions=True
        ) as error_ctx:

            try:
                # Validate max_concurrent_tasks
                max_concurrent_tasks = max(1, min(max_concurrent_tasks, 10))

                logger.info(
                    f"Starting quantum execution cycle with max {max_concurrent_tasks} concurrent tasks",
                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                          "max_concurrent": max_concurrent_tasks}
                )

                # Get optimized schedule with error handling
                try:
                    optimized_schedule = self.optimize_schedule()
                except Exception as e:
                    logger.error(
                        f"Error optimizing schedule: {e}",
                        extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle"}
                    )
                    optimized_schedule = []
                    results["errors"].append({"stage": "schedule_optimization", "error": str(e)})

                if not optimized_schedule:
                    logger.info(
                        "No tasks ready for execution in this cycle",
                        extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle"}
                    )
                    results["metrics"] = self._calculate_cycle_metrics(results)
                    return results

                # Execute tasks with resource constraints and error handling
                executing_tasks = []
                resource_allocated_tasks = []

                for task in optimized_schedule:
                    try:
                        if len(executing_tasks) >= max_concurrent_tasks:
                            logger.debug(
                                f"Reached maximum concurrent tasks ({max_concurrent_tasks}), skipping remaining",
                                extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle"}
                            )
                            break

                        # Check and allocate resources
                        if self.can_allocate_resources(task):
                            if self.allocate_task_resources(task):
                                resource_allocated_tasks.append(task)

                                # Start task execution
                                task_coroutine = self.execute_task(task)
                                executing_tasks.append((task, task_coroutine))

                                logger.info(
                                    f"Started execution of quantum task {task.id}",
                                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                          "task_id": task.id}
                                )
                            else:
                                logger.debug(
                                    f"Failed to allocate resources for task {task.id}",
                                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                          "task_id": task.id}
                                )
                        else:
                            logger.debug(
                                f"Insufficient resources for task {task.id}",
                                extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                      "task_id": task.id}
                            )
                    except Exception as e:
                        logger.error(
                            f"Error preparing task {task.id} for execution: {e}",
                            extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                  "task_id": task.id}
                        )
                        results["errors"].append({"stage": "task_preparation", "task_id": task.id, "error": str(e)})
                        continue

                # Wait for task completions with timeout and error handling
                if executing_tasks:
                    try:
                        logger.info(
                            f"Waiting for {len(executing_tasks)} tasks to complete",
                            extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                  "executing_count": len(executing_tasks)}
                        )

                        task_results = await asyncio.gather(
                            *[task_coro for _, task_coro in executing_tasks],
                            return_exceptions=True
                        )

                        # Process results with comprehensive error handling
                        for (task, _), result in zip(executing_tasks, task_results):
                            try:
                                if isinstance(result, dict):
                                    if result.get("success"):
                                        results["tasks_executed"].append(result)
                                        self.completed_tasks.add(task.id)

                                        # Handle entangled tasks safely
                                        self._handle_entangled_tasks(task)

                                    elif result.get("cancelled"):
                                        results["tasks_cancelled"].append(result)
                                        logger.info(
                                            f"Task {task.id} was cancelled",
                                            extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                                  "task_id": task.id}
                                        )
                                    else:
                                        results["tasks_failed"].append(result)
                                        logger.warning(
                                            f"Task {task.id} failed: {result.get('error', 'Unknown error')}",
                                            extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                                  "task_id": task.id}
                                        )

                                elif isinstance(result, Exception):
                                    error_result = {
                                        "task_id": task.id,
                                        "error": str(result),
                                        "exception_type": type(result).__name__
                                    }
                                    results["tasks_failed"].append(error_result)
                                    logger.error(
                                        f"Task {task.id} raised exception: {result}",
                                        extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                              "task_id": task.id, "exception_type": type(result).__name__}
                                    )

                                else:
                                    # Unexpected result type
                                    error_result = {
                                        "task_id": task.id,
                                        "error": f"Unexpected result type: {type(result)}",
                                        "result": str(result)
                                    }
                                    results["tasks_failed"].append(error_result)

                            except Exception as e:
                                logger.error(
                                    f"Error processing result for task {task.id}: {e}",
                                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                          "task_id": task.id}
                                )
                                results["errors"].append({"stage": "result_processing", "task_id": task.id, "error": str(e)})

                            finally:
                                # Always release resources
                                try:
                                    self.release_task_resources(task)
                                except Exception as e:
                                    logger.error(
                                        f"Error releasing resources for task {task.id}: {e}",
                                        extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                                              "task_id": task.id}
                                    )

                    except Exception as e:
                        logger.error(
                            f"Error during task execution phase: {e}",
                            extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle"}
                        )
                        results["errors"].append({"stage": "task_execution", "error": str(e)})

                        # Release resources for all tasks in case of failure
                        for task in resource_allocated_tasks:
                            try:
                                self.release_task_resources(task)
                            except Exception:
                                pass  # Best effort cleanup

                # Calculate resource utilization safely
                self._calculate_resource_utilization(results)

                # Calculate quantum coherence safely
                self._calculate_quantum_coherence(results)

                # Calculate comprehensive metrics
                results["metrics"] = self._calculate_cycle_metrics(results)

                cycle_duration = time.time() - cycle_start
                results["cycle_duration"] = cycle_duration

                logger.info(
                    f"Quantum execution cycle completed in {cycle_duration:.2f}s: "
                    f"{len(results['tasks_executed'])} executed, "
                    f"{len(results['tasks_failed'])} failed, "
                    f"{len(results['tasks_cancelled'])} cancelled",
                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle",
                          "duration": cycle_duration, "executed": len(results['tasks_executed']),
                          "failed": len(results['tasks_failed']), "cancelled": len(results['tasks_cancelled'])}
                )

                return results

            except Exception as e:
                logger.error(
                    f"Critical error in quantum execution cycle: {e}",
                    extra={"component": "quantum_planner", "operation": "run_quantum_execution_cycle"}
                )
                results["errors"].append({"stage": "critical", "error": str(e)})
                results["cycle_duration"] = time.time() - cycle_start
                return results

    def _handle_entangled_tasks(self, completed_task: QuantumTask) -> None:
        """Handle quantum entanglement effects when a task completes"""
        try:
            if not hasattr(completed_task, 'entangled_tasks'):
                return

            for entangled_id in completed_task.entangled_tasks:
                try:
                    if entangled_id in self.tasks:
                        entangled_task = self.tasks[entangled_id]
                        if hasattr(entangled_task, 'probability_amplitude'):
                            # Quantum measurement affects entangled tasks
                            old_amplitude = entangled_task.probability_amplitude
                            entangled_task.probability_amplitude *= 0.9

                            logger.debug(
                                f"Entanglement effect: task {entangled_id} amplitude reduced from "
                                f"{abs(old_amplitude):.3f} to {abs(entangled_task.probability_amplitude):.3f}",
                                extra={"component": "quantum_planner", "operation": "_handle_entangled_tasks",
                                      "completed_task": completed_task.id, "entangled_task": entangled_id}
                            )
                except Exception as e:
                    logger.warning(
                        f"Error handling entanglement for task {entangled_id}: {e}",
                        extra={"component": "quantum_planner", "operation": "_handle_entangled_tasks"}
                    )
                    continue
        except Exception as e:
            logger.error(
                f"Error in entanglement handling: {e}",
                extra={"component": "quantum_planner", "operation": "_handle_entangled_tasks"}
            )

    def _calculate_resource_utilization(self, results: Dict[str, Any]) -> None:
        """Calculate resource utilization safely"""
        try:
            resource_utilization = {}
            for name, resource in self.resources.items():
                try:
                    if hasattr(resource, 'available_capacity') and hasattr(resource, 'total_capacity'):
                        if resource.total_capacity > 0:
                            utilization = 1.0 - (resource.available_capacity / resource.total_capacity)
                            resource_utilization[name] = max(0.0, min(1.0, utilization))
                        else:
                            resource_utilization[name] = 0.0
                    else:
                        resource_utilization[name] = 0.0
                except Exception as e:
                    logger.warning(
                        f"Error calculating utilization for resource {name}: {e}",
                        extra={"component": "quantum_planner", "operation": "_calculate_resource_utilization"}
                    )
                    resource_utilization[name] = 0.0

            results["resource_utilization"] = resource_utilization

        except Exception as e:
            logger.error(
                f"Error calculating resource utilization: {e}",
                extra={"component": "quantum_planner", "operation": "_calculate_resource_utilization"}
            )
            results["resource_utilization"] = {}

    def _calculate_quantum_coherence(self, results: Dict[str, Any]) -> None:
        """Calculate quantum coherence safely"""
        try:
            coherent_tasks = []
            total_coherence = 0.0

            for task in self.tasks.values():
                try:
                    if hasattr(task, 'state') and task.state != QuantumState.DECOHERENT:
                        coherent_tasks.append(task)
                        if hasattr(task, 'probability_amplitude'):
                            amplitude = getattr(task, 'probability_amplitude', 1.0)
                            if isinstance(amplitude, (int, float, complex)):
                                total_coherence += abs(amplitude) ** 2
                            else:
                                total_coherence += 1.0  # Default coherence
                except Exception:
                    continue  # Skip problematic tasks

            if coherent_tasks:
                results["quantum_coherence"] = total_coherence / len(coherent_tasks)
            else:
                results["quantum_coherence"] = 0.0

        except Exception as e:
            logger.error(
                f"Error calculating quantum coherence: {e}",
                extra={"component": "quantum_planner", "operation": "_calculate_quantum_coherence"}
            )
            results["quantum_coherence"] = 0.0

    def _calculate_cycle_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive cycle metrics"""
        try:
            return {
                "total_tasks_scheduled": len(results["tasks_executed"]) + len(results["tasks_failed"]) + len(results["tasks_cancelled"]),
                "success_rate": len(results["tasks_executed"]) / max(1, len(results["tasks_executed"]) + len(results["tasks_failed"])),
                "cancellation_rate": len(results["tasks_cancelled"]) / max(1, len(results["tasks_executed"]) + len(results["tasks_failed"]) + len(results["tasks_cancelled"])),
                "error_count": len(results["errors"]),
                "avg_execution_time": (
                    sum(task["duration"] for task in results["tasks_executed"] if "duration" in task) /
                    max(1, len(results["tasks_executed"]))
                ) if results["tasks_executed"] else 0.0,
                "resource_efficiency": (
                    sum(results["resource_utilization"].values()) /
                    max(1, len(results["resource_utilization"]))
                ) if results["resource_utilization"] else 0.0,
                "coherence_level": results.get("quantum_coherence", 0.0)
            }
        except Exception as e:
            logger.error(
                f"Error calculating cycle metrics: {e}",
                extra={"component": "quantum_planner", "operation": "_calculate_cycle_metrics"}
            )
            return {"error": str(e)}

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
