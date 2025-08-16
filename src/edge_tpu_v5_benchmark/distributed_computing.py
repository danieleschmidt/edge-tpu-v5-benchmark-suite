"""Distributed Computing Framework for TPU v5 Benchmark Suite

This module implements scalable distributed computing capabilities including
cloud computing integration, edge computing coordination, federated learning,
and distributed benchmark execution across multiple nodes.
"""

import asyncio
import json
import logging
import threading
import time
import socket
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import psutil
from collections import defaultdict, deque
import concurrent.futures

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logging.warning("Ray not available - distributed computing features limited")

try:
    import kubernetes
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logging.warning("Kubernetes client not available - cluster features disabled")

from .security import SecurityContext
from .robust_error_handling import robust_operation, ErrorRecoveryManager


class NodeType(Enum):
    """Types of compute nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    EDGE_DEVICE = "edge_device"
    CLOUD_INSTANCE = "cloud_instance"
    GPU_NODE = "gpu_node"
    TPU_NODE = "tpu_node"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class ExecutionMode(Enum):
    """Execution modes for distributed tasks."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    FEDERATED = "federated"


@dataclass
class ComputeNode:
    """Representation of a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    hostname: str
    port: int
    capabilities: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    current_load: float = 0.0
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    total_tasks_completed: int = 0
    avg_task_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "hostname": self.hostname,
            "port": self.port,
            "capabilities": self.capabilities,
            "resource_limits": self.resource_limits,
            "current_load": self.current_load,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "total_tasks_completed": self.total_tasks_completed,
            "avg_task_duration": self.avg_task_duration
        }


@dataclass
class DistributedTask:
    """Representation of a distributed computation task."""
    task_id: str
    function_name: str
    function_code: Optional[str] = None
    input_data: Any = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    assigned_node: Optional[str] = None
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "function_name": self.function_name,
            "function_code": self.function_code,
            "input_data": self.input_data,
            "parameters": self.parameters,
            "priority": self.priority.value,
            "execution_mode": self.execution_mode.value,
            "resource_requirements": self.resource_requirements,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "assigned_node": self.assigned_node,
            "status": self.status
        }


@dataclass
class TaskResult:
    """Result of a distributed task execution."""
    task_id: str
    node_id: str
    result_data: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    memory_used: int = 0
    cpu_usage: float = 0.0
    success: bool = True
    completed_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "cpu_usage": self.cpu_usage,
            "success": self.success,
            "completed_at": self.completed_at
        }


class LoadBalancer:
    """Advanced load balancing for distributed tasks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load balancing strategies
        self.strategies = {
            "round_robin": self._round_robin,
            "least_loaded": self._least_loaded,
            "weighted_random": self._weighted_random,
            "capability_based": self._capability_based,
            "performance_based": self._performance_based
        }
        
        self.current_strategy = "performance_based"
        self.round_robin_index = 0
        
    def select_node(self, task: DistributedTask, 
                   available_nodes: List[ComputeNode]) -> Optional[ComputeNode]:
        """Select best node for task execution."""
        if not available_nodes:
            return None
        
        # Filter nodes by capabilities
        capable_nodes = self._filter_capable_nodes(task, available_nodes)
        if not capable_nodes:
            return None
        
        # Apply load balancing strategy
        strategy_func = self.strategies.get(self.current_strategy, self._least_loaded)
        return strategy_func(task, capable_nodes)
    
    def _filter_capable_nodes(self, task: DistributedTask, 
                            nodes: List[ComputeNode]) -> List[ComputeNode]:
        """Filter nodes by capability requirements."""
        capable_nodes = []
        
        for node in nodes:
            if node.status != "active":
                continue
            
            # Check resource requirements
            resource_reqs = task.resource_requirements
            node_limits = node.resource_limits
            
            # Memory check
            required_memory = resource_reqs.get("memory_mb", 0)
            available_memory = node_limits.get("memory_mb", float('inf'))
            if required_memory > available_memory:
                continue
            
            # CPU check
            required_cpu = resource_reqs.get("cpu_cores", 0)
            available_cpu = node_limits.get("cpu_cores", float('inf'))
            if required_cpu > available_cpu:
                continue
            
            # Capability check (e.g., GPU, TPU)
            required_caps = resource_reqs.get("capabilities", [])
            node_caps = node.capabilities.get("hardware", [])
            if not all(cap in node_caps for cap in required_caps):
                continue
            
            capable_nodes.append(node)
        
        return capable_nodes
    
    def _round_robin(self, task: DistributedTask, 
                    nodes: List[ComputeNode]) -> ComputeNode:
        """Round-robin load balancing."""
        selected_node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected_node
    
    def _least_loaded(self, task: DistributedTask, 
                     nodes: List[ComputeNode]) -> ComputeNode:
        """Select node with least current load."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_random(self, task: DistributedTask, 
                        nodes: List[ComputeNode]) -> ComputeNode:
        """Weighted random selection based on inverse load."""
        if not nodes:
            return None
        
        # Calculate weights (inverse of load)
        weights = [1.0 / (node.current_load + 0.1) for node in nodes]
        total_weight = sum(weights)
        
        # Random selection
        import random
        r = random.random() * total_weight
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return nodes[i]
        
        return nodes[-1]  # Fallback
    
    def _capability_based(self, task: DistributedTask, 
                         nodes: List[ComputeNode]) -> ComputeNode:
        """Select node based on specific capabilities."""
        # Score nodes based on capability match
        scored_nodes = []
        
        for node in nodes:
            score = 0
            
            # Bonus for TPU nodes if task requires TPU
            if ("tpu" in task.resource_requirements.get("capabilities", []) and
                "tpu" in node.capabilities.get("hardware", [])):
                score += 10
            
            # Bonus for GPU nodes if task requires GPU
            if ("gpu" in task.resource_requirements.get("capabilities", []) and
                "gpu" in node.capabilities.get("hardware", [])):
                score += 8
            
            # Penalty for high load
            score -= node.current_load * 5
            
            # Bonus for good performance history
            if node.avg_task_duration > 0:
                score += 5.0 / node.avg_task_duration
            
            scored_nodes.append((score, node))
        
        # Select highest scoring node
        return max(scored_nodes, key=lambda x: x[0])[1]
    
    def _performance_based(self, task: DistributedTask, 
                          nodes: List[ComputeNode]) -> ComputeNode:
        """Select node based on historical performance."""
        if not nodes:
            return None
        
        # Calculate performance score for each node
        best_node = None
        best_score = float('-inf')
        
        for node in nodes:
            # Base score inversely related to load
            score = 10.0 / (node.current_load + 1.0)
            
            # Bonus for completed tasks
            if node.total_tasks_completed > 0:
                completion_bonus = min(node.total_tasks_completed / 100.0, 5.0)
                score += completion_bonus
            
            # Bonus for fast execution
            if node.avg_task_duration > 0:
                speed_bonus = min(10.0 / node.avg_task_duration, 5.0)
                score += speed_bonus
            
            # Recent activity bonus
            time_since_heartbeat = time.time() - node.last_heartbeat
            if time_since_heartbeat < 30:  # Active within last 30 seconds
                score += 2.0
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node


class TaskScheduler:
    """Advanced task scheduling with dependency management."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.task_queue: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
    
    def submit_task(self, task: DistributedTask):
        """Submit task for scheduling."""
        with self.lock:
            # Add to appropriate priority queue
            self.task_queue[task.priority].append(task)
            
            # Track dependencies
            if task.dependencies:
                self.task_dependencies[task.task_id] = task.dependencies.copy()
            
            self.logger.info(f"Task {task.task_id} submitted with priority {task.priority.value}")
    
    def get_next_task(self, available_nodes: List[ComputeNode]) -> Optional[Tuple[DistributedTask, ComputeNode]]:
        """Get next task ready for execution."""
        with self.lock:
            # Check queues in priority order (highest first)
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue = self.task_queue[priority]
                
                # Look for ready tasks in this priority level
                for i, task in enumerate(queue):
                    if self._is_task_ready(task):
                        # Remove from queue
                        queue.remove(task)
                        
                        # Select node for execution
                        selected_node = self.load_balancer.select_node(task, available_nodes)
                        
                        if selected_node:
                            # Mark as running
                            task.assigned_node = selected_node.node_id
                            task.status = "running"
                            self.running_tasks[task.task_id] = task
                            
                            return task, selected_node
                        else:
                            # No suitable node available, put task back
                            queue.appendleft(task)
                            break
            
            return None
    
    def _is_task_ready(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if task.task_id not in self.task_dependencies:
            return True
        
        dependencies = self.task_dependencies[task.task_id]
        return all(dep_id in self.completed_tasks for dep_id in dependencies)
    
    def complete_task(self, task_result: TaskResult):
        """Mark task as completed."""
        with self.lock:
            task_id = task_result.task_id
            
            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            # Add to completed tasks
            self.completed_tasks[task_id] = task_result
            
            # Clean up dependencies
            if task_id in self.task_dependencies:
                del self.task_dependencies[task_id]
            
            self.logger.info(f"Task {task_id} completed on node {task_result.node_id}")
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        with self.lock:
            pending_counts = {
                priority.name: len(queue) 
                for priority, queue in self.task_queue.items()
            }
            
            return {
                "pending_tasks": pending_counts,
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "total_submitted": sum(pending_counts.values()) + len(self.running_tasks) + len(self.completed_tasks)
            }


class NodeManager:
    """Manages compute nodes in the distributed system."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.node_health_monitors: Dict[str, threading.Thread] = {}
        self.heartbeat_timeout = 60.0  # seconds
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        self._monitoring = True
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def register_node(self, node: ComputeNode):
        """Register a new compute node."""
        with self.lock:
            self.nodes[node.node_id] = node
            
            # Start health monitoring for this node
            monitor_thread = threading.Thread(
                target=self._monitor_node_health,
                args=(node.node_id,),
                daemon=True
            )
            monitor_thread.start()
            self.node_health_monitors[node.node_id] = monitor_thread
            
            self.logger.info(f"Registered node {node.node_id} ({node.node_type.value})")
    
    def unregister_node(self, node_id: str):
        """Unregister a compute node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
            
            if node_id in self.node_health_monitors:
                # Health monitoring thread will stop naturally
                del self.node_health_monitors[node_id]
            
            self.logger.info(f"Unregistered node {node_id}")
    
    def update_node_heartbeat(self, node_id: str, load: float = None, 
                            task_stats: Dict[str, Any] = None):
        """Update node heartbeat and status."""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.last_heartbeat = time.time()
                
                if load is not None:
                    node.current_load = load
                
                if task_stats:
                    node.total_tasks_completed = task_stats.get("completed", node.total_tasks_completed)
                    node.avg_task_duration = task_stats.get("avg_duration", node.avg_task_duration)
                
                node.status = "active"
    
    def get_active_nodes(self) -> List[ComputeNode]:
        """Get list of active compute nodes."""
        with self.lock:
            current_time = time.time()
            active_nodes = []
            
            for node in self.nodes.values():
                if (node.status == "active" and 
                    current_time - node.last_heartbeat < self.heartbeat_timeout):
                    active_nodes.append(node)
            
            return active_nodes
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[ComputeNode]:
        """Get nodes of specific type."""
        return [node for node in self.get_active_nodes() if node.node_type == node_type]
    
    def _start_health_monitoring(self):
        """Start global health monitoring."""
        def health_monitor():
            while self._monitoring:
                self._check_node_health()
                time.sleep(30)  # Check every 30 seconds
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    def _check_node_health(self):
        """Check health of all nodes."""
        current_time = time.time()
        
        with self.lock:
            for node_id, node in list(self.nodes.items()):
                if current_time - node.last_heartbeat > self.heartbeat_timeout:
                    if node.status == "active":
                        node.status = "inactive"
                        self.logger.warning(f"Node {node_id} marked as inactive (no heartbeat)")
                
                elif current_time - node.last_heartbeat > self.heartbeat_timeout * 2:
                    # Remove completely unresponsive nodes
                    self.logger.error(f"Removing unresponsive node {node_id}")
                    self.unregister_node(node_id)
    
    def _monitor_node_health(self, node_id: str):
        """Monitor health of specific node."""
        while self._monitoring and node_id in self.nodes:
            try:
                # Could implement actual health checks here
                # For now, just rely on heartbeat updates
                time.sleep(60)
            except Exception as e:
                self.logger.error(f"Health monitoring error for node {node_id}: {e}")
                break
    
    def get_node_stats(self) -> Dict[str, Any]:
        """Get node management statistics."""
        with self.lock:
            active_nodes = self.get_active_nodes()
            
            stats = {
                "total_nodes": len(self.nodes),
                "active_nodes": len(active_nodes),
                "node_types": {},
                "total_load": 0.0,
                "avg_load": 0.0
            }
            
            # Count by type
            for node in active_nodes:
                node_type = node.node_type.value
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
                stats["total_load"] += node.current_load
            
            if active_nodes:
                stats["avg_load"] = stats["total_load"] / len(active_nodes)
            
            return stats


class DistributedComputingFramework:
    """Main distributed computing framework."""
    
    def __init__(self, 
                 node_type: NodeType = NodeType.COORDINATOR,
                 security_context: Optional[SecurityContext] = None,
                 error_manager: Optional[ErrorRecoveryManager] = None):
        self.node_type = node_type
        self.security_context = security_context or SecurityContext()
        self.error_manager = error_manager or ErrorRecoveryManager()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.load_balancer = LoadBalancer()
        self.task_scheduler = TaskScheduler(self.load_balancer)
        self.node_manager = NodeManager()
        
        # Current node info
        self.node_id = str(uuid.uuid4())
        self.hostname = socket.gethostname()
        self.port = 8080
        
        # Task execution
        self.task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        
        # Distributed computing backend
        self.ray_enabled = RAY_AVAILABLE
        if self.ray_enabled and node_type == NodeType.COORDINATOR:
            self._initialize_ray()
        
        # Performance tracking
        self.execution_stats = {
            "tasks_executed": 0,
            "total_execution_time": 0.0,
            "failed_tasks": 0,
            "avg_execution_time": 0.0
        }
        
        self.lock = threading.RLock()
        
        # Register self as a node
        if node_type != NodeType.COORDINATOR:
            self._register_self()
    
    def _initialize_ray(self):
        """Initialize Ray for distributed computing."""
        try:
            ray.init(ignore_reinit_error=True)
            self.logger.info("Ray initialized for distributed computing")
        except Exception as e:
            self.ray_enabled = False
            self.logger.warning(f"Failed to initialize Ray: {e}")
    
    def _register_self(self):
        """Register this instance as a compute node."""
        # Detect capabilities
        capabilities = self._detect_node_capabilities()
        resource_limits = self._get_resource_limits()
        
        node = ComputeNode(
            node_id=self.node_id,
            node_type=self.node_type,
            hostname=self.hostname,
            port=self.port,
            capabilities=capabilities,
            resource_limits=resource_limits
        )
        
        self.node_manager.register_node(node)
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
    
    def _detect_node_capabilities(self) -> Dict[str, Any]:
        """Detect node hardware capabilities."""
        capabilities = {
            "hardware": [],
            "software": [],
            "frameworks": []
        }
        
        # Check for GPU
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                capabilities["hardware"].append("gpu")
                capabilities["gpu_count"] = len(gpus)
        except:
            pass
        
        # Check for TPU (simplified check)
        tpu_devices = ["/dev/apex_0", "/dev/apex_1", "/dev/tpu0", "/dev/tpu1"]
        for device in tpu_devices:
            if Path(device).exists():
                capabilities["hardware"].append("tpu")
                break
        
        # Check software frameworks
        frameworks = ["tensorflow", "pytorch", "jax", "numpy", "scipy"]
        for framework in frameworks:
            try:
                __import__(framework)
                capabilities["frameworks"].append(framework)
            except ImportError:
                pass
        
        return capabilities
    
    def _get_resource_limits(self) -> Dict[str, Any]:
        """Get node resource limits."""
        memory_info = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return {
            "memory_mb": memory_info.total // (1024 * 1024),
            "cpu_cores": cpu_count,
            "disk_gb": psutil.disk_usage("/").total // (1024**3)
        }
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat updates."""
        while True:
            try:
                # Calculate current load
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                load = (cpu_percent + memory_percent) / 200.0  # Normalize to 0-1
                
                # Update heartbeat
                self.node_manager.update_node_heartbeat(
                    self.node_id, 
                    load=load,
                    task_stats={
                        "completed": self.execution_stats["tasks_executed"],
                        "avg_duration": self.execution_stats["avg_execution_time"]
                    }
                )
                
                time.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                time.sleep(30)
    
    @robust_operation(timeout=300.0)
    def submit_distributed_task(self, 
                              function: Callable,
                              input_data: Any,
                              task_priority: TaskPriority = TaskPriority.NORMAL,
                              execution_mode: ExecutionMode = ExecutionMode.ASYNCHRONOUS,
                              resource_requirements: Optional[Dict[str, Any]] = None) -> str:
        """Submit task for distributed execution."""
        
        task_id = str(uuid.uuid4())
        
        # Create distributed task
        task = DistributedTask(
            task_id=task_id,
            function_name=function.__name__,
            function_code=self._serialize_function(function),
            input_data=input_data,
            priority=task_priority,
            execution_mode=execution_mode,
            resource_requirements=resource_requirements or {}
        )
        
        # Submit to scheduler
        self.task_scheduler.submit_task(task)
        
        self.logger.info(f"Submitted distributed task {task_id}")
        return task_id
    
    def _serialize_function(self, function: Callable) -> str:
        """Serialize function for distributed execution."""
        try:
            import inspect
            return inspect.getsource(function)
        except Exception as e:
            self.logger.warning(f"Could not serialize function {function.__name__}: {e}")
            return ""
    
    async def execute_distributed_benchmark(self, 
                                          benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark across distributed nodes."""
        
        # Prepare benchmark tasks
        benchmark_tasks = self._prepare_benchmark_tasks(benchmark_config)
        
        # Submit tasks
        task_ids = []
        for task_config in benchmark_tasks:
            task_id = self.submit_distributed_task(
                function=task_config["function"],
                input_data=task_config["data"],
                task_priority=TaskPriority.HIGH,
                resource_requirements=task_config.get("requirements", {})
            )
            task_ids.append(task_id)
        
        # Wait for completion and collect results
        results = await self._wait_for_tasks(task_ids)
        
        # Aggregate results
        aggregated_results = self._aggregate_benchmark_results(results)
        
        return aggregated_results
    
    def _prepare_benchmark_tasks(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare benchmark tasks for distributed execution."""
        tasks = []
        
        # Example: Split large benchmark into smaller tasks
        model_path = config.get("model_path")
        iterations = config.get("iterations", 1000)
        batch_size = config.get("batch_size", 1)
        
        # Split iterations across available nodes
        active_nodes = self.node_manager.get_active_nodes()
        if not active_nodes:
            active_nodes = [None]  # Execute locally
        
        iterations_per_node = max(1, iterations // len(active_nodes))
        
        for i, node in enumerate(active_nodes):
            start_iter = i * iterations_per_node
            end_iter = min((i + 1) * iterations_per_node, iterations)
            
            if start_iter >= iterations:
                break
            
            task_config = {
                "function": self._benchmark_task_function,
                "data": {
                    "model_path": model_path,
                    "start_iteration": start_iter,
                    "end_iteration": end_iter,
                    "batch_size": batch_size
                },
                "requirements": {
                    "memory_mb": 1024,
                    "cpu_cores": 1,
                    "capabilities": ["tpu"] if "tpu" in config.get("device_type", "") else []
                }
            }
            
            tasks.append(task_config)
        
        return tasks
    
    @staticmethod
    def _benchmark_task_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Function to execute benchmark task on remote node."""
        # This would be the actual benchmark execution
        # Simplified for demonstration
        import time
        import random
        
        start_time = time.time()
        
        # Simulate benchmark execution
        iterations = task_data["end_iteration"] - task_data["start_iteration"]
        batch_size = task_data["batch_size"]
        
        # Mock results
        latencies = [random.uniform(0.001, 0.01) for _ in range(iterations)]
        throughput = iterations / (time.time() - start_time)
        
        return {
            "iterations": iterations,
            "execution_time": time.time() - start_time,
            "latencies": latencies,
            "throughput": throughput,
            "avg_latency": sum(latencies) / len(latencies),
            "batch_size": batch_size
        }
    
    async def _wait_for_tasks(self, task_ids: List[str], 
                            timeout: float = 600.0) -> Dict[str, TaskResult]:
        """Wait for distributed tasks to complete."""
        start_time = time.time()
        completed_tasks = {}
        
        while len(completed_tasks) < len(task_ids):
            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout waiting for tasks: {len(completed_tasks)}/{len(task_ids)} completed")
                break
            
            # Check for completed tasks
            for task_id in task_ids:
                if task_id not in completed_tasks and task_id in self.task_scheduler.completed_tasks:
                    completed_tasks[task_id] = self.task_scheduler.completed_tasks[task_id]
            
            await asyncio.sleep(1.0)
        
        return completed_tasks
    
    def _aggregate_benchmark_results(self, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Aggregate distributed benchmark results."""
        if not results:
            return {"error": "No results to aggregate"}
        
        # Combine results from all nodes
        total_iterations = 0
        total_execution_time = 0.0
        all_latencies = []
        total_throughput = 0.0
        
        successful_results = 0
        
        for task_result in results.values():
            if task_result.success and task_result.result_data:
                data = task_result.result_data
                total_iterations += data.get("iterations", 0)
                total_execution_time = max(total_execution_time, data.get("execution_time", 0))
                all_latencies.extend(data.get("latencies", []))
                total_throughput += data.get("throughput", 0)
                successful_results += 1
        
        if successful_results == 0:
            return {"error": "No successful task results"}
        
        # Calculate aggregated metrics
        aggregated = {
            "total_iterations": total_iterations,
            "total_execution_time": total_execution_time,
            "overall_throughput": total_throughput,
            "avg_latency": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
            "latency_p50": np.percentile(all_latencies, 50) if all_latencies else 0,
            "latency_p95": np.percentile(all_latencies, 95) if all_latencies else 0,
            "latency_p99": np.percentile(all_latencies, 99) if all_latencies else 0,
            "successful_nodes": successful_results,
            "total_nodes": len(results),
            "distributed_efficiency": successful_results / len(results)
        }
        
        return aggregated
    
    def start_task_execution_loop(self):
        """Start the task execution loop for worker nodes."""
        if self.node_type == NodeType.COORDINATOR:
            self.logger.info("Starting coordinator task distribution loop")
            threading.Thread(target=self._coordinator_loop, daemon=True).start()
        else:
            self.logger.info("Starting worker task execution loop")
            threading.Thread(target=self._worker_loop, daemon=True).start()
    
    def _coordinator_loop(self):
        """Main loop for coordinator node."""
        while True:
            try:
                # Get available worker nodes
                worker_nodes = [
                    node for node in self.node_manager.get_active_nodes()
                    if node.node_type in [NodeType.WORKER, NodeType.EDGE_DEVICE, NodeType.TPU_NODE]
                ]
                
                if not worker_nodes:
                    time.sleep(5)
                    continue
                
                # Get next task to distribute
                task_assignment = self.task_scheduler.get_next_task(worker_nodes)
                
                if task_assignment:
                    task, assigned_node = task_assignment
                    self._distribute_task_to_node(task, assigned_node)
                else:
                    time.sleep(1)  # No tasks ready
                
            except Exception as e:
                self.logger.error(f"Coordinator loop error: {e}")
                time.sleep(5)
    
    def _worker_loop(self):
        """Main loop for worker node."""
        while True:
            try:
                # Worker nodes would receive tasks from coordinator
                # For now, process local task queue
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")
                time.sleep(5)
    
    def _distribute_task_to_node(self, task: DistributedTask, node: ComputeNode):
        """Distribute task to specific node."""
        # In a real implementation, this would send the task over network
        # For now, execute locally if it's the same node
        if node.node_id == self.node_id:
            future = self.task_executor.submit(self._execute_task_locally, task)
            self.running_tasks[task.task_id] = future
        else:
            self.logger.info(f"Would distribute task {task.task_id} to node {node.node_id}")
    
    def _execute_task_locally(self, task: DistributedTask) -> TaskResult:
        """Execute task on local node."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            # Execute the task function
            if task.function_code:
                # In practice, would need safer execution environment
                local_vars = {}
                exec(task.function_code, globals(), local_vars)
                function = local_vars.get(task.function_name)
            else:
                # Use registered function
                function = getattr(self, task.function_name, None)
            
            if not function:
                raise ValueError(f"Function {task.function_name} not found")
            
            result_data = function(task.input_data)
            
            # Create success result
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - start_memory
            
            result = TaskResult(
                task_id=task.task_id,
                node_id=self.node_id,
                result_data=result_data,
                execution_time=execution_time,
                memory_used=memory_used,
                cpu_usage=psutil.cpu_percent(),
                success=True
            )
            
            # Update stats
            with self.lock:
                self.execution_stats["tasks_executed"] += 1
                self.execution_stats["total_execution_time"] += execution_time
                self.execution_stats["avg_execution_time"] = (
                    self.execution_stats["total_execution_time"] / 
                    self.execution_stats["tasks_executed"]
                )
            
            # Mark task as completed
            self.task_scheduler.complete_task(result)
            
            return result
            
        except Exception as e:
            # Create error result
            result = TaskResult(
                task_id=task.task_id,
                node_id=self.node_id,
                error_message=str(e),
                execution_time=time.time() - start_time,
                success=False
            )
            
            with self.lock:
                self.execution_stats["failed_tasks"] += 1
            
            self.task_scheduler.complete_task(result)
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
            return result
    
    def get_distributed_system_status(self) -> Dict[str, Any]:
        """Get comprehensive distributed system status."""
        return {
            "node_info": {
                "node_id": self.node_id,
                "node_type": self.node_type.value,
                "hostname": self.hostname,
                "port": self.port
            },
            "cluster_stats": self.node_manager.get_node_stats(),
            "scheduler_stats": self.task_scheduler.get_scheduler_stats(),
            "execution_stats": self.execution_stats.copy(),
            "ray_enabled": self.ray_enabled,
            "active_nodes": len(self.node_manager.get_active_nodes())
        }
    
    def export_distributed_report(self, filepath: Path):
        """Export comprehensive distributed computing report."""
        report = {
            "system_status": self.get_distributed_system_status(),
            "node_details": [
                node.to_dict() for node in self.node_manager.nodes.values()
            ],
            "completed_tasks": [
                result.to_dict() for result in self.task_scheduler.completed_tasks.values()
            ],
            "load_balancer_strategy": self.load_balancer.current_strategy,
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Distributed computing report exported to {filepath}")
    
    def shutdown(self):
        """Shutdown distributed computing framework."""
        # Shutdown task executor
        self.task_executor.shutdown(wait=True)
        
        # Shutdown Ray if enabled
        if self.ray_enabled:
            try:
                ray.shutdown()
            except:
                pass
        
        # Stop monitoring
        self.node_manager._monitoring = False
        
        self.logger.info("Distributed computing framework shutdown complete")


def create_distributed_framework(node_type: str = "coordinator",
                                security_context: Optional[SecurityContext] = None,
                                error_manager: Optional[ErrorRecoveryManager] = None) -> DistributedComputingFramework:
    """Factory function to create distributed computing framework."""
    node_type_enum = NodeType(node_type)
    return DistributedComputingFramework(node_type_enum, security_context, error_manager)