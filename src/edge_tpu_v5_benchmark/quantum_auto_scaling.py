"""Quantum Auto-Scaling and Load Balancing

Advanced auto-scaling system for quantum task execution with TPU v5 cluster management.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from enum import Enum
import statistics
import math

from .quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumResource
from .quantum_monitoring import MetricsCollector, PerformanceMetrics, HealthStatus
from .quantum_performance import OptimizedQuantumTaskPlanner, PerformanceProfile

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Auto-scaling direction"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    QUANTUM_AFFINITY = "quantum_affinity"
    RESOURCE_OPTIMAL = "resource_optimal"
    LATENCY_AWARE = "latency_aware"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions"""
    timestamp: float = field(default_factory=time.time)
    
    # Queue metrics
    queue_length: int = 0
    avg_queue_wait_time: float = 0.0
    queue_growth_rate: float = 0.0
    
    # Performance metrics
    throughput_tasks_per_second: float = 0.0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    tpu_utilization: float = 0.0
    
    # Quantum metrics
    avg_coherence: float = 1.0
    decoherence_rate: float = 0.0
    entanglement_efficiency: float = 1.0
    
    # Predictive metrics
    predicted_load: float = 0.0
    capacity_headroom: float = 1.0


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    enabled: bool = True
    
    # Scaling thresholds
    scale_up_cpu_threshold: float = 0.8
    scale_down_cpu_threshold: float = 0.3
    scale_up_memory_threshold: float = 0.8
    scale_down_memory_threshold: float = 0.3
    scale_up_queue_threshold: int = 20
    scale_down_queue_threshold: int = 5
    
    # Performance thresholds
    min_success_rate: float = 0.85
    max_avg_execution_time: float = 30.0
    min_throughput: float = 1.0
    
    # Quantum thresholds
    min_coherence_threshold: float = 0.4
    max_decoherence_rate: float = 0.1
    
    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes
    
    # Predictive scaling
    enable_predictive_scaling: bool = True
    prediction_horizon: float = 300.0  # 5 minutes
    
    def is_scale_up_needed(self, metrics: ScalingMetrics) -> bool:
        """Check if scale up is needed based on metrics"""
        return (
            metrics.cpu_utilization > self.scale_up_cpu_threshold or
            metrics.memory_utilization > self.scale_up_memory_threshold or
            metrics.queue_length > self.scale_up_queue_threshold or
            metrics.success_rate < self.min_success_rate or
            metrics.avg_execution_time > self.max_avg_execution_time or
            metrics.avg_coherence < self.min_coherence_threshold
        )
    
    def is_scale_down_possible(self, metrics: ScalingMetrics) -> bool:
        """Check if scale down is possible based on metrics"""
        return (
            metrics.cpu_utilization < self.scale_down_cpu_threshold and
            metrics.memory_utilization < self.scale_down_memory_threshold and
            metrics.queue_length < self.scale_down_queue_threshold and
            metrics.success_rate > self.min_success_rate and
            metrics.avg_execution_time < self.max_avg_execution_time and
            metrics.avg_coherence > self.min_coherence_threshold
        )


@dataclass
class QuantumNode:
    """Represents a quantum processing node in the cluster"""
    node_id: str
    planner: OptimizedQuantumTaskPlanner
    max_capacity: int
    current_load: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    health_status: HealthStatus = HealthStatus.HEALTHY
    
    # Node-specific metrics
    total_tasks_executed: int = 0
    avg_execution_time: float = 0.0
    success_rate: float = 1.0
    
    # Resource utilization
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    tpu_utilization: float = 0.0
    
    # Quantum metrics
    coherence_level: float = 1.0
    decoherence_rate: float = 0.0
    
    def get_load_factor(self) -> float:
        """Get current load factor (0.0 to 1.0)"""
        return self.current_load / max(self.max_capacity, 1)
    
    def get_health_score(self) -> float:
        """Get node health score (0.0 to 1.0)"""
        if self.health_status == HealthStatus.HEALTHY:
            base_score = 1.0
        elif self.health_status == HealthStatus.WARNING:
            base_score = 0.7
        elif self.health_status == HealthStatus.DEGRADED:
            base_score = 0.5
        else:
            base_score = 0.1
        
        # Adjust based on performance
        performance_factor = self.success_rate * 0.5 + (1.0 - min(self.get_load_factor(), 1.0)) * 0.3 + self.coherence_level * 0.2
        
        return base_score * performance_factor
    
    def can_accept_task(self, task: QuantumTask) -> bool:
        """Check if node can accept a new task"""
        return (
            self.health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING] and
            self.current_load < self.max_capacity and
            time.time() - self.last_heartbeat < 30.0  # Node is responsive
        )
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update node metrics"""
        self.cpu_utilization = metrics.get('cpu_utilization', 0.0)
        self.memory_utilization = metrics.get('memory_utilization', 0.0)
        self.tpu_utilization = metrics.get('tpu_utilization', 0.0)
        self.coherence_level = metrics.get('coherence_level', 1.0)
        self.decoherence_rate = metrics.get('decoherence_rate', 0.0)
        self.last_heartbeat = time.time()


class LoadBalancer:
    """Advanced load balancer for quantum task distribution"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.QUANTUM_AFFINITY):
        self.strategy = strategy
        self.task_assignments: Dict[str, str] = {}  # task_id -> node_id
        self.node_task_counts: Dict[str, int] = {}
        self.round_robin_index = 0
    
    def select_node(self, task: QuantumTask, available_nodes: List[QuantumNode]) -> Optional[QuantumNode]:
        """Select optimal node for task execution"""
        if not available_nodes:
            return None
        
        # Filter healthy nodes that can accept the task
        suitable_nodes = [node for node in available_nodes if node.can_accept_task(task)]
        
        if not suitable_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(suitable_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(suitable_nodes)
        elif self.strategy == LoadBalancingStrategy.QUANTUM_AFFINITY:
            return self._quantum_affinity_selection(task, suitable_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_OPTIMAL:
            return self._resource_optimal_selection(task, suitable_nodes)
        elif self.strategy == LoadBalancingStrategy.LATENCY_AWARE:
            return self._latency_aware_selection(task, suitable_nodes)
        else:
            return suitable_nodes[0]  # Default to first available
    
    def _round_robin_selection(self, nodes: List[QuantumNode]) -> QuantumNode:
        """Round-robin node selection"""
        selected_node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected_node
    
    def _least_loaded_selection(self, nodes: List[QuantumNode]) -> QuantumNode:
        """Select node with least current load"""
        return min(nodes, key=lambda node: node.get_load_factor())
    
    def _quantum_affinity_selection(self, task: QuantumTask, nodes: List[QuantumNode]) -> QuantumNode:
        """Select node based on quantum affinity and coherence"""
        def quantum_score(node: QuantumNode) -> float:
            # Higher coherence is better
            coherence_score = node.coherence_level
            
            # Lower decoherence rate is better
            decoherence_score = 1.0 - min(node.decoherence_rate, 1.0)
            
            # Lower load is better
            load_score = 1.0 - node.get_load_factor()
            
            # Check if task has entangled tasks already on this node
            entanglement_bonus = 0.0
            for entangled_id in task.entangled_tasks:
                if self.task_assignments.get(entangled_id) == node.node_id:
                    entanglement_bonus += 0.2
            
            return coherence_score * 0.4 + decoherence_score * 0.3 + load_score * 0.2 + entanglement_bonus
        
        return max(nodes, key=quantum_score)
    
    def _resource_optimal_selection(self, task: QuantumTask, nodes: List[QuantumNode]) -> QuantumNode:
        """Select node with optimal resource fit"""
        def resource_fit_score(node: QuantumNode) -> float:
            score = 0.0
            
            # CPU fit
            cpu_req = task.resource_requirements.get('cpu_cores', 0)
            if cpu_req > 0:
                cpu_available = (1.0 - node.cpu_utilization) * 100  # Assume 100 cores max
                if cpu_available >= cpu_req:
                    score += 0.3 * (1.0 - abs(cpu_available - cpu_req) / cpu_available)
            
            # Memory fit
            memory_req = task.resource_requirements.get('memory_gb', 0)
            if memory_req > 0:
                memory_available = (1.0 - node.memory_utilization) * 128  # Assume 128GB max
                if memory_available >= memory_req:
                    score += 0.3 * (1.0 - abs(memory_available - memory_req) / memory_available)
            
            # TPU availability
            if 'tpu_v5_primary' in task.resource_requirements:
                if node.tpu_utilization < 0.8:
                    score += 0.4 * (1.0 - node.tpu_utilization)
            
            return score
        
        return max(nodes, key=resource_fit_score)
    
    def _latency_aware_selection(self, task: QuantumTask, nodes: List[QuantumNode]) -> QuantumNode:
        """Select node to minimize execution latency"""
        def latency_score(node: QuantumNode) -> float:
            # Lower execution time is better
            time_score = 1.0 / max(node.avg_execution_time, 0.1)
            
            # Lower load reduces queueing delay
            load_score = 1.0 - node.get_load_factor()
            
            # Higher success rate reduces retry latency
            success_score = node.success_rate
            
            return time_score * 0.4 + load_score * 0.3 + success_score * 0.3
        
        return max(nodes, key=latency_score)
    
    def assign_task(self, task: QuantumTask, node: QuantumNode) -> None:
        """Assign task to node and update tracking"""
        self.task_assignments[task.id] = node.node_id
        self.node_task_counts[node.node_id] = self.node_task_counts.get(node.node_id, 0) + 1
        node.current_load += 1
    
    def release_task(self, task_id: str) -> None:
        """Release task assignment and update tracking"""
        if task_id in self.task_assignments:
            node_id = self.task_assignments[task_id]
            del self.task_assignments[task_id]
            
            if node_id in self.node_task_counts:
                self.node_task_counts[node_id] = max(0, self.node_task_counts[node_id] - 1)
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get current load distribution across nodes"""
        return self.node_task_counts.copy()


class PredictiveScaler:
    """Predictive scaling based on historical patterns and trends"""
    
    def __init__(self, horizon_seconds: float = 300.0):
        self.horizon_seconds = horizon_seconds
        self.historical_metrics: deque = deque(maxlen=1000)
        self.seasonal_patterns: Dict[str, List[float]] = {}
        
    def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics for predictive analysis"""
        self.historical_metrics.append(metrics)
    
    def predict_load(self, current_time: Optional[float] = None) -> float:
        """Predict future load based on patterns"""
        if len(self.historical_metrics) < 10:
            # Not enough data for prediction
            return self.historical_metrics[-1].queue_length if self.historical_metrics else 0.0
        
        current_time = current_time or time.time()
        
        # Simple trend analysis
        recent_metrics = list(self.historical_metrics)[-10:]
        queue_lengths = [m.queue_length for m in recent_metrics]
        
        # Calculate trend
        x = np.arange(len(queue_lengths))
        z = np.polyfit(x, queue_lengths, 1)
        trend_slope = z[0]
        
        # Project forward
        prediction_steps = self.horizon_seconds / 30.0  # Assume 30-second intervals
        predicted_load = queue_lengths[-1] + (trend_slope * prediction_steps)
        
        # Apply seasonal adjustment if available
        predicted_load = self._apply_seasonal_adjustment(predicted_load, current_time)
        
        return max(0.0, predicted_load)
    
    def _apply_seasonal_adjustment(self, base_prediction: float, current_time: float) -> float:
        """Apply seasonal patterns to prediction"""
        # Simple hourly pattern detection
        hour_of_day = int((current_time % (24 * 3600)) // 3600)
        
        if 'hourly' in self.seasonal_patterns:
            hourly_pattern = self.seasonal_patterns['hourly']
            if len(hourly_pattern) > hour_of_day:
                seasonal_factor = hourly_pattern[hour_of_day]
                return base_prediction * seasonal_factor
        
        return base_prediction
    
    def update_seasonal_patterns(self) -> None:
        """Update seasonal patterns from historical data"""
        if len(self.historical_metrics) < 100:
            return
        
        # Build hourly patterns
        hourly_loads: Dict[int, List[float]] = {}
        
        for metrics in self.historical_metrics:
            hour = int((metrics.timestamp % (24 * 3600)) // 3600)
            if hour not in hourly_loads:
                hourly_loads[hour] = []
            hourly_loads[hour].append(metrics.queue_length)
        
        # Calculate average load for each hour
        hourly_pattern = []
        overall_avg = statistics.mean([m.queue_length for m in self.historical_metrics])
        
        for hour in range(24):
            if hour in hourly_loads and hourly_loads[hour]:
                hour_avg = statistics.mean(hourly_loads[hour])
                seasonal_factor = hour_avg / max(overall_avg, 0.1)
            else:
                seasonal_factor = 1.0
            
            hourly_pattern.append(seasonal_factor)
        
        self.seasonal_patterns['hourly'] = hourly_pattern
        logger.info("Updated seasonal patterns for predictive scaling")


class QuantumAutoScaler:
    """Main auto-scaling system for quantum cluster management"""
    
    def __init__(self, policy: Optional[ScalingPolicy] = None):
        self.policy = policy or ScalingPolicy()
        self.nodes: Dict[str, QuantumNode] = {}
        self.load_balancer = LoadBalancer()
        self.predictive_scaler = PredictiveScaler()
        self.metrics_collector = MetricsCollector()
        
        # Scaling state
        self.last_scale_up: float = 0.0
        self.last_scale_down: float = 0.0
        self.scaling_in_progress = False
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def add_node(self, node: QuantumNode) -> None:
        """Add node to cluster"""
        self.nodes[node.node_id] = node
        logger.info(f"Added node {node.node_id} to cluster (capacity: {node.max_capacity})")
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from cluster"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Removed node {node_id} from cluster")
    
    def get_cluster_capacity(self) -> int:
        """Get total cluster capacity"""
        return sum(node.max_capacity for node in self.nodes.values())
    
    def get_cluster_load(self) -> int:
        """Get current cluster load"""
        return sum(node.current_load for node in self.nodes.values())
    
    def collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect metrics for scaling decisions"""
        if not self.nodes:
            return ScalingMetrics()
        
        # Aggregate metrics from all nodes
        total_queue = sum(len(node.planner.get_ready_tasks()) for node in self.nodes.values())
        cpu_utils = [node.cpu_utilization for node in self.nodes.values()]
        memory_utils = [node.memory_utilization for node in self.nodes.values()]
        tpu_utils = [node.tpu_utilization for node in self.nodes.values()]
        coherence_levels = [node.coherence_level for node in self.nodes.values()]
        
        # Get recent performance data
        recent_stats = self.metrics_collector.get_task_execution_stats(window_seconds=300)
        
        # Calculate predictive load
        predicted_load = 0.0
        if self.policy.enable_predictive_scaling:
            predicted_load = self.predictive_scaler.predict_load()
        
        metrics = ScalingMetrics(
            queue_length=total_queue,
            throughput_tasks_per_second=recent_stats.get('tasks_per_second', 0.0),
            avg_execution_time=recent_stats.get('avg_duration', 0.0),
            success_rate=recent_stats.get('success_rate', 1.0),
            cpu_utilization=statistics.mean(cpu_utils) if cpu_utils else 0.0,
            memory_utilization=statistics.mean(memory_utils) if memory_utils else 0.0,
            tpu_utilization=statistics.mean(tpu_utils) if tpu_utils else 0.0,
            avg_coherence=statistics.mean(coherence_levels) if coherence_levels else 1.0,
            predicted_load=predicted_load,
            capacity_headroom=(self.get_cluster_capacity() - self.get_cluster_load()) / max(self.get_cluster_capacity(), 1)
        )
        
        # Add to predictive scaler
        self.predictive_scaler.add_metrics(metrics)
        
        return metrics
    
    def make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDirection:
        """Make scaling decision based on metrics"""
        if not self.policy.enabled or self.scaling_in_progress:
            return ScalingDirection.MAINTAIN
        
        current_time = time.time()
        current_nodes = len(self.nodes)
        
        # Check cooldown periods
        scale_up_cooled_down = current_time - self.last_scale_up > self.policy.scale_up_cooldown
        scale_down_cooled_down = current_time - self.last_scale_down > self.policy.scale_down_cooldown
        
        # Scale up conditions
        if (scale_up_cooled_down and 
            current_nodes < self.policy.max_instances and
            self.policy.is_scale_up_needed(metrics)):
            
            # Additional check for predictive scaling
            if (self.policy.enable_predictive_scaling and 
                metrics.predicted_load > self.policy.scale_up_queue_threshold):
                logger.info(f"Predictive scaling triggered: predicted load {metrics.predicted_load}")
                return ScalingDirection.SCALE_UP
            
            return ScalingDirection.SCALE_UP
        
        # Scale down conditions
        elif (scale_down_cooled_down and 
              current_nodes > self.policy.min_instances and
              self.policy.is_scale_down_possible(metrics)):
            
            # Ensure we have capacity headroom before scaling down
            if metrics.capacity_headroom > 0.5:  # At least 50% headroom
                return ScalingDirection.SCALE_DOWN
        
        return ScalingDirection.MAINTAIN
    
    async def execute_scaling_action(self, direction: ScalingDirection) -> bool:
        """Execute scaling action"""
        if direction == ScalingDirection.MAINTAIN:
            return True
        
        self.scaling_in_progress = True
        current_time = time.time()
        
        try:
            if direction == ScalingDirection.SCALE_UP:
                success = await self._scale_up()
                if success:
                    self.last_scale_up = current_time
                    logger.info("Cluster scaled up successfully")
            
            elif direction == ScalingDirection.SCALE_DOWN:
                success = await self._scale_down()
                if success:
                    self.last_scale_down = current_time
                    logger.info("Cluster scaled down successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Scaling action failed: {e}")
            return False
        
        finally:
            self.scaling_in_progress = False
    
    async def _scale_up(self) -> bool:
        """Scale up cluster by adding new node"""
        try:
            # Create new node
            node_id = f"quantum_node_{len(self.nodes)}"
            new_planner = OptimizedQuantumTaskPlanner()
            
            new_node = QuantumNode(
                node_id=node_id,
                planner=new_planner,
                max_capacity=10,  # Default capacity
                health_status=HealthStatus.HEALTHY
            )
            
            self.add_node(new_node)
            
            # Start node health monitoring
            await self._start_node_monitoring(new_node)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
            return False
    
    async def _scale_down(self) -> bool:
        """Scale down cluster by removing least utilized node"""
        try:
            if len(self.nodes) <= self.policy.min_instances:
                return False
            
            # Find node with lowest utilization and no active tasks
            candidate_node = None
            min_utilization = float('inf')
            
            for node in self.nodes.values():
                if (node.current_load == 0 and 
                    node.cpu_utilization < min_utilization):
                    min_utilization = node.cpu_utilization
                    candidate_node = node
            
            if candidate_node:
                # Gracefully shutdown node
                await self._shutdown_node(candidate_node)
                self.remove_node(candidate_node.node_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to scale down: {e}")
            return False
    
    async def _start_node_monitoring(self, node: QuantumNode) -> None:
        """Start monitoring for a node"""
        # In a real implementation, this would start monitoring services
        pass
    
    async def _shutdown_node(self, node: QuantumNode) -> None:
        """Gracefully shutdown a node"""
        # In a real implementation, this would handle graceful shutdown
        if hasattr(node.planner, 'shutdown'):
            await node.planner.shutdown()
    
    async def assign_task_to_cluster(self, task: QuantumTask) -> Optional[QuantumNode]:
        """Assign task to optimal node in cluster"""
        available_nodes = [node for node in self.nodes.values() 
                          if node.can_accept_task(task)]
        
        if not available_nodes:
            logger.warning(f"No available nodes for task {task.id}")
            return None
        
        # Use load balancer to select optimal node
        selected_node = self.load_balancer.select_node(task, available_nodes)
        
        if selected_node:
            self.load_balancer.assign_task(task, selected_node)
            logger.debug(f"Assigned task {task.id} to node {selected_node.node_id}")
        
        return selected_node
    
    async def start_auto_scaling(self, monitoring_interval: float = 30.0) -> None:
        """Start auto-scaling monitoring"""
        logger.info("Starting quantum auto-scaling system")
        
        async def scaling_loop():
            while not self._shutdown:
                try:
                    # Collect metrics
                    metrics = self.collect_scaling_metrics()
                    
                    # Make scaling decision
                    scaling_decision = self.make_scaling_decision(metrics)
                    
                    # Execute scaling if needed
                    if scaling_decision != ScalingDirection.MAINTAIN:
                        await self.execute_scaling_action(scaling_decision)
                    
                    # Update seasonal patterns periodically
                    if len(self.predictive_scaler.historical_metrics) % 100 == 0:
                        self.predictive_scaler.update_seasonal_patterns()
                    
                    # Wait for next monitoring cycle
                    await asyncio.sleep(monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Auto-scaling loop error: {e}")
                    await asyncio.sleep(min(monitoring_interval, 30.0))
        
        self._monitoring_task = asyncio.create_task(scaling_loop())
    
    async def stop_auto_scaling(self) -> None:
        """Stop auto-scaling monitoring"""
        self._shutdown = True
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling system stopped")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status"""
        metrics = self.collect_scaling_metrics()
        
        return {
            'cluster_info': {
                'total_nodes': len(self.nodes),
                'total_capacity': self.get_cluster_capacity(),
                'current_load': self.get_cluster_load(),
                'load_factor': self.get_cluster_load() / max(self.get_cluster_capacity(), 1)
            },
            'scaling_metrics': {
                'queue_length': metrics.queue_length,
                'avg_cpu_utilization': metrics.cpu_utilization,
                'avg_memory_utilization': metrics.memory_utilization,
                'avg_tpu_utilization': metrics.tpu_utilization,
                'avg_coherence': metrics.avg_coherence,
                'predicted_load': metrics.predicted_load
            },
            'scaling_policy': {
                'enabled': self.policy.enabled,
                'min_instances': self.policy.min_instances,
                'max_instances': self.policy.max_instances,
                'scale_up_threshold': self.policy.scale_up_queue_threshold,
                'scale_down_threshold': self.policy.scale_down_queue_threshold
            },
            'node_status': [
                {
                    'node_id': node.node_id,
                    'capacity': node.max_capacity,
                    'current_load': node.current_load,
                    'health_status': node.health_status.value,
                    'health_score': node.get_health_score(),
                    'cpu_utilization': node.cpu_utilization,
                    'coherence_level': node.coherence_level
                }
                for node in self.nodes.values()
            ],
            'load_balancer': {
                'strategy': self.load_balancer.strategy.value,
                'task_distribution': self.load_balancer.get_load_distribution()
            },
            'last_scaling_actions': {
                'last_scale_up': self.last_scale_up,
                'last_scale_down': self.last_scale_down,
                'scaling_in_progress': self.scaling_in_progress
            }
        }