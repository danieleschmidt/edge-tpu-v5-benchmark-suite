"""Auto-scaling and adaptive resource management for TPU v5 benchmark suite."""

import asyncio
import logging
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple

import psutil


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceType(Enum):
    """Types of scalable resources."""
    THREADS = "threads"
    PROCESSES = "processes"
    MEMORY = "memory"
    CACHE = "cache"


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    queue_size: int
    active_tasks: int
    throughput: float  # tasks/second
    latency_p95: float  # ms
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "queue_size": self.queue_size,
            "active_tasks": self.active_tasks,
            "throughput": self.throughput,
            "latency_p95": self.latency_p95,
            "error_rate": self.error_rate
        }


@dataclass
class ScalingRule:
    """Rule for determining when to scale resources."""
    name: str
    resource_type: ResourceType
    metric: str  # cpu_usage, memory_usage, queue_size, etc.
    threshold_up: float
    threshold_down: float
    min_samples: int = 3
    cooldown_seconds: int = 300  # 5 minutes
    scale_factor: float = 1.5
    min_value: int = 1
    max_value: int = 100
    enabled: bool = True

    def should_scale_up(self, metrics: List[MetricSnapshot]) -> bool:
        """Check if should scale up based on metrics."""
        if not self.enabled or len(metrics) < self.min_samples:
            return False

        recent_metrics = metrics[-self.min_samples:]
        values = [getattr(metric, self.metric) for metric in recent_metrics]

        # All recent values should exceed threshold
        return all(value > self.threshold_up for value in values)

    def should_scale_down(self, metrics: List[MetricSnapshot]) -> bool:
        """Check if should scale down based on metrics."""
        if not self.enabled or len(metrics) < self.min_samples:
            return False

        recent_metrics = metrics[-self.min_samples:]
        values = [getattr(metric, self.metric) for metric in recent_metrics]

        # All recent values should be below threshold
        return all(value < self.threshold_down for value in values)


@dataclass
class ScalingAction:
    """Represents a scaling action taken."""
    timestamp: datetime
    resource_type: ResourceType
    direction: ScalingDirection
    old_value: int
    new_value: int
    trigger_metric: str
    trigger_value: float
    rule_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "resource_type": self.resource_type.value,
            "direction": self.direction.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "trigger_metric": self.trigger_metric,
            "trigger_value": self.trigger_value,
            "rule_name": self.rule_name
        }


class AdaptiveResourceManager:
    """Manages adaptive scaling of benchmark resources."""

    def __init__(self,
                 metrics_window_size: int = 60,  # Keep 60 metric snapshots
                 evaluation_interval: float = 30.0):  # Evaluate every 30 seconds

        self.metrics_window_size = metrics_window_size
        self.evaluation_interval = evaluation_interval

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=metrics_window_size)
        self.metrics_lock = threading.Lock()

        # Scaling rules
        self.scaling_rules: List[ScalingRule] = []
        self.last_scaling_actions: Dict[ResourceType, datetime] = {}

        # Current resource levels
        self.current_resources: Dict[ResourceType, int] = {
            ResourceType.THREADS: psutil.cpu_count() or 4,
            ResourceType.PROCESSES: psutil.cpu_count() or 4,
            ResourceType.MEMORY: 100,  # MB
            ResourceType.CACHE: 50     # MB
        }

        # Scaling history
        self.scaling_history: List[ScalingAction] = []

        # Control
        self.running = False
        self.monitor_task = None
        self.logger = logging.getLogger(__name__)

        # Callbacks for resource changes
        self.resource_callbacks: Dict[ResourceType, List[Callable]] = {
            resource_type: [] for resource_type in ResourceType
        }

        # Setup default scaling rules
        self._setup_default_scaling_rules()

    def _setup_default_scaling_rules(self):
        """Setup default auto-scaling rules."""
        # Thread pool scaling based on CPU usage and queue size
        self.scaling_rules.extend([
            ScalingRule(
                name="thread_cpu_scaling",
                resource_type=ResourceType.THREADS,
                metric="cpu_usage",
                threshold_up=80.0,
                threshold_down=40.0,
                min_samples=3,
                cooldown_seconds=180,
                scale_factor=1.5,
                min_value=2,
                max_value=32
            ),
            ScalingRule(
                name="thread_queue_scaling",
                resource_type=ResourceType.THREADS,
                metric="queue_size",
                threshold_up=10.0,
                threshold_down=2.0,
                min_samples=2,
                cooldown_seconds=120,
                scale_factor=2.0,
                min_value=2,
                max_value=32
            ),
            ScalingRule(
                name="process_scaling",
                resource_type=ResourceType.PROCESSES,
                metric="cpu_usage",
                threshold_up=70.0,
                threshold_down=30.0,
                min_samples=5,
                cooldown_seconds=300,
                scale_factor=1.3,
                min_value=1,
                max_value=psutil.cpu_count() or 4
            ),
            ScalingRule(
                name="memory_scaling",
                resource_type=ResourceType.MEMORY,
                metric="memory_usage",
                threshold_up=85.0,
                threshold_down=50.0,
                min_samples=3,
                cooldown_seconds=240,
                scale_factor=1.5,
                min_value=50,
                max_value=2000
            ),
            ScalingRule(
                name="cache_scaling",
                resource_type=ResourceType.CACHE,
                metric="throughput",
                threshold_up=50.0,  # High throughput = need more cache
                threshold_down=10.0,
                min_samples=4,
                cooldown_seconds=300,
                scale_factor=1.4,
                min_value=25,
                max_value=500
            )
        ])

    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        self.logger.info(f"Added scaling rule: {rule.name}")

    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule by name."""
        self.scaling_rules = [rule for rule in self.scaling_rules if rule.name != rule_name]
        self.logger.info(f"Removed scaling rule: {rule_name}")

    def add_resource_callback(self, resource_type: ResourceType, callback: Callable[[int, int], None]):
        """Add callback for resource changes."""
        self.resource_callbacks[resource_type].append(callback)

    def record_metrics(self,
                      cpu_usage: float,
                      memory_usage: float,
                      queue_size: int,
                      active_tasks: int,
                      throughput: float,
                      latency_p95: float,
                      error_rate: float):
        """Record current system metrics."""
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_size=queue_size,
            active_tasks=active_tasks,
            throughput=throughput,
            latency_p95=latency_p95,
            error_rate=error_rate
        )

        with self.metrics_lock:
            self.metrics_history.append(snapshot)

    async def start(self):
        """Start the adaptive resource manager."""
        if self.running:
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Adaptive resource manager started")

    async def stop(self):
        """Stop the adaptive resource manager."""
        self.running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Adaptive resource manager stopped")

    async def _monitoring_loop(self):
        """Main monitoring and scaling loop."""
        while self.running:
            try:
                await asyncio.sleep(self.evaluation_interval)

                # Evaluate scaling rules
                await self._evaluate_scaling_rules()

            except Exception as e:
                self.logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(self.evaluation_interval)

    async def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and take actions."""
        with self.metrics_lock:
            if len(self.metrics_history) < 2:
                return  # Need at least 2 metrics snapshots

            metrics_list = list(self.metrics_history)

        for rule in self.scaling_rules:
            if not rule.enabled:
                continue

            # Check cooldown period
            last_action = self.last_scaling_actions.get(rule.resource_type)
            if last_action:
                time_since_last = datetime.now() - last_action
                if time_since_last.total_seconds() < rule.cooldown_seconds:
                    continue

            current_value = self.current_resources[rule.resource_type]

            # Check for scale up
            if rule.should_scale_up(metrics_list):
                new_value = min(
                    rule.max_value,
                    max(current_value + 1, int(current_value * rule.scale_factor))
                )

                if new_value > current_value:
                    await self._execute_scaling_action(
                        rule, ScalingDirection.UP, current_value, new_value, metrics_list[-1]
                    )

            # Check for scale down
            elif rule.should_scale_down(metrics_list):
                new_value = max(
                    rule.min_value,
                    max(current_value - 1, int(current_value / rule.scale_factor))
                )

                if new_value < current_value:
                    await self._execute_scaling_action(
                        rule, ScalingDirection.DOWN, current_value, new_value, metrics_list[-1]
                    )

    async def _execute_scaling_action(self,
                                    rule: ScalingRule,
                                    direction: ScalingDirection,
                                    old_value: int,
                                    new_value: int,
                                    trigger_metric: MetricSnapshot):
        """Execute a scaling action."""
        # Create scaling action record
        action = ScalingAction(
            timestamp=datetime.now(),
            resource_type=rule.resource_type,
            direction=direction,
            old_value=old_value,
            new_value=new_value,
            trigger_metric=rule.metric,
            trigger_value=getattr(trigger_metric, rule.metric),
            rule_name=rule.name
        )

        # Update resource level
        self.current_resources[rule.resource_type] = new_value
        self.last_scaling_actions[rule.resource_type] = action.timestamp
        self.scaling_history.append(action)

        # Notify callbacks
        for callback in self.resource_callbacks[rule.resource_type]:
            try:
                callback(old_value, new_value)
            except Exception as e:
                self.logger.error(f"Error in resource callback: {e}")

        self.logger.info(
            f"Scaled {rule.resource_type.value} {direction.value}: {old_value} -> {new_value} "
            f"(trigger: {rule.metric}={action.trigger_value:.2f})"
        )

    def get_current_resources(self) -> Dict[ResourceType, int]:
        """Get current resource levels."""
        return self.current_resources.copy()

    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        if not self.scaling_history:
            return {
                "total_actions": 0,
                "scale_ups": 0,
                "scale_downs": 0,
                "most_active_resource": None,
                "avg_time_between_actions": 0
            }

        # Count actions by direction
        scale_ups = len([a for a in self.scaling_history if a.direction == ScalingDirection.UP])
        scale_downs = len([a for a in self.scaling_history if a.direction == ScalingDirection.DOWN])

        # Most active resource
        resource_counts = {}
        for action in self.scaling_history:
            resource_counts[action.resource_type] = resource_counts.get(action.resource_type, 0) + 1

        most_active = max(resource_counts, key=resource_counts.get) if resource_counts else None

        # Average time between actions
        if len(self.scaling_history) > 1:
            times = [(action.timestamp - self.scaling_history[i].timestamp).total_seconds()
                    for i, action in enumerate(self.scaling_history[1:])]
            avg_time = statistics.mean(times)
        else:
            avg_time = 0

        return {
            "total_actions": len(self.scaling_history),
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "most_active_resource": most_active.value if most_active else None,
            "avg_time_between_actions": avg_time,
            "current_resources": {rt.value: val for rt, val in self.current_resources.items()},
            "recent_actions": [a.to_dict() for a in self.scaling_history[-5:]]
        }

    def predict_scaling_needs(self, forecast_minutes: int = 30) -> Dict[ResourceType, Tuple[int, float]]:
        """Predict future scaling needs based on trends."""
        predictions = {}

        with self.metrics_lock:
            if len(self.metrics_history) < 10:
                return predictions  # Need more data for prediction

            metrics_list = list(self.metrics_history)

        # For each resource type, analyze trends
        for resource_type in ResourceType:
            # Find relevant rules for this resource
            relevant_rules = [rule for rule in self.scaling_rules
                            if rule.resource_type == resource_type and rule.enabled]

            if not relevant_rules:
                continue

            # Analyze trend for primary metric
            primary_rule = relevant_rules[0]  # Use first rule as primary
            metric_values = [getattr(m, primary_rule.metric) for m in metrics_list[-10:]]

            # Simple linear trend analysis
            x_values = list(range(len(metric_values)))
            if len(x_values) > 1:
                # Calculate trend slope
                x_mean = statistics.mean(x_values)
                y_mean = statistics.mean(metric_values)

                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, metric_values))
                denominator = sum((x - x_mean) ** 2 for x in x_values)

                if denominator != 0:
                    slope = numerator / denominator

                    # Project forward
                    current_value = metric_values[-1]
                    projected_value = current_value + slope * (forecast_minutes / (self.evaluation_interval / 60))

                    # Determine predicted scaling action
                    current_resource = self.current_resources[resource_type]
                    predicted_resource = current_resource
                    confidence = 0.0

                    if projected_value > primary_rule.threshold_up:
                        scale_factor = min(2.0, projected_value / primary_rule.threshold_up)
                        predicted_resource = min(
                            primary_rule.max_value,
                            int(current_resource * scale_factor)
                        )
                        confidence = min(1.0, (projected_value - primary_rule.threshold_up) / primary_rule.threshold_up)

                    elif projected_value < primary_rule.threshold_down:
                        scale_factor = max(0.5, projected_value / primary_rule.threshold_down)
                        predicted_resource = max(
                            primary_rule.min_value,
                            int(current_resource * scale_factor)
                        )
                        confidence = min(1.0, (primary_rule.threshold_down - projected_value) / primary_rule.threshold_down)

                    predictions[resource_type] = (predicted_resource, confidence)

        return predictions

    def optimize_rules_from_history(self):
        """Optimize scaling rules based on historical performance."""
        if len(self.scaling_history) < 10:
            return  # Need more history

        # Analyze effectiveness of each rule
        for rule in self.scaling_rules:
            rule_actions = [a for a in self.scaling_history if a.rule_name == rule.name]

            if len(rule_actions) < 3:
                continue

            # Calculate average time between scale up and scale down
            up_actions = [a for a in rule_actions if a.direction == ScalingDirection.UP]
            down_actions = [a for a in rule_actions if a.direction == ScalingDirection.DOWN]

            if up_actions and down_actions:
                # If we're scaling up and down too frequently, increase cooldown
                avg_cycle_time = statistics.mean([
                    (down.timestamp - up.timestamp).total_seconds()
                    for up in up_actions
                    for down in down_actions
                    if down.timestamp > up.timestamp
                ][:5])  # Last 5 cycles

                if avg_cycle_time < rule.cooldown_seconds / 2:
                    rule.cooldown_seconds = int(rule.cooldown_seconds * 1.2)
                    self.logger.info(f"Increased cooldown for rule {rule.name} to {rule.cooldown_seconds}s")

            # Adjust thresholds based on trigger values
            trigger_values = [a.trigger_value for a in rule_actions[-10:]]  # Last 10 actions
            if trigger_values:
                avg_trigger = statistics.mean(trigger_values)

                # If average trigger is much higher than threshold, increase threshold
                if avg_trigger > rule.threshold_up * 1.2:
                    rule.threshold_up *= 1.1
                    rule.threshold_down *= 1.1
                    self.logger.info(f"Adjusted thresholds for rule {rule.name}")


class LoadBalancer:
    """Intelligent load balancer for distributing benchmark tasks."""

    def __init__(self, resource_manager: AdaptiveResourceManager):
        self.resource_manager = resource_manager
        self.worker_pools: Dict[str, Dict[str, Any]] = {}
        self.task_routing_rules: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

        # Setup default routing rules
        self._setup_default_routing_rules()

    def _setup_default_routing_rules(self):
        """Setup default task routing rules."""
        self.task_routing_rules = [
            {
                "name": "cpu_intensive_to_processes",
                "condition": lambda task: task.metadata.get("cpu_intensive", False),
                "target_pool": "process_pool",
                "priority_boost": 1
            },
            {
                "name": "io_intensive_to_threads",
                "condition": lambda task: task.metadata.get("io_intensive", False),
                "target_pool": "thread_pool",
                "priority_boost": 0
            },
            {
                "name": "large_models_to_processes",
                "condition": lambda task: task.metadata.get("model_size_mb", 0) > 100,
                "target_pool": "process_pool",
                "priority_boost": 1
            },
            {
                "name": "quick_tasks_to_threads",
                "condition": lambda task: task.metadata.get("estimated_duration", 60) < 30,
                "target_pool": "thread_pool",
                "priority_boost": 0
            }
        ]

    def register_worker_pool(self, name: str, pool_info: Dict[str, Any]):
        """Register a worker pool with the load balancer."""
        self.worker_pools[name] = {
            **pool_info,
            "current_load": 0,
            "total_tasks": 0,
            "avg_task_time": 0.0,
            "last_updated": datetime.now()
        }

        self.logger.info(f"Registered worker pool: {name}")

    def route_task(self, task) -> str:
        """Determine the best worker pool for a task."""
        # Apply routing rules
        for rule in self.task_routing_rules:
            try:
                if rule["condition"](task):
                    target_pool = rule["target_pool"]
                    if target_pool in self.worker_pools:
                        # Boost task priority if specified
                        if rule["priority_boost"] > 0:
                            task.priority = TaskPriority(min(4, task.priority.value + rule["priority_boost"]))

                        self.logger.debug(f"Routed task {task.id} to {target_pool} via rule {rule['name']}")
                        return target_pool
            except Exception as e:
                self.logger.error(f"Error applying routing rule {rule['name']}: {e}")

        # Fallback to load-based routing
        return self._select_least_loaded_pool()

    def _select_least_loaded_pool(self) -> str:
        """Select the worker pool with the lowest current load."""
        if not self.worker_pools:
            return "default_pool"

        # Calculate load scores for each pool
        pool_scores = {}
        for name, info in self.worker_pools.items():
            # Load score = (current_load / max_workers) + avg_task_time_factor
            max_workers = info.get("max_workers", 1)
            current_load = info.get("current_load", 0)
            avg_task_time = info.get("avg_task_time", 1.0)

            load_ratio = current_load / max_workers
            time_factor = min(1.0, avg_task_time / 60.0)  # Normalize to 60 seconds

            pool_scores[name] = load_ratio + (time_factor * 0.3)  # Weight time factor at 30%

        # Select pool with lowest score
        best_pool = min(pool_scores, key=pool_scores.get)
        self.logger.debug(f"Selected least loaded pool: {best_pool}")
        return best_pool

    def update_pool_metrics(self, pool_name: str, current_load: int, avg_task_time: float):
        """Update metrics for a worker pool."""
        if pool_name in self.worker_pools:
            self.worker_pools[pool_name].update({
                "current_load": current_load,
                "avg_task_time": avg_task_time,
                "last_updated": datetime.now()
            })

    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across pools."""
        distribution = {}
        total_load = 0

        for name, info in self.worker_pools.items():
            load = info.get("current_load", 0)
            max_workers = info.get("max_workers", 1)

            distribution[name] = {
                "current_load": load,
                "max_workers": max_workers,
                "utilization": load / max_workers if max_workers > 0 else 0,
                "avg_task_time": info.get("avg_task_time", 0)
            }
            total_load += load

        return {
            "pools": distribution,
            "total_active_tasks": total_load,
            "most_utilized": max(distribution.keys(),
                               key=lambda k: distribution[k]["utilization"]) if distribution else None
        }


# Enhanced classes for ML-driven scaling
class AnomalyDetector:
    """Statistical anomaly detector for system metrics."""

    def __init__(self, window_size: int = 50, sensitivity: float = 2.5):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

    def calculate_anomaly_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate anomaly score for current metrics."""
        anomaly_scores = []

        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                history = self.metric_history[metric_name]
                history.append(value)

                if len(history) >= 5:
                    mean = statistics.mean(history)
                    std = statistics.stdev(history) if len(history) > 1 else 0.1

                    if std > 0:
                        z_score = abs(value - mean) / std
                        anomaly_score = max(0, (z_score - 1) / self.sensitivity)
                        anomaly_scores.append(anomaly_score)

        return statistics.mean(anomaly_scores) if anomaly_scores else 0.0


class CostOptimizer:
    """Cost optimization for resource scaling decisions."""

    def __init__(self, budget_per_hour: float = 100.0):
        self.budget_per_hour = budget_per_hour
        self.current_cost = 0.0

    def calculate_scaling_cost(self, resource_changes: Dict[ResourceType, int]) -> float:
        """Calculate cost of proposed resource changes."""
        cost_mapping = {
            ResourceType.THREADS: 1.0,
            ResourceType.PROCESSES: 2.0,
            ResourceType.MEMORY: 0.1,
            ResourceType.CACHE: 0.05
        }

        total_cost = 0.0
        for resource_type, change in resource_changes.items():
            cost_per_unit = cost_mapping.get(resource_type, 1.0)
            total_cost += abs(change) * cost_per_unit

        return total_cost

    def is_within_budget(self, additional_cost: float) -> bool:
        """Check if scaling action is within budget."""
        return (self.current_cost + additional_cost) <= self.budget_per_hour


class PredictiveScalingManager:
    """Enhanced scaling manager with ML prediction capabilities."""

    def __init__(self):
        self.base_manager = AdaptiveResourceManager()
        self.anomaly_detector = AnomalyDetector()
        self.cost_optimizer = CostOptimizer()
        self.ml_enabled = True
        self.prediction_models = {}
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the predictive scaling manager."""
        await self.base_manager.start()
        self.logger.info("Predictive scaling manager started")

    async def stop(self):
        """Stop the predictive scaling manager."""
        await self.base_manager.stop()
        self.logger.info("Predictive scaling manager stopped")

    def record_enhanced_metrics(self, metrics: Dict[str, Any]):
        """Record metrics with anomaly detection."""
        # Calculate anomaly score
        anomaly_score = self.anomaly_detector.calculate_anomaly_score(metrics)
        metrics['anomaly_score'] = anomaly_score

        # Record in base manager
        self.base_manager.record_metrics(
            cpu_usage=metrics.get('cpu_usage', 0),
            memory_usage=metrics.get('memory_usage', 0),
            queue_size=metrics.get('queue_size', 0),
            active_tasks=metrics.get('active_tasks', 0),
            throughput=metrics.get('throughput', 0),
            latency_p95=metrics.get('latency_p95', 0),
            error_rate=metrics.get('error_rate', 0)
        )

        # Handle anomalies
        if anomaly_score > 0.8:
            asyncio.create_task(self._handle_anomaly(metrics))

    async def _handle_anomaly(self, metrics: Dict[str, Any]):
        """Handle detected anomalies with emergency scaling."""
        self.logger.warning(f"Anomaly detected: {metrics.get('anomaly_score', 0):.3f}")

        # Emergency scaling - increase threads if CPU anomaly
        if metrics.get('cpu_usage', 0) > 90:
            current_threads = self.base_manager.current_resources[ResourceType.THREADS]
            emergency_threads = min(64, current_threads * 2)

            if emergency_threads > current_threads:
                self.base_manager.current_resources[ResourceType.THREADS] = emergency_threads
                self.logger.info(f"Emergency scaling: threads {current_threads} -> {emergency_threads}")

    def get_predictions(self, minutes_ahead: int = 10) -> Dict[str, float]:
        """Get scaling predictions (simplified implementation)."""
        if not self.ml_enabled or len(self.base_manager.metrics_history) < 10:
            return {}

        # Simple trend-based prediction
        recent_metrics = list(self.base_manager.metrics_history)[-10:]

        # Calculate CPU usage trend
        cpu_values = [getattr(m, 'cpu_usage', 0) for m in recent_metrics]
        if len(cpu_values) >= 3:
            trend = (cpu_values[-1] - cpu_values[0]) / len(cpu_values)
            predicted_cpu = cpu_values[-1] + trend * minutes_ahead

            return {
                'cpu_usage_predicted': max(0, min(100, predicted_cpu)),
                'confidence': 0.7 if abs(trend) > 1 else 0.5
            }

        return {}

    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get ML-driven scaling recommendations."""
        recommendations = []
        predictions = self.get_predictions()

        if predictions.get('confidence', 0) > 0.6:
            predicted_cpu = predictions.get('cpu_usage_predicted', 0)

            if predicted_cpu > 80:
                cost = self.cost_optimizer.calculate_scaling_cost({ResourceType.THREADS: 2})
                if self.cost_optimizer.is_within_budget(cost):
                    recommendations.append({
                        'resource_type': ResourceType.THREADS,
                        'action': 'scale_up',
                        'reason': f'Predicted CPU: {predicted_cpu:.1f}%',
                        'confidence': predictions['confidence'],
                        'cost': cost
                    })
            elif predicted_cpu < 30:
                recommendations.append({
                    'resource_type': ResourceType.THREADS,
                    'action': 'scale_down',
                    'reason': f'Predicted CPU: {predicted_cpu:.1f}%',
                    'confidence': predictions['confidence'],
                    'cost': 0
                })

        return recommendations

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including ML metrics."""
        base_stats = self.base_manager.get_scaling_statistics()

        # Add ML-specific stats
        ml_stats = {
            'ml_enabled': self.ml_enabled,
            'prediction_models': len(self.prediction_models),
            'anomaly_threshold': self.anomaly_detector.sensitivity,
            'budget_per_hour': self.cost_optimizer.budget_per_hour,
            'current_cost': self.cost_optimizer.current_cost
        }

        return {**base_stats, **ml_stats}


# Global resource manager instance
_resource_manager = None


async def get_resource_manager() -> PredictiveScalingManager:
    """Get global predictive resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = PredictiveScalingManager()
        await _resource_manager.start()
    return _resource_manager


# Utility functions for scaling optimization
def analyze_scaling_effectiveness(scaling_history: List[ScalingAction]) -> Dict[str, float]:
    """Analyze the effectiveness of past scaling actions."""
    if not scaling_history:
        return {'effectiveness_score': 0.0}

    # Simple effectiveness metric based on action frequency and direction changes
    scale_ups = sum(1 for action in scaling_history if action.direction == ScalingDirection.UP)
    scale_downs = sum(1 for action in scaling_history if action.direction == ScalingDirection.DOWN)

    # Balanced scaling is generally better
    balance_score = 1.0 - abs(scale_ups - scale_downs) / len(scaling_history)

    # Recent actions should be weighted more
    recent_actions = [action for action in scaling_history
                     if (datetime.now() - action.timestamp).total_seconds() < 3600]
    recency_score = len(recent_actions) / len(scaling_history) if scaling_history else 0

    effectiveness_score = (balance_score + recency_score) / 2

    return {
        'effectiveness_score': effectiveness_score,
        'total_actions': len(scaling_history),
        'scale_ups': scale_ups,
        'scale_downs': scale_downs,
        'recent_actions': len(recent_actions)
    }


def optimize_scaling_parameters(metrics_history: List, current_rules: List[ScalingRule]) -> List[ScalingRule]:
    """Optimize scaling rule parameters based on historical performance."""
    optimized_rules = []

    for rule in current_rules:
        # Create optimized copy of rule
        optimized_rule = ScalingRule(
            name=f"optimized_{rule.name}",
            resource_type=rule.resource_type,
            metric=rule.metric,
            threshold_up=rule.threshold_up,
            threshold_down=rule.threshold_down,
            min_samples=rule.min_samples,
            cooldown_seconds=rule.cooldown_seconds,
            scale_factor=rule.scale_factor,
            min_value=rule.min_value,
            max_value=rule.max_value,
            enabled=rule.enabled
        )

        # Simple optimization - reduce thresholds if system is stable
        if len(metrics_history) > 50:
            recent_variance = statistics.variance([
                getattr(m, rule.metric, 0) for m in metrics_history[-20:]
            ]) if len(metrics_history) >= 20 else 0

            if recent_variance < 5.0:  # Low variance = stable system
                optimized_rule.threshold_up *= 0.95  # More sensitive scaling
                optimized_rule.cooldown_seconds = int(optimized_rule.cooldown_seconds * 0.8)  # Faster response

        optimized_rules.append(optimized_rule)

    return optimized_rules
