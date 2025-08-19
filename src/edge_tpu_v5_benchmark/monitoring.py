"""Comprehensive monitoring and observability for TPU v5 benchmark suite."""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "type": self.metric_type.value
        }


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "level": self.level.value,
            "title": self.title,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "labels": self.labels,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def record_counter(self, name: str, value: Union[int, float] = 1,
                      labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (cumulative value)."""
        self._record_metric(name, value, MetricType.COUNTER, labels or {})

    def record_gauge(self, name: str, value: Union[int, float],
                    labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (current value)."""
        self._record_metric(name, value, MetricType.GAUGE, labels or {})

    def record_histogram(self, name: str, value: Union[int, float],
                        labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (distribution of values)."""
        self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})

    def record_timer(self, name: str, duration: float,
                    labels: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration in seconds)."""
        self._record_metric(f"{name}_duration", duration, MetricType.TIMER, labels or {})

    def _record_metric(self, name: str, value: Union[int, float],
                      metric_type: MetricType, labels: Dict[str, str]):
        """Internal method to record a metric."""
        with self.lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.now(),
                labels=labels,
                metric_type=metric_type
            )

            # Create composite key for metrics with labels
            key = self._create_metric_key(name, labels)
            self.metrics[key].append(metric_point)

            self.logger.debug(f"Recorded metric: {name}={value} {labels}")

    def _create_metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"

    def get_metrics(self, name_pattern: Optional[str] = None,
                   since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metrics matching pattern and time range."""
        with self.lock:
            results = []

            for key, points in self.metrics.items():
                metric_name = key.split('[')[0]  # Extract name without labels

                # Filter by name pattern
                if name_pattern and name_pattern not in metric_name:
                    continue

                for point in points:
                    # Filter by time
                    if since and point.timestamp < since:
                        continue

                    results.append(point)

            # Sort by timestamp
            results.sort(key=lambda p: p.timestamp)
            return results

    def get_latest_values(self, name_pattern: Optional[str] = None) -> Dict[str, MetricPoint]:
        """Get latest values for metrics matching pattern."""
        with self.lock:
            results = {}

            for key, points in self.metrics.items():
                metric_name = key.split('[')[0]

                if name_pattern and name_pattern not in metric_name:
                    continue

                if points:
                    results[key] = points[-1]

            return results

    def clear_metrics(self, older_than: Optional[datetime] = None):
        """Clear old metrics to free memory."""
        with self.lock:
            if older_than is None:
                older_than = datetime.now() - timedelta(hours=24)

            for key in list(self.metrics.keys()):
                points = self.metrics[key]
                # Remove old points
                while points and points[0].timestamp < older_than:
                    points.popleft()

                # Remove empty metrics
                if not points:
                    del self.metrics[key]

            self.logger.info(f"Cleared metrics older than {older_than}")

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        with self.lock:
            all_metrics = []
            for points in self.metrics.values():
                all_metrics.extend(points)

            if format.lower() == "json":
                return json.dumps([point.to_dict() for point in all_metrics], indent=2)
            elif format.lower() == "prometheus":
                return self._export_prometheus_format(all_metrics)
            else:
                raise ValueError(f"Unsupported export format: {format}")

    def _export_prometheus_format(self, metrics: List[MetricPoint]) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Group metrics by name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric)

        for name, points in by_name.items():
            # Add help and type comments
            lines.append(f"# HELP {name} Benchmark metric")
            lines.append(f"# TYPE {name} {points[0].metric_type.value}")

            # Add metric points
            for point in points:
                labels_str = ""
                if point.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                    labels_str = "{" + ",".join(label_pairs) + "}"

                lines.append(f"{name}{labels_str} {point.value} {int(point.timestamp.timestamp() * 1000)}")

            lines.append("")

        return "\n".join(lines)


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                      level: AlertLevel, title: str, description: str):
        """Add an alert rule."""
        rule = {
            "name": name,
            "condition": condition,
            "level": level,
            "title": title,
            "description": description
        }
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {name}")

    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        with self.lock:
            for rule in self.alert_rules:
                try:
                    if rule["condition"](metrics):
                        self._trigger_alert(rule, metrics)
                    else:
                        self._resolve_alert(rule["name"])
                except Exception as e:
                    self.logger.error(f"Error checking alert rule {rule['name']}: {e}")

    def _trigger_alert(self, rule: Dict[str, Any], metrics: Dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"{rule['name']}_{int(time.time())}"

        # Check if already active
        if rule["name"] in self.active_alerts:
            return  # Don't duplicate active alerts

        alert = Alert(
            id=alert_id,
            level=rule["level"],
            title=rule["title"],
            description=rule["description"],
            timestamp=datetime.now(),
            source="benchmark_monitor",
            labels={"rule": rule["name"]}
        )

        self.alerts.append(alert)
        self.active_alerts[rule["name"]] = alert

        self.logger.warning(f"Alert triggered: {alert.title}")

    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[rule_name]

            self.logger.info(f"Alert resolved: {alert.title}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, since: Optional[datetime] = None) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            if since is None:
                return list(self.alerts)

            return [alert for alert in self.alerts if alert.timestamp >= since]


class PerformanceMonitor:
    """Monitors benchmark performance and system health."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None,
                 alert_manager: Optional[AlertManager] = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.alerts = alert_manager or AlertManager()
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds

        # Setup default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Setup default monitoring alert rules."""
        # High memory usage alert
        self.alerts.add_alert_rule(
            name="high_memory_usage",
            condition=lambda m: m.get("system.memory_usage_percent", 0) > 90,
            level=AlertLevel.WARNING,
            title="High Memory Usage",
            description="System memory usage is above 90%"
        )

        # High CPU usage alert
        self.alerts.add_alert_rule(
            name="high_cpu_usage",
            condition=lambda m: m.get("system.cpu_usage_percent", 0) > 95,
            level=AlertLevel.WARNING,
            title="High CPU Usage",
            description="System CPU usage is above 95%"
        )

        # Benchmark failure rate alert
        self.alerts.add_alert_rule(
            name="high_failure_rate",
            condition=lambda m: m.get("benchmark.failure_rate", 0) > 0.1,
            level=AlertLevel.ERROR,
            title="High Benchmark Failure Rate",
            description="Benchmark failure rate is above 10%"
        )

        # Thermal throttling alert
        self.alerts.add_alert_rule(
            name="thermal_throttling",
            condition=lambda m: m.get("system.thermal_state", "") == "throttling",
            level=AlertLevel.CRITICAL,
            title="Thermal Throttling Detected",
            description="System is thermal throttling, performance may be degraded"
        )

    def start_monitoring(self):
        """Start background monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        self.logger.info("Performance monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()

                # Record metrics
                for name, value in system_metrics.items():
                    self.metrics.record_gauge(name, value)

                # Check alerts
                self.alerts.check_alerts(system_metrics)

                # Sleep until next monitoring cycle
                time.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}

        try:
            # Simulate system metrics collection
            # In real implementation, this would use psutil or similar
            import random

            metrics.update({
                "system.memory_usage_percent": random.uniform(30, 85),
                "system.cpu_usage_percent": random.uniform(10, 60),
                "system.disk_usage_percent": random.uniform(20, 80),
                "system.temperature_celsius": random.uniform(35, 75),
                "system.load_average": random.uniform(0.5, 2.0),
                "benchmark.active_count": random.randint(0, 5),
                "benchmark.queue_size": random.randint(0, 10),
                "benchmark.failure_rate": random.uniform(0, 0.05)
            })

            # Add thermal state
            temp = metrics["system.temperature_celsius"]
            if temp > 80:
                metrics["system.thermal_state"] = "throttling"
            elif temp > 70:
                metrics["system.thermal_state"] = "warm"
            else:
                metrics["system.thermal_state"] = "normal"

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

        return metrics

    def record_benchmark_start(self, benchmark_id: str, model_name: str):
        """Record benchmark start event."""
        self.metrics.record_counter("benchmark.started", 1, {
            "benchmark_id": benchmark_id,
            "model": model_name
        })
        self.logger.info(f"Benchmark started: {benchmark_id} ({model_name})")

    def record_benchmark_completion(self, benchmark_id: str, model_name: str,
                                  duration: float, success: bool):
        """Record benchmark completion event."""
        labels = {
            "benchmark_id": benchmark_id,
            "model": model_name,
            "status": "success" if success else "failure"
        }

        self.metrics.record_counter("benchmark.completed", 1, labels)
        self.metrics.record_timer("benchmark.duration", duration, labels)

        if success:
            self.metrics.record_counter("benchmark.success", 1, {"model": model_name})
        else:
            self.metrics.record_counter("benchmark.failure", 1, {"model": model_name})

        self.logger.info(f"Benchmark completed: {benchmark_id} ({'success' if success else 'failure'})")

    def record_model_performance(self, model_name: str, throughput: float,
                               latency: float, power: float):
        """Record model performance metrics."""
        labels = {"model": model_name}

        self.metrics.record_gauge("model.throughput", throughput, labels)
        self.metrics.record_gauge("model.latency", latency, labels)
        self.metrics.record_gauge("model.power", power, labels)
        self.metrics.record_gauge("model.efficiency", throughput / power if power > 0 else 0, labels)

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        active_alerts = self.alerts.get_active_alerts()
        latest_metrics = self.metrics.get_latest_values()

        # Determine overall health
        health_score = 100
        if any(alert.level == AlertLevel.CRITICAL for alert in active_alerts):
            health_score = 0
        elif any(alert.level == AlertLevel.ERROR for alert in active_alerts):
            health_score = 25
        elif any(alert.level == AlertLevel.WARNING for alert in active_alerts):
            health_score = 75

        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 75 else "degraded" if health_score > 25 else "critical",
            "active_alerts_count": len(active_alerts),
            "metrics_count": len(latest_metrics),
            "monitoring_active": self.monitoring_active,
            "timestamp": datetime.now().isoformat()
        }

    def generate_report(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if since is None:
            since = datetime.now() - timedelta(hours=1)

        metrics = self.metrics.get_metrics(since=since)
        alerts = self.alerts.get_alert_history(since=since)
        health = self.get_health_status()

        # Calculate summary statistics
        benchmark_count = len([m for m in metrics if m.name == "benchmark.completed"])
        success_count = len([m for m in metrics if m.name == "benchmark.success"])
        failure_count = len([m for m in metrics if m.name == "benchmark.failure"])

        success_rate = success_count / max(1, benchmark_count)

        return {
            "report_period": {
                "start": since.isoformat(),
                "end": datetime.now().isoformat(),
                "duration_hours": (datetime.now() - since).total_seconds() / 3600
            },
            "summary": {
                "benchmarks_run": benchmark_count,
                "success_rate": success_rate,
                "alerts_triggered": len(alerts),
                "health_score": health["health_score"]
            },
            "health_status": health,
            "alert_summary": {
                "total": len(alerts),
                "by_level": {
                    level.value: len([a for a in alerts if a.level == level])
                    for level in AlertLevel
                }
            },
            "metrics_summary": {
                "total_points": len(metrics),
                "unique_metrics": len(set(m.name for m in metrics))
            }
        }
