"""Advanced Telemetry and Observability System - Generation 3

This module provides comprehensive telemetry, observability, and real-time monitoring
capabilities for the TERRAGON quantum-enhanced TPU benchmark system, including:

- Real-time performance dashboards and metrics visualization
- Distributed tracing across quantum-enhanced workflows  
- Predictive anomaly detection and alerting
- Advanced performance profiling and bottleneck analysis
- Multi-dimensional time series analysis
- Quantum state monitoring and decoherence tracking
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import statistics

import numpy as np
import psutil

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available - advanced visualization disabled")

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available - metrics export disabled")

from .security import SecurityContext


class MetricType(Enum):
    """Types of metrics for telemetry."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    QUANTUM_STATE = "quantum_state"
    TIME_SERIES = "time_series"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    QUANTUM_DECOHERENCE = "quantum_decoherence"


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: Union[float, Dict[str, float]]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    parent_span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    quantum_context: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """High-performance metric collector with quantum-enhanced capabilities."""
    
    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.metric_types = {}
        self.retention_hours = retention_hours
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Prometheus integration
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self.prom_counters = {}
            self.prom_gauges = {}
            self.prom_histograms = {}
        
        # Start cleanup task
        self._start_cleanup_task()
    
    def record_metric(self, name: str, value: Union[float, Dict[str, float]], 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Dict[str, str] = None, 
                     metadata: Dict[str, Any] = None):
        """Record a metric value."""
        with self.lock:
            timestamp = time.time()
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                labels=labels or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(point)
            self.metric_types[name] = metric_type
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self._update_prometheus_metric(name, value, metric_type, labels)
    
    def _update_prometheus_metric(self, name: str, value: Union[float, Dict[str, float]], 
                                 metric_type: MetricType, labels: Dict[str, str]):
        """Update Prometheus metric."""
        try:
            if metric_type == MetricType.COUNTER:
                if name not in self.prom_counters:
                    self.prom_counters[name] = Counter(
                        name, f'Counter metric: {name}', 
                        labelnames=list(labels.keys()) if labels else [],
                        registry=self.registry
                    )
                if isinstance(value, (int, float)):
                    if labels:
                        self.prom_counters[name].labels(**labels).inc(value)
                    else:
                        self.prom_counters[name].inc(value)
                        
            elif metric_type == MetricType.GAUGE:
                if name not in self.prom_gauges:
                    self.prom_gauges[name] = Gauge(
                        name, f'Gauge metric: {name}',
                        labelnames=list(labels.keys()) if labels else [],
                        registry=self.registry
                    )
                if isinstance(value, (int, float)):
                    if labels:
                        self.prom_gauges[name].labels(**labels).set(value)
                    else:
                        self.prom_gauges[name].set(value)
                        
            elif metric_type == MetricType.HISTOGRAM:
                if name not in self.prom_histograms:
                    self.prom_histograms[name] = Histogram(
                        name, f'Histogram metric: {name}',
                        labelnames=list(labels.keys()) if labels else [],
                        registry=self.registry
                    )
                if isinstance(value, (int, float)):
                    if labels:
                        self.prom_histograms[name].labels(**labels).observe(value)
                    else:
                        self.prom_histograms[name].observe(value)
                        
        except Exception as e:
            self.logger.warning(f"Failed to update Prometheus metric {name}: {e}")
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time period."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = time.time() - (hours * 3600)
            return [point for point in self.metrics[name] if point.timestamp >= cutoff_time]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self.lock:
            current = {}
            for name, points in self.metrics.items():
                if points:
                    latest = points[-1]
                    current[name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp,
                        'labels': latest.labels,
                        'type': self.metric_types.get(name, MetricType.GAUGE).value
                    }
            return current
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def _start_cleanup_task(self):
        """Start background task to clean up old metrics."""
        def cleanup():
            while True:
                try:
                    cutoff_time = time.time() - (self.retention_hours * 3600)
                    with self.lock:
                        for name in list(self.metrics.keys()):
                            points = self.metrics[name]
                            # Remove old points
                            while points and points[0].timestamp < cutoff_time:
                                points.popleft()
                    
                    time.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    self.logger.error(f"Metric cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()


class AnomalyDetector:
    """ML-based anomaly detection for performance metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baselines = {}
        self.anomaly_history = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def detect_anomalies(self, metric_name: str, values: List[float]) -> List[Tuple[int, float]]:
        """Detect anomalies in metric values using statistical methods."""
        if len(values) < 10:
            return []  # Need minimum data for anomaly detection
        
        anomalies = []
        
        # Calculate rolling statistics
        window_size = min(20, len(values) // 2)
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            current_value = values[i]
            
            # Statistical anomaly detection
            mean_val = statistics.mean(window)
            std_val = statistics.stdev(window) if len(set(window)) > 1 else 0.1
            
            # Z-score based detection
            z_score = abs(current_value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > self.sensitivity:
                anomalies.append((i, current_value))
        
        # Update baseline
        if values:
            self.baselines[metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(set(values)) > 1 else 0.1,
                'last_updated': time.time()
            }
        
        return anomalies
    
    def is_anomalous(self, metric_name: str, value: float) -> bool:
        """Check if a single value is anomalous."""
        if metric_name not in self.baselines:
            return False
        
        baseline = self.baselines[metric_name]
        z_score = abs(value - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
        
        return z_score > self.sensitivity


class AlertManager:
    """Comprehensive alerting system with quantum-aware thresholds."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_callbacks = []
        self.logger = logging.getLogger(__name__)
        
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      severity: AlertSeverity, condition: str = "greater",
                      quantum_context: bool = False):
        """Add an alert rule for a metric."""
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'severity': severity,
            'condition': condition,
            'quantum_context': quantum_context
        }
    
    def check_alerts(self, metric_name: str, value: Union[float, Dict[str, float]]):
        """Check if metric value triggers any alerts."""
        if metric_name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric_name]
        
        # Handle different value types
        check_value = value
        if isinstance(value, dict):
            check_value = value.get('primary', 0.0)
        
        # Check alert condition
        triggered = False
        if rule['condition'] == 'greater' and check_value > rule['threshold']:
            triggered = True
        elif rule['condition'] == 'less' and check_value < rule['threshold']:
            triggered = True
        elif rule['condition'] == 'equal' and abs(check_value - rule['threshold']) < 0.001:
            triggered = True
        
        if triggered:
            self._trigger_alert(metric_name, rule, check_value)
        else:
            self._resolve_alert(metric_name)
    
    def _trigger_alert(self, metric_name: str, rule: Dict[str, Any], value: float):
        """Trigger an alert."""
        alert_id = f"{metric_name}_{rule['severity'].value}"
        
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return  # Alert already active
        
        alert = Alert(
            alert_id=alert_id,
            severity=rule['severity'],
            message=f"Metric {metric_name} {rule['condition']} threshold {rule['threshold']}: current value {value:.3f}",
            metric_name=metric_name,
            threshold=rule['threshold'],
            current_value=value,
            timestamp=time.time()
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        self.logger.warning(f"ALERT TRIGGERED: {alert.message}")
        
        # Notify callbacks
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert notification callback failed: {e}")
    
    def _resolve_alert(self, metric_name: str):
        """Resolve alerts for a metric."""
        for alert_id, alert in self.alerts.items():
            if alert.metric_name == metric_name and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = time.time()
                self.logger.info(f"ALERT RESOLVED: {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.notification_callbacks.append(callback)


class DistributedTracer:
    """Distributed tracing for quantum-enhanced workflows."""
    
    def __init__(self):
        self.traces = {}
        self.spans = {}
        self.logger = logging.getLogger(__name__)
    
    def start_trace(self, operation_name: str) -> str:
        """Start a new trace."""
        trace_id = f"trace_{int(time.time() * 1000000)}_{np.random.randint(1000, 9999)}"
        span_id = f"span_{int(time.time() * 1000000)}_{np.random.randint(1000, 9999)}"
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self.traces[trace_id] = [span_id]
        self.spans[span_id] = span
        
        return span_id
    
    def start_child_span(self, parent_span_id: str, operation_name: str) -> str:
        """Start a child span."""
        if parent_span_id not in self.spans:
            return self.start_trace(operation_name)
        
        parent_span = self.spans[parent_span_id]
        span_id = f"span_{int(time.time() * 1000000)}_{np.random.randint(1000, 9999)}"
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=parent_span.trace_id,
            operation_name=operation_name,
            start_time=time.time(),
            parent_span_id=parent_span_id
        )
        
        self.traces[parent_span.trace_id].append(span_id)
        self.spans[span_id] = span
        
        return span_id
    
    def finish_span(self, span_id: str, tags: Dict[str, str] = None, 
                   quantum_context: Dict[str, Any] = None):
        """Finish a span."""
        if span_id not in self.spans:
            return
        
        span = self.spans[span_id]
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time
        
        if tags:
            span.tags.update(tags)
        
        if quantum_context:
            span.quantum_context.update(quantum_context)
    
    def add_log(self, span_id: str, message: str, level: str = "INFO"):
        """Add a log entry to a span."""
        if span_id not in self.spans:
            return
        
        span = self.spans[span_id]
        span.logs.append({
            'timestamp': time.time(),
            'level': level,
            'message': message
        })
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get complete trace information."""
        if trace_id not in self.traces:
            return {}
        
        span_ids = self.traces[trace_id]
        spans = [self.spans[sid] for sid in span_ids if sid in self.spans]
        
        # Calculate trace duration
        start_times = [s.start_time for s in spans]
        end_times = [s.end_time for s in spans if s.end_time]
        
        trace_duration = max(end_times) - min(start_times) if end_times else 0
        
        return {
            'trace_id': trace_id,
            'duration': trace_duration,
            'span_count': len(spans),
            'spans': [s.__dict__ for s in spans]
        }


class PerformanceDashboard:
    """Real-time performance dashboard with quantum metrics visualization."""
    
    def __init__(self, metric_collector: MetricCollector, 
                 alert_manager: AlertManager,
                 tracer: DistributedTracer):
        self.metric_collector = metric_collector
        self.alert_manager = alert_manager
        self.tracer = tracer
        self.logger = logging.getLogger(__name__)
    
    def generate_dashboard(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        dashboard_data = {
            'timestamp': time.time(),
            'time_range_hours': time_range_hours,
            'metrics': self._get_metric_summaries(time_range_hours),
            'alerts': [alert.__dict__ for alert in self.alert_manager.get_active_alerts()],
            'system_health': self._get_system_health(),
            'quantum_state': self._get_quantum_metrics(),
            'performance_trends': self._get_performance_trends(time_range_hours)
        }
        
        return dashboard_data
    
    def _get_metric_summaries(self, hours: int) -> Dict[str, Any]:
        """Get metric summaries for dashboard."""
        summaries = {}
        current_metrics = self.metric_collector.get_current_metrics()
        
        for name, current in current_metrics.items():
            history = self.metric_collector.get_metric_history(name, hours)
            
            if history:
                values = []
                for point in history:
                    if isinstance(point.value, (int, float)):
                        values.append(point.value)
                    elif isinstance(point.value, dict):
                        values.append(point.value.get('primary', 0.0))
                
                if values:
                    summaries[name] = {
                        'current': current['value'],
                        'min': min(values),
                        'max': max(values),
                        'avg': statistics.mean(values),
                        'trend': self._calculate_trend(values),
                        'data_points': len(values)
                    }
        
        return summaries
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values."""
        if len(values) < 2:
            return "stable"
        
        recent = values[-min(10, len(values)):]
        older = values[:len(values)//2]
        
        if not older:
            return "stable"
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        change_percent = (recent_avg - older_avg) / older_avg * 100 if older_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]]
        
        health_score = 100.0
        health_score -= len(critical_alerts) * 20
        health_score -= len(active_alerts) * 5
        health_score = max(0, health_score)
        
        status = "healthy"
        if health_score < 50:
            status = "critical"
        elif health_score < 70:
            status = "warning"
        elif health_score < 90:
            status = "degraded"
        
        return {
            'score': health_score,
            'status': status,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts)
        }
    
    def _get_quantum_metrics(self) -> Dict[str, Any]:
        """Get quantum-specific metrics."""
        quantum_metrics = {}
        
        for name, current in self.metric_collector.get_current_metrics().items():
            if 'quantum' in name.lower():
                quantum_metrics[name] = current
        
        return quantum_metrics
    
    def _get_performance_trends(self, hours: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance trend data for charts."""
        trends = {}
        
        key_metrics = ['throughput', 'latency', 'error_rate', 'cpu_usage', 'memory_usage']
        
        for metric in key_metrics:
            history = self.metric_collector.get_metric_history(metric, hours)
            if history:
                trend_data = []
                for point in history:
                    value = point.value
                    if isinstance(value, dict):
                        value = value.get('primary', 0.0)
                    
                    trend_data.append({
                        'timestamp': point.timestamp,
                        'value': value
                    })
                
                trends[metric] = trend_data
        
        return trends
    
    def create_visualization(self, dashboard_data: Dict[str, Any]) -> Optional[str]:
        """Create HTML visualization of dashboard data."""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance Trends', 'System Health', 'Alert Status', 'Quantum Metrics'),
                specs=[[{"secondary_y": True}, {"type": "indicator"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Performance trends
            trends = dashboard_data.get('performance_trends', {})
            for metric, data in trends.items():
                if data:
                    timestamps = [datetime.fromtimestamp(p['timestamp']) for p in data]
                    values = [p['value'] for p in data]
                    
                    fig.add_trace(
                        go.Scatter(x=timestamps, y=values, name=metric, mode='lines'),
                        row=1, col=1
                    )
            
            # System health indicator
            health = dashboard_data.get('system_health', {})
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=health.get('score', 0),
                    title={'text': "System Health"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [
                               {'range': [0, 50], 'color': "lightgray"},
                               {'range': [50, 80], 'color': "yellow"},
                               {'range': [80, 100], 'color': "green"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=2
            )
            
            # Alert status
            alerts = dashboard_data.get('alerts', [])
            alert_counts = defaultdict(int)
            for alert in alerts:
                alert_counts[alert['severity']] += 1
            
            if alert_counts:
                fig.add_trace(
                    go.Bar(x=list(alert_counts.keys()), y=list(alert_counts.values()),
                          name="Active Alerts"),
                    row=2, col=1
                )
            
            # Quantum metrics
            quantum = dashboard_data.get('quantum_state', {})
            if quantum:
                names = list(quantum.keys())
                values = [q.get('value', 0) for q in quantum.values()]
                
                fig.add_trace(
                    go.Scatter(x=names, y=values, mode='markers+text',
                             text=values, textposition="top center",
                             name="Quantum Metrics"),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                title="TERRAGON Quantum-Enhanced Performance Dashboard",
                showlegend=True
            )
            
            return fig.to_html(include_plotlyjs='cdn')
            
        except Exception as e:
            self.logger.error(f"Dashboard visualization failed: {e}")
            return None


class TelemetrySystem:
    """Main telemetry and observability system."""
    
    def __init__(self, retention_hours: int = 24):
        self.metric_collector = MetricCollector(retention_hours=retention_hours)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.tracer = DistributedTracer()
        self.dashboard = PerformanceDashboard(
            self.metric_collector, self.alert_manager, self.tracer
        )
        
        self.security_context = SecurityContext()
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def start(self):
        """Start the telemetry system."""
        self.running = True
        self.logger.info("Telemetry system started")
        
        # Start background monitoring
        def start_monitoring():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._monitor_system_metrics())
            except Exception as e:
                self.logger.error(f"Telemetry monitoring failed: {e}")
        
        self.monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
        self.monitoring_thread.start()
    
    def stop(self):
        """Stop the telemetry system."""
        self.running = False
        self.logger.info("Telemetry system stopped")
    
    def record_performance_metric(self, name: str, value: Union[float, Dict[str, float]], 
                                labels: Dict[str, str] = None):
        """Record a performance metric."""
        self.metric_collector.record_metric(name, value, MetricType.GAUGE, labels)
        
        # Check for anomalies
        if isinstance(value, (int, float)):
            history = self.metric_collector.get_metric_history(name, 1)
            values = [p.value for p in history if isinstance(p.value, (int, float))]
            
            if len(values) > 10:
                anomalies = self.anomaly_detector.detect_anomalies(name, values)
                if anomalies and self.anomaly_detector.is_anomalous(name, value):
                    self.logger.warning(f"Anomaly detected in {name}: {value}")
        
        # Check alerts
        self.alert_manager.check_alerts(name, value)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            'cpu_usage', 80.0, AlertSeverity.WARNING, 'greater'
        )
        self.alert_manager.add_alert_rule(
            'memory_usage', 90.0, AlertSeverity.ERROR, 'greater'
        )
        self.alert_manager.add_alert_rule(
            'error_rate', 0.05, AlertSeverity.ERROR, 'greater'
        )
        self.alert_manager.add_alert_rule(
            'quantum_coherence', 0.7, AlertSeverity.QUANTUM_DECOHERENCE, 'less', True
        )
    
    async def _monitor_system_metrics(self):
        """Monitor system-level metrics."""
        while self.running:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                self.record_performance_metric('cpu_usage', cpu_percent)
                self.record_performance_metric('memory_usage', memory.percent)
                self.record_performance_metric('memory_available_gb', memory.available / (1024**3))
                
                # Process metrics
                process = psutil.Process()
                self.record_performance_metric('process_cpu', process.cpu_percent())
                self.record_performance_metric('process_memory_mb', process.memory_info().rss / (1024**2))
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_data(self, hours: int = 1) -> Dict[str, Any]:
        """Get dashboard data."""
        return self.dashboard.generate_dashboard(hours)
    
    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            return self.metric_collector.export_prometheus_metrics()
        elif format == "json":
            return json.dumps(self.metric_collector.get_current_metrics(), indent=2)
        else:
            return "Unsupported format"


# Export main classes
__all__ = [
    'TelemetrySystem',
    'MetricCollector', 
    'AlertManager',
    'DistributedTracer',
    'PerformanceDashboard',
    'AnomalyDetector',
    'MetricType',
    'AlertSeverity',
    'Alert',
    'MetricPoint',
    'TraceSpan'
]