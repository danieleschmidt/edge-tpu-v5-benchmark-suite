"""Quantum System Monitoring and Health Checks

Advanced monitoring system for quantum task execution with TPU metrics integration.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import logging
from enum import Enum
import statistics

from .quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumState
from .quantum_validation import ValidationReport, QuantumSystemValidator

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float = field(default_factory=time.time)
    
    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    
    # Execution metrics
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = 0.0
    
    # Queue metrics
    queue_length: int = 0
    avg_queue_wait_time: float = 0.0
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    tpu_utilization: float = 0.0
    
    # Quantum metrics
    avg_coherence: float = 0.0
    entanglement_count: int = 0
    decoherence_rate: float = 0.0
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    successful_tasks_per_second: float = 0.0


class MetricsCollector:
    """Collects and stores system metrics over time"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.task_execution_history: deque = deque(maxlen=max_history * 10)
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Record a custom metric value"""
        with self._lock:
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=value,
                metadata=metadata or {}
            )
            self.custom_metrics[name].append(metric_point)
    
    def record_task_execution(self, task_id: str, duration: float, success: bool, metadata: Optional[Dict] = None) -> None:
        """Record task execution event"""
        with self._lock:
            execution_record = {
                'timestamp': time.time(),
                'task_id': task_id,
                'duration': duration,
                'success': success,
                'metadata': metadata or {}
            }
            self.task_execution_history.append(execution_record)
    
    def record_performance_snapshot(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics snapshot"""
        with self._lock:
            self.metrics_history.append(metrics)
    
    def get_metric_history(self, name: str, window_seconds: Optional[float] = None) -> List[MetricPoint]:
        """Get history for specific metric"""
        with self._lock:
            if name not in self.custom_metrics:
                return []
            
            history = list(self.custom_metrics[name])
            
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                history = [point for point in history if point.timestamp >= cutoff_time]
            
            return history
    
    def get_performance_history(self, window_seconds: Optional[float] = None) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        with self._lock:
            history = list(self.metrics_history)
            
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                history = [metrics for metrics in history if metrics.timestamp >= cutoff_time]
            
            return history
    
    def get_task_execution_stats(self, window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get task execution statistics"""
        with self._lock:
            executions = list(self.task_execution_history)
            
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                executions = [ex for ex in executions if ex['timestamp'] >= cutoff_time]
            
            if not executions:
                return {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'success_rate': 0.0,
                    'avg_duration': 0.0,
                    'min_duration': 0.0,
                    'max_duration': 0.0,
                    'tasks_per_second': 0.0
                }
            
            successful = [ex for ex in executions if ex['success']]
            failed = [ex for ex in executions if not ex['success']]
            durations = [ex['duration'] for ex in executions]
            
            # Calculate time span for throughput
            if len(executions) > 1:
                time_span = executions[-1]['timestamp'] - executions[0]['timestamp']
                tasks_per_second = len(executions) / max(time_span, 1.0)
            else:
                tasks_per_second = 0.0
            
            return {
                'total_executions': len(executions),
                'successful_executions': len(successful),
                'failed_executions': len(failed),
                'success_rate': len(successful) / len(executions) if executions else 0.0,
                'avg_duration': statistics.mean(durations) if durations else 0.0,
                'min_duration': min(durations) if durations else 0.0,
                'max_duration': max(durations) if durations else 0.0,
                'tasks_per_second': tasks_per_second
            }


class QuantumHealthMonitor:
    """Comprehensive health monitoring for quantum systems"""
    
    def __init__(self, planner: QuantumTaskPlanner):
        self.planner = planner
        self.validator = QuantumSystemValidator()
        self.metrics_collector = MetricsCollector()
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds = {}
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Register default health checks
        self._register_default_health_checks()
        self._set_default_thresholds()
    
    def _register_default_health_checks(self) -> None:
        """Register default health check functions"""
        self.health_checks = {
            'quantum_coherence': self._check_quantum_coherence,
            'resource_utilization': self._check_resource_utilization,
            'task_queue_health': self._check_task_queue_health,
            'execution_performance': self._check_execution_performance,
            'system_validation': self._check_system_validation,
            'decoherence_levels': self._check_decoherence_levels,
            'entanglement_integrity': self._check_entanglement_integrity,
            'memory_usage': self._check_memory_usage
        }
    
    def _set_default_thresholds(self) -> None:
        """Set default alert thresholds"""
        self.alert_thresholds = {
            'min_coherence': 0.3,
            'max_resource_utilization': 0.9,
            'max_queue_length': 100,
            'max_decoherence': 0.8,
            'min_success_rate': 0.8,
            'max_avg_execution_time': 60.0,
            'max_memory_utilization': 0.85
        }
    
    def set_alert_threshold(self, metric: str, value: float) -> None:
        """Set custom alert threshold"""
        self.alert_thresholds[metric] = value
        logger.info(f"Set alert threshold {metric} = {value}")
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous monitoring"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        logger.info(f"Starting quantum health monitoring (interval: {interval}s)")
        
        async def monitor_loop():
            while self._monitoring_active:
                try:
                    # Collect performance metrics
                    await self._collect_performance_metrics()
                    
                    # Run health checks
                    health_results = await self._run_all_health_checks()
                    
                    # Check for alerts
                    alerts = self._check_for_alerts(health_results)
                    if alerts:
                        self._handle_alerts(alerts)
                    
                    await asyncio.sleep(interval)
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(min(interval, 10.0))  # Shorter retry interval
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Quantum health monitoring stopped")
    
    async def _collect_performance_metrics(self) -> None:
        """Collect current performance metrics"""
        state = self.planner.get_system_state()
        
        # Get task execution statistics
        exec_stats = self.metrics_collector.get_task_execution_stats(window_seconds=300)  # 5 minutes
        
        metrics = PerformanceMetrics(
            total_tasks=state['total_tasks'],
            completed_tasks=state['completed_tasks'],
            failed_tasks=exec_stats.get('failed_executions', 0),
            active_tasks=len(self.planner.active_tasks),
            
            avg_execution_time=exec_stats.get('avg_duration', 0.0),
            max_execution_time=exec_stats.get('max_duration', 0.0),
            min_execution_time=exec_stats.get('min_duration', 0.0),
            
            queue_length=len(self.planner.get_ready_tasks()),
            
            # Resource utilization (from quantum planner state)
            tpu_utilization=state['resource_utilization'].get('tpu_v5_primary', 0.0),
            cpu_utilization=state['resource_utilization'].get('cpu_cores', 0.0),
            memory_utilization=state['resource_utilization'].get('memory_gb', 0.0),
            
            # Quantum metrics
            avg_coherence=state['quantum_metrics']['average_coherence'],
            entanglement_count=state['quantum_metrics']['entanglement_pairs'],
            
            # Throughput
            tasks_per_second=exec_stats.get('tasks_per_second', 0.0),
            successful_tasks_per_second=exec_stats.get('tasks_per_second', 0.0) * exec_stats.get('success_rate', 0.0)
        )
        
        self.metrics_collector.record_performance_snapshot(metrics)
        
        # Record individual metrics for trending
        self.metrics_collector.record_metric('coherence', metrics.avg_coherence)
        self.metrics_collector.record_metric('queue_length', metrics.queue_length)
        self.metrics_collector.record_metric('tasks_per_second', metrics.tasks_per_second)
    
    async def _run_all_health_checks(self) -> List[HealthCheck]:
        """Run all registered health checks"""
        health_results = []
        
        for check_name, check_func in self.health_checks.items():
            try:
                result = await asyncio.create_task(asyncio.to_thread(check_func))
                health_results.append(result)
            except Exception as e:
                error_result = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}",
                    details={'error': str(e)}
                )
                health_results.append(error_result)
                logger.error(f"Health check {check_name} failed: {e}")
        
        return health_results
    
    def _check_quantum_coherence(self) -> HealthCheck:
        """Check quantum coherence levels"""
        state = self.planner.get_system_state()
        coherence = state['quantum_metrics']['average_coherence']
        
        if coherence < self.alert_thresholds.get('min_coherence', 0.3):
            status = HealthStatus.CRITICAL
            message = f"Low quantum coherence: {coherence:.1%}"
        elif coherence < 0.5:
            status = HealthStatus.WARNING
            message = f"Moderate quantum coherence: {coherence:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Good quantum coherence: {coherence:.1%}"
        
        return HealthCheck(
            name="quantum_coherence",
            status=status,
            message=message,
            details={
                'coherence_level': coherence,
                'threshold': self.alert_thresholds.get('min_coherence', 0.3)
            }
        )
    
    def _check_resource_utilization(self) -> HealthCheck:
        """Check resource utilization levels"""
        state = self.planner.get_system_state()
        utilizations = state['resource_utilization']
        
        max_util = max(utilizations.values()) if utilizations else 0.0
        max_threshold = self.alert_thresholds.get('max_resource_utilization', 0.9)
        
        if max_util > max_threshold:
            status = HealthStatus.CRITICAL
            message = f"High resource utilization: {max_util:.1%}"
        elif max_util > 0.75:
            status = HealthStatus.WARNING
            message = f"Moderate resource utilization: {max_util:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Normal resource utilization: {max_util:.1%}"
        
        return HealthCheck(
            name="resource_utilization",
            status=status,
            message=message,
            details={
                'utilizations': utilizations,
                'max_utilization': max_util,
                'threshold': max_threshold
            }
        )
    
    def _check_task_queue_health(self) -> HealthCheck:
        """Check task queue health"""
        ready_tasks = self.planner.get_ready_tasks()
        queue_length = len(ready_tasks)
        max_queue = self.alert_thresholds.get('max_queue_length', 100)
        
        if queue_length > max_queue:
            status = HealthStatus.CRITICAL
            message = f"Queue backed up: {queue_length} tasks"
        elif queue_length > max_queue * 0.7:
            status = HealthStatus.WARNING
            message = f"Queue getting full: {queue_length} tasks"
        else:
            status = HealthStatus.HEALTHY
            message = f"Queue healthy: {queue_length} tasks"
        
        # Check for very old tasks
        oldest_task_age = 0.0
        if ready_tasks:
            current_time = time.time()
            oldest_task_age = max(current_time - task.created_at for task in ready_tasks)
        
        return HealthCheck(
            name="task_queue_health",
            status=status,
            message=message,
            details={
                'queue_length': queue_length,
                'oldest_task_age': oldest_task_age,
                'threshold': max_queue
            }
        )
    
    def _check_execution_performance(self) -> HealthCheck:
        """Check task execution performance"""
        exec_stats = self.metrics_collector.get_task_execution_stats(window_seconds=300)
        
        success_rate = exec_stats.get('success_rate', 1.0)
        avg_duration = exec_stats.get('avg_duration', 0.0)
        
        min_success_rate = self.alert_thresholds.get('min_success_rate', 0.8)
        max_avg_time = self.alert_thresholds.get('max_avg_execution_time', 60.0)
        
        if success_rate < min_success_rate:
            status = HealthStatus.CRITICAL
            message = f"Low success rate: {success_rate:.1%}"
        elif avg_duration > max_avg_time:
            status = HealthStatus.WARNING
            message = f"Slow execution: {avg_duration:.1f}s average"
        elif exec_stats.get('total_executions', 0) == 0:
            status = HealthStatus.WARNING
            message = "No recent task executions"
        else:
            status = HealthStatus.HEALTHY
            message = f"Good performance: {success_rate:.1%} success, {avg_duration:.1f}s avg"
        
        return HealthCheck(
            name="execution_performance",
            status=status,
            message=message,
            details=exec_stats
        )
    
    def _check_system_validation(self) -> HealthCheck:
        """Check system validation status"""
        try:
            validation_report = self.validator.validate_system(
                self.planner.tasks,
                self.planner.resources
            )
            
            if validation_report.critical_issues > 0:
                status = HealthStatus.CRITICAL
                message = f"Critical validation issues: {validation_report.critical_issues}"
            elif validation_report.error_issues > 0:
                status = HealthStatus.DEGRADED
                message = f"Validation errors: {validation_report.error_issues}"
            elif validation_report.warning_issues > 0:
                status = HealthStatus.WARNING
                message = f"Validation warnings: {validation_report.warning_issues}"
            else:
                status = HealthStatus.HEALTHY
                message = "System validation passed"
            
            return HealthCheck(
                name="system_validation",
                status=status,
                message=message,
                details={
                    'total_issues': validation_report.total_issues,
                    'critical_issues': validation_report.critical_issues,
                    'error_issues': validation_report.error_issues,
                    'warning_issues': validation_report.warning_issues,
                    'validation_time': validation_report.validation_time
                }
            )
        
        except Exception as e:
            return HealthCheck(
                name="system_validation",
                status=HealthStatus.UNKNOWN,
                message=f"Validation check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _check_decoherence_levels(self) -> HealthCheck:
        """Check quantum decoherence levels"""
        decoherent_tasks = []
        high_decoherence_tasks = []
        
        for task in self.planner.tasks.values():
            decoherence = task.measure_decoherence()
            if decoherence > 0.9:
                decoherent_tasks.append((task.id, decoherence))
            elif decoherence > self.alert_thresholds.get('max_decoherence', 0.8):
                high_decoherence_tasks.append((task.id, decoherence))
        
        if decoherent_tasks:
            status = HealthStatus.CRITICAL
            message = f"{len(decoherent_tasks)} tasks highly decoherent"
        elif high_decoherence_tasks:
            status = HealthStatus.WARNING
            message = f"{len(high_decoherence_tasks)} tasks showing decoherence"
        else:
            status = HealthStatus.HEALTHY
            message = "Decoherence levels normal"
        
        return HealthCheck(
            name="decoherence_levels",
            status=status,
            message=message,
            details={
                'decoherent_tasks': decoherent_tasks,
                'high_decoherence_tasks': high_decoherence_tasks,
                'threshold': self.alert_thresholds.get('max_decoherence', 0.8)
            }
        )
    
    def _check_entanglement_integrity(self) -> HealthCheck:
        """Check quantum entanglement integrity"""
        broken_entanglements = []
        
        for task in self.planner.tasks.values():
            for entangled_id in task.entangled_tasks:
                if entangled_id not in self.planner.tasks:
                    broken_entanglements.append((task.id, entangled_id))
                elif task.id not in self.planner.tasks[entangled_id].entangled_tasks:
                    broken_entanglements.append((task.id, entangled_id))
        
        if broken_entanglements:
            status = HealthStatus.WARNING
            message = f"{len(broken_entanglements)} broken entanglements"
        else:
            status = HealthStatus.HEALTHY
            message = "Entanglement integrity maintained"
        
        return HealthCheck(
            name="entanglement_integrity",
            status=status,
            message=message,
            details={'broken_entanglements': broken_entanglements}
        )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage patterns"""
        # Simple memory utilization check
        state = self.planner.get_system_state()
        memory_util = state['resource_utilization'].get('memory_gb', 0.0)
        max_memory_util = self.alert_thresholds.get('max_memory_utilization', 0.85)
        
        if memory_util > max_memory_util:
            status = HealthStatus.CRITICAL
            message = f"High memory usage: {memory_util:.1%}"
        elif memory_util > 0.7:
            status = HealthStatus.WARNING
            message = f"Moderate memory usage: {memory_util:.1%}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Normal memory usage: {memory_util:.1%}"
        
        return HealthCheck(
            name="memory_usage",
            status=status,
            message=message,
            details={
                'memory_utilization': memory_util,
                'threshold': max_memory_util
            }
        )
    
    def _check_for_alerts(self, health_results: List[HealthCheck]) -> List[HealthCheck]:
        """Check health results for alert conditions"""
        alerts = []
        for result in health_results:
            if result.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alerts.append(result)
        return alerts
    
    def _handle_alerts(self, alerts: List[HealthCheck]) -> None:
        """Handle system alerts"""
        for alert in alerts:
            logger.warning(f"ALERT [{alert.name}]: {alert.message}")
            
            # Record alert metric
            self.metrics_collector.record_metric(
                f"alert_{alert.name}",
                1.0,
                {'status': alert.status.value, 'message': alert.message}
            )
    
    def get_current_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        health_results = []
        
        # Run all health checks synchronously
        for check_name, check_func in self.health_checks.items():
            try:
                result = check_func()
                health_results.append(result)
            except Exception as e:
                error_result = HealthCheck(
                    name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}"
                )
                health_results.append(error_result)
        
        # Determine overall status
        status_priority = {
            HealthStatus.CRITICAL: 4,
            HealthStatus.DEGRADED: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.HEALTHY: 1,
            HealthStatus.UNKNOWN: 0
        }
        
        overall_status = HealthStatus.HEALTHY
        for result in health_results:
            if status_priority[result.status] > status_priority[overall_status]:
                overall_status = result.status
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'health_checks': [
                {
                    'name': result.name,
                    'status': result.status.value,
                    'message': result.message,
                    'details': result.details
                }
                for result in health_results
            ],
            'metrics_summary': self._get_metrics_summary()
        }
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics"""
        recent_metrics = self.metrics_collector.get_performance_history(window_seconds=300)
        if not recent_metrics:
            return {}
        
        latest = recent_metrics[-1]
        return {
            'tasks_per_second': latest.tasks_per_second,
            'success_rate': latest.successful_tasks_per_second / max(latest.tasks_per_second, 0.001),
            'avg_coherence': latest.avg_coherence,
            'queue_length': latest.queue_length,
            'resource_utilization': {
                'cpu': latest.cpu_utilization,
                'memory': latest.memory_utilization,
                'tpu': latest.tpu_utilization
            }
        }
    
    def export_health_report(self, filename: str) -> None:
        """Export comprehensive health report"""
        report_data = {
            'timestamp': time.time(),
            'health_status': self.get_current_health_status(),
            'performance_history': [
                {
                    'timestamp': m.timestamp,
                    'tasks_per_second': m.tasks_per_second,
                    'coherence': m.avg_coherence,
                    'queue_length': m.queue_length,
                    'resource_utilization': {
                        'cpu': m.cpu_utilization,
                        'memory': m.memory_utilization,
                        'tpu': m.tpu_utilization
                    }
                }
                for m in self.metrics_collector.get_performance_history()
            ],
            'execution_statistics': self.metrics_collector.get_task_execution_stats(),
            'alert_thresholds': self.alert_thresholds
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Health report exported to {filename}")


def create_health_dashboard_data(monitor: QuantumHealthMonitor) -> Dict[str, Any]:
    """Create data structure for health dashboard visualization"""
    health_status = monitor.get_current_health_status()
    recent_metrics = monitor.metrics_collector.get_performance_history(window_seconds=3600)  # 1 hour
    
    # Prepare time series data
    time_series = {
        'timestamps': [m.timestamp for m in recent_metrics],
        'coherence': [m.avg_coherence for m in recent_metrics],
        'tasks_per_second': [m.tasks_per_second for m in recent_metrics],
        'queue_length': [m.queue_length for m in recent_metrics],
        'cpu_utilization': [m.cpu_utilization for m in recent_metrics],
        'memory_utilization': [m.memory_utilization for m in recent_metrics],
        'tpu_utilization': [m.tpu_utilization for m in recent_metrics]
    }
    
    return {
        'overall_status': health_status['overall_status'],
        'health_checks': health_status['health_checks'],
        'time_series': time_series,
        'current_metrics': health_status.get('metrics_summary', {}),
        'alerts': [
            check for check in health_status['health_checks']
            if check['status'] in ['critical', 'degraded']
        ]
    }