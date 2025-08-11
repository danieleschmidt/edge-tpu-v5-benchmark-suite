"""Quantum System Monitoring and Health Checks

Advanced monitoring system for quantum task execution with TPU metrics integration,
enhanced with circuit breakers, retry mechanisms, and comprehensive error handling.
"""

import time
import asyncio
import threading
import weakref
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import logging
from enum import Enum
import statistics
from contextlib import asynccontextmanager

from .quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumState
from .quantum_validation import ValidationReport, QuantumSystemValidator
from .exceptions import (
    QuantumError, QuantumMonitoringError, QuantumCircuitBreakerError,
    QuantumRetryExhaustedError, ErrorContext, handle_quantum_error,
    quantum_operation, validate_input, CircuitBreaker, CircuitBreakerConfig,
    RetryManager, RetryConfig,
    ErrorHandlingContext, AsyncErrorHandlingContext
)
from .security import InputValidator, DataSanitizer

# Configure structured logging for monitoring
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter for structured monitoring logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(component)s:%(operation)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Individual metric data point with validation and metadata"""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_id: str = ""
    source: str = "unknown"
    quality: float = 1.0  # Data quality score (0-1)
    
    def __post_init__(self):
        # Validate and sanitize data
        if not isinstance(self.timestamp, (int, float)) or self.timestamp <= 0:
            self.timestamp = time.time()
            self.quality *= 0.5
        
        if not isinstance(self.value, (int, float)):
            try:
                self.value = float(self.value)
            except (ValueError, TypeError):
                self.value = 0.0
                self.quality = 0.0
        
        # Sanitize metadata
        if self.metadata:
            self.metadata = DataSanitizer.sanitize_dict(self.metadata, max_keys=10)
        
        # Generate metric ID if not provided
        if not self.metric_id:
            self.metric_id = f"metric_{int(self.timestamp)}_{hash(str(self.value)) % 10000}"


@dataclass
class HealthCheck:
    """Health check result with enhanced metadata and validation"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    check_id: str = ""
    execution_time: float = 0.0
    next_check_time: Optional[float] = None
    check_count: int = 0
    failure_count: int = 0
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate and sanitize
        self.name = DataSanitizer.sanitize_string(self.name, max_length=100)
        self.message = DataSanitizer.sanitize_string(self.message, max_length=500)
        
        if self.details:
            self.details = DataSanitizer.sanitize_dict(self.details, max_keys=20)
        
        if not self.check_id:
            self.check_id = f"check_{self.name}_{int(self.timestamp)}"
    
    def is_healthy(self) -> bool:
        """Check if health status indicates system is healthy"""
        return self.status == HealthStatus.HEALTHY
    
    def is_critical(self) -> bool:
        """Check if health status indicates critical issues"""
        return self.status == HealthStatus.CRITICAL
    
    def get_severity_score(self) -> float:
        """Get numeric severity score (0-1, higher is worse)"""
        severity_map = {
            HealthStatus.HEALTHY: 0.0,
            HealthStatus.WARNING: 0.25,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.CRITICAL: 1.0,
            HealthStatus.UNKNOWN: 0.75
        }
        return severity_map.get(self.status, 0.75)


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
    """Collects and stores system metrics over time with enhanced error handling and validation"""
    
    def __init__(self, max_history: int = 1000, enable_persistence: bool = False):
        # Validate and sanitize parameters
        self.max_history = max(100, min(max_history, 100000))  # Reasonable bounds
        self.enable_persistence = enable_persistence
        
        # Thread-safe data structures
        self._lock = threading.RLock()
        self.metrics_history: deque = deque(maxlen=self.max_history)
        self.task_execution_history: deque = deque(maxlen=self.max_history * 10)
        self.custom_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        
        # Error tracking
        self.collection_errors: deque = deque(maxlen=100)
        self.metrics_collected = 0
        self.collection_failures = 0
        
        # Circuit breaker for metric collection resilience
        cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=30.0,
            expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(cb_config)
        
        logger.info(
            f"MetricsCollector initialized with max_history={self.max_history}, "
            f"persistence={enable_persistence}",
            extra={"component": "metrics_collector", "operation": "__init__"}
        )
    
    @validate_input(
        lambda self, name, value, metadata=None: (
            isinstance(name, str) and len(name) > 0 and 
            isinstance(value, (int, float))
        ),
        "Invalid metric name or value"
    )
    def record_metric(self, name: str, value: float, metadata: Optional[Dict] = None) -> None:
        """Record a custom metric value with comprehensive validation and error handling"""
        try:
            # Apply circuit breaker
            self.circuit_breaker.call(self._record_metric_internal, name, value, metadata)
        except QuantumCircuitBreakerError as e:
            logger.warning(
                f"Metrics collection circuit breaker open: {e}",
                extra={"component": "metrics_collector", "operation": "record_metric"}
            )
            self.collection_failures += 1
        except Exception as e:
            logger.error(
                f"Error recording metric {name}: {e}",
                extra={"component": "metrics_collector", "operation": "record_metric",
                      "metric_name": name}
            )
            self.collection_failures += 1
            self._record_collection_error("record_metric", name, str(e))
    
    def _record_metric_internal(self, name: str, value: float, metadata: Optional[Dict]) -> None:
        """Internal metric recording with validation"""
        with self._lock:
            # Sanitize inputs
            name = DataSanitizer.sanitize_string(name, max_length=100)
            
            # Validate value
            if not isinstance(value, (int, float)) or not (-1e10 <= value <= 1e10):
                raise ValueError(f"Invalid metric value: {value}")
            
            # Create metric point with validation
            metric_point = MetricPoint(
                timestamp=time.time(),
                value=float(value),
                metadata=metadata or {},
                source="custom"
            )
            
            # Store metric
            self.custom_metrics[name].append(metric_point)
            self.metrics_collected += 1
            
            logger.debug(
                f"Recorded metric {name}={value}",
                extra={"component": "metrics_collector", "operation": "_record_metric_internal",
                      "metric_name": name, "metric_value": value}
            )
    
    @validate_input(
        lambda self, task_id, duration, success, metadata=None: (
            isinstance(task_id, str) and len(task_id) > 0 and
            isinstance(duration, (int, float)) and duration >= 0 and
            isinstance(success, bool)
        ),
        "Invalid task execution parameters"
    )
    def record_task_execution(self, task_id: str, duration: float, success: bool, metadata: Optional[Dict] = None) -> None:
        """Record task execution event with comprehensive validation"""
        try:
            with self._lock:
                # Sanitize inputs
                task_id = DataSanitizer.sanitize_string(task_id, max_length=200)
                duration = max(0.0, min(duration, 86400.0))  # Cap at 24 hours
                
                execution_record = {
                    'record_id': f"exec_{task_id}_{int(time.time())}_{hash(task_id) % 1000}",
                    'timestamp': time.time(),
                    'task_id': task_id,
                    'duration': duration,
                    'success': success,
                    'metadata': DataSanitizer.sanitize_dict(metadata or {}, max_keys=15)
                }
                
                self.task_execution_history.append(execution_record)
                self.metrics_collected += 1
                
                logger.debug(
                    f"Recorded task execution: {task_id} ({'success' if success else 'failed'}) in {duration:.3f}s",
                    extra={"component": "metrics_collector", "operation": "record_task_execution",
                          "task_id": task_id, "duration": duration, "success": success}
                )
                
        except Exception as e:
            logger.error(
                f"Error recording task execution for {task_id}: {e}",
                extra={"component": "metrics_collector", "operation": "record_task_execution",
                      "task_id": task_id}
            )
            self.collection_failures += 1
            self._record_collection_error("record_task_execution", task_id, str(e))
    
    @validate_input(
        lambda self, metrics: hasattr(metrics, 'timestamp'),
        "Invalid performance metrics object"
    )
    def record_performance_snapshot(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics snapshot with validation"""
        try:
            with self._lock:
                # Validate metrics object
                if not hasattr(metrics, 'timestamp'):
                    raise ValueError("Performance metrics missing timestamp")
                
                # Sanitize metrics values
                sanitized_metrics = PerformanceMetrics(
                    timestamp=getattr(metrics, 'timestamp', time.time()),
                    total_tasks=max(0, getattr(metrics, 'total_tasks', 0)),
                    completed_tasks=max(0, getattr(metrics, 'completed_tasks', 0)),
                    failed_tasks=max(0, getattr(metrics, 'failed_tasks', 0)),
                    active_tasks=max(0, getattr(metrics, 'active_tasks', 0)),
                    avg_execution_time=max(0.0, getattr(metrics, 'avg_execution_time', 0.0)),
                    max_execution_time=max(0.0, getattr(metrics, 'max_execution_time', 0.0)),
                    min_execution_time=max(0.0, getattr(metrics, 'min_execution_time', 0.0)),
                    queue_length=max(0, getattr(metrics, 'queue_length', 0)),
                    avg_queue_wait_time=max(0.0, getattr(metrics, 'avg_queue_wait_time', 0.0)),
                    cpu_utilization=max(0.0, min(1.0, getattr(metrics, 'cpu_utilization', 0.0))),
                    memory_utilization=max(0.0, min(1.0, getattr(metrics, 'memory_utilization', 0.0))),
                    tpu_utilization=max(0.0, min(1.0, getattr(metrics, 'tpu_utilization', 0.0))),
                    avg_coherence=max(0.0, min(1.0, getattr(metrics, 'avg_coherence', 0.0))),
                    entanglement_count=max(0, getattr(metrics, 'entanglement_count', 0)),
                    decoherence_rate=max(0.0, min(1.0, getattr(metrics, 'decoherence_rate', 0.0))),
                    tasks_per_second=max(0.0, getattr(metrics, 'tasks_per_second', 0.0)),
                    successful_tasks_per_second=max(0.0, getattr(metrics, 'successful_tasks_per_second', 0.0))
                )
                
                self.metrics_history.append(sanitized_metrics)
                self.metrics_collected += 1
                
                logger.debug(
                    f"Recorded performance snapshot: {sanitized_metrics.total_tasks} total tasks, "
                    f"{sanitized_metrics.tasks_per_second:.2f} tasks/sec",
                    extra={"component": "metrics_collector", "operation": "record_performance_snapshot",
                          "total_tasks": sanitized_metrics.total_tasks,
                          "tasks_per_second": sanitized_metrics.tasks_per_second}
                )
                
        except Exception as e:
            logger.error(
                f"Error recording performance snapshot: {e}",
                extra={"component": "metrics_collector", "operation": "record_performance_snapshot"}
            )
            self.collection_failures += 1
            self._record_collection_error("record_performance_snapshot", "system", str(e))
    
    def _record_collection_error(self, operation: str, target: str, error: str) -> None:
        """Record collection error for debugging"""
        try:
            error_record = {
                'timestamp': time.time(),
                'operation': operation,
                'target': target,
                'error': error,
                'error_count': self.collection_failures
            }
            self.collection_errors.append(error_record)
        except Exception:
            pass  # Best effort error recording
    
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
        """Get task execution statistics with comprehensive error handling"""
        try:
            with self._lock:
                executions = list(self.task_execution_history)
                
                # Apply time window filter with validation
                if window_seconds is not None:
                    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
                        logger.warning(
                            f"Invalid window_seconds: {window_seconds}, using None",
                            extra={"component": "metrics_collector", "operation": "get_task_execution_stats"}
                        )
                        window_seconds = None
                    else:
                        cutoff_time = time.time() - window_seconds
                        executions = [ex for ex in executions if ex.get('timestamp', 0) >= cutoff_time]
                
                # Return empty stats for no executions
                if not executions:
                    return {
                        'total_executions': 0,
                        'successful_executions': 0,
                        'failed_executions': 0,
                        'success_rate': 0.0,
                        'avg_duration': 0.0,
                        'min_duration': 0.0,
                        'max_duration': 0.0,
                        'tasks_per_second': 0.0,
                        'percentile_95_duration': 0.0,
                        'error_rate': 0.0,
                        'window_seconds': window_seconds,
                        'data_quality': 1.0
                    }
                
                # Separate successful and failed executions with validation
                successful = []
                failed = []
                valid_durations = []
                
                for ex in executions:
                    try:
                        if isinstance(ex, dict) and 'success' in ex:
                            if ex['success']:
                                successful.append(ex)
                            else:
                                failed.append(ex)
                            
                            # Collect valid durations
                            duration = ex.get('duration', 0)
                            if isinstance(duration, (int, float)) and duration >= 0:
                                valid_durations.append(duration)
                    except Exception as e:
                        logger.debug(
                            f"Skipping invalid execution record: {e}",
                            extra={"component": "metrics_collector", "operation": "get_task_execution_stats"}
                        )
                        continue
                
                # Calculate throughput with error handling
                tasks_per_second = 0.0
                try:
                    if len(executions) > 1:
                        timestamps = [ex.get('timestamp', 0) for ex in executions if 'timestamp' in ex]
                        if len(timestamps) > 1:
                            time_span = max(timestamps) - min(timestamps)
                            if time_span > 0:
                                tasks_per_second = len(executions) / time_span
                except Exception as e:
                    logger.debug(
                        f"Error calculating tasks per second: {e}",
                        extra={"component": "metrics_collector", "operation": "get_task_execution_stats"}
                    )
                
                # Calculate statistics with error handling
                stats = {
                    'total_executions': len(executions),
                    'successful_executions': len(successful),
                    'failed_executions': len(failed),
                    'success_rate': len(successful) / len(executions) if executions else 0.0,
                    'error_rate': len(failed) / len(executions) if executions else 0.0,
                    'tasks_per_second': tasks_per_second,
                    'window_seconds': window_seconds,
                    'data_quality': len(valid_durations) / max(len(executions), 1)
                }
                
                # Duration statistics with error handling
                if valid_durations:
                    try:
                        stats.update({
                            'avg_duration': statistics.mean(valid_durations),
                            'min_duration': min(valid_durations),
                            'max_duration': max(valid_durations),
                            'median_duration': statistics.median(valid_durations),
                            'percentile_95_duration': (
                                sorted(valid_durations)[int(0.95 * len(valid_durations))] 
                                if len(valid_durations) > 1 else valid_durations[0]
                            )
                        })
                    except Exception as e:
                        logger.warning(
                            f"Error calculating duration statistics: {e}",
                            extra={"component": "metrics_collector", "operation": "get_task_execution_stats"}
                        )
                        stats.update({
                            'avg_duration': 0.0,
                            'min_duration': 0.0,
                            'max_duration': 0.0,
                            'median_duration': 0.0,
                            'percentile_95_duration': 0.0
                        })
                else:
                    stats.update({
                        'avg_duration': 0.0,
                        'min_duration': 0.0,
                        'max_duration': 0.0,
                        'median_duration': 0.0,
                        'percentile_95_duration': 0.0
                    })
                
                return stats
                
        except Exception as e:
            logger.error(
                f"Error getting task execution stats: {e}",
                extra={"component": "metrics_collector", "operation": "get_task_execution_stats"}
            )
            # Return safe default stats
            return {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'error_rate': 0.0,
                'avg_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'tasks_per_second': 0.0,
                'error': str(e),
                'data_quality': 0.0
            }
    
    def get_collection_health(self) -> Dict[str, Any]:
        """Get metrics collection health status"""
        try:
            with self._lock:
                total_operations = self.metrics_collected + self.collection_failures
                success_rate = self.metrics_collected / max(total_operations, 1)
                
                return {
                    'metrics_collected': self.metrics_collected,
                    'collection_failures': self.collection_failures,
                    'success_rate': success_rate,
                    'circuit_breaker_state': self.circuit_breaker.get_state(),
                    'recent_errors': list(self.collection_errors)[-5:],  # Last 5 errors
                    'health_status': 'healthy' if success_rate > 0.9 else 'degraded' if success_rate > 0.5 else 'unhealthy'
                }
        except Exception as e:
            return {
                'error': str(e),
                'health_status': 'unknown'
            }


class QuantumHealthMonitor:
    """Comprehensive health monitoring for quantum systems with enhanced resilience"""
    
    def __init__(self, planner: QuantumTaskPlanner, 
                 monitoring_interval: float = 30.0,
                 enable_alerts: bool = True,
                 max_concurrent_checks: int = 5):
        
        # Validate parameters
        self.monitoring_interval = max(1.0, min(monitoring_interval, 300.0))  # 1s to 5min
        self.max_concurrent_checks = max(1, min(max_concurrent_checks, 20))
        self.enable_alerts = enable_alerts
        
        # Core components with weak reference to avoid circular dependencies
        try:
            self.planner_ref = weakref.ref(planner)
        except TypeError:
            logger.warning(
                "Unable to create weak reference to planner, using direct reference",
                extra={"component": "quantum_health_monitor", "operation": "__init__"}
            )
            self.planner = planner
            self.planner_ref = None
        
        self.validator = QuantumSystemValidator()
        self.metrics_collector = MetricsCollector(max_history=2000)
        
        # Monitoring state
        self._lock = threading.RLock()
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds = {}
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._check_semaphore = asyncio.Semaphore(max_concurrent_checks)
        
        # Health check history and statistics
        self.check_history = defaultdict(lambda: deque(maxlen=100))
        self.check_statistics = defaultdict(lambda: {
            'total_runs': 0,
            'failures': 0,
            'avg_duration': 0.0,
            'last_success': None,
            'last_failure': None
        })
        
        # Circuit breakers for health checks
        self.check_circuit_breakers = {}
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=500)
        self.alert_callbacks: List[Callable] = []
        
        # Register default components
        self._register_default_health_checks()
        self._set_default_thresholds()
        
        logger.info(
            f"QuantumHealthMonitor initialized with interval={monitoring_interval}s, "
            f"alerts={'enabled' if enable_alerts else 'disabled'}",
            extra={"component": "quantum_health_monitor", "operation": "__init__",
                  "monitoring_interval": monitoring_interval, "enable_alerts": enable_alerts}
        )
    
    @property
    def planner(self) -> Optional[QuantumTaskPlanner]:
        """Get planner from weak reference if available"""
        if self.planner_ref:
            return self.planner_ref()
        return getattr(self, '_planner', None)
    
    @planner.setter
    def planner(self, value: QuantumTaskPlanner) -> None:
        """Set planner reference"""
        self._planner = value
    
    def _register_default_health_checks(self) -> None:
        """Register default health check functions with metadata and circuit breakers"""
        default_checks = {
            'quantum_coherence': {
                'function': self._check_quantum_coherence,
                'description': 'Monitor quantum coherence levels',
                'interval': 30.0,
                'timeout': 5.0,
                'critical': True,
                'enabled': True
            },
            'resource_utilization': {
                'function': self._check_resource_utilization,
                'description': 'Monitor resource utilization levels',
                'interval': 15.0,
                'timeout': 3.0,
                'critical': True,
                'enabled': True
            },
            'task_queue_health': {
                'function': self._check_task_queue_health,
                'description': 'Monitor task queue status',
                'interval': 20.0,
                'timeout': 5.0,
                'critical': True,
                'enabled': True
            },
            'execution_performance': {
                'function': self._check_execution_performance,
                'description': 'Monitor task execution performance',
                'interval': 30.0,
                'timeout': 5.0,
                'critical': False,
                'enabled': True
            },
            'system_validation': {
                'function': self._check_system_validation,
                'description': 'Run system validation checks',
                'interval': 60.0,
                'timeout': 10.0,
                'critical': False,
                'enabled': True
            },
            'decoherence_levels': {
                'function': self._check_decoherence_levels,
                'description': 'Monitor quantum decoherence levels',
                'interval': 25.0,
                'timeout': 3.0,
                'critical': False,
                'enabled': True
            },
            'entanglement_integrity': {
                'function': self._check_entanglement_integrity,
                'description': 'Verify quantum entanglement integrity',
                'interval': 45.0,
                'timeout': 5.0,
                'critical': False,
                'enabled': True
            },
            'memory_usage': {
                'function': self._check_memory_usage,
                'description': 'Monitor system memory usage',
                'interval': 20.0,
                'timeout': 2.0,
                'critical': False,
                'enabled': True
            },
            'metrics_collection_health': {
                'function': self._check_metrics_collection_health,
                'description': 'Monitor metrics collection system health',
                'interval': 60.0,
                'timeout': 5.0,
                'critical': False,
                'enabled': True
            }
        }
        
        # Register checks and create circuit breakers
        for check_name, check_config in default_checks.items():
            self.health_checks[check_name] = check_config
            
            # Create circuit breaker for each check
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=check_config['interval'] * 2,
                expected_exception=Exception
            )
            self.check_circuit_breakers[check_name] = CircuitBreaker(cb_config)
            
            logger.debug(
                f"Registered health check: {check_name}",
                extra={"component": "quantum_health_monitor", "operation": "_register_default_health_checks",
                      "check_name": check_name}
            )
    
    def _set_default_thresholds(self) -> None:
        """Set default alert thresholds with validation"""
        default_thresholds = {
            'min_coherence': 0.3,
            'max_resource_utilization': 0.9,
            'max_queue_length': 100,
            'max_decoherence': 0.8,
            'min_success_rate': 0.8,
            'max_avg_execution_time': 60.0,
            'max_memory_utilization': 0.85,
            'max_failure_rate': 0.2,
            'min_tasks_per_second': 0.1,
            'max_validation_errors': 5,
            'max_circuit_breaker_open_time': 300.0
        }
        
        # Validate and set thresholds
        self.alert_thresholds = {}
        for threshold_name, threshold_value in default_thresholds.items():
            try:
                if isinstance(threshold_value, (int, float)) and threshold_value >= 0:
                    self.alert_thresholds[threshold_name] = threshold_value
                else:
                    logger.warning(
                        f"Invalid threshold value for {threshold_name}: {threshold_value}",
                        extra={"component": "quantum_health_monitor", "operation": "_set_default_thresholds"}
                    )
            except Exception as e:
                logger.error(
                    f"Error setting threshold {threshold_name}: {e}",
                    extra={"component": "quantum_health_monitor", "operation": "_set_default_thresholds"}
                )
        
        logger.info(
            f"Set {len(self.alert_thresholds)} alert thresholds",
            extra={"component": "quantum_health_monitor", "operation": "_set_default_thresholds",
                  "threshold_count": len(self.alert_thresholds)}
        )
    
    @validate_input(
        lambda self, metric, value: (
            isinstance(metric, str) and len(metric) > 0 and
            isinstance(value, (int, float))
        ),
        "Invalid metric name or threshold value"
    )
    def set_alert_threshold(self, metric: str, value: float) -> None:
        """Set custom alert threshold with validation"""
        try:
            # Sanitize inputs
            metric = DataSanitizer.sanitize_string(metric, max_length=100)
            value = float(value)
            
            # Validate threshold value ranges
            if metric.startswith('min_') and value < 0:
                raise ValueError(f"Minimum threshold cannot be negative: {value}")
            elif metric.startswith('max_') and value <= 0:
                raise ValueError(f"Maximum threshold must be positive: {value}")
            elif metric.endswith('_rate') and not (0.0 <= value <= 1.0):
                raise ValueError(f"Rate threshold must be between 0 and 1: {value}")
            
            old_value = self.alert_thresholds.get(metric)
            self.alert_thresholds[metric] = value
            
            logger.info(
                f"Updated alert threshold {metric}: {old_value} -> {value}",
                extra={"component": "quantum_health_monitor", "operation": "set_alert_threshold",
                      "metric": metric, "old_value": old_value, "new_value": value}
            )
            
        except Exception as e:
            logger.error(
                f"Error setting alert threshold {metric}={value}: {e}",
                extra={"component": "quantum_health_monitor", "operation": "set_alert_threshold",
                      "metric": metric, "value": value}
            )
            raise QuantumMonitoringError(f"Failed to set alert threshold: {e}") from e
    
    @validate_input(
        lambda self, interval=30.0: isinstance(interval, (int, float)) and interval > 0,
        "Invalid monitoring interval"
    )
    async def start_monitoring(self, interval: Optional[float] = None) -> None:
        """Start continuous monitoring with enhanced error handling and resilience"""
        try:
            if self._monitoring_active:
                logger.warning(
                    "Monitoring already active",
                    extra={"component": "quantum_health_monitor", "operation": "start_monitoring"}
                )
                return
            
            # Use provided interval or default
            monitoring_interval = interval if interval is not None else self.monitoring_interval
            monitoring_interval = max(1.0, min(monitoring_interval, 300.0))
            
            with self._lock:
                self._monitoring_active = True
            
            logger.info(
                f"Starting quantum health monitoring with interval {monitoring_interval}s",
                extra={"component": "quantum_health_monitor", "operation": "start_monitoring",
                      "interval": monitoring_interval, "max_concurrent_checks": self.max_concurrent_checks}
            )
            
            # Create monitoring coroutine with comprehensive error handling
            async def monitor_loop():
                consecutive_failures = 0
                max_consecutive_failures = 5
                
                while self._monitoring_active:
                    loop_start = time.time()
                    
                    try:
                        # Collect performance metrics with timeout
                        await asyncio.wait_for(
                            self._collect_performance_metrics(),
                            timeout=monitoring_interval / 2
                        )
                        
                        # Run health checks with concurrency control
                        health_results = await asyncio.wait_for(
                            self._run_all_health_checks(),
                            timeout=monitoring_interval * 0.8
                        )
                        
                        # Process alerts
                        if self.enable_alerts:
                            alerts = self._check_for_alerts(health_results)
                            if alerts:
                                await self._handle_alerts(alerts)
                        
                        # Reset failure counter on success
                        consecutive_failures = 0
                        
                        logger.debug(
                            f"Monitoring cycle completed in {time.time() - loop_start:.2f}s",
                            extra={"component": "quantum_health_monitor", "operation": "monitor_loop",
                                  "cycle_duration": time.time() - loop_start,
                                  "health_checks": len(health_results)}
                        )
                    
                    except asyncio.TimeoutError:
                        consecutive_failures += 1
                        logger.warning(
                            f"Monitoring cycle timed out (attempt {consecutive_failures})",
                            extra={"component": "quantum_health_monitor", "operation": "monitor_loop",
                                  "consecutive_failures": consecutive_failures}
                        )
                    
                    except Exception as e:
                        consecutive_failures += 1
                        logger.error(
                            f"Error in monitoring loop (attempt {consecutive_failures}): {e}",
                            extra={"component": "quantum_health_monitor", "operation": "monitor_loop",
                                  "consecutive_failures": consecutive_failures, "error": str(e)}
                        )
                        
                        # Create system health alert for monitoring failures
                        if consecutive_failures >= max_consecutive_failures:
                            critical_alert = HealthCheck(
                                name="monitoring_system",
                                status=HealthStatus.CRITICAL,
                                message=f"Monitoring system failing: {consecutive_failures} consecutive failures",
                                details={"error": str(e), "failures": consecutive_failures},
                                recovery_suggestions=[
                                    "Check system resources",
                                    "Restart monitoring system",
                                    "Verify system health"
                                ]
                            )
                            await self._handle_alerts([critical_alert])
                    
                    # Calculate sleep time accounting for processing time
                    cycle_duration = time.time() - loop_start
                    sleep_time = max(0.1, monitoring_interval - cycle_duration)
                    
                    # Use exponential backoff for failures
                    if consecutive_failures > 0:
                        sleep_time *= min(2 ** consecutive_failures, 8)  # Max 8x backoff
                    
                    await asyncio.sleep(sleep_time)
                
                logger.info(
                    "Monitoring loop stopped",
                    extra={"component": "quantum_health_monitor", "operation": "monitor_loop"}
                )
            
            # Start monitoring task
            self._monitor_task = asyncio.create_task(monitor_loop())
            
        except Exception as e:
            with self._lock:
                self._monitoring_active = False
            
            context = ErrorContext(
                component="quantum_health_monitor",
                operation="start_monitoring"
            )
            handle_quantum_error(e, context)
    
    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring with graceful shutdown"""
        try:
            logger.info(
                "Stopping quantum health monitoring",
                extra={"component": "quantum_health_monitor", "operation": "stop_monitoring"}
            )
            
            with self._lock:
                self._monitoring_active = False
            
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                
                try:
                    await asyncio.wait_for(self._monitor_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Monitoring task did not stop gracefully within timeout",
                        extra={"component": "quantum_health_monitor", "operation": "stop_monitoring"}
                    )
                except asyncio.CancelledError:
                    logger.debug(
                        "Monitoring task cancelled successfully",
                        extra={"component": "quantum_health_monitor", "operation": "stop_monitoring"}
                    )
                except Exception as e:
                    logger.error(
                        f"Error stopping monitoring task: {e}",
                        extra={"component": "quantum_health_monitor", "operation": "stop_monitoring"}
                    )
                finally:
                    self._monitor_task = None
            
            # Clear active alerts
            self.active_alerts.clear()
            
            logger.info(
                "Quantum health monitoring stopped successfully",
                extra={"component": "quantum_health_monitor", "operation": "stop_monitoring"}
            )
            
        except Exception as e:
            context = ErrorContext(
                component="quantum_health_monitor",
                operation="stop_monitoring"
            )
            handle_quantum_error(e, context, reraise=False)
    
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