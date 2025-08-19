"""Robust Error Handling and Recovery System for TPU v5 Benchmark Suite

This module implements comprehensive error handling, automatic recovery,
circuit breakers, and fault tolerance mechanisms.
"""

import contextlib
import functools
import json
import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import psutil

from .security import SecurityContext


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    RESTART = "restart"
    ABORT = "abort"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    module_name: str = ""
    stack_trace: str = ""
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "function_name": self.function_name,
            "module_name": self.module_name,
            "stack_trace": self.stack_trace,
            "system_state": self.system_state,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    name: str
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    last_failure_time: float = 0.0
    success_count: int = 0
    success_threshold: int = 3

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "open"

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "half_open"

    def should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit breaker."""
        return (self.state == "open" and
                time.time() - self.last_failure_time > self.recovery_timeout)


class CircuitBreakerError(Exception):
    """Error raised when circuit breaker is open."""
    pass


class RetryableError(Exception):
    """Error that can be retried."""
    pass


class FatalError(Exception):
    """Fatal error that should not be retried."""
    pass


class TimeoutError(Exception):
    """Operation timeout error."""
    pass


class ResourceExhaustedError(Exception):
    """System resources exhausted."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 3):
        self.state = CircuitBreakerState(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        with self.lock:
            if self.state.is_open():
                if self.state.should_attempt_reset():
                    self.state.state = "half_open"
                    self.state.success_count = 0
                    self.logger.info(f"Circuit breaker {self.state.name} moved to half-open")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.state.name} is open")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except Exception as e:
                self._on_failure(e)
                raise

    def _on_success(self):
        """Handle successful operation."""
        if self.state.is_half_open():
            self.state.success_count += 1
            if self.state.success_count >= self.state.success_threshold:
                self.state.state = "closed"
                self.state.failure_count = 0
                self.logger.info(f"Circuit breaker {self.state.name} reset to closed")
        elif self.state.state == "closed":
            self.state.failure_count = 0

    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()

        if self.state.failure_count >= self.state.failure_threshold:
            self.state.state = "open"
            self.logger.warning(f"Circuit breaker {self.state.name} opened due to failures")

        self.logger.debug(f"Circuit breaker {self.state.name} failure count: {self.state.failure_count}")


class RetryPolicy:
    """Configurable retry policy."""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_backoff: bool = True,
                 jitter: bool = True,
                 retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Check if should retry the operation."""
        if attempt >= self.max_attempts:
            return False

        return isinstance(error, self.retryable_exceptions)

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

        return delay


def retry(policy: Optional[RetryPolicy] = None):
    """Decorator for retrying failed operations."""
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(attempt, e):
                        break

                    if attempt < policy.max_attempts:
                        delay = policy.get_delay(attempt)
                        logging.getLogger(__name__).info(
                            f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt}/{policy.max_attempts})"
                        )
                        time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


class TimeoutManager:
    """Manages operation timeouts."""

    def __init__(self, timeout: float, operation_name: str = "operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        self.start_time = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                self.logger.warning(
                    f"{self.operation_name} took {elapsed:.2f}s (timeout: {self.timeout}s)"
                )

    def check_timeout(self):
        """Check if operation has timed out."""
        if self.start_time and time.time() - self.start_time > self.timeout:
            raise TimeoutError(f"{self.operation_name} timed out after {self.timeout}s")


class ResourceMonitor:
    """Monitors system resources and prevents resource exhaustion."""

    def __init__(self,
                 max_memory_percent: float = 85.0,
                 max_cpu_percent: float = 90.0,
                 max_disk_percent: float = 95.0,
                 check_interval: float = 5.0):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.max_disk_percent = max_disk_percent
        self.check_interval = check_interval

        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10.0)
        self.logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._check_resources()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.check_interval * 2)

    def _check_resources(self):
        """Check system resources."""
        # Check memory
        memory_info = psutil.virtual_memory()
        if memory_info.percent > self.max_memory_percent:
            raise ResourceExhaustedError(
                f"Memory usage {memory_info.percent:.1f}% exceeds threshold {self.max_memory_percent}%"
            )

        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1.0)
        if cpu_percent > self.max_cpu_percent:
            self.logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

        # Check disk
        disk_info = psutil.disk_usage('/')
        disk_percent = (disk_info.used / disk_info.total) * 100
        if disk_percent > self.max_disk_percent:
            raise ResourceExhaustedError(
                f"Disk usage {disk_percent:.1f}% exceeds threshold {self.max_disk_percent}%"
            )

    def check_resources_sync(self):
        """Synchronously check resources."""
        self._check_resources()


class ErrorRecoveryManager:
    """Manages error recovery strategies."""

    def __init__(self, security_context: Optional[SecurityContext] = None):
        self.security_context = security_context or SecurityContext()
        self.logger = logging.getLogger(__name__)

        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.resource_monitor = ResourceMonitor()

        self.lock = threading.RLock()

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

    def register_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)

    def register_recovery_strategy(self, error_type: str, strategy: RecoveryStrategy):
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"Registered recovery strategy for {error_type}: {strategy.value}")

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle error with appropriate recovery strategy."""
        error_context = self._create_error_context(error, context)

        with self.lock:
            self.error_history.append(error_context)

        self.logger.error(f"Handling error: {error_context.error_message}")

        # Apply recovery strategy
        strategy = self.recovery_strategies.get(
            error_context.error_type,
            RecoveryStrategy.RETRY
        )

        try:
            self._apply_recovery_strategy(strategy, error_context)
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            error_context.recovery_attempts += 1

        return error_context

    def _create_error_context(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create error context from exception."""
        severity = self._determine_severity(error)

        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            function_name=context.get("function_name", "") if context else "",
            module_name=context.get("module_name", "") if context else "",
            stack_trace=traceback.format_exc(),
            system_state=self._collect_system_state()
        )

        return error_context

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, (FatalError, ResourceExhaustedError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (TimeoutError, CircuitBreakerError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, RetryableError):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state information."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                "active_threads": threading.active_count(),
                "timestamp": time.time()
            }
        except Exception:
            return {"error": "Failed to collect system state"}

    def _apply_recovery_strategy(self, strategy: RecoveryStrategy, error_context: ErrorContext):
        """Apply specified recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            self._retry_recovery(error_context)
        elif strategy == RecoveryStrategy.FALLBACK:
            self._fallback_recovery(error_context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self._circuit_breaker_recovery(error_context)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            self._graceful_degradation_recovery(error_context)
        elif strategy == RecoveryStrategy.RESTART:
            self._restart_recovery(error_context)
        elif strategy == RecoveryStrategy.ABORT:
            self._abort_recovery(error_context)

    def _retry_recovery(self, error_context: ErrorContext):
        """Implement retry recovery strategy."""
        if error_context.recovery_attempts < error_context.max_recovery_attempts:
            self.logger.info(f"Retrying operation for {error_context.error_type}")
            error_context.recovery_attempts += 1
        else:
            self.logger.error(f"Max retry attempts reached for {error_context.error_type}")
            raise FatalError(f"Recovery failed after {error_context.max_recovery_attempts} attempts")

    def _fallback_recovery(self, error_context: ErrorContext):
        """Implement fallback recovery strategy."""
        self.logger.info(f"Implementing fallback for {error_context.error_type}")
        # Would implement specific fallback logic here

    def _circuit_breaker_recovery(self, error_context: ErrorContext):
        """Implement circuit breaker recovery strategy."""
        self.logger.info(f"Circuit breaker activated for {error_context.error_type}")
        # Circuit breaker handles this automatically

    def _graceful_degradation_recovery(self, error_context: ErrorContext):
        """Implement graceful degradation recovery strategy."""
        self.logger.info(f"Graceful degradation for {error_context.error_type}")
        # Reduce functionality but continue operation

    def _restart_recovery(self, error_context: ErrorContext):
        """Implement restart recovery strategy."""
        self.logger.warning(f"Restart required for {error_context.error_type}")
        # Would implement restart logic here

    def _abort_recovery(self, error_context: ErrorContext):
        """Implement abort recovery strategy."""
        self.logger.critical(f"Aborting due to {error_context.error_type}")
        raise FatalError(f"Operation aborted due to {error_context.error_type}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self.lock:
            if not self.error_history:
                return {"total_errors": 0}

            error_types = defaultdict(int)
            severity_counts = defaultdict(int)
            recent_errors = []

            for error_ctx in self.error_history:
                error_types[error_ctx.error_type] += 1
                severity_counts[error_ctx.severity.value] += 1

                if time.time() - error_ctx.timestamp < 3600:  # Last hour
                    recent_errors.append(error_ctx.to_dict())

            return {
                "total_errors": len(self.error_history),
                "error_types": dict(error_types),
                "severity_distribution": dict(severity_counts),
                "recent_errors": recent_errors[-10:],  # Last 10 recent errors
                "circuit_breaker_status": {
                    name: cb.state.state
                    for name, cb in self.circuit_breakers.items()
                }
            }

    def export_error_report(self, filepath: Path):
        """Export comprehensive error report."""
        report = {
            "error_statistics": self.get_error_statistics(),
            "error_history": [
                error_ctx.to_dict() for error_ctx in self.error_history
            ],
            "circuit_breakers": {
                name: {
                    "state": cb.state.state,
                    "failure_count": cb.state.failure_count,
                    "failure_threshold": cb.state.failure_threshold,
                    "last_failure_time": cb.state.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            },
            "recovery_strategies": {
                error_type: strategy.value
                for error_type, strategy in self.recovery_strategies.items()
            },
            "report_timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Error report exported to {filepath}")

    def __del__(self):
        """Cleanup resources."""
        try:
            self.resource_monitor.stop_monitoring()
        except:
            pass


# Decorators for robust error handling
def robust_operation(circuit_breaker_name: Optional[str] = None,
                    retry_policy: Optional[RetryPolicy] = None,
                    timeout: Optional[float] = None,
                    error_manager: Optional[ErrorRecoveryManager] = None):
    """Decorator for robust operation with comprehensive error handling."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Setup error manager
            manager = error_manager or ErrorRecoveryManager()

            # Setup circuit breaker
            circuit_breaker = None
            if circuit_breaker_name:
                circuit_breaker = manager.get_circuit_breaker(circuit_breaker_name)
                if not circuit_breaker:
                    circuit_breaker = manager.register_circuit_breaker(circuit_breaker_name)

            # Setup retry policy
            policy = retry_policy or RetryPolicy()

            # Setup timeout
            timeout_manager = TimeoutManager(timeout, func.__name__) if timeout else None

            # Execute with all protections
            def protected_execution():
                with timeout_manager if timeout_manager else contextlib.nullcontext():
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)

            # Apply retry policy
            last_exception = None
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return protected_execution()

                except Exception as e:
                    last_exception = e

                    # Handle error through manager
                    error_context = manager.handle_error(e, {
                        "function_name": func.__name__,
                        "module_name": func.__module__,
                        "attempt": attempt
                    })

                    if not policy.should_retry(attempt, e) or attempt >= policy.max_attempts:
                        break

                    delay = policy.get_delay(attempt)
                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# Global error recovery manager instance
_global_error_manager = None

def get_global_error_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ErrorRecoveryManager()
    return _global_error_manager
