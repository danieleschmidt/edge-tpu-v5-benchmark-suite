"""Custom exceptions for TPU v5 benchmark suite with quantum-specific errors and circuit breaker patterns."""

import time
import threading
import functools
import asyncio
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class TPUBenchmarkError(Exception):
    """Base exception for all TPU benchmark errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class DeviceError(TPUBenchmarkError):
    """Errors related to TPU device access and initialization."""
    pass


class DeviceNotFoundError(DeviceError):
    """TPU device not found or not accessible."""
    pass


class DeviceInitializationError(DeviceError):
    """Error during TPU device initialization."""
    pass


class ModelError(TPUBenchmarkError):
    """Errors related to model loading and compilation."""
    pass


class ModelNotFoundError(ModelError):
    """Model file not found."""
    pass


class ModelLoadError(ModelError):
    """Error loading model file."""
    pass


class ModelCompilationError(ModelError):
    """Error during model compilation for TPU."""
    pass


class UnsupportedModelError(ModelError):
    """Model format or architecture not supported."""
    pass


class BenchmarkError(TPUBenchmarkError):
    """Errors during benchmark execution."""
    pass


class BenchmarkConfigurationError(BenchmarkError):
    """Invalid benchmark configuration."""
    pass


class BenchmarkExecutionError(BenchmarkError):
    """Error during benchmark execution."""
    pass


class InferenceError(BenchmarkError):
    """Error during model inference."""
    pass


class ValidationError(TPUBenchmarkError):
    """Input validation errors."""
    pass


class ConversionError(TPUBenchmarkError):
    """Model conversion errors."""
    pass


class OptimizationError(TPUBenchmarkError):
    """Model optimization errors."""
    pass


class ResourceError(TPUBenchmarkError):
    """System resource errors."""
    pass


class MemoryError(ResourceError):
    """Memory-related errors."""
    pass


class DiskSpaceError(ResourceError):
    """Disk space errors."""
    pass


class NetworkError(TPUBenchmarkError):
    """Network-related errors."""
    pass


class TimeoutError(TPUBenchmarkError):
    """Operation timeout errors."""
    pass


class SecurityError(TPUBenchmarkError):
    """Security-related errors."""
    pass


class ConfigurationError(TPUBenchmarkError):
    """Configuration errors."""
    pass


# Quantum-specific exceptions
class QuantumError(TPUBenchmarkError):
    """Base exception for quantum-related errors."""
    pass


class QuantumCoherenceError(QuantumError):
    """Errors related to quantum coherence loss."""
    pass


class QuantumEntanglementError(QuantumError):
    """Errors related to quantum entanglement."""
    pass


class QuantumDecoherenceError(QuantumError):
    """Errors due to quantum decoherence."""
    pass


class QuantumStateError(QuantumError):
    """Errors related to invalid quantum states."""
    pass


class QuantumValidationError(QuantumError):
    """Quantum task validation errors."""
    pass


class QuantumResourceAllocationError(QuantumError):
    """Quantum resource allocation failures."""
    pass


class QuantumCircuitBreakerError(QuantumError):
    """Circuit breaker is open, operations not allowed."""
    pass


class QuantumRetryExhaustedError(QuantumError):
    """All retry attempts have been exhausted."""
    pass


class QuantumTimeoutError(QuantumError):
    """Quantum operation timed out."""
    pass


class QuantumTaskExecutionError(QuantumError):
    """Error during quantum task execution."""
    pass


class QuantumMonitoringError(QuantumError):
    """Error in quantum monitoring systems."""
    pass


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout: float = 60.0
    expected_exception: type = Exception


class CircuitBreaker:
    """Circuit breaker implementation for quantum operations."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker."""
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise QuantumCircuitBreakerError(
                        f"Circuit breaker is open. Last failure: {self.last_failure_time}"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


class RetryManager:
    """Retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry logic to function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.retry(func, *args, **kwargs)
        return wrapper
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Final attempt failed
                    break
                
                delay = self._calculate_delay(attempt)
                time.sleep(delay)
        
        # All attempts failed
        raise QuantumRetryExhaustedError(
            f"Failed after {self.config.max_attempts} attempts. Last error: {last_exception}"
        ) from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next attempt."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            import random
            delay *= random.uniform(0.5, 1.5)
        
        return delay


@dataclass
class ErrorContext:
    """Context information for error reporting."""
    component: str
    operation: str
    benchmark_id: Optional[str] = None
    model_name: Optional[str] = None
    task_id: Optional[str] = None
    timestamp: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    quantum_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "operation": self.operation,
            "benchmark_id": self.benchmark_id,
            "model_name": self.model_name,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "quantum_state": self.quantum_state
        }


class ErrorHandler:
    """Centralized error handling and reporting with quantum-aware features."""
    
    def __init__(self):
        self.error_callbacks: List[callable] = []
        self.error_history: deque = deque(maxlen=1000)
        self.error_counts: defaultdict = defaultdict(int)
        self.lock = threading.RLock()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
    
    def add_error_callback(self, callback: callable):
        """Add error callback for notifications."""
        self.error_callbacks.append(callback)
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for named operation."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def get_retry_manager(self, name: str, config: Optional[RetryConfig] = None) -> RetryManager:
        """Get or create retry manager for named operation."""
        if name not in self.retry_managers:
            if config is None:
                config = RetryConfig()
            self.retry_managers[name] = RetryManager(config)
        return self.retry_managers[name]
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None,
                    reraise: bool = True) -> Dict[str, Any]:
        """Handle and report error with context and enhanced quantum features."""
        import logging
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        with self.lock:
            # Update error counts
            error_key = f"{type(error).__name__}:{context.component if context else 'unknown'}"
            self.error_counts[error_key] += 1
        
        # Create comprehensive error report
        error_report = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context.to_dict() if context else {},
            "traceback": None,
            "error_frequency": self.error_counts[error_key],
            "quantum_specific": isinstance(error, QuantumError)
        }
        
        # Add traceback for debugging
        if hasattr(error, '__traceback__') and error.__traceback__:
            import traceback
            error_report["traceback"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )
        
        # Add custom error details if available
        if isinstance(error, TPUBenchmarkError):
            error_report["error_code"] = error.error_code
            error_report["details"] = error.details
        
        # Add quantum-specific error analysis
        if isinstance(error, QuantumError):
            error_report["quantum_analysis"] = self._analyze_quantum_error(error, context)
        
        # Store in error history
        with self.lock:
            self.error_history.append(error_report)
        
        # Log error with appropriate level
        if isinstance(error, (QuantumCircuitBreakerError, QuantumRetryExhaustedError)):
            logger.critical(f"Critical quantum error in {context.component if context else 'unknown'}: {error}",
                          extra={"error_report": error_report})
        elif isinstance(error, QuantumError):
            logger.error(f"Quantum error in {context.component if context else 'unknown'}: {error}",
                        extra={"error_report": error_report})
        else:
            logger.error(f"Error in {context.component if context else 'unknown'}: {error}",
                        extra={"error_report": error_report})
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error, error_report)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
        
        # Reraise if requested
        if reraise:
            raise error
        
        return error_report
    
    def _analyze_quantum_error(self, error: QuantumError, context: Optional[ErrorContext]) -> Dict[str, Any]:
        """Analyze quantum-specific error patterns."""
        analysis = {
            "error_category": self._categorize_quantum_error(error),
            "severity": self._assess_quantum_error_severity(error),
            "recovery_suggestions": self._get_quantum_recovery_suggestions(error)
        }
        
        if context and context.quantum_state:
            analysis["quantum_state_analysis"] = self._analyze_quantum_state_impact(error, context.quantum_state)
        
        return analysis
    
    def _categorize_quantum_error(self, error: QuantumError) -> str:
        """Categorize quantum error type."""
        if isinstance(error, QuantumCoherenceError):
            return "coherence"
        elif isinstance(error, QuantumEntanglementError):
            return "entanglement"
        elif isinstance(error, QuantumDecoherenceError):
            return "decoherence"
        elif isinstance(error, QuantumStateError):
            return "state"
        elif isinstance(error, QuantumValidationError):
            return "validation"
        elif isinstance(error, QuantumResourceAllocationError):
            return "resource"
        else:
            return "general"
    
    def _assess_quantum_error_severity(self, error: QuantumError) -> str:
        """Assess severity of quantum error."""
        if isinstance(error, (QuantumCircuitBreakerError, QuantumRetryExhaustedError)):
            return "critical"
        elif isinstance(error, (QuantumCoherenceError, QuantumResourceAllocationError)):
            return "high"
        elif isinstance(error, (QuantumDecoherenceError, QuantumStateError)):
            return "medium"
        else:
            return "low"
    
    def _get_quantum_recovery_suggestions(self, error: QuantumError) -> List[str]:
        """Get recovery suggestions for quantum errors."""
        suggestions = []
        
        if isinstance(error, QuantumCoherenceError):
            suggestions.extend([
                "Reduce system load to maintain coherence",
                "Check for environmental interference",
                "Consider increasing decoherence time limits"
            ])
        elif isinstance(error, QuantumEntanglementError):
            suggestions.extend([
                "Verify entanglement relationships are bidirectional",
                "Check for orphaned entangled tasks",
                "Reduce entanglement complexity"
            ])
        elif isinstance(error, QuantumResourceAllocationError):
            suggestions.extend([
                "Increase resource capacity",
                "Optimize resource allocation strategy",
                "Consider task prioritization adjustments"
            ])
        elif isinstance(error, QuantumValidationError):
            suggestions.extend([
                "Review task configuration parameters",
                "Check input data validity",
                "Verify system dependencies"
            ])
        
        return suggestions
    
    def _analyze_quantum_state_impact(self, error: QuantumError, quantum_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze impact of error on quantum state."""
        return {
            "coherence_level": quantum_state.get("coherence", 0.0),
            "entanglement_count": quantum_state.get("entanglements", 0),
            "decoherence_rate": quantum_state.get("decoherence", 0.0),
            "state_stability": "unstable" if quantum_state.get("coherence", 0.0) < 0.5 else "stable"
        }
    
    def get_error_statistics(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get error statistics for specified time window."""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            recent_errors = [
                error for error in self.error_history
                if datetime.fromisoformat(error["timestamp"]) >= cutoff_time
            ]
        
        # Analyze error patterns
        error_types = defaultdict(int)
        quantum_errors = 0
        components = defaultdict(int)
        
        for error in recent_errors:
            error_types[error["error_type"]] += 1
            if error.get("quantum_specific", False):
                quantum_errors += 1
            
            context = error.get("context", {})
            if context.get("component"):
                components[context["component"]] += 1
        
        return {
            "total_errors": len(recent_errors),
            "quantum_errors": quantum_errors,
            "error_types": dict(error_types),
            "affected_components": dict(components),
            "error_rate": len(recent_errors) / max(window_minutes, 1),
            "circuit_breaker_states": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            }
        }
    
    def create_user_friendly_message(self, error: Exception) -> str:
        """Create user-friendly error message with quantum-specific guidance."""
        if isinstance(error, DeviceNotFoundError):
            return ("TPU device not found. Please check device connections or "
                   "run in simulation mode for development.")
        
        elif isinstance(error, ModelNotFoundError):
            return ("Model file not found. Please check the file path and "
                   "ensure the model exists.")
        
        elif isinstance(error, UnsupportedModelError):
            return ("Model format not supported. Please use ONNX, TensorFlow Lite, "
                   "or other supported formats.")
        
        elif isinstance(error, BenchmarkConfigurationError):
            return ("Invalid benchmark configuration. Please check your parameters "
                   "and try again.")
        
        elif isinstance(error, MemoryError):
            return ("Insufficient memory. Try reducing batch size or model complexity.")
        
        elif isinstance(error, TimeoutError):
            return ("Operation timed out. The benchmark may be taking longer than expected.")
        
        elif isinstance(error, NetworkError):
            return ("Network error occurred. Please check your internet connection.")
        
        elif isinstance(error, SecurityError):
            return ("Security error. Operation not allowed for safety reasons.")
        
        # Quantum-specific error messages
        elif isinstance(error, QuantumCoherenceError):
            return ("Quantum coherence lost. The system may be experiencing interference "
                   "or excessive load. Try reducing concurrent operations.")
        
        elif isinstance(error, QuantumEntanglementError):
            return ("Quantum entanglement error. Check that all entangled tasks exist "
                   "and relationships are properly configured.")
        
        elif isinstance(error, QuantumDecoherenceError):
            return ("Quantum decoherence detected. Tasks may have exceeded their "
                   "coherence time. Consider refreshing or recreating affected tasks.")
        
        elif isinstance(error, QuantumStateError):
            return ("Invalid quantum state. The task may be in an inconsistent state. "
                   "Try resetting the task or checking its configuration.")
        
        elif isinstance(error, QuantumValidationError):
            return ("Quantum task validation failed. Check task parameters, dependencies, "
                   "and resource requirements.")
        
        elif isinstance(error, QuantumResourceAllocationError):
            return ("Quantum resource allocation failed. Insufficient resources available "
                   "or allocation conflicts detected.")
        
        elif isinstance(error, QuantumCircuitBreakerError):
            return ("Circuit breaker is open due to repeated failures. The system is "
                   "protecting itself from cascading errors. Please wait and try again.")
        
        elif isinstance(error, QuantumRetryExhaustedError):
            return ("All retry attempts have been exhausted. The operation has failed "
                   "multiple times. Check system health and configuration.")
        
        elif isinstance(error, QuantumTimeoutError):
            return ("Quantum operation timed out. The system may be overloaded or "
                   "experiencing performance issues.")
        
        elif isinstance(error, QuantumTaskExecutionError):
            return ("Quantum task execution failed. Check task configuration and "
                   "system resources.")
        
        elif isinstance(error, QuantumMonitoringError):
            return ("Quantum monitoring system error. Health checks and monitoring "
                   "may be impaired.")
        
        else:
            return f"An unexpected error occurred: {str(error)}"


# Enhanced error handler factory
def create_quantum_error_handler() -> ErrorHandler:
    """Create error handler optimized for quantum operations."""
    handler = ErrorHandler()
    
    # Add quantum-specific circuit breakers
    quantum_cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0,
        expected_exception=QuantumError
    )
    
    handler.get_circuit_breaker("quantum_operations", quantum_cb_config)
    
    # Add quantum-specific retry configuration
    quantum_retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        retryable_exceptions=(QuantumCoherenceError, QuantumResourceAllocationError)
    )
    
    handler.get_retry_manager("quantum_operations", quantum_retry_config)
    
    return handler


# Global error handler instances
_error_handler = ErrorHandler()
_quantum_error_handler = create_quantum_error_handler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def get_quantum_error_handler() -> ErrorHandler:
    """Get quantum-optimized error handler instance."""
    return _quantum_error_handler


def handle_error(error: Exception, context: Optional[ErrorContext] = None,
                reraise: bool = True) -> Dict[str, Any]:
    """Handle error using global error handler."""
    return _error_handler.handle_error(error, context, reraise)


def handle_quantum_error(error: Exception, context: Optional[ErrorContext] = None,
                        reraise: bool = True) -> Dict[str, Any]:
    """Handle quantum error using quantum-optimized error handler."""
    return _quantum_error_handler.handle_error(error, context, reraise)


# Enhanced decorators for error handling
def handle_errors(component: str, operation: str, 
                 use_quantum_handler: bool = False,
                 circuit_breaker: Optional[str] = None,
                 retry_config: Optional[str] = None):
    """Enhanced decorator for automatic error handling with quantum support."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                timestamp=None  # Will be set by error handler
            )
            
            error_handler = _quantum_error_handler if use_quantum_handler else _error_handler
            
            # Apply circuit breaker if specified
            actual_func = func
            if circuit_breaker:
                cb = error_handler.get_circuit_breaker(circuit_breaker)
                actual_func = cb(actual_func)
            
            # Apply retry logic if specified
            if retry_config:
                retry_manager = error_handler.get_retry_manager(retry_config)
                actual_func = retry_manager(actual_func)
            
            try:
                return actual_func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, context, reraise=True)
        
        return wrapper
    return decorator


def quantum_operation(operation_name: str, 
                     circuit_breaker: bool = True,
                     retry_attempts: int = 3,
                     validate_state: bool = True,
                     timeout_seconds: Optional[float] = None):
    """Comprehensive decorator for quantum operations."""
    def decorator(func: Callable) -> Callable:
        decorated_func = func
        
        # Apply quantum state validation
        if validate_state:
            decorated_func = validate_quantum_state(decorated_func)
        
        # Apply timeout if specified
        if timeout_seconds:
            decorated_func = timeout_operation(timeout_seconds)(decorated_func)
        
        # Apply circuit breaker and retry logic
        if circuit_breaker or retry_attempts > 1:
            cb_name = f"quantum_{operation_name}"
            retry_name = f"quantum_{operation_name}"
            
            decorated_func = handle_errors(
                component="quantum_system",
                operation=operation_name,
                use_quantum_handler=True,
                circuit_breaker=cb_name if circuit_breaker else None,
                retry_config=retry_name if retry_attempts > 1 else None
            )(decorated_func)
        
        return decorated_func
    return decorator


# Async decorator for error handling
def handle_async_errors(component: str, operation: str,
                       use_quantum_handler: bool = False):
    """Decorator for async error handling."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                timestamp=None
            )
            
            error_handler = _quantum_error_handler if use_quantum_handler else _error_handler
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return error_handler.handle_error(e, context, reraise=True)
        
        return wrapper
    return decorator


# Validation decorators
def validate_input(validator_func: Callable, error_message: str = "Invalid input"):
    """Decorator for input validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validator_func(*args, **kwargs):
                    raise ValidationError(error_message)
                return func(*args, **kwargs)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(f"Validation failed: {str(e)}") from e
        return wrapper
    return decorator


def validate_quantum_state(func: Callable) -> Callable:
    """Decorator to validate quantum state before operation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract quantum task if present
        quantum_task = None
        for arg in args:
            if hasattr(arg, 'state') and hasattr(arg, 'probability_amplitude'):
                quantum_task = arg
                break
        
        if quantum_task:
            # Basic quantum state validation
            if hasattr(quantum_task, 'measure_decoherence'):
                decoherence = quantum_task.measure_decoherence()
                if decoherence > 0.9:
                    raise QuantumDecoherenceError(
                        f"Task {quantum_task.id} is highly decoherent: {decoherence:.1%}"
                    )
            
            if abs(quantum_task.probability_amplitude) < 0.001:
                raise QuantumCoherenceError(
                    f"Task {quantum_task.id} has very low probability amplitude"
                )
        
        return func(*args, **kwargs)
    return wrapper


def sanitize_input(sanitizer_func: Callable):
    """Decorator for input sanitization."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                sanitized_args, sanitized_kwargs = sanitizer_func(*args, **kwargs)
                return func(*sanitized_args, **sanitized_kwargs)
            except Exception as e:
                raise ValidationError(f"Input sanitization failed: {str(e)}") from e
        return wrapper
    return decorator


def timeout_operation(timeout_seconds: float, error_message: str = "Operation timed out"):
    """Decorator to add timeout to operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise QuantumTimeoutError(f"{error_message} ({timeout_seconds}s)")
            
            # Set the signal alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


# Context manager for error handling
class ErrorHandlingContext:
    """Enhanced context manager for error handling with quantum awareness."""
    
    def __init__(self, component: str, operation: str, 
                 benchmark_id: Optional[str] = None,
                 model_name: Optional[str] = None,
                 task_id: Optional[str] = None,
                 quantum_state: Optional[Dict[str, Any]] = None,
                 suppress_exceptions: bool = False):
        self.context = ErrorContext(
            component=component,
            operation=operation,
            benchmark_id=benchmark_id,
            model_name=model_name,
            task_id=task_id,
            quantum_state=quantum_state
        )
        self.suppress_exceptions = suppress_exceptions
        self.error_report = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            self.error_report = handle_error(exc_val, self.context, reraise=not self.suppress_exceptions)
            return self.suppress_exceptions  # Suppress exception if requested
        return False
    
    def get_error_report(self) -> Optional[Dict[str, Any]]:
        """Get the error report from the last handled error."""
        return self.error_report


# Async context manager for error handling
class AsyncErrorHandlingContext:
    """Async context manager for error handling."""
    
    def __init__(self, component: str, operation: str, 
                 benchmark_id: Optional[str] = None,
                 model_name: Optional[str] = None,
                 task_id: Optional[str] = None,
                 quantum_state: Optional[Dict[str, Any]] = None,
                 suppress_exceptions: bool = False):
        self.context = ErrorContext(
            component=component,
            operation=operation,
            benchmark_id=benchmark_id,
            model_name=model_name,
            task_id=task_id,
            quantum_state=quantum_state
        )
        self.suppress_exceptions = suppress_exceptions
        self.error_report = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            self.error_report = handle_error(exc_val, self.context, reraise=not self.suppress_exceptions)
            return self.suppress_exceptions
        return False
    
    def get_error_report(self) -> Optional[Dict[str, Any]]:
        """Get the error report from the last handled error."""
        return self.error_report


# Resource cleanup utilities
class ResourceManager:
    """Resource manager with automatic cleanup."""
    
    def __init__(self):
        self.resources: List[Any] = []
        self.cleanup_callbacks: List[Callable] = []
        self.lock = threading.RLock()
    
    def register_resource(self, resource: Any, cleanup_callback: Optional[Callable] = None) -> None:
        """Register a resource for cleanup."""
        with self.lock:
            self.resources.append(resource)
            if cleanup_callback:
                self.cleanup_callbacks.append(cleanup_callback)
    
    def cleanup_all(self) -> None:
        """Clean up all registered resources."""
        with self.lock:
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    # Log cleanup error but don't fail
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Error during resource cleanup: {e}")
            
            self.resources.clear()
            self.cleanup_callbacks.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all()
        return False


# Enhanced error handler factory
def create_quantum_error_handler() -> ErrorHandler:
    """Create error handler optimized for quantum operations."""
    handler = ErrorHandler()
    
    # Add quantum-specific circuit breakers
    quantum_cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0,
        expected_exception=QuantumError
    )
    
    handler.get_circuit_breaker("quantum_operations", quantum_cb_config)
    
    # Add quantum-specific retry configuration
    quantum_retry_config = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=1.5,
        retryable_exceptions=(QuantumCoherenceError, QuantumResourceAllocationError)
    )
    
    handler.get_retry_manager("quantum_operations", quantum_retry_config)
    
    return handler