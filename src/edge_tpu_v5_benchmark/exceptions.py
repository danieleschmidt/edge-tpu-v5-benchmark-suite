"""Custom exceptions for TPU v5 benchmark suite."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass


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


@dataclass
class ErrorContext:
    """Context information for error reporting."""
    component: str
    operation: str
    benchmark_id: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component": self.component,
            "operation": self.operation,
            "benchmark_id": self.benchmark_id,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "system_info": self.system_info
        }


class ErrorHandler:
    """Centralized error handling and reporting."""
    
    def __init__(self):
        self.error_callbacks: List[callable] = []
    
    def add_error_callback(self, callback: callable):
        """Add error callback for notifications."""
        self.error_callbacks.append(callback)
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None,
                    reraise: bool = True) -> Dict[str, Any]:
        """Handle and report error with context."""
        import logging
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        # Create error report
        error_report = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context.to_dict() if context else {},
            "traceback": None
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
        
        # Log error
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
    
    def create_user_friendly_message(self, error: Exception) -> str:
        """Create user-friendly error message."""
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
        
        else:
            return f"An unexpected error occurred: {str(error)}"


# Global error handler instance
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def handle_error(error: Exception, context: Optional[ErrorContext] = None,
                reraise: bool = True) -> Dict[str, Any]:
    """Handle error using global error handler."""
    return _error_handler.handle_error(error, context, reraise)


# Decorator for error handling
def handle_errors(component: str, operation: str):
    """Decorator for automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                component=component,
                operation=operation,
                timestamp=None  # Will be set by error handler
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle_error(e, context, reraise=True)
        
        return wrapper
    return decorator


# Context manager for error handling
class ErrorHandlingContext:
    """Context manager for error handling."""
    
    def __init__(self, component: str, operation: str, 
                 benchmark_id: Optional[str] = None,
                 model_name: Optional[str] = None):
        self.context = ErrorContext(
            component=component,
            operation=operation,
            benchmark_id=benchmark_id,
            model_name=model_name
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            handle_error(exc_val, self.context, reraise=False)
            return True  # Suppress exception
        return False