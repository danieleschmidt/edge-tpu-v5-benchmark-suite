"""Testing utilities and helper functions."""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest


def requires_hardware(func: Callable) -> Callable:
    """Decorator to skip tests that require TPU hardware."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import os
        if os.getenv("RUN_HARDWARE_TESTS", "false").lower() != "true":
            pytest.skip("Hardware tests disabled (set RUN_HARDWARE_TESTS=true to enable)")
        return func(*args, **kwargs)
    return wrapper


def requires_network(func: Callable) -> Callable:
    """Decorator to skip tests that require network access."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
        except OSError:
            pytest.skip("Network access required for this test")
        return func(*args, **kwargs)
    return wrapper


def slow_test(func: Callable) -> Callable:
    """Decorator to mark tests as slow."""
    return pytest.mark.slow(func)


def integration_test(func: Callable) -> Callable:
    """Decorator to mark tests as integration tests."""
    return pytest.mark.integration(func)


@contextmanager
def timer() -> Generator[Dict[str, float], None, None]:
    """Context manager for timing code execution.
    
    Yields:
        Dictionary that will contain 'elapsed' time in seconds
    """
    result = {}
    start_time = time.perf_counter()
    try:
        yield result
    finally:
        result['elapsed'] = time.perf_counter() - start_time


@contextmanager
def assert_timing(min_time: float = 0, max_time: float = float('inf')):
    """Context manager to assert execution time bounds.
    
    Args:
        min_time: Minimum expected execution time in seconds
        max_time: Maximum expected execution time in seconds
    """
    start_time = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start_time
    
    assert min_time <= elapsed <= max_time, \
        f"Execution time {elapsed:.3f}s not in range [{min_time:.3f}s, {max_time:.3f}s]"


def create_mock_benchmark_result(
    model_name: str = "test_model",
    iterations: int = 100,
    avg_latency_ms: float = 10.0,
    **kwargs
) -> Dict[str, Any]:
    """Create a mock benchmark result for testing.
    
    Args:
        model_name: Name of the model
        iterations: Number of benchmark iterations
        avg_latency_ms: Average latency in milliseconds
        **kwargs: Additional fields to include in the result
        
    Returns:
        Mock benchmark result dictionary
    """
    result = {
        "model_name": model_name,
        "device_info": {
            "device_path": "/dev/apex_0",
            "version": "v5",
            "memory_mb": 8192
        },
        "performance_metrics": {
            "iterations": iterations,
            "total_time_ms": iterations * avg_latency_ms,
            "avg_latency_ms": avg_latency_ms,
            "throughput_fps": 1000.0 / avg_latency_ms,
            "latency_percentiles": {
                "p50": avg_latency_ms * 0.95,
                "p90": avg_latency_ms * 1.1,
                "p95": avg_latency_ms * 1.2,
                "p99": avg_latency_ms * 1.5
            }
        },
        "power_metrics": {
            "avg_power_w": 2.5,
            "peak_power_w": 3.1,
            "total_energy_j": 2.5 * avg_latency_ms * iterations / 1000.0,
            "efficiency_fps_per_w": (1000.0 / avg_latency_ms) / 2.5
        },
        "system_info": {
            "python_version": "3.11.0",
            "numpy_version": "1.24.0",
            "timestamp": "2025-01-15T10:30:00Z"
        }
    }
    
    # Update with any additional fields
    result.update(kwargs)
    
    return result


def assert_array_properties(
    array: np.ndarray,
    expected_shape: tuple,
    expected_dtype: np.dtype,
    value_range: tuple = None
) -> None:
    """Assert properties of a numpy array.
    
    Args:
        array: Array to check
        expected_shape: Expected shape
        expected_dtype: Expected data type
        value_range: Optional (min, max) value range to check
    """
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"
    assert array.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {array.dtype}"
    
    if value_range is not None:
        min_val, max_val = value_range
        assert array.min() >= min_val, f"Array minimum {array.min()} below expected {min_val}"
        assert array.max() <= max_val, f"Array maximum {array.max()} above expected {max_val}"


def create_realistic_latency_distribution(
    num_samples: int = 1000,
    base_latency_ms: float = 10.0,
    variance: float = 0.1
) -> np.ndarray:
    """Create a realistic latency distribution for testing.
    
    Args:
        num_samples: Number of samples to generate
        base_latency_ms: Base latency in milliseconds
        variance: Variance as fraction of base latency
        
    Returns:
        Array of latency values in milliseconds
    """
    # Generate gamma distribution for realistic latency distribution
    # (latencies are typically right-skewed)
    shape = 4.0  # Shape parameter for gamma distribution
    scale = base_latency_ms / shape
    
    latencies = np.random.gamma(shape, scale, num_samples)
    
    # Add some occasional outliers (simulating system jitter)
    num_outliers = int(num_samples * 0.05)  # 5% outliers
    outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
    latencies[outlier_indices] *= np.random.uniform(2.0, 5.0, num_outliers)
    
    return latencies


def mock_tpu_device_with_thermal_throttling():
    """Create a mock TPU device that simulates thermal throttling."""
    mock_device = Mock()
    mock_device.device_path = "/dev/apex_0"
    mock_device.is_available.return_value = True
    
    # Simulate thermal state
    mock_device._temperature = 35.0  # Start at room temperature
    mock_device._is_throttling = False
    
    def get_temperature():
        # Simulate temperature increase during operation
        mock_device._temperature += np.random.uniform(0.5, 1.5)
        if mock_device._temperature > 85.0:
            mock_device._is_throttling = True
        elif mock_device._temperature < 75.0:
            mock_device._is_throttling = False
        return mock_device._temperature
    
    def is_throttling():
        return mock_device._is_throttling
    
    def cool_down():
        # Simulate cooling
        mock_device._temperature *= 0.95
        
    mock_device.get_temperature = get_temperature
    mock_device.is_throttling = is_throttling
    mock_device.cool_down = cool_down
    
    return mock_device


class MockPowerMeter:
    """Enhanced mock power meter with realistic behavior."""
    
    def __init__(self, base_power: float = 2.5):
        self.base_power = base_power
        self.is_measuring = False
        self.measurements = []
        self._measurement_start_time = None
        
    def start_measurement(self):
        """Start power measurement."""
        self.is_measuring = True
        self.measurements = []
        self._measurement_start_time = time.time()
        
    def stop_measurement(self):
        """Stop power measurement."""
        self.is_measuring = False
        
    def get_current_power(self) -> float:
        """Get current power consumption with realistic variation."""
        if not self.is_measuring:
            return 0.0
            
        # Simulate realistic power variation
        variation = np.random.normal(0, 0.1)  # 10% standard deviation
        current_power = self.base_power + variation
        
        # Add some load-dependent variation
        if len(self.measurements) > 0:
            # Simulate brief power spikes during computation
            if np.random.random() < 0.1:  # 10% chance of spike
                current_power *= 1.3
                
        self.measurements.append(current_power)
        return max(0.1, current_power)  # Ensure positive power
        
    def get_measurements(self) -> List[float]:
        """Get all power measurements."""
        return self.measurements.copy()
        
    def get_statistics(self) -> Dict[str, float]:
        """Get power measurement statistics."""
        if not self.measurements:
            return {}
            
        measurements = np.array(self.measurements)
        return {
            "mean": float(np.mean(measurements)),
            "std": float(np.std(measurements)),
            "min": float(np.min(measurements)),
            "max": float(np.max(measurements)),
            "total_energy": float(np.sum(measurements)) * 0.001,  # Assume 1ms sampling
        }


def parametrize_models(models: List[str] = None):
    """Parametrize decorator for testing multiple models.
    
    Args:
        models: List of model names to test. If None, uses default models.
    """
    if models is None:
        models = ["mobilenet_v3", "efficientnet_lite", "resnet50"]
        
    return pytest.mark.parametrize("model_name", models)


def parametrize_batch_sizes(batch_sizes: List[int] = None):
    """Parametrize decorator for testing multiple batch sizes.
    
    Args:
        batch_sizes: List of batch sizes to test. If None, uses default sizes.
    """
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]
        
    return pytest.mark.parametrize("batch_size", batch_sizes)


@contextmanager
def capture_logs(logger_name: str = None, level: str = "INFO"):
    """Context manager to capture log messages during test execution.
    
    Args:
        logger_name: Name of logger to capture. If None, captures root logger.
        level: Minimum log level to capture.
        
    Yields:
        List that will contain captured log records
    """
    import logging
    
    captured_logs = []
    
    class TestLogHandler(logging.Handler):
        def emit(self, record):
            captured_logs.append(record)
            
    handler = TestLogHandler()
    handler.setLevel(getattr(logging, level.upper()))
    
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    
    try:
        yield captured_logs
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


def assert_log_contains(logs: List, message: str, level: str = None):
    """Assert that captured logs contain a specific message.
    
    Args:
        logs: List of log records from capture_logs
        message: Message to search for
        level: Optional log level to filter by
    """
    matching_logs = []
    
    for log in logs:
        if message in log.getMessage():
            if level is None or log.levelname == level.upper():
                matching_logs.append(log)
                
    assert len(matching_logs) > 0, f"No log messages found containing '{message}'"