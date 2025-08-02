"""Test utilities and helper functions."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock
import numpy as np
import json


class TestDataGenerator:
    """Generate test data for benchmarking tests."""
    
    @staticmethod
    def create_random_input(shape: tuple, dtype: str = 'float32') -> np.ndarray:
        """Create random input data with specified shape and dtype."""
        if dtype == 'float32':
            return np.random.random(shape).astype(np.float32)
        elif dtype == 'int8':
            return np.random.randint(-128, 127, shape, dtype=np.int8)
        elif dtype == 'uint8':
            return np.random.randint(0, 255, shape, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    
    @staticmethod
    def create_classification_output(batch_size: int, num_classes: int) -> np.ndarray:
        """Create mock classification output."""
        logits = np.random.randn(batch_size, num_classes).astype(np.float32)
        # Apply softmax to create valid probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    @staticmethod
    def create_regression_output(batch_size: int, output_dim: int) -> np.ndarray:
        """Create mock regression output."""
        return np.random.randn(batch_size, output_dim).astype(np.float32)
    
    @staticmethod
    def create_latency_data(num_samples: int, mean_latency: float = 0.1, 
                          std_dev: float = 0.02) -> List[float]:
        """Create realistic latency data with some outliers."""
        # Generate base latencies with normal distribution
        latencies = np.random.normal(mean_latency, std_dev, num_samples)
        
        # Add some outliers (5% of samples)
        num_outliers = max(1, num_samples // 20)
        outlier_indices = np.random.choice(num_samples, num_outliers, replace=False)
        latencies[outlier_indices] *= np.random.uniform(2, 5, num_outliers)
        
        # Ensure all latencies are positive
        latencies = np.abs(latencies)
        
        return latencies.tolist()
    
    @staticmethod
    def create_power_trace(duration: float, sampling_rate: int = 1000) -> List[Dict[str, float]]:
        """Create mock power consumption trace."""
        num_samples = int(duration * sampling_rate)
        base_power = 1.5  # Base power consumption in watts
        
        trace = []
        for i in range(num_samples):
            timestamp = i / sampling_rate
            # Add some realistic variation
            power = base_power + 0.3 * np.sin(2 * np.pi * timestamp) + np.random.normal(0, 0.1)
            power = max(0.5, power)  # Minimum power consumption
            
            trace.append({
                'timestamp': timestamp,
                'power_watts': power,
                'voltage': 3.3 + np.random.normal(0, 0.05),
                'current': power / 3.3
            })
        
        return trace
    
    @staticmethod
    def create_thermal_data(duration: float, base_temp: float = 45.0) -> List[Dict[str, Any]]:
        """Create mock thermal monitoring data."""
        num_samples = int(duration * 10)  # 10 Hz sampling
        thermal_data = []
        
        for i in range(num_samples):
            timestamp = i / 10
            # Simulate gradual temperature increase during operation
            temp = base_temp + 0.1 * timestamp + np.random.normal(0, 1)
            temp = max(20, min(85, temp))  # Reasonable temperature range
            
            thermal_data.append({
                'timestamp': timestamp,
                'temperature': temp,
                'state': 'normal' if temp < 70 else 'warm' if temp < 80 else 'hot',
                'throttled': temp > 80
            })
        
        return thermal_data


class MockModelFactory:
    """Factory for creating mock models with different characteristics."""
    
    @staticmethod
    def create_cv_model(model_name: str = "test_cv_model", 
                       input_shape: tuple = (1, 3, 224, 224),
                       num_classes: int = 1000) -> Mock:
        """Create a mock computer vision model."""
        model = Mock()
        model.name = model_name
        model.input_shape = input_shape
        model.output_shape = (input_shape[0], num_classes)
        model.model_type = "classification"
        model.size_bytes = 5_000_000  # 5MB
        
        def mock_inference(input_data):
            batch_size = input_data.shape[0]
            return TestDataGenerator.create_classification_output(batch_size, num_classes)
        
        model.run_inference = mock_inference
        return model
    
    @staticmethod
    def create_nlp_model(model_name: str = "test_nlp_model",
                        sequence_length: int = 512,
                        vocab_size: int = 30000) -> Mock:
        """Create a mock NLP model."""
        model = Mock()
        model.name = model_name
        model.input_shape = (1, sequence_length)
        model.output_shape = (1, sequence_length, vocab_size)
        model.model_type = "language_model"
        model.size_bytes = 50_000_000  # 50MB
        
        def mock_inference(input_data):
            batch_size, seq_len = input_data.shape
            return TestDataGenerator.create_classification_output(
                batch_size * seq_len, vocab_size
            ).reshape(batch_size, seq_len, vocab_size)
        
        model.run_inference = mock_inference
        return model
    
    @staticmethod
    def create_slow_model(model_name: str = "slow_model",
                         inference_time: float = 0.5) -> Mock:
        """Create a mock model with configurable inference time."""
        model = Mock()
        model.name = model_name
        model.input_shape = (1, 3, 224, 224)
        model.output_shape = (1, 1000)
        model.model_type = "slow_classification"
        
        def slow_inference(input_data):
            import time
            time.sleep(inference_time)
            batch_size = input_data.shape[0]
            return TestDataGenerator.create_classification_output(batch_size, 1000)
        
        model.run_inference = slow_inference
        return model
    
    @staticmethod
    def create_failing_model(model_name: str = "failing_model",
                            failure_rate: float = 0.1) -> Mock:
        """Create a mock model that fails occasionally."""
        model = Mock()
        model.name = model_name
        model.input_shape = (1, 3, 224, 224)
        model.output_shape = (1, 1000)
        model.model_type = "unreliable_classification"
        
        def unreliable_inference(input_data):
            if np.random.random() < failure_rate:
                raise RuntimeError("Simulated model inference failure")
            batch_size = input_data.shape[0]
            return TestDataGenerator.create_classification_output(batch_size, 1000)
        
        model.run_inference = unreliable_inference
        return model


class FileTestHelper:
    """Helper for file-based testing operations."""
    
    @staticmethod
    def create_temp_model_file(model_data: bytes = b"mock_model_data", 
                              suffix: str = ".tflite") -> Path:
        """Create a temporary model file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(model_data)
            return Path(f.name)
    
    @staticmethod
    def create_temp_config_file(config_data: Dict[str, Any]) -> Path:
        """Create a temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f, indent=2)
            return Path(f.name)
    
    @staticmethod
    def create_temp_results_file(results_data: Dict[str, Any]) -> Path:
        """Create a temporary results file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(results_data, f, indent=2)
            return Path(f.name)
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[Path]) -> None:
        """Clean up temporary files."""
        for path in file_paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass  # Ignore cleanup errors


class BenchmarkResultsValidator:
    """Validator for benchmark results."""
    
    @staticmethod
    def validate_basic_results(results) -> bool:
        """Validate basic benchmark results structure."""
        required_fields = [
            'model_name', 'iterations', 'total_time', 'avg_latency',
            'throughput', 'latency_p50', 'latency_p90', 'latency_p95', 'latency_p99'
        ]
        
        for field in required_fields:
            if not hasattr(results, field):
                return False
            if getattr(results, field) is None:
                return False
        
        return True
    
    @staticmethod
    def validate_performance_metrics(results) -> bool:
        """Validate performance metrics are reasonable."""
        # Check that latencies are positive and make sense
        if results.avg_latency <= 0:
            return False
        
        if results.throughput <= 0:
            return False
        
        # Check percentile ordering
        percentiles = [results.latency_p50, results.latency_p90, 
                      results.latency_p95, results.latency_p99]
        
        for i in range(1, len(percentiles)):
            if percentiles[i] < percentiles[i-1]:
                return False
        
        return True
    
    @staticmethod
    def validate_power_metrics(results) -> bool:
        """Validate power-related metrics."""
        if hasattr(results, 'power_consumption'):
            if results.power_consumption < 0:
                return False
        
        if hasattr(results, 'energy_per_inference'):
            if results.energy_per_inference < 0:
                return False
        
        return True
    
    @staticmethod
    def validate_thermal_metrics(results) -> bool:
        """Validate thermal-related metrics."""
        if hasattr(results, 'temperature_avg'):
            if results.temperature_avg < 0 or results.temperature_avg > 100:
                return False
        
        if hasattr(results, 'temperature_max'):
            if results.temperature_max < results.temperature_avg:
                return False
        
        return True


class EnvironmentHelper:
    """Helper for managing test environment."""
    
    @staticmethod
    def setup_test_environment(config: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Set up test environment variables."""
        default_config = {
            'EDGE_TPU_DEV_MODE': '1',
            'EDGE_TPU_LOG_LEVEL': 'DEBUG',
            'BENCHMARK_DEFAULT_ITERATIONS': '10',
            'POWER_MEASUREMENT_ENABLED': 'false',
            'TELEMETRY_ENABLED': 'false'
        }
        
        if config:
            default_config.update(config)
        
        original_env = {}
        for key, value in default_config.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        return original_env
    
    @staticmethod
    def restore_environment(original_env: Dict[str, str]) -> None:
        """Restore original environment variables."""
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    @staticmethod
    def is_hardware_available() -> bool:
        """Check if TPU hardware is available for testing."""
        # In real implementation, this would check for actual TPU devices
        return os.environ.get('RUN_HARDWARE_TESTS', '').lower() == 'true'
    
    @staticmethod
    def is_network_available() -> bool:
        """Check if network access is available for testing."""
        return os.environ.get('RUN_NETWORK_TESTS', '').lower() == 'true'


# Test decorators and markers
def requires_hardware(func):
    """Decorator to mark tests that require hardware."""
    return pytest.mark.hardware(func)


def requires_network(func):
    """Decorator to mark tests that require network access."""
    return pytest.mark.network(func)


def slow_test(func):
    """Decorator to mark slow tests."""
    return pytest.mark.slow(func)


def performance_test(func):
    """Decorator to mark performance tests."""
    return pytest.mark.performance(func)