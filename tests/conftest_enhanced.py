"""Enhanced pytest configuration and shared fixtures for TPU v5 benchmark suite tests."""

import pytest
import tempfile
import logging
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os
import time

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require TPU hardware"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests that measure performance"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and skip conditions."""
    # Skip hardware tests by default
    skip_hardware = pytest.mark.skip(reason="Hardware tests require TPU device")
    
    # Skip network tests if no network
    skip_network = pytest.mark.skip(reason="Network tests require internet connection")
    
    for item in items:
        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Skip hardware tests unless explicitly requested
        if "hardware" in item.keywords:
            if not config.getoption("--run-hardware"):
                item.add_marker(skip_hardware)
        
        # Skip network tests unless explicitly requested
        if "network" in item.keywords:
            if not config.getoption("--run-network"):
                item.add_marker(skip_network)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require TPU hardware"
    )
    parser.addoption(
        "--run-network",
        action="store_true", 
        default=False,
        help="Run tests that require network access"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Reduce noise from verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Mock numpy for tests that don't have it installed
@pytest.fixture(autouse=True)
def mock_numpy():
    """Mock numpy module for testing environments without numpy."""
    try:
        import numpy
        yield numpy
    except ImportError:
        # Create mock numpy
        mock_numpy = Mock()
        mock_numpy.array = lambda x: x
        mock_numpy.mean = lambda x: sum(x) / len(x) if x else 0
        mock_numpy.std = lambda x: 0  # Simplified
        mock_numpy.percentile = lambda x, p: sorted(x)[int(len(x) * p / 100)] if x else 0
        mock_numpy.random.randn = lambda *shape: [0.5] * (shape[0] if shape else 1)
        mock_numpy.random.uniform = lambda low, high, size: [low + (high-low)/2] * (size[0] if hasattr(size, '__iter__') else size)
        mock_numpy.random.normal = lambda mean, std: mean
        mock_numpy.random.randint = lambda low, high: (low + high) // 2
        mock_numpy.float32 = float
        mock_numpy.ndarray = list
        
        with patch.dict('sys.modules', {'numpy': mock_numpy, 'np': mock_numpy}):
            yield mock_numpy


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache_dir():
    """Provide a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory(prefix="cache_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture  
def temp_model_dir():
    """Provide a temporary directory for model files."""
    with tempfile.TemporaryDirectory(prefix="model_test_") as tmpdir:
        yield Path(tmpdir)


# Mock data fixtures
@pytest.fixture
def mock_tpu_device():
    """Mock TPU device for testing."""
    with patch("edge_tpu_v5_benchmark.benchmark.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.stat.return_value.st_mode = 0o644
        yield mock_path


@pytest.fixture
def mock_model_file(temp_model_dir):
    """Create a mock model file for testing."""
    model_file = temp_model_dir / "test_model.onnx"
    model_file.write_bytes(b"mock_onnx_model_data")
    yield model_file
    
    # Cleanup
    if model_file.exists():
        model_file.unlink()


@pytest.fixture
def mock_large_model_file(temp_model_dir):
    """Create a mock large model file for testing."""
    model_file = temp_model_dir / "large_model.onnx"
    # Create a larger file (1MB)
    model_file.write_bytes(b"x" * (1024 * 1024))
    yield model_file
    
    if model_file.exists():
        model_file.unlink()


@pytest.fixture
def sample_benchmark_results():
    """Provide sample benchmark results for testing."""
    from edge_tpu_v5_benchmark.benchmark import BenchmarkResults
    from datetime import datetime
    
    return BenchmarkResults(
        throughput=850.5,
        latency_p99=1.25,
        latency_p95=1.15,
        latency_p50=1.05,
        latency_mean=1.08,
        latency_std=0.12,
        avg_power=0.85,
        peak_power=1.10,
        min_power=0.65,
        energy_consumed=2.55,
        inferences_per_watt=1000.0,
        total_iterations=1000,
        warmup_iterations=50,
        duration_seconds=1.18,
        success_rate=0.998,
        memory_usage_mb=128.5,
        cpu_utilization=45.2,
        thermal_state="normal",
        raw_latencies=[1.05 + (i % 10) * 0.01 for i in range(100)],
        raw_power_samples=[0.85 + (i % 20) * 0.005 for i in range(200)]
    )


# Async fixtures
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_scheduler():
    """Provide an async task scheduler for testing."""
    from edge_tpu_v5_benchmark.concurrency import TaskScheduler
    
    scheduler = TaskScheduler()
    await scheduler.start()
    
    yield scheduler
    
    await scheduler.stop()


@pytest.fixture
async def async_job_manager():
    """Provide an async job manager for testing."""
    from edge_tpu_v5_benchmark.concurrency import BenchmarkJobManager
    
    manager = BenchmarkJobManager()
    await manager.start()
    
    yield manager
    
    await manager.stop()


@pytest.fixture
async def async_resource_manager():
    """Provide an async resource manager for testing."""
    from edge_tpu_v5_benchmark.auto_scaling import AdaptiveResourceManager
    
    manager = AdaptiveResourceManager(
        metrics_window_size=10,
        evaluation_interval=1.0
    )
    await manager.start()
    
    yield manager
    
    await manager.stop()


# Mock component fixtures
@pytest.fixture
def mock_benchmark():
    """Mock benchmark instance for testing."""
    from edge_tpu_v5_benchmark.benchmark import TPUv5Benchmark
    
    with patch.object(TPUv5Benchmark, "_is_device_available", return_value=False):
        benchmark = TPUv5Benchmark()
        yield benchmark


@pytest.fixture
def mock_model():
    """Mock compiled TPU model for testing."""
    from edge_tpu_v5_benchmark.models import CompiledTPUModel
    
    model = CompiledTPUModel(
        path="mock_model.onnx",
        optimization_level=3,
        target="tpu_v5_edge"
    )
    
    # Mock the run method to avoid numpy dependency
    def mock_run(input_data):
        time.sleep(0.001)  # Simulate 1ms inference
        return {"output": [0.5] * 1000}  # Mock output without numpy
    
    model.run = mock_run
    yield model


@pytest.fixture
def mock_cache_manager(temp_cache_dir):
    """Mock cache manager for testing."""
    from edge_tpu_v5_benchmark.cache import CacheManager
    
    manager = CacheManager(temp_cache_dir)
    yield manager
    
    # Cleanup
    manager.clear_all()


@pytest.fixture  
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    from edge_tpu_v5_benchmark.monitoring import MetricsCollector
    
    collector = MetricsCollector(max_points=1000)
    yield collector


@pytest.fixture
def mock_health_monitor():
    """Mock health monitor for testing."""
    from edge_tpu_v5_benchmark.health import HealthMonitor
    
    monitor = HealthMonitor()
    yield monitor


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer for performance measurements."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
        
        def __enter__(self):
            self.start()
            return self
        
        def __exit__(self, *args):
            self.stop()
    
    return Timer()


@pytest.fixture
def memory_profiler():
    """Memory profiler for memory usage testing."""
    try:
        import psutil
        
        class MemoryProfiler:
            def __init__(self):
                self.process = psutil.Process()
                self.initial_memory = None
                self.peak_memory = None
            
            def start(self):
                self.initial_memory = self.process.memory_info().rss
                self.peak_memory = self.initial_memory
            
            def sample(self):
                current_memory = self.process.memory_info().rss
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                return current_memory
            
            @property
            def memory_increase(self):
                if self.initial_memory is None:
                    return None
                return self.peak_memory - self.initial_memory
            
            def __enter__(self):
                self.start()
                return self
            
            def __exit__(self, *args):
                self.sample()
        
        return MemoryProfiler()
        
    except ImportError:
        # Mock memory profiler if psutil not available
        class MockMemoryProfiler:
            def __init__(self):
                self.initial_memory = 100 * 1024 * 1024  # 100MB
                self.peak_memory = self.initial_memory
            
            def start(self):
                pass
            
            def sample(self):
                return self.peak_memory
            
            @property
            def memory_increase(self):
                return 10 * 1024 * 1024  # 10MB increase
            
            def __enter__(self):
                self.start()
                return self
            
            def __exit__(self, *args):
                self.sample()
        
        return MockMemoryProfiler()


# Data generation fixtures
@pytest.fixture
def random_data_generator():
    """Generate random test data."""
    import random
    import string
    
    class DataGenerator:
        @staticmethod
        def random_string(length=10):
            return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
        
        @staticmethod
        def random_dict(size=5):
            return {
                DataGenerator.random_string(8): DataGenerator.random_string(16)
                for _ in range(size)
            }
        
        @staticmethod
        def random_list(size=10):
            return [random.randint(1, 1000) for _ in range(size)]
        
        @staticmethod
        def large_data(size_mb=1):
            """Generate large data of specified size in MB."""
            size_bytes = size_mb * 1024 * 1024
            return "x" * size_bytes
    
    return DataGenerator()


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Clean up global state after each test."""
    yield
    
    # Clear any global singletons or caches
    try:
        from edge_tpu_v5_benchmark.cache import _cache_manager
        if _cache_manager:
            _cache_manager.clear_all()
    except:
        pass
    
    try:
        from edge_tpu_v5_benchmark.auto_scaling import _resource_manager
        if _resource_manager and _resource_manager.running:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_resource_manager.stop())
            except:
                pass
    except:
        pass


# Skip conditions
def requires_tpu():
    """Decorator to skip tests that require TPU hardware."""
    return pytest.mark.skipif(
        not os.path.exists("/dev/apex_0"),
        reason="TPU hardware not available"
    )


def requires_network():
    """Decorator to skip tests that require network access."""
    import socket
    
    def check_network():
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    return pytest.mark.skipif(
        not check_network(),
        reason="Network access not available"
    )


# Test data constants
TEST_MODEL_SHAPES = [
    (1, 3, 224, 224),    # Standard vision model
    (1, 3, 512, 512),    # High resolution vision
    (1, 1000),           # Classification output
    (1, 768),            # BERT-like model
    (1, 512, 768),       # Transformer model
]

TEST_BATCH_SIZES = [1, 2, 4, 8, 16]

TEST_ITERATION_COUNTS = [10, 50, 100, 500, 1000]