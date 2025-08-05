"""Pytest configuration and shared fixtures for TPU v5 benchmark suite tests."""

import pytest
import tempfile
import logging
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Test configuration constants
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_MODELS_DIR = TEST_DATA_DIR / "models"
TEST_RESULTS_DIR = TEST_DATA_DIR / "results"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)
TEST_RESULTS_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration settings."""
    return {
        "device_path": "/dev/mock_tpu",
        "default_iterations": 10,
        "warmup_iterations": 2,
        "timeout": 30,
        "models_dir": str(TEST_MODELS_DIR),
        "results_dir": str(TEST_RESULTS_DIR),
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_tpu_device():
    """Mock TPU device for testing."""
    device = Mock()
    device.device_path = "/dev/mock_tpu"
    device.is_available.return_value = True
    device.get_info.return_value = {
        "version": "v5_edge",
        "serial": "mock_serial_123",
        "temperature": 45.0,
        "power_consumption": 1.2,
    }
    return device


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.name = "test_model"
    model.input_shape = (1, 3, 224, 224)
    model.output_shape = (1, 1000)
    model.model_path = str(TEST_MODELS_DIR / "test.tflite")
    model.size_bytes = 5_000_000
    
    # Mock inference method
    def mock_inference(input_data):
        return np.random.random(model.output_shape).astype(np.float32)
    
    model.run_inference = mock_inference
    return model


@pytest.fixture
def sample_input_data():
    """Provide sample input data for testing."""
    return np.random.random((1, 3, 224, 224)).astype(np.float32)


@pytest.fixture
def sample_benchmark_results():
    """Provide sample benchmark results for testing."""
    from edge_tpu_v5_benchmark.benchmark import BenchmarkResults
    
    return BenchmarkResults(
        model_name="test_model",
        iterations=100,
        total_time=10.0,
        avg_latency=0.1,
        throughput=10.0,
        latency_p50=0.095,
        latency_p90=0.12,
        latency_p95=0.13,
        latency_p99=0.15,
        power_consumption=1.5,
        energy_per_inference=0.15,
        temperature_avg=50.0,
        temperature_max=55.0,
        memory_usage=128_000_000,
        accuracy_metrics={"top1": 0.75, "top5": 0.92},
    )


@pytest.fixture
def mock_power_profiler():
    """Mock power profiler for testing."""
    profiler = Mock()
    profiler.start_measurement.return_value = None
    profiler.stop_measurement.return_value = {
        "avg_power": 1.5,
        "max_power": 2.0,
        "total_energy": 15.0,
        "duration": 10.0,
    }
    profiler.is_available.return_value = True
    return profiler


@pytest.fixture
def mock_thermal_monitor():
    """Mock thermal monitor for testing."""
    monitor = Mock()
    monitor.get_temperature.return_value = 45.0
    monitor.get_thermal_state.return_value = {
        "temperature": 45.0,
        "state": "normal",
        "throttled": False,
    }
    monitor.is_available.return_value = True
    return monitor


@pytest.fixture
def sample_onnx_model_path(temp_dir: Path) -> Path:
    """Create a sample ONNX model file for testing."""
    import onnx
    from onnx import helper, TensorProto
    
    # Create a simple ONNX model
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 1000]
    )
    
    # Create a simple Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    
    # Create weight initializer
    weight = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [1000, 3, 3, 3],
        np.random.random((1000, 3, 3, 3)).flatten().tolist(),
    )
    
    # Create the graph
    graph = helper.make_graph(
        [conv_node],
        "test_model",
        [input_tensor],
        [output_tensor],
        [weight],
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name="test")
    
    # Save the model
    model_path = temp_dir / "test_model.onnx"
    onnx.save(model, str(model_path))
    
    return model_path


@pytest.fixture
def sample_tflite_model_path(temp_dir: Path) -> Path:
    """Create a sample TFLite model file for testing."""
    try:
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow not available for TFLite model creation")
    
    # Create a simple TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1000, activation='softmax')
    ])
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    model_path = temp_dir / "test_model.tflite"
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    return model_path


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "EDGE_TPU_DEV_MODE": "1",
        "EDGE_TPU_LOG_LEVEL": "DEBUG",
        "EDGE_TPU_DEVICE_PATH": "/dev/mock_tpu",
        "BENCHMARK_DEFAULT_ITERATIONS": "10",
        "BENCHMARK_WARMUP_ITERATIONS": "2",
        "POWER_MEASUREMENT_ENABLED": "false",
        "LEADERBOARD_SUBMISSION_ENABLED": "false",
        "TELEMETRY_ENABLED": "false",
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_leaderboard_client():
    """Mock leaderboard client for testing."""
    client = Mock()
    client.submit_results.return_value = {
        "submission_id": "test_123",
        "status": "accepted",
        "rank": 42,
        "total_submissions": 100,
    }
    client.get_leaderboard.return_value = [
        {"rank": 1, "model": "model_a", "score": 95.5},
        {"rank": 2, "model": "model_b", "score": 93.2},
        {"rank": 3, "model": "model_c", "score": 91.8},
    ]
    return client


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "hardware: marks tests that require TPU hardware")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "network: marks tests that require network access")
    config.addinivalue_line("markers", "external: marks tests that require external services")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers and skip conditions."""
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_hardware = pytest.mark.skip(reason="need --runhardware option to run")
    skip_network = pytest.mark.skip(reason="need --runnetwork option to run")
    
    for item in items:
        # Handle slow tests
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)
        
        # Handle hardware tests
        if "hardware" in item.keywords and not config.getoption("--runhardware"):
            item.add_marker(skip_hardware)
        
        # Handle network tests
        if "network" in item.keywords and not config.getoption("--runnetwork"):
            item.add_marker(skip_network)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--runhardware", 
        action="store_true", 
        default=False, 
        help="run hardware tests (requires TPU device)"
    )
    parser.addoption(
        "--runnetwork", 
        action="store_true", 
        default=False, 
        help="run network tests (requires internet access)"
    )


# Test utilities
class TestHelpers:
    """Helper utilities for tests."""
    
    @staticmethod
    def assert_results_valid(results):
        """Assert that benchmark results are valid."""
        assert results is not None
        assert results.iterations > 0
        assert results.total_time > 0
        assert results.avg_latency > 0
        assert results.throughput > 0
        assert 0 <= results.latency_p50 <= results.latency_p99
        assert results.power_consumption >= 0
        assert results.energy_per_inference >= 0
        assert results.temperature_avg > 0
        assert results.memory_usage >= 0
    
    @staticmethod
    def create_mock_model_file(path: Path, content: bytes = b"mock_model_data"):
        """Create a mock model file for testing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(content)
        return path


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers