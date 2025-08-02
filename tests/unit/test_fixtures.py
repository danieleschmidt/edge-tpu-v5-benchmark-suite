"""Unit tests for test fixtures and utilities."""

import numpy as np
import pytest


class TestFixtures:
    """Test that test fixtures work correctly."""
    
    def test_mock_tpu_device(self, mock_tpu_device):
        """Test mock TPU device fixture."""
        assert mock_tpu_device.device_path == "/dev/apex_0"
        assert mock_tpu_device.is_available() is True
        
        info = mock_tpu_device.get_info()
        assert info["version"] == "v5"
        assert info["memory_mb"] == 8192
        assert info["max_power_w"] == 4.0
        
    def test_mock_model(self, mock_model):
        """Test mock model fixture."""
        assert mock_model.name == "test_mobilenet_v3"
        assert mock_model.input_shape == (1, 224, 224, 3)
        assert mock_model.output_shape == (1, 1000)
        assert mock_model.size_bytes == 1024 * 1024
        assert mock_model.is_compiled is True
        
    def test_sample_input_data(self, sample_input_data):
        """Test sample input data fixture."""
        assert isinstance(sample_input_data, np.ndarray)
        assert sample_input_data.shape == (1, 224, 224, 3)
        assert sample_input_data.dtype == np.uint8
        assert 0 <= sample_input_data.min() <= sample_input_data.max() <= 255
        
    def test_sample_batch_data(self, sample_batch_data):
        """Test sample batch data fixture."""
        assert isinstance(sample_batch_data, np.ndarray)
        assert sample_batch_data.shape == (8, 224, 224, 3)
        assert sample_batch_data.dtype == np.uint8
        
    def test_benchmark_config(self, benchmark_config):
        """Test benchmark config fixture."""
        required_keys = [
            "iterations", "warmup_iterations", "timeout_seconds",
            "measure_power", "measure_thermal", "output_format", "save_results"
        ]
        for key in required_keys:
            assert key in benchmark_config
            
        assert benchmark_config["iterations"] == 10
        assert benchmark_config["warmup_iterations"] == 2
        assert benchmark_config["timeout_seconds"] == 30
        
    def test_mock_power_meter(self, mock_power_meter):
        """Test mock power meter fixture."""
        assert mock_power_meter.get_current_power() == 2.5
        
        measurements = mock_power_meter.get_measurements()
        assert len(measurements) == 5
        assert all(isinstance(m, (int, float)) for m in measurements)
        
    def test_mock_thermal_sensor(self, mock_thermal_sensor):
        """Test mock thermal sensor fixture."""
        assert mock_thermal_sensor.get_temperature() == 45.0
        assert mock_thermal_sensor.get_thermal_state() == "normal"
        assert mock_thermal_sensor.is_throttling() is False
        
    def test_benchmark_results_sample(self, benchmark_results_sample):
        """Test benchmark results sample fixture."""
        assert benchmark_results_sample["model_name"] == "test_mobilenet_v3"
        assert "device_info" in benchmark_results_sample
        assert "performance_metrics" in benchmark_results_sample
        assert "power_metrics" in benchmark_results_sample
        assert "system_info" in benchmark_results_sample
        
    def test_mock_tpu_runtime(self, mock_tpu_runtime):
        """Test mock TPU runtime fixture."""
        assert not mock_tpu_runtime.is_initialized
        
        success = mock_tpu_runtime.initialize()
        assert success is True
        assert mock_tpu_runtime.is_initialized is True
        
        devices = mock_tpu_runtime.get_devices()
        assert "/dev/apex_0" in devices


class TestDataGenerators:
    """Test data generation utilities."""
    
    def test_test_image_fixture(self, test_image):
        """Test generated test image."""
        assert isinstance(test_image, np.ndarray)
        assert test_image.shape == (224, 224, 3)
        assert test_image.dtype == np.uint8
        
    def test_test_batch_fixture(self, test_batch):
        """Test generated test batch."""
        assert isinstance(test_batch, np.ndarray)
        assert test_batch.shape == (8, 224, 224, 3)
        assert test_batch.dtype == np.uint8
        
    def test_performance_timer(self, performance_timer):
        """Test performance timer fixture."""
        import time
        
        performance_timer.start()
        time.sleep(0.01)  # Sleep for 10ms
        elapsed = performance_timer.stop()
        
        assert elapsed >= 0.01
        assert elapsed < 0.1  # Should be much less than 100ms


class TestCustomAssertions:
    """Test custom assertion functions."""
    
    def test_benchmark_results_validation(self, benchmark_results_sample, assert_benchmark_valid):
        """Test benchmark results validation function."""
        # Should not raise any assertions for valid data
        assert_benchmark_valid(benchmark_results_sample)
        
    def test_benchmark_results_validation_invalid(self, assert_benchmark_valid):
        """Test benchmark results validation with invalid data."""
        invalid_results = {"invalid": "data"}
        
        with pytest.raises(AssertionError):
            assert_benchmark_valid(invalid_results)
            
    def test_power_metrics_validation(self, benchmark_results_sample, assert_power_valid):
        """Test power metrics validation function."""
        power_metrics = benchmark_results_sample["power_metrics"]
        
        # Should not raise any assertions for valid data
        assert_power_valid(power_metrics)
        
    def test_power_metrics_validation_invalid(self, assert_power_valid):
        """Test power metrics validation with invalid data."""
        invalid_power = {
            "avg_power_w": -1.0,  # Invalid negative power
            "peak_power_w": 0.5,  # Peak less than average
            "total_energy_j": 0,  # Zero energy
            "efficiency_fps_per_w": -10  # Negative efficiency
        }
        
        with pytest.raises(AssertionError):
            assert_power_valid(invalid_power)