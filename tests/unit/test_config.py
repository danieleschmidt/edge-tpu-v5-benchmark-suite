"""Tests for configuration module."""

import pytest
import os
import tempfile
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.config import (
    BenchmarkSuiteConfig,
    ConfigManager,
    TPUConfig,
    BenchmarkConfig,
    ModelConfig,
    LoggingConfig,
    LogLevel,
    OptimizationLevel,
    get_config,
    initialize_config,
    get_config_manager
)


class TestConfigDataClasses:
    """Test configuration dataclasses."""
    
    def test_tpu_config_defaults(self):
        """Test TPU configuration defaults."""
        config = TPUConfig()
        
        assert config.device_path == "/dev/apex_0"
        assert config.enable_simulation is False
        assert config.runtime_version == "v5.0"
        assert config.max_memory_gb == 16.0
        assert config.thermal_throttle_threshold == 85.0
    
    def test_benchmark_config_defaults(self):
        """Test benchmark configuration defaults."""
        config = BenchmarkConfig()
        
        assert config.default_iterations == 1000
        assert config.default_warmup_iterations == 100
        assert config.timeout_seconds == 600
        assert config.confidence_level == 0.95
        assert config.enable_power_monitoring is True
        assert config.power_sampling_rate_hz == 1000
    
    def test_model_config_defaults(self):
        """Test model configuration defaults."""
        config = ModelConfig()
        
        assert config.cache_dir == "~/.edge_tpu_v5_cache"
        assert config.cache_max_size_gb == 10.0
        assert config.auto_cleanup_cache is True
        assert config.default_optimization_level == OptimizationLevel.MODERATE
        assert config.enable_quantization is True
        assert config.max_model_size_mb == 1000
        assert ".onnx" in config.allowed_extensions
        assert ".tflite" in config.allowed_extensions
    
    def test_logging_config_defaults(self):
        """Test logging configuration defaults."""
        config = LoggingConfig()
        
        assert config.level == LogLevel.INFO
        assert config.format == "json"
        assert config.file_path == "logs/benchmark.log"
        assert config.enable_structured_logging is True
        assert config.max_file_size_mb == 100
        assert config.backup_count == 5
    
    def test_benchmark_suite_config_creation(self):
        """Test creating complete benchmark suite configuration."""
        config = BenchmarkSuiteConfig()
        
        assert isinstance(config.tpu, TPUConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.logging, LoggingConfig)
        
        # Test to_dict conversion
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "tpu" in config_dict
        assert "benchmark" in config_dict
        assert "model" in config_dict
        assert "logging" in config_dict
    
    def test_config_save_yaml(self, temp_dir):
        """Test saving configuration to YAML file."""
        config = BenchmarkSuiteConfig()
        config.tpu.device_path = "/dev/custom_tpu"
        config.benchmark.default_iterations = 500
        
        yaml_path = temp_dir / "config.yaml"
        config.save(yaml_path, format="yaml")
        
        assert yaml_path.exists()
        
        # Verify content
        with open(yaml_path, 'r') as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["tpu"]["device_path"] == "/dev/custom_tpu"
        assert loaded_data["benchmark"]["default_iterations"] == 500
    
    def test_config_save_json(self, temp_dir):
        """Test saving configuration to JSON file."""
        config = BenchmarkSuiteConfig()
        config.tpu.device_path = "/dev/custom_tpu"
        
        json_path = temp_dir / "config.json"
        config.save(json_path, format="json")
        
        assert json_path.exists()
        
        # Verify content
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["tpu"]["device_path"] == "/dev/custom_tpu"
    
    def test_config_save_invalid_format(self, temp_dir):
        """Test saving configuration with invalid format raises error."""
        config = BenchmarkSuiteConfig()
        
        with pytest.raises(ValueError, match="Unsupported format"):
            config.save(temp_dir / "config.txt", format="txt")


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_init_default(self):
        """Test ConfigManager initialization with defaults."""
        manager = ConfigManager()
        
        assert manager.config_file is None
        assert isinstance(manager.config, BenchmarkSuiteConfig)
        assert manager.config.tpu.device_path == "/dev/apex_0"
    
    def test_init_with_config_file(self, temp_dir):
        """Test ConfigManager initialization with config file."""
        # Create a config file
        config_data = {
            "tpu": {"device_path": "/dev/custom_tpu"},
            "benchmark": {"default_iterations": 500}
        }
        
        config_file = temp_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        manager = ConfigManager(config_file)
        
        assert manager.config.tpu.device_path == "/dev/custom_tpu"
        assert manager.config.benchmark.default_iterations == 500
    
    def test_load_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "TPU_DEVICE_PATH": "/dev/env_tpu",
            "DEFAULT_ITERATIONS": "750",
            "POWER_MONITORING_ENABLED": "false",
            "LOG_LEVEL": "DEBUG",
            "CACHE_MAX_SIZE_GB": "5.5"
        }
        
        with patch.dict(os.environ, env_vars):
            manager = ConfigManager()
            
            assert manager.config.tpu.device_path == "/dev/env_tpu"
            assert manager.config.benchmark.default_iterations == 750
            assert manager.config.benchmark.enable_power_monitoring is False
            assert manager.config.logging.level == "DEBUG"
            assert manager.config.model.cache_max_size_gb == 5.5


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_workflow(self, temp_dir):
        """Test complete configuration workflow."""
        # Step 1: Create config file with custom settings
        config_data = {
            "tpu": {
                "device_path": "/dev/integration_tpu",
                "enable_simulation": True
            },
            "benchmark": {
                "default_iterations": 2000,
                "enable_power_monitoring": False
            }
        }
        
        config_file = temp_dir / "integration_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Step 2: Initialize config manager
        manager = ConfigManager(config_file)
        
        # Step 3: Verify file settings were loaded
        assert manager.config.tpu.device_path == "/dev/integration_tpu"
        assert manager.config.tpu.enable_simulation is True
        assert manager.config.benchmark.enable_power_monitoring is False