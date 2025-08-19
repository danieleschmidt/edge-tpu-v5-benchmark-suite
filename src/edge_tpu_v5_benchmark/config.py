"""Configuration management for the Edge TPU v5 Benchmark Suite."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizationLevel(Enum):
    """Model optimization level enumeration."""
    BASIC = 1
    MODERATE = 2
    AGGRESSIVE = 3


@dataclass
class TPUConfig:
    """TPU hardware configuration."""
    device_path: str = "/dev/apex_0"
    enable_simulation: bool = False
    runtime_version: str = "v5.0"
    compiler_path: str = "/usr/local/bin/edgetpu_compiler"
    runtime_path: str = "/usr/local/lib/libedgetpu.so"
    max_memory_gb: float = 16.0
    thermal_throttle_threshold: float = 85.0


@dataclass
class BenchmarkConfig:
    """Benchmark execution configuration."""
    default_iterations: int = 1000
    default_warmup_iterations: int = 100
    timeout_seconds: int = 600
    confidence_level: float = 0.95
    min_statistical_runs: int = 10
    max_concurrent_benchmarks: int = 1
    enable_power_monitoring: bool = True
    power_sampling_rate_hz: int = 1000
    thermal_monitoring_enabled: bool = True


@dataclass
class ModelConfig:
    """Model compilation and optimization configuration."""
    cache_dir: str = "~/.edge_tpu_v5_cache"
    cache_max_size_gb: float = 10.0
    auto_cleanup_cache: bool = True
    default_optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    enable_quantization: bool = True
    quantization_method: str = "post_training"
    validation_level: str = "moderate"
    max_model_size_mb: int = 1000
    allowed_extensions: List[str] = field(default_factory=lambda: [".onnx", ".tflite", ".pb"])


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "json"
    file_path: str = "logs/benchmark.log"
    enable_structured_logging: bool = True
    log_dir: str = "./logs"
    max_file_size_mb: int = 100
    backup_count: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///benchmark_results.db"
    result_retention_days: int = 365
    enable_compression: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24


@dataclass
class SecurityConfig:
    """Security configuration."""
    allow_remote_models: bool = False
    verify_model_signatures: bool = True
    sandbox_execution: bool = True
    allow_unsafe_models: bool = False
    enable_model_verification: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    numpy_threads: int = 4
    openblas_num_threads: int = 4
    mkl_num_threads: int = 4
    omp_num_threads: int = 4
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    disk_space_warning_gb: float = 5.0


@dataclass
class NetworkConfig:
    """Network and API configuration."""
    api_base_url: str = "https://api.edge-tpu-benchmark.org"
    api_key: str = ""
    api_timeout_seconds: int = 30
    leaderboard_enabled: bool = False
    model_download_timeout: int = 300
    model_download_retries: int = 3
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = "localhost,127.0.0.1"


@dataclass
class ReportingConfig:
    """Reporting and output configuration."""
    default_output_formats: List[str] = field(default_factory=lambda: ["json", "html"])
    include_raw_data: bool = False
    auto_generate_reports: bool = True
    report_template_dir: str = "templates/reports"
    results_dir: str = "./results"


@dataclass
class ExperimentalConfig:
    """Experimental features configuration."""
    enable_experimental_features: bool = False
    auto_optimization: bool = False
    parallel_benchmarking: bool = False
    enable_gpu_fallback: bool = False
    gpu_device_id: int = 0


@dataclass
class BenchmarkSuiteConfig:
    """Complete benchmark suite configuration."""
    tpu: TPUConfig = field(default_factory=TPUConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    experimental: ExperimentalConfig = field(default_factory=ExperimentalConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file.
        
        Args:
            path: File path to save configuration
            format: File format ('yaml' or 'json')
        """
        data = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "yaml":
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved configuration to {path}")


class ConfigManager:
    """Configuration manager for loading and managing settings."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file) if config_file else None
        self._config = BenchmarkSuiteConfig()
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load configuration from various sources."""
        # 1. Load from file if specified
        if self.config_file and self.config_file.exists():
            self._load_from_file(self.config_file)

        # 2. Load from environment variables
        self._load_from_environment()

        # 3. Apply any command-line overrides (if available)
        self._apply_overrides()

        logger.info("Configuration loaded successfully")

    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file) as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")

            self._update_config_from_dict(data)
            logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            # TPU Configuration
            "TPU_DEVICE_PATH": ("tpu", "device_path"),
            "TPU_ENABLE_SIMULATION": ("tpu", "enable_simulation", bool),
            "TPU_RUNTIME_VERSION": ("tpu", "runtime_version"),
            "TPU_COMPILER_PATH": ("tpu", "compiler_path"),
            "TPU_RUNTIME_PATH": ("tpu", "runtime_path"),

            # Benchmark Configuration
            "DEFAULT_ITERATIONS": ("benchmark", "default_iterations", int),
            "DEFAULT_WARMUP_ITERATIONS": ("benchmark", "default_warmup_iterations", int),
            "BENCHMARK_TIMEOUT_SECONDS": ("benchmark", "timeout_seconds", int),
            "POWER_SAMPLING_RATE_HZ": ("benchmark", "power_sampling_rate_hz", int),
            "POWER_MONITORING_ENABLED": ("benchmark", "enable_power_monitoring", bool),
            "THERMAL_MONITORING_ENABLED": ("benchmark", "thermal_monitoring_enabled", bool),

            # Model Configuration
            "MODEL_CACHE_DIR": ("model", "cache_dir"),
            "CACHE_MAX_SIZE_GB": ("model", "cache_max_size_gb", float),
            "AUTO_CLEANUP_CACHE": ("model", "auto_cleanup_cache", bool),
            "DEFAULT_OPTIMIZATION_LEVEL": ("model", "default_optimization_level", int),
            "ENABLE_QUANTIZATION": ("model", "enable_quantization", bool),
            "MAX_MODEL_SIZE_MB": ("model", "max_model_size_mb", int),

            # Logging Configuration
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FORMAT": ("logging", "format"),
            "LOG_FILE_PATH": ("logging", "file_path"),
            "LOG_DIR": ("logging", "log_dir"),

            # Database Configuration
            "DATABASE_URL": ("database", "url"),
            "RESULT_RETENTION_DAYS": ("database", "result_retention_days", int),

            # Security Configuration
            "ALLOW_REMOTE_MODELS": ("security", "allow_remote_models", bool),
            "VERIFY_MODEL_SIGNATURES": ("security", "verify_model_signatures", bool),
            "SANDBOX_EXECUTION": ("security", "sandbox_execution", bool),

            # Performance Configuration
            "NUMPY_THREADS": ("performance", "numpy_threads", int),
            "MAX_MEMORY_GB": ("performance", "max_memory_gb", float),
            "MAX_CPU_CORES": ("performance", "max_cpu_cores", int),

            # Network Configuration
            "API_BASE_URL": ("network", "api_base_url"),
            "API_KEY": ("network", "api_key"),
            "API_TIMEOUT_SECONDS": ("network", "api_timeout_seconds", int),
            "LEADERBOARD_ENABLED": ("network", "leaderboard_enabled", bool),
            "HTTP_PROXY": ("network", "http_proxy"),
            "HTTPS_PROXY": ("network", "https_proxy"),

            # Experimental Configuration
            "ENABLE_EXPERIMENTAL_FEATURES": ("experimental", "enable_experimental_features", bool),
            "AUTO_OPTIMIZATION": ("experimental", "auto_optimization", bool),
            "PARALLEL_BENCHMARKING": ("experimental", "parallel_benchmarking", bool),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_config_value(config_path, value)

    def _set_config_value(self, config_path: tuple, value: str) -> None:
        """Set configuration value from environment variable."""
        section = config_path[0]
        key = config_path[1]
        type_converter = config_path[2] if len(config_path) > 2 else str

        # Convert string value to appropriate type
        if type_converter == bool:
            converted_value = value.lower() in ('true', '1', 'yes', 'on')
        elif type_converter == int:
            converted_value = int(value)
        elif type_converter == float:
            converted_value = float(value)
        else:
            converted_value = value

        # Set the value in the configuration
        section_obj = getattr(self._config, section)
        setattr(section_obj, key, converted_value)

    def _update_config_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary data."""
        for section_name, section_data in data.items():
            if hasattr(self._config, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self._config, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

    def _apply_overrides(self) -> None:
        """Apply any additional configuration overrides."""
        # This method can be extended to handle command-line arguments
        # or other runtime configuration sources
        pass

    @property
    def config(self) -> BenchmarkSuiteConfig:
        """Get the current configuration."""
        return self._config

    def get_section(self, section_name: str):
        """Get a specific configuration section."""
        return getattr(self._config, section_name, None)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._update_config_from_dict(updates)
        logger.info("Configuration updated")

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate TPU configuration
        tpu_device = Path(self.config.tpu.device_path)
        if not self.config.tpu.enable_simulation and not tpu_device.exists():
            issues.append(f"TPU device not found: {self.config.tpu.device_path}")

        # Validate directories
        for dir_path in [
            self.config.model.cache_dir,
            self.config.logging.log_dir,
            self.config.reporting.results_dir
        ]:
            expanded_path = Path(dir_path).expanduser()
            try:
                expanded_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {dir_path}: {e}")

        # Validate numeric ranges
        if self.config.benchmark.default_iterations < 1:
            issues.append("default_iterations must be >= 1")

        if self.config.benchmark.confidence_level <= 0 or self.config.benchmark.confidence_level >= 1:
            issues.append("confidence_level must be between 0 and 1")

        if self.config.model.cache_max_size_gb <= 0:
            issues.append("cache_max_size_gb must be > 0")

        # Validate optimization level
        if not (1 <= self.config.model.default_optimization_level.value <= 3):
            issues.append("default_optimization_level must be 1, 2, or 3")

        return issues

    def setup_environment(self) -> None:
        """Setup environment based on configuration."""
        # Set performance-related environment variables
        os.environ["NUMPY_THREADS"] = str(self.config.performance.numpy_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.performance.openblas_num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.performance.mkl_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(self.config.performance.omp_num_threads)

        # Set proxy environment variables if configured
        if self.config.network.http_proxy:
            os.environ["HTTP_PROXY"] = self.config.network.http_proxy
        if self.config.network.https_proxy:
            os.environ["HTTPS_PROXY"] = self.config.network.https_proxy
        if self.config.network.no_proxy:
            os.environ["NO_PROXY"] = self.config.network.no_proxy

        # Create necessary directories
        for dir_path in [
            self.config.model.cache_dir,
            self.config.logging.log_dir,
            self.config.reporting.results_dir
        ]:
            Path(dir_path).expanduser().mkdir(parents=True, exist_ok=True)

        logger.info("Environment setup completed")

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration with all sources merged."""
        return self.config.to_dict()

    def save_config(self, path: Union[str, Path], format: str = "yaml") -> None:
        """Save current configuration to file."""
        self.config.save(path, format)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> BenchmarkSuiteConfig:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def initialize_config(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
