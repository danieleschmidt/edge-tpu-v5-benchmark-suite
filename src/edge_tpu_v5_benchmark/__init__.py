"""Edge TPU v5 Benchmark Suite

First comprehensive open-source benchmark harness for Google's TPU v5 edge cards.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .benchmark import TPUv5Benchmark, BenchmarkResults
from .models import ModelLoader, CompiledTPUModel, ModelRegistry, ModelOptimizer
from .power import PowerProfiler, PowerMeasurement, PowerStatistics
from .database import BenchmarkDatabase, DataManager, ResultsCache
from .config import (
    BenchmarkSuiteConfig, 
    ConfigManager,
    get_config,
    initialize_config,
    get_config_manager
)

__all__ = [
    # Core classes
    "TPUv5Benchmark",
    "BenchmarkResults",
    "ModelLoader",
    "CompiledTPUModel", 
    "ModelRegistry",
    "ModelOptimizer",
    "PowerProfiler",
    "PowerMeasurement",
    "PowerStatistics",
    
    # Data management
    "BenchmarkDatabase",
    "DataManager",
    "ResultsCache",
    
    # Configuration
    "BenchmarkSuiteConfig",
    "ConfigManager",
    "get_config",
    "initialize_config",
    "get_config_manager",
    
    # Metadata
    "__version__"
]