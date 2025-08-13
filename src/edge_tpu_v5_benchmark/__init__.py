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
from .cache import CacheManager, PredictiveSmartCache
from .config import (
    BenchmarkSuiteConfig, 
    ConfigManager,
    get_config,
    initialize_config,
    get_config_manager
)
from .compiler import CompilerAnalyzer, TPUv5Optimizer, CompilerAnalysis
from .converter import ONNXToTPUv5, TensorFlowToTPUv5, PyTorchToTPUv5, prepare_for_tpu_v5
from .security import SecurityContext, InputValidator, DataSanitizer, SecurityLoggingFilter
from .quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumResource,
    QuantumState,
    QuantumAnnealer
)
from .quantum_validation import (
    QuantumTaskValidator,
    QuantumSystemValidator,
    ValidationReport,
    ValidationSeverity
)
from .quantum_monitoring import (
    QuantumHealthMonitor,
    MetricsCollector,
    HealthStatus,
    PerformanceMetrics
)
from .quantum_security import (
    QuantumSecurityManager,
    SecureQuantumTaskPlanner,
    SecurityPolicy,
    SecurityLevel
)
from .quantum_performance import (
    OptimizedQuantumTaskPlanner,
    PerformanceProfile,
    OptimizationStrategy,
    AdaptiveCache
)
from .quantum_auto_scaling import (
    QuantumAutoScaler,
    QuantumNode,
    ScalingPolicy,
    LoadBalancer,
    LoadBalancingStrategy
)
from .quantum_i18n import (
    QuantumLocalizer,
    LocalizationConfig,
    SupportedLanguage,
    t,
    detect_and_set_locale
)
from .quantum_compliance import (
    QuantumComplianceManager,
    DataCategory,
    ProcessingPurpose,
    ConsentManager
)
from .quantum_standalone import (
    StandaloneQuantumPlanner,
    StandaloneConfig,
    create_simple_planner,
    create_secure_planner,
    create_performance_planner
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
    "CacheManager",
    "PredictiveSmartCache",
    
    # Configuration
    "BenchmarkSuiteConfig",
    "ConfigManager",
    "get_config",
    "initialize_config",
    "get_config_manager",
    
    # Compiler and optimization
    "CompilerAnalyzer",
    "TPUv5Optimizer", 
    "CompilerAnalysis",
    
    # Model conversion
    "ONNXToTPUv5",
    "TensorFlowToTPUv5", 
    "PyTorchToTPUv5",
    "prepare_for_tpu_v5",
    
    # Security utilities
    "SecurityContext",
    "InputValidator",
    "DataSanitizer",
    "SecurityLoggingFilter",
    
    # Quantum task planning
    "QuantumTaskPlanner",
    "QuantumTask",
    "QuantumResource", 
    "QuantumState",
    "QuantumAnnealer",
    
    # Quantum validation
    "QuantumTaskValidator",
    "QuantumSystemValidator", 
    "ValidationReport",
    "ValidationSeverity",
    
    # Quantum monitoring
    "QuantumHealthMonitor",
    "MetricsCollector",
    "HealthStatus", 
    "PerformanceMetrics",
    
    # Quantum security
    "QuantumSecurityManager",
    "SecureQuantumTaskPlanner",
    "SecurityPolicy",
    "SecurityLevel",
    
    # Quantum performance
    "OptimizedQuantumTaskPlanner",
    "PerformanceProfile",
    "OptimizationStrategy",
    "AdaptiveCache",
    
    # Quantum auto-scaling
    "QuantumAutoScaler",
    "QuantumNode",
    "ScalingPolicy",
    "LoadBalancer",
    "LoadBalancingStrategy",
    
    # Quantum internationalization
    "QuantumLocalizer",
    "LocalizationConfig",
    "SupportedLanguage",
    "t",
    "detect_and_set_locale",
    
    # Quantum compliance
    "QuantumComplianceManager",
    "DataCategory",
    "ProcessingPurpose",
    "ConsentManager",
    
    # Standalone planner
    "StandaloneQuantumPlanner",
    "StandaloneConfig",
    "create_simple_planner",
    "create_secure_planner",
    "create_performance_planner",
    
    # Metadata
    "__version__"
]