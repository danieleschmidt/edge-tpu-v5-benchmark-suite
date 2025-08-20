"""Edge TPU v5 Benchmark Suite

First comprehensive open-source benchmark harness for Google's TPU v5 edge cards.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .benchmark import BenchmarkResults, TPUv5Benchmark
from .cache import CacheManager, PredictiveSmartCache
from .compiler import CompilerAnalysis, CompilerAnalyzer, TPUv5Optimizer
from .config import (
    BenchmarkSuiteConfig,
    ConfigManager,
    get_config,
    get_config_manager,
    initialize_config,
)
from .converter import (
    ONNXToTPUv5,
    PyTorchToTPUv5,
    TensorFlowToTPUv5,
    prepare_for_tpu_v5,
)
from .database import BenchmarkDatabase, DataManager, ResultsCache
from .models import CompiledTPUModel, ModelLoader, ModelOptimizer, ModelRegistry
from .power import PowerMeasurement, PowerProfiler, PowerStatistics
from .quantum_auto_scaling import (
    LoadBalancer,
    LoadBalancingStrategy,
    QuantumAutoScaler,
    QuantumNode,
    ScalingPolicy,
)
from .quantum_compliance import (
    ConsentManager,
    DataCategory,
    ProcessingPurpose,
    QuantumComplianceManager,
)
from .quantum_i18n import (
    LocalizationConfig,
    QuantumLocalizer,
    SupportedLanguage,
    detect_and_set_locale,
    t,
)
from .quantum_monitoring import (
    HealthStatus,
    MetricsCollector,
    PerformanceMetrics,
    QuantumHealthMonitor,
)
from .quantum_performance import (
    AdaptiveCache,
    OptimizationStrategy,
    OptimizedQuantumTaskPlanner,
    PerformanceProfile,
)
from .quantum_planner import (
    QuantumAnnealer,
    QuantumResource,
    QuantumState,
    QuantumTask,
    QuantumTaskPlanner,
)
from .quantum_security import (
    QuantumSecurityManager,
    SecureQuantumTaskPlanner,
    SecurityLevel,
    SecurityPolicy,
)
from .quantum_standalone import (
    StandaloneConfig,
    StandaloneQuantumPlanner,
    create_performance_planner,
    create_secure_planner,
    create_simple_planner,
)
from .quantum_validation import (
    QuantumSystemValidator,
    QuantumTaskValidator,
    ValidationReport,
    ValidationSeverity,
)
from .security import (
    DataSanitizer,
    InputValidator,
    SecurityContext,
    SecurityLoggingFilter,
)
from .adaptive_quantum_error_mitigation import (
    AdaptiveErrorMitigationFramework,
    MLWorkloadProfiler,
    ErrorMitigationType,
    MLWorkloadType,
    WorkloadCharacteristics,
    ErrorProfile,
    MitigationStrategy,
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

    # Adaptive error mitigation
    "AdaptiveErrorMitigationFramework",
    "MLWorkloadProfiler",
    "ErrorMitigationType",
    "MLWorkloadType",
    "WorkloadCharacteristics",
    "ErrorProfile",
    "MitigationStrategy",

    # Metadata
    "__version__"
]
