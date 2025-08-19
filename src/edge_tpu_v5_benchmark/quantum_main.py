"""Main entry point for standalone quantum task planner.

This module provides direct access to quantum functionality without TPU dependencies.
"""

# Direct imports of quantum modules to avoid dependency issues
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
from .quantum_security import QuantumSecurityManager, SecurityLevel, SecurityPolicy
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

# Version info
__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

# Export main classes for direct import
__all__ = [
    # Core quantum classes
    "QuantumTaskPlanner",
    "QuantumTask",
    "QuantumResource",
    "QuantumState",
    "QuantumAnnealer",

    # Validation
    "QuantumTaskValidator",
    "QuantumSystemValidator",
    "ValidationReport",
    "ValidationSeverity",

    # Monitoring
    "QuantumHealthMonitor",
    "MetricsCollector",
    "HealthStatus",
    "PerformanceMetrics",

    # Security
    "QuantumSecurityManager",
    "SecurityPolicy",
    "SecurityLevel",

    # Performance
    "OptimizedQuantumTaskPlanner",
    "PerformanceProfile",
    "OptimizationStrategy",
    "AdaptiveCache",

    # Auto-scaling
    "QuantumAutoScaler",
    "QuantumNode",
    "ScalingPolicy",
    "LoadBalancer",
    "LoadBalancingStrategy",

    # Internationalization
    "QuantumLocalizer",
    "LocalizationConfig",
    "SupportedLanguage",
    "t",
    "detect_and_set_locale",

    # Compliance
    "QuantumComplianceManager",
    "DataCategory",
    "ProcessingPurpose",
    "ConsentManager",

    # Standalone
    "StandaloneQuantumPlanner",
    "StandaloneConfig",
    "create_simple_planner",
    "create_secure_planner",
    "create_performance_planner",

    # Metadata
    "__version__",
    "__author__"
]
