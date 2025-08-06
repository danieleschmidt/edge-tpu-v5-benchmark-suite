"""Main entry point for standalone quantum task planner.

This module provides direct access to quantum functionality without TPU dependencies.
"""

# Direct imports of quantum modules to avoid dependency issues
from .quantum_planner import (
    QuantumTaskPlanner, QuantumTask, QuantumResource, QuantumState, QuantumAnnealer
)
from .quantum_validation import (
    QuantumTaskValidator, QuantumSystemValidator, ValidationReport, ValidationSeverity
)
from .quantum_monitoring import (
    QuantumHealthMonitor, MetricsCollector, HealthStatus, PerformanceMetrics
)
from .quantum_security import (
    QuantumSecurityManager, SecurityPolicy, SecurityLevel
)
from .quantum_performance import (
    OptimizedQuantumTaskPlanner, PerformanceProfile, OptimizationStrategy, AdaptiveCache
)
from .quantum_auto_scaling import (
    QuantumAutoScaler, QuantumNode, ScalingPolicy, LoadBalancer, LoadBalancingStrategy
)
from .quantum_i18n import (
    QuantumLocalizer, LocalizationConfig, SupportedLanguage, t, detect_and_set_locale
)
from .quantum_compliance import (
    QuantumComplianceManager, DataCategory, ProcessingPurpose, ConsentManager
)
from .quantum_standalone import (
    StandaloneQuantumPlanner, StandaloneConfig,
    create_simple_planner, create_secure_planner, create_performance_planner
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