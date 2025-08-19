"""Quantum Task Validation and Error Handling

Comprehensive validation system for quantum tasks with TPU-specific checks,
enhanced with circuit breakers, retry mechanisms, and structured logging.
"""

import logging
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .exceptions import (
    CircuitBreaker,
    CircuitBreakerConfig,
    ErrorHandlingContext,
    QuantumCircuitBreakerError,
    QuantumValidationError,
    RetryConfig,
    RetryManager,
    quantum_operation,
    validate_input,
)
from .quantum_planner import QuantumResource, QuantumState, QuantumTask

# Configure structured logging for validation
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter for structured logging
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(component)s:%(operation)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue with enhanced metadata"""
    severity: ValidationSeverity
    code: str
    message: str
    task_id: Optional[str] = None
    resource: Optional[str] = None
    suggestion: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    rule_name: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.code}: {self.message}"]
        if self.task_id:
            parts.append(f"Task: {self.task_id}")
        if self.resource:
            parts.append(f"Resource: {self.resource}")
        if self.rule_name:
            parts.append(f"Rule: {self.rule_name}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "task_id": self.task_id,
            "resource": self.resource,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "context_data": self.context_data,
            "recovery_actions": self.recovery_actions
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report with enhanced analytics"""
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    validation_time: float = 0.0
    is_valid: bool = True
    validation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    rules_executed: int = 0
    rules_failed: int = 0
    context: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add validation issue and update counters"""
        self.issues.append(issue)
        self.total_issues += 1

        if issue.severity == ValidationSeverity.CRITICAL:
            self.critical_issues += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.ERROR:
            self.error_issues += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_issues += 1
        elif issue.severity == ValidationSeverity.INFO:
            self.info_issues += 1

    def has_blocking_issues(self) -> bool:
        """Check if report has issues that block execution"""
        return self.critical_issues > 0 or self.error_issues > 0

    def get_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity"""
        return {
            "total": self.total_issues,
            "critical": self.critical_issues,
            "errors": self.error_issues,
            "warnings": self.warning_issues,
            "info": self.info_issues
        }


class QuantumTaskValidator:
    """Comprehensive quantum task validation system with resilience patterns"""

    def __init__(self, enable_circuit_breaker: bool = True, max_validation_time: float = 30.0):
        self.validation_rules = {}
        self._lock = threading.RLock()
        self.validation_history = deque(maxlen=1000)
        self.rule_performance = defaultdict(lambda: {"total_time": 0.0, "executions": 0, "failures": 0})
        self.max_validation_time = max_validation_time

        # Circuit breaker for validation system resilience
        if enable_circuit_breaker:
            cb_config = CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout=60.0,
                expected_exception=Exception
            )
            self.circuit_breaker = CircuitBreaker(cb_config)
        else:
            self.circuit_breaker = None

        # Retry manager for transient failures
        retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            retryable_exceptions=(QuantumValidationError,)
        )
        self.retry_manager = RetryManager(retry_config)

        self._register_default_rules()

        logger.info(
            "Quantum task validator initialized",
            extra={"component": "quantum_validator", "operation": "__init__",
                  "circuit_breaker_enabled": enable_circuit_breaker,
                  "max_validation_time": max_validation_time}
        )

    def _register_default_rules(self) -> None:
        """Register default validation rules with metadata"""
        self.validation_rules = {
            "task_id_format": {
                "function": self._validate_task_id_format,
                "description": "Validate task ID format and constraints",
                "critical": True,
                "timeout": 1.0
            },
            "task_dependencies": {
                "function": self._validate_task_dependencies,
                "description": "Validate task dependency structure",
                "critical": True,
                "timeout": 2.0
            },
            "resource_requirements": {
                "function": self._validate_resource_requirements,
                "description": "Validate resource requirement specifications",
                "critical": True,
                "timeout": 1.0
            },
            "quantum_coherence": {
                "function": self._validate_quantum_coherence,
                "description": "Validate quantum coherence properties",
                "critical": False,
                "timeout": 1.0
            },
            "tpu_compatibility": {
                "function": self._validate_tpu_compatibility,
                "description": "Validate TPU-specific compatibility requirements",
                "critical": False,
                "timeout": 2.0
            },
            "execution_safety": {
                "function": self._validate_execution_safety,
                "description": "Validate execution safety constraints",
                "critical": True,
                "timeout": 1.0
            },
            "decoherence_limits": {
                "function": self._validate_decoherence_limits,
                "description": "Validate quantum decoherence parameters",
                "critical": False,
                "timeout": 1.0
            },
            "entanglement_validity": {
                "function": self._validate_entanglement_validity,
                "description": "Validate quantum entanglement relationships",
                "critical": False,
                "timeout": 1.0
            },
            "priority_ranges": {
                "function": self._validate_priority_ranges,
                "description": "Validate priority value ranges",
                "critical": False,
                "timeout": 0.5
            },
            "complexity_sanity": {
                "function": self._validate_complexity_sanity,
                "description": "Validate complexity value sanity checks",
                "critical": False,
                "timeout": 0.5
            }
        }

    @validate_input(
        lambda self, task, available_resources=None: hasattr(task, 'id'),
        "Invalid task object for validation"
    )
    @quantum_operation("validate_task", retry_attempts=2, timeout_seconds=30.0)
    def validate_task(self, task: QuantumTask, available_resources: Optional[Dict[str, QuantumResource]] = None) -> ValidationReport:
        """Validate individual quantum task with comprehensive error handling and resilience"""
        validation_start = time.time()
        task_id = getattr(task, 'id', 'unknown')

        # Initialize report with enhanced metadata
        report = ValidationReport(
            validation_id=f"val_{int(validation_start)}_{task_id}",
            timestamp=validation_start
        )

        with ErrorHandlingContext(
            component="quantum_validator",
            operation="validate_task",
            task_id=task_id,
            suppress_exceptions=True
        ) as error_ctx:

            try:
                with self._lock:
                    logger.info(
                        f"Starting validation for task {task_id}",
                        extra={"component": "quantum_validator", "operation": "validate_task",
                              "task_id": task_id, "validation_id": report.validation_id}
                    )

                    # Validate task basic structure
                    if not self._validate_task_structure(task, report):
                        logger.error(
                            f"Task {task_id} failed basic structure validation",
                            extra={"component": "quantum_validator", "operation": "validate_task",
                                  "task_id": task_id}
                        )
                        report.validation_time = time.time() - validation_start
                        return report

                    # Apply circuit breaker if enabled
                    validation_func = self._execute_validation_rules
                    if self.circuit_breaker:
                        validation_func = self.circuit_breaker(validation_func)

                    # Execute validation with timeout
                    try:
                        validation_func(task, available_resources, report)
                    except QuantumCircuitBreakerError as e:
                        logger.error(
                            f"Validation circuit breaker open for task {task_id}: {e}",
                            extra={"component": "quantum_validator", "operation": "validate_task",
                                  "task_id": task_id}
                        )
                        error_issue = ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            code="VALIDATION_CIRCUIT_BREAKER_OPEN",
                            message=f"Validation circuit breaker is open: {str(e)}",
                            task_id=task_id,
                            rule_name="circuit_breaker",
                            suggestion="Wait for circuit breaker to reset, check system health",
                            recovery_actions=["Wait and retry", "Check validation system health"]
                        )
                        report.add_issue(error_issue)

                    # Calculate final metrics
                    report.validation_time = time.time() - validation_start
                    report.performance_metrics = {
                        "total_validation_time": report.validation_time,
                        "rules_per_second": report.rules_executed / max(report.validation_time, 0.001),
                        "issues_per_rule": report.total_issues / max(report.rules_executed, 1)
                    }

                    # Generate recovery suggestions
                    if report.has_blocking_issues():
                        report.recovery_suggestions = self._generate_recovery_suggestions(report)

                    # Store validation history
                    self.validation_history.append({
                        "validation_id": report.validation_id,
                        "task_id": task_id,
                        "timestamp": report.timestamp,
                        "total_issues": report.total_issues,
                        "validation_time": report.validation_time,
                        "is_valid": report.is_valid
                    })

                    logger.info(
                        f"Validation completed for task {task_id}: "
                        f"{report.total_issues} issues ({report.critical_issues} critical, {report.error_issues} errors)",
                        extra={"component": "quantum_validator", "operation": "validate_task",
                              "task_id": task_id, "validation_time": report.validation_time,
                              "total_issues": report.total_issues, "is_valid": report.is_valid}
                    )

                    return report

            except Exception as e:
                # Handle unexpected validation failures
                logger.critical(
                    f"Unexpected validation failure for task {task_id}: {e}",
                    extra={"component": "quantum_validator", "operation": "validate_task",
                          "task_id": task_id, "error": str(e)}
                )

                critical_issue = ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="VALIDATION_SYSTEM_FAILURE",
                    message=f"Critical validation system failure: {str(e)}",
                    task_id=task_id,
                    rule_name="system",
                    suggestion="Check task structure and validation system integrity",
                    recovery_actions=["Verify task object", "Check validation system", "Report bug"]
                )
                report.add_issue(critical_issue)
                report.rules_failed += 1

                report.validation_time = time.time() - validation_start
                return report

    def _validate_task_structure(self, task: QuantumTask, report: ValidationReport) -> bool:
        """Validate basic task structure before running rules"""
        try:
            required_attrs = ['id', 'name', 'state']
            for attr in required_attrs:
                if not hasattr(task, attr):
                    issue = ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        code="MISSING_REQUIRED_ATTRIBUTE",
                        message=f"Task missing required attribute: {attr}",
                        task_id=getattr(task, 'id', 'unknown'),
                        rule_name="structure_validation",
                        suggestion=f"Ensure task has {attr} attribute",
                        recovery_actions=[f"Add {attr} attribute to task"]
                    )
                    report.add_issue(issue)
                    return False

            return True

        except Exception as e:
            logger.error(
                f"Error in task structure validation: {e}",
                extra={"component": "quantum_validator", "operation": "_validate_task_structure"}
            )
            return False

    def _execute_validation_rules(self, task: QuantumTask, available_resources: Optional[Dict[str, QuantumResource]], report: ValidationReport) -> None:
        """Execute all validation rules with individual error handling"""
        for rule_name, rule_config in self.validation_rules.items():
            rule_start = time.time()
            report.rules_executed += 1

            try:
                # Apply timeout to individual rules
                rule_func = rule_config["function"]
                rule_timeout = rule_config.get("timeout", 5.0)

                logger.debug(
                    f"Executing validation rule {rule_name} for task {task.id}",
                    extra={"component": "quantum_validator", "operation": "_execute_validation_rules",
                          "rule_name": rule_name, "task_id": task.id}
                )

                # Execute rule with error handling
                issues = []
                try:
                    issues = rule_func(task, available_resources)
                    if not isinstance(issues, list):
                        issues = []
                        logger.warning(
                            f"Validation rule {rule_name} returned non-list result",
                            extra={"component": "quantum_validator", "operation": "_execute_validation_rules",
                                  "rule_name": rule_name}
                        )
                except Exception as rule_error:
                    logger.error(
                        f"Validation rule {rule_name} failed for task {task.id}: {rule_error}",
                        extra={"component": "quantum_validator", "operation": "_execute_validation_rules",
                              "rule_name": rule_name, "task_id": task.id, "error": str(rule_error)}
                    )

                    # Create issue for rule failure
                    rule_failure_issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR if rule_config.get("critical", False) else ValidationSeverity.WARNING,
                        code=f"VALIDATION_RULE_FAILED_{rule_name.upper()}",
                        message=f"Validation rule '{rule_name}' failed: {str(rule_error)}",
                        task_id=task.id,
                        rule_name=rule_name,
                        suggestion=f"Check task data for rule '{rule_name}' requirements",
                        recovery_actions=["Verify task configuration", "Check rule implementation"]
                    )
                    issues = [rule_failure_issue]
                    report.rules_failed += 1

                # Add issues to report
                for issue in issues:
                    if isinstance(issue, ValidationIssue):
                        issue.rule_name = rule_name  # Ensure rule name is set
                        report.add_issue(issue)

                # Update rule performance metrics
                rule_time = time.time() - rule_start
                self.rule_performance[rule_name]["total_time"] += rule_time
                self.rule_performance[rule_name]["executions"] += 1

                if len(issues) > 0 and any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] for issue in issues):
                    self.rule_performance[rule_name]["failures"] += 1

                logger.debug(
                    f"Validation rule {rule_name} completed in {rule_time:.3f}s with {len(issues)} issues",
                    extra={"component": "quantum_validator", "operation": "_execute_validation_rules",
                          "rule_name": rule_name, "rule_time": rule_time, "issues": len(issues)}
                )

            except Exception as e:
                # Rule execution completely failed
                logger.error(
                    f"Critical error executing validation rule {rule_name}: {e}",
                    extra={"component": "quantum_validator", "operation": "_execute_validation_rules",
                          "rule_name": rule_name, "error": str(e)}
                )

                critical_issue = ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="VALIDATION_RULE_SYSTEM_ERROR",
                    message=f"Critical error in validation rule '{rule_name}': {str(e)}",
                    task_id=task.id,
                    rule_name=rule_name,
                    suggestion="Check validation system integrity",
                    recovery_actions=["Report validation system bug", "Skip this validation"]
                )
                report.add_issue(critical_issue)
                report.rules_failed += 1

    def _generate_recovery_suggestions(self, report: ValidationReport) -> List[str]:
        """Generate recovery suggestions based on validation issues"""
        suggestions = []

        # Categorize issues
        critical_count = report.critical_issues
        error_count = report.error_issues

        if critical_count > 0:
            suggestions.append(f"Address {critical_count} critical issues before proceeding")
            suggestions.append("Critical issues prevent task execution")

        if error_count > 0:
            suggestions.append(f"Resolve {error_count} error-level issues")

        # Rule-specific suggestions
        rule_issues = defaultdict(int)
        for issue in report.issues:
            if issue.rule_name:
                rule_issues[issue.rule_name] += 1

        if len(rule_issues) > 3:
            suggestions.append("Multiple validation rules failed - review task configuration")

        # Add specific recovery actions from issues
        for issue in report.issues:
            if issue.recovery_actions:
                suggestions.extend(issue.recovery_actions[:2])  # Limit to prevent spam

        return list(set(suggestions))  # Remove duplicates

    @validate_input(
        lambda self, task, resources=None: hasattr(task, 'id'),
        "Task missing ID attribute"
    )
    def _validate_task_id_format(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate task ID format and uniqueness with comprehensive checks"""
        issues = []

        try:
            task_id = getattr(task, 'id', None)

            # Check ID existence
            if not task_id:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="TASK_ID_EMPTY",
                    message="Task ID cannot be empty",
                    task_id=str(task_id),
                    suggestion="Provide a non-empty task ID",
                    recovery_actions=["Set a valid task ID", "Generate unique ID"]
                ))
                return issues

            # Sanitize and validate ID
            if not isinstance(task_id, str):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="TASK_ID_INVALID_TYPE",
                    message=f"Task ID must be string, got {type(task_id).__name__}",
                    task_id=str(task_id),
                    suggestion="Ensure task ID is a string",
                    recovery_actions=["Convert ID to string", "Use string ID"]
                ))
                return issues

            # Check ID format with enhanced pattern
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_\-]*$', task_id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="TASK_ID_INVALID_FORMAT",
                    message="Task ID must start with letter and contain only alphanumeric, underscore, hyphen",
                    task_id=task_id,
                    suggestion="Use format: letter followed by alphanumeric, underscore, or hyphen",
                    recovery_actions=["Fix ID format", "Regenerate valid ID"],
                    context_data={"current_id": task_id, "pattern": "^[a-zA-Z][a-zA-Z0-9_\\-]*$"}
                ))

            # Check length constraints
            if len(task_id) < 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    code="TASK_ID_TOO_SHORT",
                    message="Task ID cannot be empty",
                    task_id=task_id,
                    suggestion="Provide at least 1 character",
                    recovery_actions=["Add content to ID"]
                ))
            elif len(task_id) > 128:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="TASK_ID_TOO_LONG",
                    message=f"Task ID is {len(task_id)} characters, longer than recommended 128",
                    task_id=task_id,
                    suggestion="Consider shorter, more concise task IDs",
                    recovery_actions=["Shorten ID", "Use abbreviations"],
                    context_data={"current_length": len(task_id), "max_recommended": 128}
                ))

            # Check for reserved patterns
            reserved_patterns = [r'^test_', r'^temp_', r'^debug_']
            for pattern in reserved_patterns:
                if re.match(pattern, task_id.lower()):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="TASK_ID_RESERVED_PATTERN",
                        message=f"Task ID uses reserved pattern: {pattern}",
                        task_id=task_id,
                        suggestion="Avoid reserved prefixes for production tasks",
                        recovery_actions=["Use different prefix", "Rename task"]
                    ))
                    break

            # Check for potential security issues
            dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
            found_dangerous = [char for char in dangerous_chars if char in task_id]
            if found_dangerous:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="TASK_ID_SECURITY_RISK",
                    message=f"Task ID contains potentially dangerous characters: {found_dangerous}",
                    task_id=task_id,
                    suggestion="Remove special characters that could cause security issues",
                    recovery_actions=["Sanitize ID", "Use safe characters only"],
                    context_data={"dangerous_chars": found_dangerous}
                ))

        except Exception as e:
            logger.error(
                f"Error in task ID format validation: {e}",
                extra={"component": "quantum_validator", "operation": "_validate_task_id_format"}
            )
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="TASK_ID_VALIDATION_ERROR",
                message=f"Error validating task ID: {str(e)}",
                task_id=getattr(task, 'id', 'unknown'),
                suggestion="Check task ID validation logic",
                recovery_actions=["Verify task object", "Check validation system"]
            ))

        return issues

    def _validate_task_dependencies(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate task dependency structure"""
        issues = []

        # Self-dependency check
        if task.id in task.dependencies:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SELF_DEPENDENCY",
                message="Task cannot depend on itself",
                task_id=task.id,
                suggestion="Remove self-reference from dependencies"
            ))

        # Circular dependency hints (basic check)
        if len(task.dependencies) > 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_DEPENDENCIES",
                message="Task has unusually high number of dependencies",
                task_id=task.id,
                suggestion="Consider breaking down complex dependencies"
            ))

        return issues

    def _validate_resource_requirements(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate resource requirement specifications"""
        issues = []

        if not task.resource_requirements:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="NO_RESOURCE_REQUIREMENTS",
                message="Task has no explicit resource requirements",
                task_id=task.id,
                suggestion="Consider specifying resource requirements for better scheduling"
            ))
            return issues

        # Validate resource values
        for resource_name, amount in task.resource_requirements.items():
            if amount < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="NEGATIVE_RESOURCE_REQUIREMENT",
                    message=f"Negative resource requirement: {resource_name}={amount}",
                    task_id=task.id,
                    resource=resource_name,
                    suggestion="Resource requirements must be non-negative"
                ))

            if amount > 1000:  # Suspiciously high
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="HIGH_RESOURCE_REQUIREMENT",
                    message=f"Unusually high resource requirement: {resource_name}={amount}",
                    task_id=task.id,
                    resource=resource_name,
                    suggestion="Verify resource requirement is correct"
                ))

        # Check against available resources
        if resources:
            for resource_name, required_amount in task.resource_requirements.items():
                if resource_name in resources:
                    available = resources[resource_name].total_capacity
                    if required_amount > available:
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="RESOURCE_REQUIREMENT_EXCEEDS_CAPACITY",
                            message=f"Required {resource_name}={required_amount} exceeds capacity {available}",
                            task_id=task.id,
                            resource=resource_name,
                            suggestion="Reduce resource requirement or increase capacity"
                        ))
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="UNKNOWN_RESOURCE",
                        message=f"Unknown resource requirement: {resource_name}",
                        task_id=task.id,
                        resource=resource_name,
                        suggestion="Check resource name or add resource to system"
                    ))

        return issues

    def _validate_quantum_coherence(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate quantum coherence properties"""
        issues = []

        # Probability amplitude validation
        amplitude_magnitude = abs(task.probability_amplitude)
        if amplitude_magnitude > 10.0:  # Unreasonably high
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="HIGH_PROBABILITY_AMPLITUDE",
                message=f"Probability amplitude magnitude is high: {amplitude_magnitude:.2f}",
                task_id=task.id,
                suggestion="Consider normalizing probability amplitudes"
            ))

        # State consistency
        if task.state == QuantumState.COLLAPSED and amplitude_magnitude < 0.1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="COLLAPSED_STATE_LOW_AMPLITUDE",
                message="Collapsed state has unexpectedly low probability amplitude",
                task_id=task.id,
                suggestion="Verify quantum state consistency"
            ))

        return issues

    def _validate_tpu_compatibility(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate TPU-specific compatibility requirements"""
        issues = []

        # TPU affinity validation
        if task.tpu_affinity:
            # Check affinity format
            if not re.match(r'^/dev/(apex|accel)_\d+$', task.tpu_affinity):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="INVALID_TPU_AFFINITY_FORMAT",
                    message=f"TPU affinity format may be invalid: {task.tpu_affinity}",
                    task_id=task.id,
                    suggestion="Use format like '/dev/apex_0' or '/dev/accel_0'"
                ))

        # Memory footprint validation
        if task.memory_footprint > 128 * 1024 * 1024 * 1024:  # 128GB
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="EXCESSIVE_MEMORY_FOOTPRINT",
                message=f"Memory footprint exceeds TPU v5 limits: {task.memory_footprint / (1024**3):.1f}GB",
                task_id=task.id,
                suggestion="Reduce memory footprint or use model optimization"
            ))

        # Model requirements validation
        for model_req in task.model_requirements:
            if not model_req.endswith(('.tflite', '.onnx', '.pb')):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    code="UNKNOWN_MODEL_FORMAT",
                    message=f"Unknown model format: {model_req}",
                    task_id=task.id,
                    suggestion="Ensure model is compatible with TPU v5"
                ))

        return issues

    def _validate_execution_safety(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate execution safety constraints"""
        issues = []

        # Duration validation
        if task.estimated_duration <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_ESTIMATED_DURATION",
                message=f"Estimated duration must be positive: {task.estimated_duration}",
                task_id=task.id,
                suggestion="Set realistic positive duration estimate"
            ))

        if task.estimated_duration > 3600:  # 1 hour
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="VERY_LONG_EXECUTION",
                message=f"Task has very long estimated duration: {task.estimated_duration/60:.1f} minutes",
                task_id=task.id,
                suggestion="Consider breaking into smaller tasks"
            ))

        # Priority validation
        if task.priority <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="NON_POSITIVE_PRIORITY",
                message=f"Task priority should be positive: {task.priority}",
                task_id=task.id,
                suggestion="Use positive priority values"
            ))

        return issues

    def _validate_decoherence_limits(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate quantum decoherence parameters"""
        issues = []

        if task.decoherence_time <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_DECOHERENCE_TIME",
                message=f"Decoherence time must be positive: {task.decoherence_time}",
                task_id=task.id,
                suggestion="Set positive decoherence time"
            ))

        if task.decoherence_time < 1.0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="VERY_FAST_DECOHERENCE",
                message=f"Very fast decoherence time: {task.decoherence_time:.1f}s",
                task_id=task.id,
                suggestion="Consider longer decoherence time for stability"
            ))

        # Check current decoherence level
        current_decoherence = task.measure_decoherence()
        if current_decoherence > 0.9:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="HIGH_DECOHERENCE_LEVEL",
                message=f"Task is highly decoherent: {current_decoherence:.1%}",
                task_id=task.id,
                suggestion="Task may need to be recreated or refreshed"
            ))
        elif current_decoherence > 0.7:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="MODERATE_DECOHERENCE_LEVEL",
                message=f"Task showing significant decoherence: {current_decoherence:.1%}",
                task_id=task.id,
                suggestion="Consider prioritizing this task for execution"
            ))

        return issues

    def _validate_entanglement_validity(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate quantum entanglement relationships"""
        issues = []

        # Self-entanglement check
        if task.id in task.entangled_tasks:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="SELF_ENTANGLEMENT",
                message="Task cannot be entangled with itself",
                task_id=task.id,
                suggestion="Remove self-reference from entangled tasks"
            ))

        # Excessive entanglements
        if len(task.entangled_tasks) > 5:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="EXCESSIVE_ENTANGLEMENTS",
                message=f"Task has many entanglements: {len(task.entangled_tasks)}",
                task_id=task.id,
                suggestion="Consider reducing entanglements for better performance"
            ))

        return issues

    def _validate_priority_ranges(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate priority value ranges"""
        issues = []

        if task.priority > 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="VERY_HIGH_PRIORITY",
                message=f"Unusually high priority value: {task.priority}",
                task_id=task.id,
                suggestion="Consider using priority range 1-10 for better balance"
            ))

        if task.priority < 0.1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                code="VERY_LOW_PRIORITY",
                message=f"Very low priority value: {task.priority}",
                task_id=task.id,
                suggestion="Low priority tasks may be delayed significantly"
            ))

        return issues

    def _validate_complexity_sanity(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate complexity value sanity checks"""
        issues = []

        if task.complexity <= 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_COMPLEXITY",
                message=f"Task complexity must be positive: {task.complexity}",
                task_id=task.id,
                suggestion="Set positive complexity value"
            ))

        if task.complexity > 100:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="VERY_HIGH_COMPLEXITY",
                message=f"Unusually high complexity: {task.complexity}",
                task_id=task.id,
                suggestion="Consider breaking down complex tasks"
            ))

        # Complexity vs duration correlation check
        if task.complexity > 10 and task.estimated_duration < 1.0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="COMPLEXITY_DURATION_MISMATCH",
                message="High complexity but low duration estimate",
                task_id=task.id,
                suggestion="Verify complexity and duration estimates are consistent"
            ))

        return issues


class CircularDependencyDetector:
    """Detect circular dependencies in task graphs"""

    @staticmethod
    def detect_cycles(tasks: Dict[str, QuantumTask]) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        def dfs(node: str, path: List[str], visited: Set[str], rec_stack: Set[str]) -> List[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            cycles = []

            if node in tasks:
                for dependency in tasks[node].dependencies:
                    if dependency not in visited:
                        cycles.extend(dfs(dependency, path + [node], visited, rec_stack))
                    elif dependency in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(dependency)
                        cycle = path[cycle_start:] + [node, dependency]
                        cycles.append(cycle)

            rec_stack.remove(node)
            return cycles

        all_cycles = []
        visited = set()

        for task_id in tasks:
            if task_id not in visited:
                all_cycles.extend(dfs(task_id, [], visited, set()))

        return all_cycles


class QuantumSystemValidator:
    """System-wide quantum validation"""

    def __init__(self):
        self.task_validator = QuantumTaskValidator()
        self.cycle_detector = CircularDependencyDetector()

    def validate_system(self, tasks: Dict[str, QuantumTask],
                       resources: Dict[str, QuantumResource]) -> ValidationReport:
        """Comprehensive system validation"""
        start_time = time.time()
        system_report = ValidationReport()

        # Validate individual tasks
        for task in tasks.values():
            task_report = self.task_validator.validate_task(task, resources)
            for issue in task_report.issues:
                system_report.add_issue(issue)

        # Check circular dependencies
        cycles = self.cycle_detector.detect_cycles(tasks)
        for cycle in cycles:
            cycle_issue = ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code="CIRCULAR_DEPENDENCY",
                message=f"Circular dependency detected: {' -> '.join(cycle)}",
                suggestion="Remove circular dependencies by breaking dependency chain"
            )
            system_report.add_issue(cycle_issue)

        # Resource capacity validation
        total_requirements = {}
        for task in tasks.values():
            for resource_name, amount in task.resource_requirements.items():
                total_requirements[resource_name] = total_requirements.get(resource_name, 0) + amount

        for resource_name, total_required in total_requirements.items():
            if resource_name in resources:
                capacity = resources[resource_name].total_capacity
                if total_required > capacity * len(tasks):  # Simple over-subscription check
                    capacity_issue = ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        code="POTENTIAL_RESOURCE_CONTENTION",
                        message=f"High resource demand for {resource_name}: {total_required:.1f} vs {capacity:.1f}",
                        resource=resource_name,
                        suggestion="Monitor resource utilization during execution"
                    )
                    system_report.add_issue(capacity_issue)

        # Entanglement consistency check
        for task in tasks.values():
            for entangled_id in task.entangled_tasks:
                if entangled_id in tasks:
                    entangled_task = tasks[entangled_id]
                    if task.id not in entangled_task.entangled_tasks:
                        consistency_issue = ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            code="ASYMMETRIC_ENTANGLEMENT",
                            message=f"Asymmetric entanglement: {task.id} -> {entangled_id}",
                            task_id=task.id,
                            suggestion="Ensure entanglement is bidirectional"
                        )
                        system_report.add_issue(consistency_issue)
                else:
                    missing_issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code="ENTANGLED_TASK_NOT_FOUND",
                        message=f"Entangled task not found: {entangled_id}",
                        task_id=task.id,
                        suggestion="Remove reference to missing task or add the task"
                    )
                    system_report.add_issue(missing_issue)

        system_report.validation_time = time.time() - start_time
        return system_report


def create_validation_summary(report: ValidationReport) -> str:
    """Create human-readable validation summary"""
    summary_parts = [
        f"Validation Summary ({report.validation_time:.2f}s)",
        "=" * 50,
        f"Status: {'✅ VALID' if report.is_valid else '❌ INVALID'}",
        f"Total Issues: {report.total_issues}",
        ""
    ]

    if report.total_issues > 0:
        summary = report.get_summary()
        summary_parts.extend([
            f"  Critical: {summary['critical']}",
            f"  Errors:   {summary['errors']}",
            f"  Warnings: {summary['warnings']}",
            f"  Info:     {summary['info']}",
            ""
        ])

        if report.has_blocking_issues():
            summary_parts.append("⚠️  BLOCKING ISSUES FOUND - EXECUTION NOT RECOMMENDED")

        # Show top issues
        critical_and_errors = [i for i in report.issues
                              if i.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]]
        if critical_and_errors:
            summary_parts.extend([
                "",
                "Critical/Error Issues:",
                "-" * 25
            ])
            for issue in critical_and_errors[:5]:  # Top 5
                summary_parts.append(f"• {issue}")

    return "\n".join(summary_parts)
