"""Quantum Task Validation and Error Handling

Comprehensive validation system for quantum tasks with TPU-specific checks.
"""

import logging
import traceback
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import time

from .quantum_planner import QuantumTask, QuantumResource, QuantumState
from .exceptions import BenchmarkError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue"""
    severity: ValidationSeverity
    code: str
    message: str
    task_id: Optional[str] = None
    resource: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.code}: {self.message}"]
        if self.task_id:
            parts.append(f"Task: {self.task_id}")
        if self.resource:
            parts.append(f"Resource: {self.resource}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


@dataclass 
class ValidationReport:
    """Comprehensive validation report"""
    total_issues: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    issues: List[ValidationIssue] = None
    validation_time: float = 0.0
    is_valid: bool = True
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
    
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
    """Comprehensive quantum task validation system"""
    
    def __init__(self):
        self.validation_rules = {}
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default validation rules"""
        self.validation_rules = {
            "task_id_format": self._validate_task_id_format,
            "task_dependencies": self._validate_task_dependencies,
            "resource_requirements": self._validate_resource_requirements,
            "quantum_coherence": self._validate_quantum_coherence,
            "tpu_compatibility": self._validate_tpu_compatibility,
            "execution_safety": self._validate_execution_safety,
            "decoherence_limits": self._validate_decoherence_limits,
            "entanglement_validity": self._validate_entanglement_validity,
            "priority_ranges": self._validate_priority_ranges,
            "complexity_sanity": self._validate_complexity_sanity
        }
    
    def validate_task(self, task: QuantumTask, available_resources: Optional[Dict[str, QuantumResource]] = None) -> ValidationReport:
        """Validate individual quantum task"""
        start_time = time.time()
        report = ValidationReport()
        
        try:
            # Run all validation rules
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    issues = rule_func(task, available_resources)
                    for issue in issues:
                        report.add_issue(issue)
                except Exception as e:
                    # Validation rule failure
                    error_issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        code=f"VALIDATION_RULE_FAILED_{rule_name.upper()}",
                        message=f"Validation rule '{rule_name}' failed: {str(e)}",
                        task_id=task.id,
                        suggestion="Check validation rule implementation"
                    )
                    report.add_issue(error_issue)
                    logger.error(f"Validation rule {rule_name} failed for task {task.id}: {e}")
        
        except Exception as e:
            # Critical validation failure
            critical_issue = ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code="VALIDATION_SYSTEM_FAILURE",
                message=f"Critical validation system failure: {str(e)}",
                task_id=task.id,
                suggestion="Check task structure and validation system"
            )
            report.add_issue(critical_issue)
            logger.critical(f"Critical validation failure for task {task.id}: {e}")
        
        report.validation_time = time.time() - start_time
        return report
    
    def _validate_task_id_format(self, task: QuantumTask, resources: Optional[Dict]) -> List[ValidationIssue]:
        """Validate task ID format and uniqueness"""
        issues = []
        
        # Check ID format
        if not task.id:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code="TASK_ID_EMPTY",
                message="Task ID cannot be empty",
                task_id=task.id,
                suggestion="Provide a non-empty task ID"
            ))
        elif not re.match(r'^[a-zA-Z0-9_\-]+$', task.id):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="TASK_ID_INVALID_FORMAT",
                message="Task ID contains invalid characters",
                task_id=task.id,
                suggestion="Use only alphanumeric characters, underscores, and hyphens"
            ))
        
        # Check length
        if len(task.id) > 128:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="TASK_ID_TOO_LONG",
                message="Task ID is longer than recommended 128 characters",
                task_id=task.id,
                suggestion="Consider shorter, more concise task IDs"
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