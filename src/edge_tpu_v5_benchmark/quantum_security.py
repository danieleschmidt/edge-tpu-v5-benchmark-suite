"""Quantum Task Security and Compliance

Security framework for quantum task execution with TPU-specific security measures.
"""

import hashlib
import hmac
import time
import secrets
import logging
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re
import json

from .quantum_planner import QuantumTask, QuantumTaskPlanner
from .exceptions import BenchmarkError

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for task execution"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    allowed_operations: Set[str] = field(default_factory=set)
    blocked_operations: Set[str] = field(default_factory=set)
    max_execution_time: float = 3600.0  # 1 hour
    max_memory_usage: float = 128 * 1024**3  # 128GB
    allowed_file_patterns: List[str] = field(default_factory=list)
    blocked_file_patterns: List[str] = field(default_factory=list)
    require_signature: bool = False
    audit_enabled: bool = True
    sandboxed_execution: bool = True
    
    def __post_init__(self):
        # Default allowed operations
        if not self.allowed_operations:
            self.allowed_operations = {
                "inference", "benchmark", "validation", "monitoring"
            }
        
        # Default blocked operations
        if not self.blocked_operations:
            self.blocked_operations = {
                "file_write", "network_access", "system_call"
            }
        
        # Default file patterns
        if not self.allowed_file_patterns:
            self.allowed_file_patterns = [
                r".*\.tflite$", r".*\.onnx$", r".*\.pb$", r".*\.json$"
            ]
        
        if not self.blocked_file_patterns:
            self.blocked_file_patterns = [
                r"/etc/.*", r"/proc/.*", r"/sys/.*", r".*\.sh$", r".*\.exe$"
            ]


@dataclass
class SecurityAuditEntry:
    """Security audit log entry"""
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    task_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: str = ""
    resource: str = ""
    outcome: str = ""  # success, failure, blocked
    threat_level: ThreatLevel = ThreatLevel.LOW
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'task_id': self.task_id,
            'user_id': self.user_id,
            'operation': self.operation,
            'resource': self.resource,
            'outcome': self.outcome,
            'threat_level': self.threat_level.value,
            'details': self.details
        }


class TaskSigner:
    """Digital signature system for quantum tasks"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
    
    def sign_task(self, task: QuantumTask) -> str:
        """Create digital signature for task"""
        # Create canonical representation
        task_data = {
            'id': task.id,
            'name': task.name,
            'priority': task.priority,
            'complexity': task.complexity,
            'dependencies': sorted(list(task.dependencies)),
            'resource_requirements': dict(sorted(task.resource_requirements.items())),
            'estimated_duration': task.estimated_duration
        }
        
        canonical_str = json.dumps(task_data, sort_keys=True, separators=(',', ':'))
        signature = hmac.new(
            self.secret_key,
            canonical_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_task_signature(self, task: QuantumTask, signature: str) -> bool:
        """Verify task signature"""
        expected_signature = self.sign_task(task)
        return hmac.compare_digest(signature, expected_signature)
    
    def get_task_hash(self, task: QuantumTask) -> str:
        """Get secure hash of task for integrity checking"""
        task_str = f"{task.id}:{task.name}:{task.priority}:{time.time()}"
        return hashlib.sha256(task_str.encode('utf-8')).hexdigest()


class SecurityValidator:
    """Validates tasks against security policies"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.compiled_patterns = {
            'allowed_files': [re.compile(pattern) for pattern in policy.allowed_file_patterns],
            'blocked_files': [re.compile(pattern) for pattern in policy.blocked_file_patterns]
        }
    
    def validate_task_security(self, task: QuantumTask) -> List[str]:
        """Validate task against security policy"""
        violations = []
        
        # Check execution time limits
        if task.estimated_duration > self.policy.max_execution_time:
            violations.append(
                f"Task execution time {task.estimated_duration}s exceeds limit {self.policy.max_execution_time}s"
            )
        
        # Check memory usage limits
        memory_requirement = task.resource_requirements.get('memory_gb', 0) * 1024**3
        if memory_requirement > self.policy.max_memory_usage:
            violations.append(
                f"Memory requirement {memory_requirement/1024**3:.1f}GB exceeds limit {self.policy.max_memory_usage/1024**3:.1f}GB"
            )
        
        # Check model file patterns
        for model_file in task.model_requirements:
            if not self._is_file_allowed(model_file):
                violations.append(f"Model file not allowed: {model_file}")
        
        # Check for suspicious task properties
        if task.complexity > 100:
            violations.append("Suspiciously high task complexity")
        
        if task.priority > 1000:
            violations.append("Suspiciously high task priority")
        
        if len(task.dependencies) > 50:
            violations.append("Excessive number of dependencies")
        
        return violations
    
    def _is_file_allowed(self, file_path: str) -> bool:
        """Check if file path is allowed by security policy"""
        # First check blocked patterns
        for pattern in self.compiled_patterns['blocked_files']:
            if pattern.match(file_path):
                return False
        
        # Then check allowed patterns
        for pattern in self.compiled_patterns['allowed_files']:
            if pattern.match(file_path):
                return True
        
        # Default deny
        return False
    
    def assess_threat_level(self, task: QuantumTask) -> ThreatLevel:
        """Assess threat level of task"""
        risk_score = 0
        
        # High complexity increases risk
        if task.complexity > 50:
            risk_score += 2
        elif task.complexity > 20:
            risk_score += 1
        
        # Long execution time increases risk
        if task.estimated_duration > 1800:  # 30 minutes
            risk_score += 2
        elif task.estimated_duration > 300:  # 5 minutes
            risk_score += 1
        
        # High resource usage increases risk
        memory_gb = task.resource_requirements.get('memory_gb', 0)
        if memory_gb > 64:
            risk_score += 2
        elif memory_gb > 16:
            risk_score += 1
        
        # Many dependencies increase risk
        if len(task.dependencies) > 20:
            risk_score += 2
        elif len(task.dependencies) > 10:
            risk_score += 1
        
        # TPU usage increases risk (more privileged access)
        if 'tpu_v5_primary' in task.resource_requirements:
            risk_score += 1
        
        # Convert to threat level
        if risk_score >= 6:
            return ThreatLevel.CRITICAL
        elif risk_score >= 4:
            return ThreatLevel.HIGH
        elif risk_score >= 2:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class SecurityAuditor:
    """Audit and logging system for security events"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.audit_entries: List[SecurityAuditEntry] = []
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup security audit logging"""
        self.audit_logger = logging.getLogger('quantum_security_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        if self.log_file:
            handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.audit_logger.addHandler(handler)
    
    def log_event(self, entry: SecurityAuditEntry) -> None:
        """Log security audit event"""
        self.audit_entries.append(entry)
        
        # Log to file if configured
        if hasattr(self, 'audit_logger'):
            log_message = (
                f"Event: {entry.event_type} | Task: {entry.task_id} | "
                f"Operation: {entry.operation} | Outcome: {entry.outcome} | "
                f"Threat: {entry.threat_level.value}"
            )
            
            if entry.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                self.audit_logger.warning(log_message)
            else:
                self.audit_logger.info(log_message)
    
    def log_task_execution(self, task_id: str, operation: str, outcome: str, 
                          details: Optional[Dict] = None) -> None:
        """Log task execution event"""
        entry = SecurityAuditEntry(
            event_type="task_execution",
            task_id=task_id,
            operation=operation,
            outcome=outcome,
            details=details or {}
        )
        self.log_event(entry)
    
    def log_security_violation(self, task_id: str, violation: str, 
                              threat_level: ThreatLevel = ThreatLevel.MEDIUM) -> None:
        """Log security violation"""
        entry = SecurityAuditEntry(
            event_type="security_violation",
            task_id=task_id,
            operation="security_check",
            outcome="blocked",
            threat_level=threat_level,
            details={'violation': violation}
        )
        self.log_event(entry)
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_entries = [e for e in self.audit_entries if e.timestamp >= cutoff_time]
        
        # Count by event type
        event_counts = {}
        outcome_counts = {}
        threat_counts = {}
        
        for entry in recent_entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
            outcome_counts[entry.outcome] = outcome_counts.get(entry.outcome, 0) + 1
            threat_counts[entry.threat_level.value] = threat_counts.get(entry.threat_level.value, 0) + 1
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_entries),
            'event_types': event_counts,
            'outcomes': outcome_counts,
            'threat_levels': threat_counts,
            'high_risk_events': [
                entry.to_dict() for entry in recent_entries
                if entry.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
            ]
        }
    
    def export_audit_log(self, filename: str, hours: Optional[int] = None) -> None:
        """Export audit log to file"""
        if hours:
            cutoff_time = time.time() - (hours * 3600)
            entries = [e for e in self.audit_entries if e.timestamp >= cutoff_time]
        else:
            entries = self.audit_entries
        
        export_data = {
            'export_timestamp': time.time(),
            'total_entries': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Audit log exported to {filename} ({len(entries)} entries)")


class QuantumSecurityManager:
    """Main security management system for quantum tasks"""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None, 
                 audit_log_file: Optional[str] = None):
        self.policy = policy or SecurityPolicy()
        self.validator = SecurityValidator(self.policy)
        self.auditor = SecurityAuditor(audit_log_file)
        self.signer = TaskSigner()
        self.task_signatures: Dict[str, str] = {}
        self.security_contexts: Dict[str, SecurityLevel] = {}
    
    def set_task_security_level(self, task_id: str, level: SecurityLevel) -> None:
        """Set security level for specific task"""
        self.security_contexts[task_id] = level
        logger.info(f"Set security level for task {task_id}: {level.value}")
    
    def validate_task_for_execution(self, task: QuantumTask) -> bool:
        """Comprehensive task validation for execution"""
        violations = self.validator.validate_task_security(task)
        threat_level = self.validator.assess_threat_level(task)
        
        # Log validation attempt
        self.auditor.log_event(SecurityAuditEntry(
            event_type="task_validation",
            task_id=task.id,
            operation="security_validation",
            outcome="success" if not violations else "blocked",
            threat_level=threat_level,
            details={
                'violations': violations,
                'security_level': self.security_contexts.get(task.id, SecurityLevel.PUBLIC).value
            }
        ))
        
        # Block if violations found
        if violations:
            for violation in violations:
                self.auditor.log_security_violation(task.id, violation, threat_level)
            return False
        
        # Additional checks for high-risk tasks
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Require signature for high-risk tasks
            if self.policy.require_signature:
                if task.id not in self.task_signatures:
                    self.auditor.log_security_violation(
                        task.id, "High-risk task requires signature", threat_level
                    )
                    return False
                
                # Verify signature
                if not self.signer.verify_task_signature(task, self.task_signatures[task.id]):
                    self.auditor.log_security_violation(
                        task.id, "Invalid task signature", ThreatLevel.CRITICAL
                    )
                    return False
        
        return True
    
    def sign_task(self, task: QuantumTask) -> str:
        """Sign task and store signature"""
        signature = self.signer.sign_task(task)
        self.task_signatures[task.id] = signature
        
        self.auditor.log_event(SecurityAuditEntry(
            event_type="task_signing",
            task_id=task.id,
            operation="sign_task",
            outcome="success"
        ))
        
        return signature
    
    def create_secure_task_context(self, task: QuantumTask) -> Dict[str, Any]:
        """Create secure execution context for task"""
        context = {
            'task_id': task.id,
            'security_level': self.security_contexts.get(task.id, SecurityLevel.PUBLIC).value,
            'threat_level': self.validator.assess_threat_level(task).value,
            'execution_limits': {
                'max_time': min(task.estimated_duration * 2, self.policy.max_execution_time),
                'max_memory': self.policy.max_memory_usage
            },
            'allowed_operations': list(self.policy.allowed_operations),
            'blocked_operations': list(self.policy.blocked_operations),
            'sandboxed': self.policy.sandboxed_execution
        }
        
        # Add signature if available
        if task.id in self.task_signatures:
            context['signature'] = self.task_signatures[task.id]
        
        return context
    
    def monitor_task_execution(self, task_id: str, execution_data: Dict[str, Any]) -> None:
        """Monitor task execution for security violations"""
        # Check for suspicious activity
        suspicious_indicators = []
        
        # Long execution time
        actual_duration = execution_data.get('duration', 0)
        if actual_duration > self.policy.max_execution_time:
            suspicious_indicators.append(f"Execution time exceeded limit: {actual_duration}s")
        
        # High memory usage
        memory_usage = execution_data.get('memory_usage', 0)
        if memory_usage > self.policy.max_memory_usage:
            suspicious_indicators.append(f"Memory usage exceeded limit: {memory_usage/1024**3:.1f}GB")
        
        # Unexpected errors
        if execution_data.get('errors'):
            suspicious_indicators.extend(execution_data['errors'])
        
        # Log monitoring results
        threat_level = ThreatLevel.HIGH if suspicious_indicators else ThreatLevel.LOW
        
        self.auditor.log_event(SecurityAuditEntry(
            event_type="execution_monitoring",
            task_id=task_id,
            operation="monitor_execution",
            outcome="suspicious" if suspicious_indicators else "normal",
            threat_level=threat_level,
            details={
                'execution_data': execution_data,
                'suspicious_indicators': suspicious_indicators
            }
        ))
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status"""
        recent_summary = self.auditor.get_audit_summary(hours=24)
        
        return {
            'policy_active': True,
            'audit_enabled': self.policy.audit_enabled,
            'sandboxing_enabled': self.policy.sandboxed_execution,
            'signature_required': self.policy.require_signature,
            'signed_tasks': len(self.task_signatures),
            'security_contexts': len(self.security_contexts),
            'recent_audit_summary': recent_summary,
            'threat_assessment': {
                'high_risk_tasks': len([
                    task_id for task_id, level in self.security_contexts.items()
                    if level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.RESTRICTED]
                ]),
                'total_managed_tasks': len(self.security_contexts)
            }
        }
    
    def create_security_report(self, filename: str) -> None:
        """Create comprehensive security report"""
        report_data = {
            'timestamp': time.time(),
            'security_status': self.get_security_status(),
            'policy_configuration': {
                'allowed_operations': list(self.policy.allowed_operations),
                'blocked_operations': list(self.policy.blocked_operations),
                'max_execution_time': self.policy.max_execution_time,
                'max_memory_usage': self.policy.max_memory_usage,
                'allowed_file_patterns': self.policy.allowed_file_patterns,
                'blocked_file_patterns': self.policy.blocked_file_patterns
            },
            'audit_summary': self.auditor.get_audit_summary(hours=168),  # 1 week
            'security_recommendations': self._generate_security_recommendations()
        }
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Security report created: {filename}")
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on audit data"""
        recommendations = []
        
        summary = self.auditor.get_audit_summary(hours=168)  # 1 week
        
        # Check for high violation rates
        total_events = summary.get('total_events', 0)
        blocked_events = summary.get('outcomes', {}).get('blocked', 0)
        
        if total_events > 0:
            violation_rate = blocked_events / total_events
            if violation_rate > 0.1:  # More than 10% violations
                recommendations.append(
                    f"High security violation rate ({violation_rate:.1%}). Consider reviewing security policies."
                )
        
        # Check for high-risk events
        high_risk_count = len(summary.get('high_risk_events', []))
        if high_risk_count > 0:
            recommendations.append(
                f"{high_risk_count} high-risk events detected. Review high-risk task patterns."
            )
        
        # Check if signatures should be required
        if not self.policy.require_signature and high_risk_count > 5:
            recommendations.append(
                "Consider enabling required signatures for high-risk tasks."
            )
        
        # Check audit log size
        if len(self.auditor.audit_entries) > 10000:
            recommendations.append(
                "Large audit log detected. Consider archiving old entries."
            )
        
        if not recommendations:
            recommendations.append("Security posture appears healthy.")
        
        return recommendations


# Integration with QuantumTaskPlanner
class SecureQuantumTaskPlanner(QuantumTaskPlanner):
    """Security-enhanced quantum task planner"""
    
    def __init__(self, resources=None, security_policy: Optional[SecurityPolicy] = None):
        super().__init__(resources)
        self.security_manager = QuantumSecurityManager(security_policy)
    
    def add_task(self, task: QuantumTask, security_level: SecurityLevel = SecurityLevel.PUBLIC) -> None:
        """Add task with security validation"""
        # Validate task security
        if not self.security_manager.validate_task_for_execution(task):
            raise BenchmarkError(f"Task {task.id} failed security validation")
        
        # Set security context
        self.security_manager.set_task_security_level(task.id, security_level)
        
        # Add to planner
        super().add_task(task)
        
        logger.info(f"Added secure task {task.id} with security level {security_level.value}")
    
    async def execute_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute task with security monitoring"""
        start_time = time.time()
        
        # Create secure execution context
        security_context = self.security_manager.create_secure_task_context(task)
        
        try:
            # Execute with monitoring
            result = await super().execute_task(task)
            
            # Monitor execution
            execution_data = {
                'duration': time.time() - start_time,
                'success': result.get('success', False),
                'security_context': security_context
            }
            self.security_manager.monitor_task_execution(task.id, execution_data)
            
            return result
            
        except Exception as e:
            # Log security-relevant execution failure
            execution_data = {
                'duration': time.time() - start_time,
                'success': False,
                'errors': [str(e)],
                'security_context': security_context
            }
            self.security_manager.monitor_task_execution(task.id, execution_data)
            raise