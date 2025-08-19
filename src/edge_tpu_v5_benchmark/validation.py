"""Input validation and error handling for TPU v5 benchmark suite."""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    code: str
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}]"]
        if self.field:
            parts.append(f"{self.field}:")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Results from validation process."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings_count: int
    errors_count: int

    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues."""
        return self.errors_count > 0

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warnings_count > 0

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "code": issue.code,
                    "message": issue.message,
                    "field": issue.field,
                    "value": str(issue.value) if issue.value is not None else None,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ],
            "summary": {
                "warnings": self.warnings_count,
                "errors": self.errors_count,
                "total_issues": len(self.issues)
            }
        }


class BenchmarkValidator:
    """Validates benchmark configuration and inputs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.issues: List[ValidationIssue] = []

    def validate_benchmark_config(
        self,
        iterations: int,
        warmup: int,
        batch_size: int,
        input_shape: tuple,
        confidence_level: float = 0.95
    ) -> ValidationResult:
        """Validate benchmark configuration parameters."""
        self.issues.clear()

        # Validate iterations
        self._validate_iterations(iterations)

        # Validate warmup
        self._validate_warmup(warmup, iterations)

        # Validate batch size
        self._validate_batch_size(batch_size)

        # Validate input shape
        self._validate_input_shape(input_shape)

        # Validate confidence level
        self._validate_confidence_level(confidence_level)

        return self._create_validation_result()

    def validate_model_path(self, model_path: Union[str, Path]) -> ValidationResult:
        """Validate model file path and format."""
        self.issues.clear()

        path = Path(model_path)

        # Check if file exists
        if not path.exists():
            self._add_error(
                "MODEL_NOT_FOUND",
                f"Model file not found: {path}",
                "model_path",
                suggestion="Check the file path and ensure the model file exists"
            )
            return self._create_validation_result()

        # Check if it's a file (not directory)
        if not path.is_file():
            self._add_error(
                "INVALID_MODEL_PATH",
                f"Path is not a file: {path}",
                "model_path",
                suggestion="Provide path to a model file, not a directory"
            )

        # Validate file extension
        supported_extensions = {'.onnx', '.tflite', '.pb', '.pt', '.pth', '.h5'}
        if path.suffix.lower() not in supported_extensions:
            self._add_warning(
                "UNSUPPORTED_EXTENSION",
                f"File extension '{path.suffix}' may not be supported",
                "model_path",
                suggestion=f"Supported extensions: {', '.join(supported_extensions)}"
            )

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1000:  # 1GB
            self._add_warning(
                "LARGE_MODEL_SIZE",
                f"Model file is very large: {file_size_mb:.1f}MB",
                "model_path",
                suggestion="Large models may cause memory issues on edge devices"
            )
        elif file_size_mb < 0.1:  # 100KB
            self._add_warning(
                "SMALL_MODEL_SIZE",
                f"Model file is very small: {file_size_mb:.1f}MB",
                "model_path",
                suggestion="Verify this is a complete model file"
            )

        # Check file permissions
        if not path.stat().st_mode & 0o444:  # Read permission
            self._add_error(
                "NO_READ_PERMISSION",
                f"No read permission for model file: {path}",
                "model_path",
                suggestion="Check file permissions"
            )

        return self._create_validation_result()

    def validate_device_path(self, device_path: str) -> ValidationResult:
        """Validate TPU device path."""
        self.issues.clear()

        # Check device path format
        valid_patterns = [
            r'^/dev/apex_\d+$',
            r'^/dev/tpu\d+$',
            r'^simulation$',
            r'^cpu$'
        ]

        if not any(re.match(pattern, device_path) for pattern in valid_patterns):
            self._add_warning(
                "UNUSUAL_DEVICE_PATH",
                f"Device path format is unusual: {device_path}",
                "device_path",
                suggestion="Expected format: /dev/apex_N or /dev/tpuN"
            )

        # Check if device exists (for real device paths)
        if device_path.startswith('/dev/'):
            device_file = Path(device_path)
            if not device_file.exists():
                self._add_info(
                    "DEVICE_NOT_FOUND",
                    f"TPU device not found: {device_path}",
                    "device_path",
                    suggestion="Will run in simulation mode"
                )

        return self._create_validation_result()

    def validate_optimization_config(
        self,
        optimization_profile: str,
        quantization_method: Optional[str] = None,
        target_precision: Optional[str] = None
    ) -> ValidationResult:
        """Validate optimization configuration."""
        self.issues.clear()

        # Validate optimization profile
        valid_profiles = {"latency", "throughput", "balanced", "efficiency"}
        if optimization_profile not in valid_profiles:
            self._add_error(
                "INVALID_OPTIMIZATION_PROFILE",
                f"Unknown optimization profile: {optimization_profile}",
                "optimization_profile",
                suggestion=f"Valid profiles: {', '.join(valid_profiles)}"
            )

        # Validate quantization method
        if quantization_method:
            valid_methods = {"static", "dynamic", "post_training"}
            if quantization_method not in valid_methods:
                self._add_error(
                    "INVALID_QUANTIZATION_METHOD",
                    f"Unknown quantization method: {quantization_method}",
                    "quantization_method",
                    suggestion=f"Valid methods: {', '.join(valid_methods)}"
                )

        # Validate target precision
        if target_precision:
            valid_precisions = {"int8", "int16", "fp16", "fp32"}
            if target_precision not in valid_precisions:
                self._add_error(
                    "INVALID_TARGET_PRECISION",
                    f"Unknown target precision: {target_precision}",
                    "target_precision",
                    suggestion=f"Valid precisions: {', '.join(valid_precisions)}"
                )

        return self._create_validation_result()

    def _validate_iterations(self, iterations: int):
        """Validate number of iterations."""
        if not isinstance(iterations, int):
            self._add_error(
                "INVALID_ITERATIONS_TYPE",
                f"Iterations must be an integer, got {type(iterations).__name__}",
                "iterations"
            )
            return

        if iterations <= 0:
            self._add_error(
                "INVALID_ITERATIONS_VALUE",
                f"Iterations must be positive, got {iterations}",
                "iterations",
                suggestion="Use at least 10 iterations for meaningful results"
            )
        elif iterations < 10:
            self._add_warning(
                "LOW_ITERATIONS_COUNT",
                f"Low iteration count may produce unreliable results: {iterations}",
                "iterations",
                suggestion="Consider using at least 100 iterations"
            )
        elif iterations > 10000:
            self._add_warning(
                "HIGH_ITERATIONS_COUNT",
                f"High iteration count will take long time: {iterations}",
                "iterations",
                suggestion="Consider reducing iterations for faster benchmarking"
            )

    def _validate_warmup(self, warmup: int, iterations: int):
        """Validate warmup iterations."""
        if not isinstance(warmup, int):
            self._add_error(
                "INVALID_WARMUP_TYPE",
                f"Warmup must be an integer, got {type(warmup).__name__}",
                "warmup"
            )
            return

        if warmup < 0:
            self._add_error(
                "INVALID_WARMUP_VALUE",
                f"Warmup cannot be negative, got {warmup}",
                "warmup"
            )
        elif warmup > iterations:
            self._add_warning(
                "WARMUP_EXCEEDS_ITERATIONS",
                f"Warmup ({warmup}) exceeds iterations ({iterations})",
                "warmup",
                suggestion="Warmup should be less than total iterations"
            )
        elif warmup < iterations * 0.1:
            self._add_info(
                "LOW_WARMUP_RATIO",
                "Warmup is less than 10% of iterations",
                "warmup",
                suggestion="Consider 10-20% warmup for stable results"
            )

    def _validate_batch_size(self, batch_size: int):
        """Validate batch size."""
        if not isinstance(batch_size, int):
            self._add_error(
                "INVALID_BATCH_SIZE_TYPE",
                f"Batch size must be an integer, got {type(batch_size).__name__}",
                "batch_size"
            )
            return

        if batch_size <= 0:
            self._add_error(
                "INVALID_BATCH_SIZE_VALUE",
                f"Batch size must be positive, got {batch_size}",
                "batch_size"
            )
        elif batch_size > 32:
            self._add_warning(
                "LARGE_BATCH_SIZE",
                f"Large batch size may cause memory issues: {batch_size}",
                "batch_size",
                suggestion="Consider batch size <= 8 for edge devices"
            )

    def _validate_input_shape(self, input_shape: tuple):
        """Validate input tensor shape."""
        if not isinstance(input_shape, (tuple, list)):
            self._add_error(
                "INVALID_INPUT_SHAPE_TYPE",
                f"Input shape must be tuple or list, got {type(input_shape).__name__}",
                "input_shape"
            )
            return

        if len(input_shape) == 0:
            self._add_error(
                "EMPTY_INPUT_SHAPE",
                "Input shape cannot be empty",
                "input_shape"
            )
            return

        # Check for valid dimensions
        for i, dim in enumerate(input_shape):
            if not isinstance(dim, int):
                self._add_error(
                    "INVALID_DIMENSION_TYPE",
                    f"Dimension {i} must be integer, got {type(dim).__name__}",
                    "input_shape"
                )
            elif dim <= 0:
                self._add_error(
                    "INVALID_DIMENSION_VALUE",
                    f"Dimension {i} must be positive, got {dim}",
                    "input_shape"
                )

        # Check tensor size
        total_elements = 1
        for dim in input_shape:
            if isinstance(dim, int) and dim > 0:
                total_elements *= dim

        # Estimate memory usage (assuming float32)
        memory_mb = (total_elements * 4) / (1024 * 1024)
        if memory_mb > 100:  # 100MB
            self._add_warning(
                "LARGE_INPUT_TENSOR",
                f"Input tensor is very large: {memory_mb:.1f}MB",
                "input_shape",
                suggestion="Large tensors may cause memory issues"
            )

    def _validate_confidence_level(self, confidence_level: float):
        """Validate statistical confidence level."""
        if not isinstance(confidence_level, (int, float)):
            self._add_error(
                "INVALID_CONFIDENCE_TYPE",
                f"Confidence level must be numeric, got {type(confidence_level).__name__}",
                "confidence_level"
            )
            return

        if confidence_level <= 0 or confidence_level >= 1:
            self._add_error(
                "INVALID_CONFIDENCE_RANGE",
                f"Confidence level must be between 0 and 1, got {confidence_level}",
                "confidence_level",
                suggestion="Use values like 0.95 for 95% confidence"
            )

    def _add_issue(self, severity: ValidationSeverity, code: str, message: str,
                   field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add a validation issue."""
        issue = ValidationIssue(
            severity=severity,
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        )
        self.issues.append(issue)

        # Log the issue
        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL
        }[severity]

        self.logger.log(log_level, str(issue))

    def _add_info(self, code: str, message: str, field: Optional[str] = None,
                  suggestion: Optional[str] = None):
        """Add an info-level issue."""
        self._add_issue(ValidationSeverity.INFO, code, message, field, suggestion)

    def _add_warning(self, code: str, message: str, field: Optional[str] = None,
                     suggestion: Optional[str] = None):
        """Add a warning-level issue."""
        self._add_issue(ValidationSeverity.WARNING, code, message, field, suggestion)

    def _add_error(self, code: str, message: str, field: Optional[str] = None,
                   suggestion: Optional[str] = None):
        """Add an error-level issue."""
        self._add_issue(ValidationSeverity.ERROR, code, message, field, suggestion)

    def _add_critical(self, code: str, message: str, field: Optional[str] = None,
                      suggestion: Optional[str] = None):
        """Add a critical-level issue."""
        self._add_issue(ValidationSeverity.CRITICAL, code, message, field, suggestion)

    def _create_validation_result(self) -> ValidationResult:
        """Create validation result from collected issues."""
        warnings_count = len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
        errors_count = len([i for i in self.issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])

        is_valid = errors_count == 0

        return ValidationResult(
            is_valid=is_valid,
            issues=self.issues.copy(),
            warnings_count=warnings_count,
            errors_count=errors_count
        )


class SecurityValidator:
    """Security-focused validation for benchmark operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_file_access(self, file_path: Path) -> ValidationResult:
        """Validate file access from security perspective."""
        issues = []

        # Check for path traversal attempts
        if ".." in str(file_path):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                code="PATH_TRAVERSAL_ATTEMPT",
                message="Path traversal detected in file path",
                field="file_path",
                suggestion="Use absolute paths without '..' components"
            ))

        # Check for suspicious file extensions
        suspicious_extensions = {'.exe', '.bat', '.sh', '.py', '.js', '.php'}
        if file_path.suffix.lower() in suspicious_extensions:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="SUSPICIOUS_FILE_EXTENSION",
                message=f"Potentially dangerous file extension: {file_path.suffix}",
                field="file_path",
                suggestion="Ensure this is a valid model file"
            ))

        # Check file size limits (prevent DoS)
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > 5000:  # 5GB limit
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    code="FILE_SIZE_LIMIT_EXCEEDED",
                    message=f"File size exceeds security limit: {size_mb:.1f}MB",
                    field="file_path",
                    suggestion="Use smaller model files or increase limits"
                ))

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            issues=issues,
            warnings_count=len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
            errors_count=len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        )

    def validate_network_access(self, url: str) -> ValidationResult:
        """Validate network access for model downloads."""
        issues = []

        # Check for allowed protocols
        allowed_protocols = {'https', 'http'}
        if not any(url.startswith(f"{proto}://") for proto in allowed_protocols):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                code="INVALID_PROTOCOL",
                message=f"Protocol not allowed in URL: {url}",
                suggestion="Use HTTP or HTTPS URLs only"
            ))

        # Check for suspicious domains
        suspicious_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '192.168.', '10.0.', '172.16.']
        if any(pattern in url.lower() for pattern in suspicious_patterns):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="LOCAL_NETWORK_ACCESS",
                message="URL points to local/private network",
                suggestion="Ensure this is intentional and secure"
            ))

        return ValidationResult(
            is_valid=len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0,
            issues=issues,
            warnings_count=len([i for i in issues if i.severity == ValidationSeverity.WARNING]),
            errors_count=len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]])
        )
