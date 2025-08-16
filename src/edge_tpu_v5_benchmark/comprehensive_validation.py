"""Comprehensive Validation Framework for TPU v5 Benchmark Suite

This module implements multi-layered validation including input validation,
data integrity checks, system validation, performance validation,
and continuous validation monitoring.
"""

import asyncio
import time
import threading
import logging
import hashlib
import json
import subprocess
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import psutil
from collections import defaultdict, deque
import re
import math

from .security import SecurityContext, InputValidator
from .robust_error_handling import ErrorRecoveryManager, robust_operation


class ValidationLevel(Enum):
    """Validation thoroughness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    EXHAUSTIVE = "exhaustive"


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    ERROR = "error"


class ValidationType(Enum):
    """Types of validation checks."""
    INPUT_VALIDATION = "input_validation"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM_VALIDATION = "system_validation"
    PERFORMANCE_VALIDATION = "performance_validation"
    SECURITY_VALIDATION = "security_validation"
    BUSINESS_LOGIC = "business_logic"
    REGRESSION = "regression"
    INTEGRATION = "integration"


@dataclass
class ValidationCheck:
    """Individual validation check definition."""
    name: str
    validation_type: ValidationType
    level: ValidationLevel
    description: str
    check_function: Callable
    expected_result: Any = None
    tolerance: float = 0.0
    timeout_seconds: float = 30.0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Validation execution report."""
    check_name: str
    result: ValidationResult
    message: str
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_name": self.check_name,
            "result": self.result.value,
            "message": self.message,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "details": self.details
        }


class AdvancedInputValidator:
    """Advanced input validation with context awareness."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules registry
        self.validation_rules: Dict[str, List[Callable]] = defaultdict(list)
        self.type_validators: Dict[Type, Callable] = {}
        
        # Setup default validators
        self._setup_default_validators()
    
    def _setup_default_validators(self):
        """Setup default validation rules."""
        # Numeric validators
        self.register_type_validator(int, self._validate_integer)
        self.register_type_validator(float, self._validate_float)
        self.register_type_validator(str, self._validate_string)
        self.register_type_validator(list, self._validate_list)
        self.register_type_validator(dict, self._validate_dict)
        
        # Domain-specific validators
        self.register_field_validator("model_path", self._validate_model_path)
        self.register_field_validator("device_path", self._validate_device_path)
        self.register_field_validator("batch_size", self._validate_batch_size)
        self.register_field_validator("iterations", self._validate_iterations)
    
    def register_type_validator(self, data_type: Type, validator: Callable):
        """Register validator for specific data type."""
        self.type_validators[data_type] = validator
    
    def register_field_validator(self, field_name: str, validator: Callable):
        """Register validator for specific field."""
        self.validation_rules[field_name].append(validator)
    
    def validate_input(self, data: Any, schema: Optional[Dict[str, Any]] = None) -> ValidationReport:
        """Comprehensive input validation."""
        start_time = time.time()
        
        try:
            # Type validation
            if schema and "type" in schema:
                expected_type = schema["type"]
                if not isinstance(data, expected_type):
                    return ValidationReport(
                        check_name="type_validation",
                        result=ValidationResult.FAIL,
                        message=f"Expected {expected_type.__name__}, got {type(data).__name__}",
                        execution_time=time.time() - start_time
                    )
            
            # Schema validation
            if schema and isinstance(data, dict):
                schema_result = self._validate_schema(data, schema)
                if schema_result.result != ValidationResult.PASS:
                    return schema_result
            
            # Custom field validation
            if isinstance(data, dict):
                for field_name, field_value in data.items():
                    field_result = self._validate_field(field_name, field_value)
                    if field_result.result != ValidationResult.PASS:
                        return field_result
            
            # Type-specific validation
            data_type = type(data)
            if data_type in self.type_validators:
                type_result = self.type_validators[data_type](data)
                if type_result.result != ValidationResult.PASS:
                    return type_result
            
            return ValidationReport(
                check_name="input_validation",
                result=ValidationResult.PASS,
                message="Input validation passed",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return ValidationReport(
                check_name="input_validation",
                result=ValidationResult.ERROR,
                message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationReport:
        """Validate data against schema."""
        start_time = time.time()
        
        # Required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                return ValidationReport(
                    check_name="schema_validation",
                    result=ValidationResult.FAIL,
                    message=f"Required field '{field}' missing",
                    execution_time=time.time() - start_time
                )
        
        # Field properties
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in data:
                field_value = data[field_name]
                
                # Type check
                if "type" in field_schema:
                    expected_type = field_schema["type"]
                    if not isinstance(field_value, expected_type):
                        return ValidationReport(
                            check_name="schema_validation",
                            result=ValidationResult.FAIL,
                            message=f"Field '{field_name}' expected {expected_type.__name__}, got {type(field_value).__name__}",
                            execution_time=time.time() - start_time
                        )
                
                # Range checks
                if isinstance(field_value, (int, float)):
                    if "minimum" in field_schema and field_value < field_schema["minimum"]:
                        return ValidationReport(
                            check_name="schema_validation",
                            result=ValidationResult.FAIL,
                            message=f"Field '{field_name}' value {field_value} below minimum {field_schema['minimum']}",
                            execution_time=time.time() - start_time
                        )
                    
                    if "maximum" in field_schema and field_value > field_schema["maximum"]:
                        return ValidationReport(
                            check_name="schema_validation",
                            result=ValidationResult.FAIL,
                            message=f"Field '{field_name}' value {field_value} above maximum {field_schema['maximum']}",
                            execution_time=time.time() - start_time
                        )
        
        return ValidationReport(
            check_name="schema_validation",
            result=ValidationResult.PASS,
            message="Schema validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_field(self, field_name: str, field_value: Any) -> ValidationReport:
        """Validate specific field using registered validators."""
        start_time = time.time()
        
        validators = self.validation_rules.get(field_name, [])
        for validator in validators:
            try:
                result = validator(field_value)
                if result.result != ValidationResult.PASS:
                    return result
            except Exception as e:
                return ValidationReport(
                    check_name=f"field_validation_{field_name}",
                    result=ValidationResult.ERROR,
                    message=f"Validator error: {str(e)}",
                    execution_time=time.time() - start_time
                )
        
        return ValidationReport(
            check_name=f"field_validation_{field_name}",
            result=ValidationResult.PASS,
            message=f"Field '{field_name}' validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_integer(self, value: int) -> ValidationReport:
        """Validate integer value."""
        start_time = time.time()
        
        if not isinstance(value, int):
            return ValidationReport(
                check_name="integer_validation",
                result=ValidationResult.FAIL,
                message="Value is not an integer",
                execution_time=time.time() - start_time
            )
        
        # Check for reasonable bounds
        if value < -2**31 or value > 2**31 - 1:
            return ValidationReport(
                check_name="integer_validation",
                result=ValidationResult.WARNING,
                message="Integer value outside 32-bit range",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="integer_validation",
            result=ValidationResult.PASS,
            message="Integer validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_float(self, value: float) -> ValidationReport:
        """Validate float value."""
        start_time = time.time()
        
        if not isinstance(value, (int, float)):
            return ValidationReport(
                check_name="float_validation",
                result=ValidationResult.FAIL,
                message="Value is not a number",
                execution_time=time.time() - start_time
            )
        
        if math.isnan(value):
            return ValidationReport(
                check_name="float_validation",
                result=ValidationResult.FAIL,
                message="Value is NaN",
                execution_time=time.time() - start_time
            )
        
        if math.isinf(value):
            return ValidationReport(
                check_name="float_validation",
                result=ValidationResult.FAIL,
                message="Value is infinite",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="float_validation",
            result=ValidationResult.PASS,
            message="Float validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_string(self, value: str) -> ValidationReport:
        """Validate string value."""
        start_time = time.time()
        
        if not isinstance(value, str):
            return ValidationReport(
                check_name="string_validation",
                result=ValidationResult.FAIL,
                message="Value is not a string",
                execution_time=time.time() - start_time
            )
        
        # Check for null bytes
        if '\x00' in value:
            return ValidationReport(
                check_name="string_validation",
                result=ValidationResult.FAIL,
                message="String contains null bytes",
                execution_time=time.time() - start_time
            )
        
        # Check reasonable length
        if len(value) > 10000:
            return ValidationReport(
                check_name="string_validation",
                result=ValidationResult.WARNING,
                message="String exceeds recommended length",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="string_validation",
            result=ValidationResult.PASS,
            message="String validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_list(self, value: list) -> ValidationReport:
        """Validate list value."""
        start_time = time.time()
        
        if not isinstance(value, list):
            return ValidationReport(
                check_name="list_validation",
                result=ValidationResult.FAIL,
                message="Value is not a list",
                execution_time=time.time() - start_time
            )
        
        # Check reasonable size
        if len(value) > 10000:
            return ValidationReport(
                check_name="list_validation",
                result=ValidationResult.WARNING,
                message="List exceeds recommended size",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="list_validation",
            result=ValidationResult.PASS,
            message="List validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_dict(self, value: dict) -> ValidationReport:
        """Validate dictionary value."""
        start_time = time.time()
        
        if not isinstance(value, dict):
            return ValidationReport(
                check_name="dict_validation",
                result=ValidationResult.FAIL,
                message="Value is not a dictionary",
                execution_time=time.time() - start_time
            )
        
        # Check reasonable size
        if len(value) > 1000:
            return ValidationReport(
                check_name="dict_validation",
                result=ValidationResult.WARNING,
                message="Dictionary exceeds recommended size",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="dict_validation",
            result=ValidationResult.PASS,
            message="Dictionary validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_model_path(self, value: Any) -> ValidationReport:
        """Validate model file path."""
        start_time = time.time()
        
        if not isinstance(value, (str, Path)):
            return ValidationReport(
                check_name="model_path_validation",
                result=ValidationResult.FAIL,
                message="Model path must be string or Path",
                execution_time=time.time() - start_time
            )
        
        path = Path(value)
        
        # Check if file exists
        if not path.exists():
            return ValidationReport(
                check_name="model_path_validation",
                result=ValidationResult.FAIL,
                message=f"Model file does not exist: {path}",
                execution_time=time.time() - start_time
            )
        
        # Check file extension
        valid_extensions = ['.tflite', '.onnx', '.pb', '.h5']
        if path.suffix.lower() not in valid_extensions:
            return ValidationReport(
                check_name="model_path_validation",
                result=ValidationResult.WARNING,
                message=f"Unexpected model file extension: {path.suffix}",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="model_path_validation",
            result=ValidationResult.PASS,
            message="Model path validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_device_path(self, value: Any) -> ValidationReport:
        """Validate TPU device path."""
        start_time = time.time()
        
        if not isinstance(value, str):
            return ValidationReport(
                check_name="device_path_validation",
                result=ValidationResult.FAIL,
                message="Device path must be string",
                execution_time=time.time() - start_time
            )
        
        # Check device path format
        valid_patterns = [r'/dev/apex_\d+', r'/dev/tpu\d+']
        if not any(re.match(pattern, value) for pattern in valid_patterns):
            return ValidationReport(
                check_name="device_path_validation",
                result=ValidationResult.WARNING,
                message=f"Unexpected device path format: {value}",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="device_path_validation",
            result=ValidationResult.PASS,
            message="Device path validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_batch_size(self, value: Any) -> ValidationReport:
        """Validate batch size parameter."""
        start_time = time.time()
        
        if not isinstance(value, int):
            return ValidationReport(
                check_name="batch_size_validation",
                result=ValidationResult.FAIL,
                message="Batch size must be integer",
                execution_time=time.time() - start_time
            )
        
        if value <= 0:
            return ValidationReport(
                check_name="batch_size_validation",
                result=ValidationResult.FAIL,
                message="Batch size must be positive",
                execution_time=time.time() - start_time
            )
        
        if value > 1024:
            return ValidationReport(
                check_name="batch_size_validation",
                result=ValidationResult.WARNING,
                message="Large batch size may cause memory issues",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="batch_size_validation",
            result=ValidationResult.PASS,
            message="Batch size validation passed",
            execution_time=time.time() - start_time
        )
    
    def _validate_iterations(self, value: Any) -> ValidationReport:
        """Validate iterations parameter."""
        start_time = time.time()
        
        if not isinstance(value, int):
            return ValidationReport(
                check_name="iterations_validation",
                result=ValidationResult.FAIL,
                message="Iterations must be integer",
                execution_time=time.time() - start_time
            )
        
        if value <= 0:
            return ValidationReport(
                check_name="iterations_validation",
                result=ValidationResult.FAIL,
                message="Iterations must be positive",
                execution_time=time.time() - start_time
            )
        
        if value > 100000:
            return ValidationReport(
                check_name="iterations_validation",
                result=ValidationResult.WARNING,
                message="Large iteration count may take very long",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="iterations_validation",
            result=ValidationResult.PASS,
            message="Iterations validation passed",
            execution_time=time.time() - start_time
        )


class DataIntegrityValidator:
    """Validates data integrity and consistency."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_data_integrity(self, data: Any, 
                              expected_checksum: Optional[str] = None) -> ValidationReport:
        """Validate data integrity using checksums."""
        start_time = time.time()
        
        try:
            # Calculate checksum
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            
            calculated_checksum = hashlib.sha256(data_bytes).hexdigest()
            
            # Compare with expected checksum if provided
            if expected_checksum:
                if calculated_checksum != expected_checksum:
                    return ValidationReport(
                        check_name="data_integrity",
                        result=ValidationResult.FAIL,
                        message=f"Checksum mismatch: expected {expected_checksum}, got {calculated_checksum}",
                        execution_time=time.time() - start_time,
                        details={"calculated_checksum": calculated_checksum}
                    )
            
            return ValidationReport(
                check_name="data_integrity",
                result=ValidationResult.PASS,
                message="Data integrity validated",
                execution_time=time.time() - start_time,
                details={"checksum": calculated_checksum}
            )
            
        except Exception as e:
            return ValidationReport(
                check_name="data_integrity",
                result=ValidationResult.ERROR,
                message=f"Integrity validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_data_consistency(self, dataset: List[Any]) -> ValidationReport:
        """Validate consistency across dataset."""
        start_time = time.time()
        
        if not dataset:
            return ValidationReport(
                check_name="data_consistency",
                result=ValidationResult.PASS,
                message="Empty dataset is consistent",
                execution_time=time.time() - start_time
            )
        
        inconsistencies = []
        
        # Check type consistency
        first_type = type(dataset[0])
        for i, item in enumerate(dataset[1:], 1):
            if type(item) != first_type:
                inconsistencies.append(f"Type mismatch at index {i}: expected {first_type.__name__}, got {type(item).__name__}")
        
        # For numeric data, check for outliers
        if first_type in (int, float) and len(dataset) > 5:
            values = np.array(dataset, dtype=float)
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
            if len(outliers) > 0:
                inconsistencies.append(f"Found {len(outliers)} outliers in numeric data")
        
        # For dictionaries, check key consistency
        if first_type == dict and dataset:
            expected_keys = set(dataset[0].keys())
            for i, item in enumerate(dataset[1:], 1):
                item_keys = set(item.keys())
                if item_keys != expected_keys:
                    missing = expected_keys - item_keys
                    extra = item_keys - expected_keys
                    if missing:
                        inconsistencies.append(f"Missing keys at index {i}: {missing}")
                    if extra:
                        inconsistencies.append(f"Extra keys at index {i}: {extra}")
        
        if inconsistencies:
            return ValidationReport(
                check_name="data_consistency",
                result=ValidationResult.WARNING,
                message=f"Found {len(inconsistencies)} consistency issues",
                execution_time=time.time() - start_time,
                details={"inconsistencies": inconsistencies}
            )
        
        return ValidationReport(
            check_name="data_consistency",
            result=ValidationResult.PASS,
            message="Data consistency validated",
            execution_time=time.time() - start_time
        )


class SystemValidator:
    """Validates system state and requirements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_system_resources(self, requirements: Dict[str, Any]) -> ValidationReport:
        """Validate system meets resource requirements."""
        start_time = time.time()
        
        issues = []
        
        try:
            # Memory validation
            memory_info = psutil.virtual_memory()
            required_memory_gb = requirements.get("memory_gb", 0)
            available_memory_gb = memory_info.available / (1024**3)
            
            if required_memory_gb > available_memory_gb:
                issues.append(f"Insufficient memory: required {required_memory_gb}GB, available {available_memory_gb:.1f}GB")
            
            # CPU validation
            cpu_count = psutil.cpu_count()
            required_cpus = requirements.get("cpu_cores", 0)
            
            if required_cpus > cpu_count:
                issues.append(f"Insufficient CPU cores: required {required_cpus}, available {cpu_count}")
            
            # Disk space validation
            disk_info = psutil.disk_usage('/')
            required_disk_gb = requirements.get("disk_gb", 0)
            available_disk_gb = disk_info.free / (1024**3)
            
            if required_disk_gb > available_disk_gb:
                issues.append(f"Insufficient disk space: required {required_disk_gb}GB, available {available_disk_gb:.1f}GB")
            
            # Check system load
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
            if load_avg > cpu_count * 0.8:
                issues.append(f"High system load: {load_avg:.2f} (recommended < {cpu_count * 0.8:.1f})")
            
            if issues:
                return ValidationReport(
                    check_name="system_resources",
                    result=ValidationResult.WARNING,
                    message=f"System resource issues detected",
                    execution_time=time.time() - start_time,
                    details={"issues": issues}
                )
            
            return ValidationReport(
                check_name="system_resources",
                result=ValidationResult.PASS,
                message="System resources validated",
                execution_time=time.time() - start_time,
                details={
                    "memory_gb": available_memory_gb,
                    "cpu_cores": cpu_count,
                    "disk_gb": available_disk_gb,
                    "load_avg": load_avg
                }
            )
            
        except Exception as e:
            return ValidationReport(
                check_name="system_resources",
                result=ValidationResult.ERROR,
                message=f"System validation error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def validate_dependencies(self, required_packages: List[str]) -> ValidationReport:
        """Validate required dependencies are available."""
        start_time = time.time()
        
        missing_packages = []
        
        for package in required_packages:
            try:
                subprocess.run([
                    'python', '-c', f'import {package}'
                ], check=True, capture_output=True, timeout=10)
            except subprocess.CalledProcessError:
                missing_packages.append(package)
            except subprocess.TimeoutExpired:
                missing_packages.append(f"{package} (timeout)")
        
        if missing_packages:
            return ValidationReport(
                check_name="dependencies",
                result=ValidationResult.FAIL,
                message=f"Missing dependencies: {missing_packages}",
                execution_time=time.time() - start_time,
                details={"missing_packages": missing_packages}
            )
        
        return ValidationReport(
            check_name="dependencies",
            result=ValidationResult.PASS,
            message="All dependencies validated",
            execution_time=time.time() - start_time,
            details={"checked_packages": required_packages}
        )


class ComprehensiveValidationFramework:
    """Main validation framework coordinating all validation types."""
    
    def __init__(self, 
                 security_context: Optional[SecurityContext] = None,
                 error_manager: Optional[ErrorRecoveryManager] = None):
        self.security_context = security_context or SecurityContext()
        self.error_manager = error_manager or ErrorRecoveryManager()
        self.logger = logging.getLogger(__name__)
        
        # Validators
        self.input_validator = AdvancedInputValidator()
        self.integrity_validator = DataIntegrityValidator()
        self.system_validator = SystemValidator()
        
        # Validation registry
        self.validation_checks: Dict[str, ValidationCheck] = {}
        self.validation_history: deque = deque(maxlen=10000)
        
        # Validation configuration
        self.default_level = ValidationLevel.STANDARD
        self.parallel_execution = True
        
        self.lock = threading.RLock()
        
        # Setup default checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default validation checks."""
        # Input validation checks
        self.register_check(ValidationCheck(
            name="input_schema_validation",
            validation_type=ValidationType.INPUT_VALIDATION,
            level=ValidationLevel.BASIC,
            description="Validate input against schema",
            check_function=self._check_input_schema
        ))
        
        # System validation checks
        self.register_check(ValidationCheck(
            name="system_resources_check",
            validation_type=ValidationType.SYSTEM_VALIDATION,
            level=ValidationLevel.STANDARD,
            description="Validate system resource availability",
            check_function=self._check_system_resources
        ))
        
        # Data integrity checks
        self.register_check(ValidationCheck(
            name="data_integrity_check",
            validation_type=ValidationType.DATA_INTEGRITY,
            level=ValidationLevel.STANDARD,
            description="Validate data integrity using checksums",
            check_function=self._check_data_integrity
        ))
        
        # Performance validation checks
        self.register_check(ValidationCheck(
            name="performance_baseline_check",
            validation_type=ValidationType.PERFORMANCE_VALIDATION,
            level=ValidationLevel.COMPREHENSIVE,
            description="Validate performance against baseline",
            check_function=self._check_performance_baseline
        ))
    
    def register_check(self, check: ValidationCheck):
        """Register a validation check."""
        with self.lock:
            self.validation_checks[check.name] = check
        self.logger.info(f"Registered validation check: {check.name}")
    
    def run_validation(self, 
                      validation_data: Dict[str, Any],
                      level: Optional[ValidationLevel] = None,
                      check_names: Optional[List[str]] = None) -> Dict[str, ValidationReport]:
        """Run validation checks."""
        level = level or self.default_level
        start_time = time.time()
        
        # Select checks to run
        if check_names:
            checks_to_run = [
                self.validation_checks[name] for name in check_names
                if name in self.validation_checks
            ]
        else:
            checks_to_run = [
                check for check in self.validation_checks.values()
                if self._should_run_check(check, level)
            ]
        
        # Sort by dependencies
        sorted_checks = self._sort_checks_by_dependencies(checks_to_run)
        
        results = {}
        
        if self.parallel_execution:
            results = self._run_checks_parallel(sorted_checks, validation_data)
        else:
            results = self._run_checks_sequential(sorted_checks, validation_data)
        
        # Store results in history
        with self.lock:
            for report in results.values():
                self.validation_history.append(report)
        
        self.logger.info(
            f"Validation completed in {time.time() - start_time:.2f}s: "
            f"{len(results)} checks, "
            f"{sum(1 for r in results.values() if r.result == ValidationResult.PASS)} passed"
        )
        
        return results
    
    def _should_run_check(self, check: ValidationCheck, level: ValidationLevel) -> bool:
        """Determine if check should run based on level."""
        level_hierarchy = {
            ValidationLevel.BASIC: 1,
            ValidationLevel.STANDARD: 2,
            ValidationLevel.COMPREHENSIVE: 3,
            ValidationLevel.EXHAUSTIVE: 4
        }
        
        return level_hierarchy[check.level] <= level_hierarchy[level]
    
    def _sort_checks_by_dependencies(self, checks: List[ValidationCheck]) -> List[ValidationCheck]:
        """Sort checks by dependencies."""
        sorted_checks = []
        remaining_checks = checks.copy()
        
        while remaining_checks:
            # Find checks with satisfied dependencies
            ready_checks = []
            for check in remaining_checks:
                dependencies_satisfied = all(
                    any(completed.name == dep for completed in sorted_checks)
                    for dep in check.dependencies
                )
                if dependencies_satisfied:
                    ready_checks.append(check)
            
            if not ready_checks:
                # Break circular dependencies by adding remaining checks
                ready_checks = remaining_checks
            
            sorted_checks.extend(ready_checks)
            for check in ready_checks:
                remaining_checks.remove(check)
        
        return sorted_checks
    
    def _run_checks_sequential(self, checks: List[ValidationCheck], 
                             validation_data: Dict[str, Any]) -> Dict[str, ValidationReport]:
        """Run validation checks sequentially."""
        results = {}
        
        for check in checks:
            try:
                report = self._execute_check(check, validation_data)
                results[check.name] = report
                
                # Stop on critical failures
                if report.result == ValidationResult.FAIL and check.validation_type == ValidationType.SECURITY_VALIDATION:
                    self.logger.warning(f"Critical security validation failed: {check.name}")
                    break
                
            except Exception as e:
                self.logger.error(f"Check execution failed: {check.name} - {e}")
                results[check.name] = ValidationReport(
                    check_name=check.name,
                    result=ValidationResult.ERROR,
                    message=f"Execution error: {str(e)}",
                    execution_time=0.0
                )
        
        return results
    
    def _run_checks_parallel(self, checks: List[ValidationCheck], 
                           validation_data: Dict[str, Any]) -> Dict[str, ValidationReport]:
        """Run validation checks in parallel."""
        results = {}
        
        # Group checks by dependency level
        dependency_levels = self._group_checks_by_level(checks)
        
        for level_checks in dependency_levels:
            level_results = {}
            
            # Run checks in this level in parallel
            with ThreadPoolExecutor(max_workers=min(len(level_checks), 10)) as executor:
                future_to_check = {
                    executor.submit(self._execute_check, check, validation_data): check
                    for check in level_checks
                }
                
                for future in future_to_check:
                    check = future_to_check[future]
                    try:
                        report = future.result(timeout=check.timeout_seconds)
                        level_results[check.name] = report
                    except Exception as e:
                        self.logger.error(f"Parallel check execution failed: {check.name} - {e}")
                        level_results[check.name] = ValidationReport(
                            check_name=check.name,
                            result=ValidationResult.ERROR,
                            message=f"Execution error: {str(e)}",
                            execution_time=0.0
                        )
            
            results.update(level_results)
            
            # Check for critical failures that should stop further validation
            critical_failures = [
                r for r in level_results.values()
                if r.result == ValidationResult.FAIL and 
                any(c.validation_type == ValidationType.SECURITY_VALIDATION 
                    for c in level_checks if c.name == r.check_name)
            ]
            
            if critical_failures:
                self.logger.warning("Critical security validation failures detected, stopping validation")
                break
        
        return results
    
    def _group_checks_by_level(self, checks: List[ValidationCheck]) -> List[List[ValidationCheck]]:
        """Group checks by dependency level."""
        levels = []
        remaining_checks = checks.copy()
        completed_checks = set()
        
        while remaining_checks:
            current_level = []
            
            for check in remaining_checks.copy():
                if all(dep in completed_checks for dep in check.dependencies):
                    current_level.append(check)
                    remaining_checks.remove(check)
                    completed_checks.add(check.name)
            
            if not current_level:
                # Handle circular dependencies
                current_level = remaining_checks.copy()
                remaining_checks.clear()
                for check in current_level:
                    completed_checks.add(check.name)
            
            levels.append(current_level)
        
        return levels
    
    @robust_operation(timeout=30.0)
    def _execute_check(self, check: ValidationCheck, 
                      validation_data: Dict[str, Any]) -> ValidationReport:
        """Execute individual validation check."""
        start_time = time.time()
        
        try:
            # Prepare check data
            check_data = validation_data.get(check.name, validation_data)
            
            # Execute check function
            result = check.check_function(check_data)
            
            # Ensure result is ValidationReport
            if not isinstance(result, ValidationReport):
                result = ValidationReport(
                    check_name=check.name,
                    result=ValidationResult.PASS if result else ValidationResult.FAIL,
                    message=str(result),
                    execution_time=time.time() - start_time
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Check execution error: {check.name} - {e}")
            return ValidationReport(
                check_name=check.name,
                result=ValidationResult.ERROR,
                message=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    # Default check implementations
    def _check_input_schema(self, data: Any) -> ValidationReport:
        """Default input schema validation."""
        return self.input_validator.validate_input(data)
    
    def _check_system_resources(self, data: Any) -> ValidationReport:
        """Default system resource validation."""
        requirements = data.get("system_requirements", {})
        return self.system_validator.validate_system_resources(requirements)
    
    def _check_data_integrity(self, data: Any) -> ValidationReport:
        """Default data integrity validation."""
        expected_checksum = data.get("expected_checksum")
        actual_data = data.get("data", data)
        return self.integrity_validator.validate_data_integrity(actual_data, expected_checksum)
    
    def _check_performance_baseline(self, data: Any) -> ValidationReport:
        """Default performance baseline validation."""
        start_time = time.time()
        
        # Mock performance check
        baseline_time = data.get("baseline_execution_time", 1.0)
        current_time = data.get("current_execution_time", 0.5)
        tolerance = data.get("tolerance", 0.1)
        
        if current_time > baseline_time * (1 + tolerance):
            return ValidationReport(
                check_name="performance_baseline",
                result=ValidationResult.FAIL,
                message=f"Performance regression: {current_time:.2f}s > {baseline_time:.2f}s (tolerance: {tolerance:.1%})",
                execution_time=time.time() - start_time
            )
        
        return ValidationReport(
            check_name="performance_baseline",
            result=ValidationResult.PASS,
            message="Performance within baseline",
            execution_time=time.time() - start_time
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        with self.lock:
            if not self.validation_history:
                return {"total_validations": 0}
            
            result_counts = defaultdict(int)
            type_counts = defaultdict(int)
            recent_validations = 0
            
            for report in self.validation_history:
                result_counts[report.result.value] += 1
                
                # Infer type from check name
                for check in self.validation_checks.values():
                    if check.name == report.check_name:
                        type_counts[check.validation_type.value] += 1
                        break
                
                if time.time() - report.timestamp < 3600:  # Last hour
                    recent_validations += 1
            
            avg_execution_time = np.mean([r.execution_time for r in self.validation_history])
            
            return {
                "total_validations": len(self.validation_history),
                "recent_validations": recent_validations,
                "result_distribution": dict(result_counts),
                "type_distribution": dict(type_counts),
                "average_execution_time": avg_execution_time,
                "registered_checks": len(self.validation_checks),
                "validation_level": self.default_level.value
            }
    
    def export_validation_report(self, filepath: Path):
        """Export comprehensive validation report."""
        with self.lock:
            report = {
                "validation_statistics": self.get_validation_statistics(),
                "registered_checks": [
                    {
                        "name": check.name,
                        "type": check.validation_type.value,
                        "level": check.level.value,
                        "description": check.description
                    }
                    for check in self.validation_checks.values()
                ],
                "recent_validations": [
                    report.to_dict() for report in 
                    list(self.validation_history)[-100:]  # Last 100 validations
                ],
                "report_timestamp": time.time()
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation report exported to {filepath}")


def create_validation_framework(security_context: Optional[SecurityContext] = None,
                              error_manager: Optional[ErrorRecoveryManager] = None) -> ComprehensiveValidationFramework:
    """Factory function to create comprehensive validation framework."""
    return ComprehensiveValidationFramework(security_context, error_manager)