"""Security utilities for TPU v5 benchmark suite."""

import hashlib
import logging
import re
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Set


class SecurityContext:
    """Security context for benchmark operations."""

    def __init__(self):
        """Initialize security context."""
        self.session_id = self._generate_session_id()
        self.allowed_paths: Set[str] = set()
        self._security_token = secrets.token_hex(32)

    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_hex(16)

    def validate_path(self, path: str) -> bool:
        """Validate if path is in allowed set."""
        normalized_path = str(Path(path).resolve())
        return normalized_path in self.allowed_paths or self._is_safe_path(normalized_path)

    def _is_safe_path(self, path: str) -> bool:
        """Check if path follows security guidelines."""
        # Prevent directory traversal
        if '..' in path or '~' in path:
            return False

        # Allow only specific device paths
        safe_patterns = [
            r'^/dev/(apex|tpu)_?\d*$',
            r'^/tmp/tpu_bench_[a-f0-9]+$'
        ]

        return any(re.match(pattern, path) for pattern in safe_patterns)


class InputValidator:
    """Input validation utilities."""

    @staticmethod
    def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None,
                              param_name: str = "value") -> float:
        """Validate numeric input with bounds checking."""
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{param_name} must be a valid number, got {type(value)}")

        if min_val is not None and num_val < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {num_val}")

        if max_val is not None and num_val > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {num_val}")

        return num_val

    @staticmethod
    def validate_string_input(value: Any, max_length: int = 1000,
                             allowed_chars: str = None, param_name: str = "value") -> str:
        """Validate string input with length and character restrictions."""
        if not isinstance(value, str):
            raise ValueError(f"{param_name} must be a string, got {type(value)}")

        if len(value) > max_length:
            raise ValueError(f"{param_name} length must be <= {max_length}, got {len(value)}")

        if allowed_chars and not all(c in allowed_chars for c in value):
            raise ValueError(f"{param_name} contains invalid characters")

        return value

    @staticmethod
    def validate_shape_input(shape: tuple) -> tuple:
        """Validate input shape tuple."""
        if not isinstance(shape, (tuple, list)):
            raise ValueError(f"Shape must be tuple or list, got {type(shape)}")

        if not shape:
            raise ValueError("Shape cannot be empty")

        for i, dim in enumerate(shape):
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Shape dimension {i} must be positive integer, got {dim}")

            if dim > 10000:  # Reasonable upper bound
                raise ValueError(f"Shape dimension {i} too large: {dim}")

        return tuple(shape)

    @staticmethod
    def validate_string(value: str, min_length: int = 0, max_length: int = 1000,
                       param_name: str = "value") -> bool:
        """Validate string with length constraints."""
        if not isinstance(value, str):
            return False

        if len(value) < min_length or len(value) > max_length:
            return False

        return True

    @staticmethod
    def validate_numeric(value: float, min_value: float = None, max_value: float = None) -> bool:
        """Validate numeric value with bounds checking."""
        try:
            num_val = float(value)
            if min_value is not None and num_val < min_value:
                return False
            if max_value is not None and num_val > max_value:
                return False
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_quantum_task_id(task_id: str) -> bool:
        """Validate quantum task ID format."""
        if not isinstance(task_id, str):
            return False

        # Task ID should be alphanumeric with hyphens, reasonable length
        pattern = r'^[a-zA-Z0-9\-_]{1,64}$'
        return bool(re.match(pattern, task_id))

    @staticmethod
    def validate_priority(priority: int) -> bool:
        """Validate task priority value."""
        try:
            pri_val = int(priority)
            return 1 <= pri_val <= 10  # Priority range 1-10
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_complexity(complexity: float) -> bool:
        """Validate task complexity value."""
        try:
            comp_val = float(complexity)
            return 0.0 <= comp_val <= 1.0  # Complexity normalized 0-1
        except (ValueError, TypeError):
            return False


class SecurityLoggingFilter(logging.Filter):
    """Logging filter to prevent sensitive data exposure."""

    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            r'password\s*=\s*\S+',
            r'token\s*=\s*\S+',
            r'key\s*=\s*\S+',
            r'/home/[^/]+',  # User home directories
            r'Bearer\s+\S+',  # API tokens
        ]

    def filter(self, record):
        """Filter out sensitive information from log records."""
        if hasattr(record, 'msg'):
            message = str(record.msg)
            for pattern in self.sensitive_patterns:
                message = re.sub(pattern, '[REDACTED]', message, flags=re.IGNORECASE)
            record.msg = message
        return True


class DataSanitizer:
    """Data sanitization utilities."""

    @staticmethod
    def sanitize_string(value: Any, max_length: int = 1000, min_length: int = 0) -> str:
        """Sanitize string input with length validation and safe character filtering."""
        if value is None:
            return ""

        # Convert to string if not already
        if not isinstance(value, str):
            value = str(value)

        # Remove potentially dangerous characters
        # Allow alphanumeric, spaces, common punctuation, but remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', value)

        # Trim whitespace
        sanitized = sanitized.strip()

        # Enforce length constraints
        if len(sanitized) < min_length:
            sanitized = sanitized.ljust(min_length, ' ')

        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized

    @staticmethod
    def sanitize_numeric(value: Any, min_value: Optional[float] = None,
                        max_value: Optional[float] = None) -> float:
        """Sanitize numeric input with bounds checking."""
        try:
            # Convert to float
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                cleaned = re.sub(r'[^0-9.\-]', '', value)
                if not cleaned or cleaned in ['.', '-', '-.']:
                    raise ValueError("Invalid numeric string")
                num_value = float(cleaned)
            else:
                num_value = float(value)

            # Check for infinity and NaN
            if not isinstance(num_value, (int, float)) or str(num_value).lower() in ['inf', '-inf', 'nan']:
                raise ValueError("Invalid numeric value")

            # Apply bounds checking
            if min_value is not None and num_value < min_value:
                num_value = min_value

            if max_value is not None and num_value > max_value:
                num_value = max_value

            return num_value

        except (ValueError, TypeError, OverflowError):
            # Return safe default value
            if min_value is not None:
                return min_value
            elif max_value is not None and max_value >= 0:
                return 0.0
            else:
                return 0.0

    @staticmethod
    def sanitize_dict(value: Any, max_keys: int = 100, max_key_length: int = 100,
                     max_value_length: int = 1000) -> Dict[str, Any]:
        """Sanitize dictionary input with size and content validation."""
        if value is None:
            return {}

        if not isinstance(value, dict):
            try:
                # Try to convert to dict if possible
                if hasattr(value, '__dict__'):
                    value = value.__dict__
                else:
                    return {}
            except:
                return {}

        sanitized_dict = {}
        key_count = 0

        for k, v in value.items():
            if key_count >= max_keys:
                break

            # Sanitize key
            clean_key = DataSanitizer.sanitize_string(str(k), max_length=max_key_length)
            if not clean_key:  # Skip empty keys
                continue

            # Sanitize value based on type
            if isinstance(v, str):
                clean_value = DataSanitizer.sanitize_string(v, max_length=max_value_length)
            elif isinstance(v, (int, float)):
                clean_value = DataSanitizer.sanitize_numeric(v)
            elif isinstance(v, dict):
                # Recursive sanitization for nested dicts (with reduced limits)
                clean_value = DataSanitizer.sanitize_dict(v, max_keys=min(10, max_keys//2))
            elif isinstance(v, (list, tuple)):
                # Sanitize list/tuple elements
                clean_value = [
                    DataSanitizer.sanitize_string(str(item), max_length=100)
                    if not isinstance(item, (int, float)) else DataSanitizer.sanitize_numeric(item)
                    for item in list(v)[:10]  # Limit list size
                ]
            else:
                # Convert other types to string and sanitize
                clean_value = DataSanitizer.sanitize_string(str(v), max_length=max_value_length)

            sanitized_dict[clean_key] = clean_value
            key_count += 1

        return sanitized_dict

    @staticmethod
    def sanitize_error_message(error: Exception) -> str:
        """Sanitize error messages to remove sensitive paths."""
        error_str = str(error)

        # Remove absolute paths
        error_str = re.sub(r'/[/\w.-]+/', '/[PATH]/', error_str)

        # Remove file extensions that might reveal system info
        error_str = re.sub(r'\.\w{2,4}(?=\s|$)', '.[EXT]', error_str)

        return error_str

    @staticmethod
    def sanitize_path_for_logging(path: str) -> str:
        """Sanitize file paths for safe logging."""
        path_obj = Path(path)
        return f"{path_obj.parent.name}/{path_obj.name}"


def create_secure_temp_dir(prefix: str = "tpu_bench") -> Path:
    """Create secure temporary directory with restricted permissions."""
    import stat
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))

    # Set restrictive permissions (owner only)
    temp_dir.chmod(stat.S_IRWXU)

    return temp_dir


def verify_file_integrity(filepath: Path, expected_hash: str = None) -> bool:
    """Verify file integrity using SHA-256 hash."""
    if not filepath.exists():
        return False

    if expected_hash is None:
        return True

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest() == expected_hash


class RateLimiter:
    """Rate limiting for API operations."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        import time

        current_time = time.time()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False

        # Record new request
        self.requests[identifier].append(current_time)
        return True
