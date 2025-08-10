"""Security utilities for TPU v5 benchmark suite."""

import logging
import hashlib
import re
from typing import Any, Dict, Set, Optional
from pathlib import Path
import secrets
import hmac


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
    import tempfile
    import stat
    
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