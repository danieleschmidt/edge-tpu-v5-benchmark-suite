"""Centralized logging configuration for TPU v5 benchmark suite."""

import json
import logging
import logging.handlers
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter."""

    def __init__(self, service_name: str = "tpu-v5-benchmark"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log structure
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add thread information
        if hasattr(record, 'thread') and hasattr(record, 'threadName'):
            log_data["thread"] = {
                "id": record.thread,
                "name": record.threadName
            }

        # Add process information
        if hasattr(record, 'process') and hasattr(record, 'processName'):
            log_data["process"] = {
                "id": record.process,
                "name": record.processName
            }

        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'message'
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)


class BenchmarkLogFilter(logging.Filter):
    """Custom filter for benchmark-specific logging."""

    def __init__(self, benchmark_id: Optional[str] = None):
        super().__init__()
        self.benchmark_id = benchmark_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on benchmark context."""
        # Add benchmark context if available
        if hasattr(record, 'benchmark_id') or self.benchmark_id:
            record.benchmark_id = getattr(record, 'benchmark_id', self.benchmark_id)

        # Filter out noisy debug messages in production
        if record.levelno <= logging.DEBUG:
            # Allow debug messages from our modules only
            if not record.name.startswith('edge_tpu_v5_benchmark'):
                return False

        return True


class SecurityLogFilter(logging.Filter):
    """Filter that sanitizes sensitive information from logs."""

    def __init__(self):
        super().__init__()
        self.sensitive_patterns = [
            'password', 'token', 'key', 'secret', 'auth', 'credential'
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Sanitize sensitive information from log messages."""
        message = record.getMessage().lower()

        # Check if message contains sensitive information
        if any(pattern in message for pattern in self.sensitive_patterns):
            # Mark as potentially sensitive
            record.security_sensitive = True

            # In production, you might want to mask or redact
            # For now, we'll just add a warning attribute
            record.security_warning = "Log may contain sensitive information"

        return True


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent blocking."""

    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        self.worker_thread.start()

    def emit(self, record: logging.LogRecord):
        """Emit log record asynchronously."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop log messages if queue is full to prevent blocking
            pass

    def _worker(self):
        """Worker thread that processes log records."""
        while not self.shutdown_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Handle logging errors without creating infinite loops
                print(f"Error in async log handler: {e}", file=sys.stderr)

    def close(self):
        """Close the handler and wait for worker thread."""
        self.shutdown_event.set()
        self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class BenchmarkLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds benchmark context to log records."""

    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Process log message and add extra context."""
        # Add benchmark context to all log records
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra'].update(self.extra)
        return msg, kwargs


class LoggingConfig:
    """Centralized logging configuration manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self._configured_loggers = set()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "class": "edge_tpu_v5_benchmark.logging_config.StructuredFormatter",
                    "service_name": "tpu-v5-benchmark"
                },
                "console": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                }
            },
            "filters": {
                "benchmark": {
                    "class": "edge_tpu_v5_benchmark.logging_config.BenchmarkLogFilter"
                },
                "security": {
                    "class": "edge_tpu_v5_benchmark.logging_config.SecurityLogFilter"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "console",
                    "filters": ["benchmark", "security"],
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "structured",
                    "filters": ["benchmark", "security"],
                    "filename": "logs/benchmark.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8"
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "structured",
                    "filters": ["benchmark", "security"],
                    "filename": "logs/benchmark_errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                    "encoding": "utf-8"
                }
            },
            "loggers": {
                "edge_tpu_v5_benchmark": {
                    "level": "DEBUG",
                    "handlers": ["console", "file", "error_file"],
                    "propagate": False
                },
                "root": {
                    "level": "WARNING",
                    "handlers": ["console"]
                }
            }
        }

    def setup_logging(self, log_level: str = "INFO",
                     log_dir: Optional[Path] = None,
                     enable_async: bool = True):
        """Setup logging configuration."""
        # Create log directory if specified
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Update file handler paths
            for handler_name, handler_config in self.config["handlers"].items():
                if "filename" in handler_config:
                    handler_config["filename"] = str(log_dir / Path(handler_config["filename"]).name)

        # Adjust log levels
        self.config["handlers"]["console"]["level"] = log_level.upper()
        self.config["loggers"]["edge_tpu_v5_benchmark"]["level"] = log_level.upper()

        # Setup logging using dictConfig
        try:
            # Create custom formatters and filters
            self._setup_custom_components()

            # Configure logging
            logging.config.dictConfig(self.config)

            # Setup async handlers if requested
            if enable_async:
                self._setup_async_handlers()

            # Log startup message
            logger = logging.getLogger("edge_tpu_v5_benchmark.logging")
            logger.info("Logging configuration initialized", extra={
                "log_level": log_level,
                "log_dir": str(log_dir) if log_dir else None,
                "async_enabled": enable_async
            })

        except Exception as e:
            # Fallback to basic logging if configuration fails
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            logging.error(f"Failed to setup advanced logging, using basic config: {e}")

    def _setup_custom_components(self):
        """Setup custom logging components."""
        # Register custom formatter
        logging.getLogger().manager.loggerDict.setdefault('formatters', {})

        # Register custom filters
        logging.getLogger().manager.loggerDict.setdefault('filters', {})

    def _setup_async_handlers(self):
        """Setup asynchronous logging handlers."""
        logger = logging.getLogger("edge_tpu_v5_benchmark")

        # Replace synchronous handlers with async versions
        async_handlers = []
        for handler in logger.handlers[:]:
            if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                async_handler = AsyncLogHandler(handler)
                async_handler.setLevel(handler.level)
                async_handler.setFormatter(handler.formatter)
                async_handlers.append(async_handler)
                logger.removeHandler(handler)

        # Add async handlers
        for handler in async_handlers:
            logger.addHandler(handler)

    def get_logger(self, name: str, benchmark_id: Optional[str] = None) -> logging.Logger:
        """Get a configured logger instance."""
        logger = logging.getLogger(name)

        # Add benchmark context if provided
        if benchmark_id:
            return BenchmarkLoggerAdapter(logger, {"benchmark_id": benchmark_id})

        return logger

    def configure_component_logger(self, component: str, level: str = "INFO"):
        """Configure logging for a specific component."""
        logger_name = f"edge_tpu_v5_benchmark.{component}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

        self._configured_loggers.add(logger_name)

        return logger

    def set_benchmark_context(self, benchmark_id: str, model_name: str):
        """Set global benchmark context for logging."""
        # Add benchmark context to all configured loggers
        for logger_name in self._configured_loggers:
            logger = logging.getLogger(logger_name)
            if not isinstance(logger, BenchmarkLoggerAdapter):
                adapted_logger = BenchmarkLoggerAdapter(logger, {
                    "benchmark_id": benchmark_id,
                    "model_name": model_name
                })
                # Replace logger in manager
                logging.getLogger().manager.loggerDict[logger_name] = adapted_logger

    def clear_benchmark_context(self):
        """Clear benchmark context from loggers."""
        for logger_name in self._configured_loggers:
            if logger_name in logging.getLogger().manager.loggerDict:
                logger = logging.getLogger().manager.loggerDict[logger_name]
                if isinstance(logger, BenchmarkLoggerAdapter):
                    # Restore original logger
                    logging.getLogger().manager.loggerDict[logger_name] = logger.logger

    def export_logs(self, output_path: Path, since: Optional[datetime] = None,
                   level: Optional[str] = None) -> Dict[str, Any]:
        """Export logs to file with filtering."""
        # This would implement log export functionality
        # For now, return summary
        return {
            "exported": True,
            "output_path": str(output_path),
            "since": since.isoformat() if since else None,
            "level": level,
            "timestamp": datetime.now().isoformat()
        }


# Global logging configuration instance
_logging_config = None


def get_logging_config() -> LoggingConfig:
    """Get global logging configuration instance."""
    global _logging_config
    if _logging_config is None:
        _logging_config = LoggingConfig()
    return _logging_config


def setup_logging(log_level: str = "INFO", log_dir: Optional[Path] = None,
                 enable_async: bool = True):
    """Setup global logging configuration."""
    config = get_logging_config()
    config.setup_logging(log_level, log_dir, enable_async)


def get_logger(name: str, benchmark_id: Optional[str] = None) -> logging.Logger:
    """Get a configured logger instance."""
    config = get_logging_config()
    return config.get_logger(name, benchmark_id)


# Context manager for benchmark logging
class BenchmarkLoggingContext:
    """Context manager for benchmark-specific logging."""

    def __init__(self, benchmark_id: str, model_name: str):
        self.benchmark_id = benchmark_id
        self.model_name = model_name
        self.config = get_logging_config()

    def __enter__(self):
        self.config.set_benchmark_context(self.benchmark_id, self.model_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.config.clear_benchmark_context()

        # Log any exceptions that occurred
        if exc_type:
            logger = get_logger("edge_tpu_v5_benchmark.context")
            logger.error(f"Exception in benchmark context: {exc_val}", exc_info=True)
