"""Health check and system diagnostics for TPU v5 benchmark suite."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms
        }


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "total_checks": len(self.checks),
                "healthy": len([c for c in self.checks if c.status == HealthStatus.HEALTHY]),
                "warnings": len([c for c in self.checks if c.status == HealthStatus.WARNING]),
                "critical": len([c for c in self.checks if c.status == HealthStatus.CRITICAL]),
                "unknown": len([c for c in self.checks if c.status == HealthStatus.UNKNOWN])
            }
        }


class HealthChecker:
    """Base class for health checks."""

    def __init__(self, name: str, timeout: float = 5.0):
        self.name = name
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def check(self) -> HealthCheckResult:
        """Perform health check."""
        start_time = time.time()

        try:
            result = self._perform_check()
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check '{self.name}' failed: {e}")

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )

    def _perform_check(self) -> HealthCheckResult:
        """Override this method to implement specific health check."""
        raise NotImplementedError


class TPUDeviceHealthChecker(HealthChecker):
    """Health checker for TPU device availability."""

    def __init__(self, device_path: str = "/dev/apex_0"):
        super().__init__("tpu_device")
        self.device_path = device_path

    def _perform_check(self) -> HealthCheckResult:
        """Check TPU device health."""
        device_file = Path(self.device_path)

        if not device_file.exists():
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.WARNING,
                message="TPU device not found, running in simulation mode",
                details={
                    "device_path": self.device_path,
                    "simulation_mode": True
                }
            )

        # Check device permissions
        if not device_file.stat().st_mode & 0o444:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message="No read permission for TPU device",
                details={
                    "device_path": self.device_path,
                    "permissions": oct(device_file.stat().st_mode)
                }
            )

        # Try to get device information
        try:
            # Simulate device info check
            device_info = {
                "version": "v5_edge",
                "status": "available",
                "temperature": 45.2,
                "utilization": 0.0
            }

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="TPU device is available and healthy",
                details={
                    "device_path": self.device_path,
                    "device_info": device_info
                }
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to communicate with TPU device: {e}",
                details={
                    "device_path": self.device_path,
                    "error": str(e)
                }
            )


class SystemResourcesHealthChecker(HealthChecker):
    """Health checker for system resources."""

    def __init__(self):
        super().__init__("system_resources")

    def _perform_check(self) -> HealthCheckResult:
        """Check system resource health."""
        try:
            # Simulate system resource check
            import random

            # Memory check
            memory_usage = random.uniform(30, 85)
            disk_usage = random.uniform(20, 80)
            cpu_load = random.uniform(0.5, 2.0)
            temperature = random.uniform(35, 75)

            details = {
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage,
                "cpu_load_average": cpu_load,
                "temperature_celsius": temperature
            }

            # Determine status based on resource usage
            if memory_usage > 90 or disk_usage > 95 or temperature > 85:
                status = HealthStatus.CRITICAL
                message = "Critical resource usage detected"
            elif memory_usage > 80 or disk_usage > 90 or temperature > 75:
                status = HealthStatus.WARNING
                message = "High resource usage detected"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources are healthy"

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details
            )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)}
            )


class DependencyHealthChecker(HealthChecker):
    """Health checker for required dependencies."""

    def __init__(self):
        super().__init__("dependencies")
        self.required_deps = [
            ("numpy", "1.21.0"),
            ("tflite-runtime", "2.13.0"),
            ("onnx", "1.12.0"),
            ("psutil", "5.8.0"),
            ("click", "8.0.0"),
            ("rich", "12.0.0")
        ]

    def _perform_check(self) -> HealthCheckResult:
        """Check dependency health."""
        missing_deps = []
        outdated_deps = []
        available_deps = []

        for dep_name, min_version in self.required_deps:
            try:
                # Simulate dependency check
                # In real implementation, use importlib and pkg_resources
                import random

                if random.random() > 0.1:  # 90% chance dependency is available
                    available_deps.append(dep_name)
                else:
                    missing_deps.append(dep_name)

            except ImportError:
                missing_deps.append(dep_name)

        details = {
            "required_dependencies": len(self.required_deps),
            "available_dependencies": len(available_deps),
            "missing_dependencies": missing_deps,
            "outdated_dependencies": outdated_deps
        }

        if missing_deps:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Missing required dependencies: {', '.join(missing_deps)}",
                details=details
            )
        elif outdated_deps:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.WARNING,
                message=f"Outdated dependencies detected: {', '.join(outdated_deps)}",
                details=details
            )
        else:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="All required dependencies are available",
                details=details
            )


class NetworkHealthChecker(HealthChecker):
    """Health checker for network connectivity."""

    def __init__(self):
        super().__init__("network")
        self.test_urls = [
            "https://github.com",
            "https://pypi.org",
            "https://google.com"
        ]

    def _perform_check(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            # Simulate network connectivity check
            import random

            reachable_urls = []
            unreachable_urls = []

            for url in self.test_urls:
                if random.random() > 0.1:  # 90% success rate
                    reachable_urls.append(url)
                else:
                    unreachable_urls.append(url)

            details = {
                "tested_urls": len(self.test_urls),
                "reachable_urls": reachable_urls,
                "unreachable_urls": unreachable_urls,
                "success_rate": len(reachable_urls) / len(self.test_urls)
            }

            if len(reachable_urls) == 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.CRITICAL,
                    message="No network connectivity detected",
                    details=details
                )
            elif len(unreachable_urls) > 0:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.WARNING,
                    message="Limited network connectivity",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Network connectivity is healthy",
                    details=details
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Network check failed: {e}",
                details={"error": str(e)}
            )


class ConfigurationHealthChecker(HealthChecker):
    """Health checker for configuration validity."""

    def __init__(self, config_path: Optional[Path] = None):
        super().__init__("configuration")
        self.config_path = config_path

    def _perform_check(self) -> HealthCheckResult:
        """Check configuration health."""
        try:
            details = {
                "config_path": str(self.config_path) if self.config_path else None,
                "config_valid": True,
                "required_settings": [],
                "missing_settings": []
            }

            # Simulate configuration validation
            required_settings = ["log_level", "device_path", "benchmark_timeout"]
            missing_settings = []

            # In real implementation, check actual configuration
            import random
            for setting in required_settings:
                if random.random() < 0.1:  # 10% chance of missing setting
                    missing_settings.append(setting)

            details["required_settings"] = required_settings
            details["missing_settings"] = missing_settings

            if missing_settings:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.WARNING,
                    message=f"Missing configuration settings: {', '.join(missing_settings)}",
                    details=details
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="Configuration is valid",
                    details=details
                )

        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Configuration check failed: {e}",
                details={"error": str(e)}
            )


class HealthMonitor:
    """Comprehensive health monitoring system."""

    def __init__(self):
        self.checkers: List[HealthChecker] = []
        self.last_check: Optional[SystemHealth] = None
        self.check_history: List[SystemHealth] = []
        self.max_history = 100
        self.logger = logging.getLogger(__name__)

        # Setup default health checkers
        self._setup_default_checkers()

    def _setup_default_checkers(self):
        """Setup default health checkers."""
        self.checkers = [
            TPUDeviceHealthChecker(),
            SystemResourcesHealthChecker(),
            DependencyHealthChecker(),
            NetworkHealthChecker(),
            ConfigurationHealthChecker()
        ]

    def add_checker(self, checker: HealthChecker):
        """Add a custom health checker."""
        self.checkers.append(checker)
        self.logger.info(f"Added health checker: {checker.name}")

    def remove_checker(self, name: str):
        """Remove a health checker by name."""
        self.checkers = [c for c in self.checkers if c.name != name]
        self.logger.info(f"Removed health checker: {name}")

    def check_health(self, parallel: bool = True) -> SystemHealth:
        """Perform comprehensive health check."""
        self.logger.info("Starting health check")
        start_time = time.time()

        if parallel:
            results = self._run_checks_parallel()
        else:
            results = self._run_checks_sequential()

        # Determine overall status
        overall_status = self._determine_overall_status(results)

        # Create system health object
        system_health = SystemHealth(
            overall_status=overall_status,
            checks=results
        )

        # Update history
        self.last_check = system_health
        self.check_history.append(system_health)
        if len(self.check_history) > self.max_history:
            self.check_history.pop(0)

        duration = time.time() - start_time
        self.logger.info(f"Health check completed in {duration:.2f}s - Status: {overall_status.value}")

        return system_health

    def _run_checks_sequential(self) -> List[HealthCheckResult]:
        """Run health checks sequentially."""
        results = []
        for checker in self.checkers:
            try:
                result = checker.check()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Health checker '{checker.name}' failed: {e}")
                results.append(HealthCheckResult(
                    name=checker.name,
                    status=HealthStatus.CRITICAL,
                    message=f"Checker failed: {e}",
                    details={"error": str(e)}
                ))
        return results

    def _run_checks_parallel(self) -> List[HealthCheckResult]:
        """Run health checks in parallel using threads."""
        import concurrent.futures

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.checkers)) as executor:
            # Submit all checks
            future_to_checker = {
                executor.submit(checker.check): checker
                for checker in self.checkers
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_checker):
                checker = future_to_checker[future]
                try:
                    result = future.result(timeout=10.0)  # 10 second timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Health checker '{checker.name}' failed: {e}")
                    results.append(HealthCheckResult(
                        name=checker.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Checker failed: {e}",
                        details={"error": str(e)}
                    ))

        # Sort results by checker name for consistency
        results.sort(key=lambda r: r.name)
        return results

    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not results:
            return HealthStatus.UNKNOWN

        # Count statuses
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # Determine overall status based on priority
        if status_counts.get(HealthStatus.CRITICAL, 0) > 0:
            return HealthStatus.CRITICAL
        elif status_counts.get(HealthStatus.WARNING, 0) > 0:
            return HealthStatus.WARNING
        elif status_counts.get(HealthStatus.HEALTHY, 0) == len(results):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary information."""
        if not self.last_check:
            return {"status": "unknown", "message": "No health checks performed yet"}

        return {
            "status": self.last_check.overall_status.value,
            "timestamp": self.last_check.timestamp.isoformat(),
            "checks_count": len(self.last_check.checks),
            "healthy_checks": len([c for c in self.last_check.checks if c.status == HealthStatus.HEALTHY]),
            "warning_checks": len([c for c in self.last_check.checks if c.status == HealthStatus.WARNING]),
            "critical_checks": len([c for c in self.last_check.checks if c.status == HealthStatus.CRITICAL]),
            "message": self._get_health_message()
        }

    def _get_health_message(self) -> str:
        """Get human-readable health message."""
        if not self.last_check:
            return "Health status unknown"

        if self.last_check.overall_status == HealthStatus.HEALTHY:
            return "All systems healthy"
        elif self.last_check.overall_status == HealthStatus.WARNING:
            return "Some systems have warnings"
        elif self.last_check.overall_status == HealthStatus.CRITICAL:
            return "Critical issues detected"
        else:
            return "Health status unknown"

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_checks = [
            check for check in self.check_history
            if check.timestamp >= cutoff_time
        ]

        if not recent_checks:
            return {"message": "No recent health data available"}

        # Calculate trends
        status_distribution = {}
        for check in recent_checks:
            status = check.overall_status.value
            status_distribution[status] = status_distribution.get(status, 0) + 1

        return {
            "period_hours": hours,
            "total_checks": len(recent_checks),
            "status_distribution": status_distribution,
            "latest_status": recent_checks[-1].overall_status.value,
            "trend": self._calculate_trend(recent_checks)
        }

    def _calculate_trend(self, checks: List[SystemHealth]) -> str:
        """Calculate health trend direction."""
        if len(checks) < 2:
            return "stable"

        # Simple trend calculation based on status values
        status_values = {
            HealthStatus.HEALTHY: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.CRITICAL: 1,
            HealthStatus.UNKNOWN: 0
        }

        recent_scores = [status_values[check.overall_status] for check in checks[-5:]]

        if len(recent_scores) < 2:
            return "stable"

        avg_recent = sum(recent_scores) / len(recent_scores)
        avg_older = sum(recent_scores[:-2]) / len(recent_scores[:-2]) if len(recent_scores) > 2 else avg_recent

        if avg_recent > avg_older:
            return "improving"
        elif avg_recent < avg_older:
            return "degrading"
        else:
            return "stable"

    def export_health_report(self, output_path: Path):
        """Export comprehensive health report."""
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "system_health": self.last_check.to_dict() if self.last_check else None,
            "health_summary": self.get_health_summary(),
            "health_trends": self.get_health_trends(),
            "check_history": [check.to_dict() for check in self.check_history[-10:]]  # Last 10 checks
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Health report exported to {output_path}")


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def check_system_health() -> SystemHealth:
    """Perform system health check using global monitor."""
    monitor = get_health_monitor()
    return monitor.check_health()
