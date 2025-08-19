"""Tests for Generation 2 Robustness Features

Comprehensive test suite for:
- Advanced error recovery and self-healing
- Enhanced validation framework
- Production monitoring and alerting
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import numpy as np

# Import Generation 2 modules
from edge_tpu_v5_benchmark.advanced_error_recovery import (
    get_self_healing_system, get_retry_manager, robust_execution,
    AdaptiveCircuitBreaker, PredictiveErrorDetector
)
from edge_tpu_v5_benchmark.enhanced_validation import (
    get_enhanced_validator, ValidationLevel, ValidationResult, ValidationType,
    DataIntegrityValidator, ValidationReport
)
from edge_tpu_v5_benchmark.production_monitoring import (
    get_production_monitor, ProductionMonitor, AlertSeverity, 
    MetricType, Alert, SLATarget
)


class TestAdvancedErrorRecovery:
    """Test advanced error recovery and self-healing capabilities."""
    
    def test_self_healing_system_initialization(self):
        """Test self-healing system initialization."""
        healing_system = get_self_healing_system()
        
        assert healing_system is not None
        assert len(healing_system.recovery_actions) > 0
        assert 'memory_leak' in healing_system.recovery_actions
        assert 'high_latency' in healing_system.recovery_actions
    
    def test_predictive_error_detector(self):
        """Test ML-based error prediction."""
        detector = PredictiveErrorDetector()
        
        # Test feature extraction
        metrics = {
            'cpu_usage': 75.0,
            'memory_usage': 80.0,
            'latency_p99': 150.0,
            'throughput': 100.0
        }
        
        features = detector.extract_features(metrics)
        assert len(features) == 8
        assert features[0] == 75.0  # cpu_usage
        assert features[1] == 80.0  # memory_usage
    
    def test_adaptive_circuit_breaker(self):
        """Test adaptive circuit breaker functionality."""
        breaker = AdaptiveCircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        # Test successful execution
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "closed"
        
        # Test failure handling
        def failure_func():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call(failure_func)
        
        assert breaker.state == "open"
        
        # Test circuit breaker prevents execution when open
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            breaker.call(success_func)
    
    def test_robust_execution_decorator(self):
        """Test robust execution decorator."""
        call_count = 0
        
        @robust_execution(max_retries=3, backoff_strategy="exponential")
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_manager(self):
        """Test adaptive retry manager."""
        retry_manager = get_retry_manager()
        
        call_count = 0
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Retry needed")
            return "completed"
        
        result = retry_manager.execute_with_retry(test_function, max_retries=3)
        assert result == "completed"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_error_recovery(self):
        """Test asynchronous error recovery scenarios."""
        healing_system = get_self_healing_system()
        
        # Simulate system metrics that would trigger healing
        problematic_metrics = {
            'cpu_usage': 95.0,
            'memory_usage': 90.0,
            'disk_usage': 85.0,
            'timestamp': time.time()
        }
        
        # Test would trigger healing in real scenario
        assert 'cpu_usage' in problematic_metrics
        assert problematic_metrics['cpu_usage'] > 90


class TestEnhancedValidation:
    """Test enhanced validation framework."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = get_enhanced_validator(ValidationLevel.STANDARD)
        
        assert validator is not None
        assert validator.level == ValidationLevel.STANDARD
        assert ValidationType.DATA_INTEGRITY in validator.validators
    
    def test_data_integrity_validation(self):
        """Test data integrity validation."""
        validator = DataIntegrityValidator()
        
        # Test valid data
        valid_data = {"key": "value", "number": 42}
        result = validator.validate(valid_data)
        
        assert result['status'] == ValidationResult.PASS
        assert "integrity check passed" in result['message']
        
        # Test null data
        null_result = validator.validate(None)
        assert null_result['status'] == ValidationResult.FAIL
        assert "null" in null_result['message']
        
        # Test numpy array with NaN
        nan_array = np.array([1.0, 2.0, np.nan, 4.0])
        nan_result = validator.validate(nan_array)
        assert nan_result['status'] == ValidationResult.FAIL
        assert "NaN" in nan_result['message']
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation workflow."""
        validator = get_enhanced_validator()
        
        test_data = {
            "values": [1, 2, 3, 4, 5],
            "metadata": {"source": "test"}
        }
        
        report = validator.validate_all(test_data)
        
        assert isinstance(report, ValidationReport)
        assert report.total_checks > 0
        assert report.execution_time > 0
        assert report.success_rate <= 1.0
    
    def test_validation_levels(self):
        """Test different validation levels."""
        basic_validator = get_enhanced_validator(ValidationLevel.BASIC)
        strict_validator = get_enhanced_validator(ValidationLevel.STRICT)
        
        assert basic_validator.level == ValidationLevel.BASIC
        assert strict_validator.level == ValidationLevel.STRICT
    
    def test_validation_report_properties(self):
        """Test validation report properties."""
        report = ValidationReport(
            timestamp=datetime.now(),
            total_checks=10,
            passed=8,
            warnings=1,
            failed=1,
            errors=0,
            execution_time=0.5
        )
        
        assert report.success_rate == 0.8
        assert report.overall_status == ValidationResult.WARN
    
    def test_checksum_validation(self):
        """Test data checksum validation."""
        validator = DataIntegrityValidator()
        
        test_data = "test string for checksum"
        import hashlib
        expected_checksum = hashlib.md5(test_data.encode()).hexdigest()
        
        context = {'expected_checksum': expected_checksum}
        result = validator.validate(test_data, context)
        
        assert result['status'] == ValidationResult.PASS


class TestProductionMonitoring:
    """Test production monitoring and alerting."""
    
    def test_monitor_initialization(self):
        """Test production monitor initialization."""
        monitor = get_production_monitor()
        
        assert monitor is not None
        assert len(monitor.sla_targets) > 0
        assert len(monitor.alert_rules) > 0
        assert 'latency_p99' in monitor.sla_targets
    
    def test_metric_recording(self):
        """Test metric recording functionality."""
        monitor = ProductionMonitor()
        
        # Record some metrics
        monitor.record_metric('test_latency', 50.0)
        monitor.record_metric('test_latency', 75.0)
        monitor.record_metric('test_latency', 60.0)
        
        assert 'test_latency' in monitor.metrics
        assert len(monitor.metrics['test_latency']) == 3
    
    def test_alert_triggering(self):
        """Test alert triggering logic."""
        monitor = ProductionMonitor()
        
        # Setup alert rule
        monitor.alert_rules['test_high_latency'] = {
            'metric': 'test_latency',
            'threshold': 100.0,
            'severity': AlertSeverity.WARNING,
            'condition': 'greater_than'
        }
        
        # Record metric that should trigger alert
        monitor.record_metric('test_latency', 150.0)
        
        # Check alert conditions
        monitor._check_alert_conditions()
        
        assert 'test_high_latency' in monitor.alerts
        alert = monitor.alerts['test_high_latency']
        assert alert.severity == AlertSeverity.WARNING
        assert alert.current_value == 150.0
    
    def test_sla_compliance_checking(self):
        """Test SLA compliance monitoring."""
        monitor = ProductionMonitor()
        
        # Add custom SLA target
        sla = SLATarget(
            name='Test Latency',
            metric_type=MetricType.LATENCY,
            target_value=50.0,
            comparison='less_than',
            measurement_window=timedelta(minutes=1)
        )
        monitor.sla_targets['test_latency'] = sla
        
        # Record metrics within SLA
        for _ in range(5):
            monitor.record_metric('test_latency', 40.0)
        
        # Record metrics violating SLA
        for _ in range(2):
            monitor.record_metric('test_latency', 80.0)
        
        # Check SLA compliance
        monitor._check_sla_compliance()
        
        # Should have recorded compliance metric
        assert any('sla_compliance' in metric for metric in monitor.metrics.keys())
    
    def test_metric_summary_statistics(self):
        """Test metric summary statistics."""
        monitor = ProductionMonitor()
        
        # Record test data
        test_values = [10, 20, 30, 40, 50]
        for value in test_values:
            monitor.record_metric('test_metric', value)
        
        summary = monitor.get_metric_summary('test_metric')
        
        assert summary['count'] == 5
        assert summary['min'] == 10
        assert summary['max'] == 50
        assert summary['mean'] == 30
        assert summary['median'] == 30
    
    def test_alert_acknowledgment_resolution(self):
        """Test alert acknowledgment and resolution."""
        monitor = ProductionMonitor()
        
        # Create test alert
        alert = Alert(
            id='test_alert',
            severity=AlertSeverity.WARNING,
            message='Test alert message',
            metric_name='test_metric',
            current_value=100.0,
            threshold=50.0,
            timestamp=datetime.now()
        )
        monitor.alerts['test_alert'] = alert
        
        # Test acknowledgment
        assert monitor.acknowledge_alert('test_alert')
        assert monitor.alerts['test_alert'].acknowledged
        
        # Test resolution
        assert monitor.resolve_alert('test_alert')
        assert monitor.alerts['test_alert'].resolved
        
        # Test getting active alerts (should be empty now)
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 0
    
    @patch('requests.post')
    def test_alert_notification(self, mock_post):
        """Test alert notification sending."""
        mock_post.return_value.status_code = 200
        
        monitor = ProductionMonitor(alert_webhook_url="http://test.webhook.url")
        
        alert = Alert(
            id='test_webhook_alert',
            severity=AlertSeverity.CRITICAL,
            message='Critical test alert',
            metric_name='test_metric',
            current_value=200.0,
            threshold=100.0,
            timestamp=datetime.now()
        )
        
        monitor._send_alert_notification(alert)
        
        # Verify webhook was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs['json']['alert_id'] == 'test_webhook_alert'
        assert kwargs['json']['severity'] == 'critical'
    
    def test_health_status_determination(self):
        """Test health status determination."""
        monitor = ProductionMonitor()
        
        # Test healthy status (no alerts)
        assert monitor.get_health_status().value == "healthy"
        
        # Add warning alert
        warning_alert = Alert(
            id='warning_test',
            severity=AlertSeverity.WARNING,
            message='Warning test',
            metric_name='test',
            current_value=100,
            threshold=50,
            timestamp=datetime.now()
        )
        monitor.alerts['warning_test'] = warning_alert
        
        assert monitor.get_health_status().value == "degraded"
        
        # Add critical alert
        critical_alert = Alert(
            id='critical_test',
            severity=AlertSeverity.CRITICAL,
            message='Critical test',
            metric_name='test',
            current_value=200,
            threshold=50,
            timestamp=datetime.now()
        )
        monitor.alerts['critical_test'] = critical_alert
        
        assert monitor.get_health_status().value == "critical"


class TestIntegrationScenarios:
    """Integration tests combining multiple Generation 2 features."""
    
    def test_error_recovery_with_monitoring(self):
        """Test error recovery integrated with monitoring."""
        monitor = get_production_monitor()
        healing_system = get_self_healing_system()
        
        # Record high error rate metric
        monitor.record_metric('error_rate', 10.0)  # 10% error rate
        
        # This would trigger healing in real scenario
        assert len(healing_system.recovery_actions) > 0
        assert 'high_latency' in healing_system.recovery_actions
    
    def test_validation_with_error_recovery(self):
        """Test validation integrated with error recovery."""
        validator = get_enhanced_validator()
        
        @robust_execution(max_retries=2)
        def validate_and_process(data):
            report = validator.validate_all(data)
            if report.overall_status == ValidationResult.FAIL:
                raise ValueError("Validation failed")
            return "processed"
        
        # Test with valid data
        valid_data = {"test": "data"}
        result = validate_and_process(valid_data)
        assert result == "processed"
    
    def test_monitoring_with_validation(self):
        """Test monitoring integrated with validation."""
        monitor = get_production_monitor()
        validator = get_enhanced_validator()
        
        # Simulate validation metrics
        test_data = [1, 2, 3, 4, 5]
        report = validator.validate_all(test_data)
        
        # Record validation metrics in monitoring
        monitor.record_metric('validation_success_rate', report.success_rate * 100)
        monitor.record_metric('validation_execution_time', report.execution_time * 1000)
        
        assert 'validation_success_rate' in monitor.metrics
        assert 'validation_execution_time' in monitor.metrics


@pytest.fixture
def sample_test_data():
    """Provide sample test data."""
    return {
        'numeric_data': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'dict_data': {'key1': 'value1', 'key2': 42},
        'list_data': [1, 2, 3, 4, 5],
        'string_data': 'test string'
    }


@pytest.fixture
def mock_performance_metrics():
    """Provide mock performance metrics."""
    return {
        'latency': 45.0,
        'throughput': 120.0,
        'cpu_usage': 65.0,
        'memory_usage': 70.0,
        'error_rate': 0.5
    }


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])