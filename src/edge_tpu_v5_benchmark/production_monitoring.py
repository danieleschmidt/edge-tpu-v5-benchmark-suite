"""Production Monitoring and Alerting System for TPU v5 Benchmark Suite

Enhanced Generation 2 monitoring capabilities:
- Real-time performance monitoring
- Automated alerting and escalation
- SLA compliance tracking
- Advanced metrics collection
- Health dashboard integration
"""

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import psutil
import requests

from .advanced_error_recovery import HealthStatus
from .enhanced_validation import ValidationResult


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being monitored."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    AVAILABILITY = "availability"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Alert definition and tracking."""
    id: str
    severity: AlertSeverity
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """Service Level Agreement target definition."""
    name: str
    metric_type: MetricType
    target_value: float
    comparison: str  # "less_than", "greater_than", "equals"
    measurement_window: timedelta
    tolerance_percent: float = 5.0
    enabled: bool = True


class ProductionMonitor:
    """Production monitoring and alerting system."""
    
    def __init__(self, alert_webhook_url: Optional[str] = None):
        self.metrics = defaultdict(deque)
        self.alerts = {}
        self.sla_targets = {}
        self.alert_rules = {}
        self.alert_webhook_url = alert_webhook_url
        
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Default SLA targets
        self._setup_default_slas()
        self._setup_default_alert_rules()
        
    def _setup_default_slas(self):
        """Setup default SLA targets."""
        self.sla_targets = {
            'latency_p99': SLATarget(
                name='P99 Latency',
                metric_type=MetricType.LATENCY,
                target_value=100.0,  # 100ms
                comparison='less_than',
                measurement_window=timedelta(minutes=5)
            ),
            'throughput': SLATarget(
                name='Minimum Throughput',
                metric_type=MetricType.THROUGHPUT,
                target_value=10.0,  # 10 requests/sec
                comparison='greater_than',
                measurement_window=timedelta(minutes=5)
            ),
            'error_rate': SLATarget(
                name='Error Rate',
                metric_type=MetricType.ERROR_RATE,
                target_value=1.0,  # 1%
                comparison='less_than',
                measurement_window=timedelta(minutes=5)
            ),
            'availability': SLATarget(
                name='System Availability',
                metric_type=MetricType.AVAILABILITY,
                target_value=99.9,  # 99.9%
                comparison='greater_than',
                measurement_window=timedelta(hours=1)
            )
        }
    
    def _setup_default_alert_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = {
            'high_latency': {
                'metric': 'latency_p99',
                'threshold': 200.0,
                'severity': AlertSeverity.WARNING,
                'condition': 'greater_than'
            },
            'critical_latency': {
                'metric': 'latency_p99',
                'threshold': 500.0,
                'severity': AlertSeverity.CRITICAL,
                'condition': 'greater_than'
            },
            'low_throughput': {
                'metric': 'throughput',
                'threshold': 5.0,
                'severity': AlertSeverity.WARNING,
                'condition': 'less_than'
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'threshold': 5.0,
                'severity': AlertSeverity.CRITICAL,
                'condition': 'greater_than'
            },
            'system_down': {
                'metric': 'availability',
                'threshold': 95.0,
                'severity': AlertSeverity.EMERGENCY,
                'condition': 'less_than'
            }
        }
    
    def start_monitoring(self):
        """Start production monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop production monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_system_metrics()
                
                # Record metrics
                for metric_name, value in current_metrics.items():
                    self.record_metric(metric_name, value)
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Check SLA compliance
                self._check_sla_compliance()
                
                # Sleep for monitoring interval
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(30)  # Back off on errors
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            }
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            return {}
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Record a metric value."""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_data = {
            'value': value,
            'timestamp': timestamp
        }
        
        # Keep only recent data (last hour)
        self.metrics[metric_name].append(metric_data)
        
        # Cleanup old data
        cutoff_time = datetime.now() - timedelta(hours=1)
        while (self.metrics[metric_name] and 
               self.metrics[metric_name][0]['timestamp'] < cutoff_time):
            self.metrics[metric_name].popleft()
    
    def _check_alert_conditions(self):
        """Check if any alert conditions are met."""
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric']
            
            if metric_name in self.metrics and self.metrics[metric_name]:
                latest_value = self.metrics[metric_name][-1]['value']
                threshold = rule['threshold']
                condition = rule['condition']
                
                should_alert = False
                if condition == 'greater_than' and latest_value > threshold:
                    should_alert = True
                elif condition == 'less_than' and latest_value < threshold:
                    should_alert = True
                elif condition == 'equals' and abs(latest_value - threshold) < 0.001:
                    should_alert = True
                
                if should_alert:
                    self._trigger_alert(
                        rule_name,
                        rule['severity'],
                        f"{metric_name} {condition} {threshold}: current value {latest_value:.2f}",
                        metric_name,
                        latest_value,
                        threshold
                    )
    
    def _trigger_alert(self, alert_id: str, severity: AlertSeverity, message: str,
                      metric_name: str, current_value: float, threshold: float):
        """Trigger an alert."""
        # Check if this alert is already active
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            return
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.alerts[alert_id] = alert
        
        # Log the alert
        logging.warning(f"ALERT [{severity.value.upper()}] {alert_id}: {message}")
        
        # Send alert notification
        self._send_alert_notification(alert)
        
        # Auto-escalate critical alerts
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            self._escalate_alert(alert)
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification."""
        if not self.alert_webhook_url:
            return
        
        try:
            payload = {
                'alert_id': alert.id,
                'severity': alert.severity.value,
                'message': alert.message,
                'metric': alert.metric_name,
                'current_value': alert.current_value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(
                self.alert_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logging.info(f"Alert notification sent for {alert.id}")
            else:
                logging.error(f"Failed to send alert notification: {response.status_code}")
                
        except Exception as e:
            logging.error(f"Error sending alert notification: {e}")
    
    def _escalate_alert(self, alert: Alert):
        """Escalate critical alerts."""
        alert.escalated = True
        logging.critical(f"ESCALATED ALERT: {alert.message}")
        
        # Additional escalation logic (emails, SMS, etc.) would go here
    
    def _check_sla_compliance(self):
        """Check SLA compliance for all targets."""
        for sla_name, sla in self.sla_targets.items():
            if not sla.enabled:
                continue
                
            metric_name = sla.name.lower().replace(' ', '_')
            if metric_name not in self.metrics:
                continue
            
            # Get metrics within the measurement window
            cutoff_time = datetime.now() - sla.measurement_window
            recent_metrics = [
                m for m in self.metrics[metric_name]
                if m['timestamp'] >= cutoff_time
            ]
            
            if not recent_metrics:
                continue
            
            # Calculate compliance
            values = [m['value'] for m in recent_metrics]
            if sla.comparison == 'less_than':
                compliance = sum(1 for v in values if v <= sla.target_value) / len(values) * 100
            elif sla.comparison == 'greater_than':
                compliance = sum(1 for v in values if v >= sla.target_value) / len(values) * 100
            else:  # equals
                tolerance = sla.target_value * sla.tolerance_percent / 100
                compliance = sum(1 for v in values 
                               if abs(v - sla.target_value) <= tolerance) / len(values) * 100
            
            # Record SLA compliance
            self.record_metric(f'sla_compliance_{sla_name}', compliance)
            
            # Alert if SLA is breached
            if compliance < (100 - sla.tolerance_percent):
                self._trigger_alert(
                    f'sla_breach_{sla_name}',
                    AlertSeverity.WARNING,
                    f"SLA breach for {sla.name}: {compliance:.1f}% compliance",
                    f'sla_compliance_{sla_name}',
                    compliance,
                    100 - sla.tolerance_percent
                )
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logging.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logging.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_metric_summary(self, metric_name: str, 
                          window: timedelta = timedelta(minutes=5)) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - window
        recent_values = [
            m['value'] for m in self.metrics[metric_name]
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        import statistics
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
        }
    
    def get_health_status(self) -> HealthStatus:
        """Get overall system health status."""
        active_alerts = self.get_active_alerts()
        
        if not active_alerts:
            return HealthStatus.HEALTHY
        
        # Check for emergency or critical alerts
        for alert in active_alerts:
            if alert.severity == AlertSeverity.EMERGENCY:
                return HealthStatus.FAILED
            elif alert.severity == AlertSeverity.CRITICAL:
                return HealthStatus.CRITICAL
        
        # Check for warning alerts
        if any(alert.severity == AlertSeverity.WARNING for alert in active_alerts):
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY


# Global monitor instance
_global_monitor = None


def get_production_monitor(alert_webhook_url: Optional[str] = None) -> ProductionMonitor:
    """Get global production monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ProductionMonitor(alert_webhook_url)
    return _global_monitor


def monitor_performance(metric_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_production_monitor()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                monitor.record_metric(f'{metric_name}_latency', execution_time)
                monitor.record_metric(f'{metric_name}_success_rate', 100.0)
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                monitor.record_metric(f'{metric_name}_latency', execution_time)
                monitor.record_metric(f'{metric_name}_success_rate', 0.0)
                monitor.record_metric(f'{metric_name}_error_rate', 100.0)
                raise
        
        return wrapper
    return decorator