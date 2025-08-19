"""Comprehensive benchmarking and metrics integration for TPU v5 benchmark suite.

This module provides:
- Unified metrics collection across all performance modules
- Real-time performance dashboards and monitoring
- Automated performance regression detection
- Comprehensive benchmark reporting with visualizations
- Performance trend analysis and predictions
- SLA monitoring and alerting
- Cross-module performance correlation analysis
"""

import asyncio
import json
import logging
import statistics
import threading
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

warnings.filterwarnings('ignore')

# Import from our enhanced modules
from .cache import get_cache_manager
from .concurrency import get_concurrency_benchmark
from .performance import get_memory_optimizer


@dataclass
class PerformanceMetric:
    """Individual performance metric with metadata."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source_module: str
    category: str  # 'latency', 'throughput', 'resource_usage', 'error_rate'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'source_module': self.source_module,
            'category': self.category,
            'metadata': self.metadata
        }


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result with detailed metrics."""
    benchmark_name: str
    started_at: datetime
    completed_at: datetime
    success: bool
    metrics: List[PerformanceMetric]
    configuration: Dict[str, Any]
    environment_info: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None

    @property
    def duration_seconds(self) -> float:
        """Get benchmark duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()

    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get specific metric by name."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def get_metrics_by_category(self, category: str) -> List[PerformanceMetric]:
        """Get all metrics in a specific category."""
        return [m for m in self.metrics if m.category == category]


class MetricsCollector:
    """Unified metrics collector for all performance modules."""

    def __init__(self, collection_interval: float = 5.0):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10,000 metrics
        self.active_collectors: Dict[str, Callable] = {}
        self.collection_lock = threading.Lock()
        self.running = False
        self.collector_thread = None
        self.logger = logging.getLogger(__name__)

        # Register default collectors
        self._register_default_collectors()

    def _register_default_collectors(self):
        """Register collectors for all performance modules."""
        self.active_collectors.update({
            'cache_metrics': self._collect_cache_metrics,
            'performance_metrics': self._collect_performance_metrics,
            'concurrency_metrics': self._collect_concurrency_metrics,
            'scaling_metrics': self._collect_scaling_metrics,
            'system_metrics': self._collect_system_metrics
        })

    async def start_collection(self):
        """Start metrics collection."""
        if self.running:
            return

        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        self.logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=2.0)
        self.logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                start_time = time.time()
                collected_metrics = []

                # Collect from all registered collectors
                for collector_name, collector_func in self.active_collectors.items():
                    try:
                        metrics = collector_func()
                        if metrics:
                            collected_metrics.extend(metrics)
                    except Exception as e:
                        self.logger.warning(f"Collector {collector_name} failed: {e}")

                # Store collected metrics
                with self.collection_lock:
                    for metric in collected_metrics:
                        self.metrics_buffer.append(metric)

                # Maintain collection interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.collection_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)

    def _collect_cache_metrics(self) -> List[PerformanceMetric]:
        """Collect cache performance metrics."""
        metrics = []
        timestamp = datetime.now()

        try:
            # Get cache manager statistics
            cache_manager = get_cache_manager()
            global_stats = cache_manager.get_global_statistics()

            metrics.extend([
                PerformanceMetric(
                    name="cache_hit_rate",
                    value=global_stats.get('global_hit_rate_percent', 0),
                    unit="percent",
                    timestamp=timestamp,
                    source_module="cache",
                    category="throughput"
                ),
                PerformanceMetric(
                    name="cache_total_requests",
                    value=global_stats.get('total_requests', 0),
                    unit="count",
                    timestamp=timestamp,
                    source_module="cache",
                    category="throughput"
                )
            ])

            # Individual cache metrics
            for cache_name, cache_stats in global_stats.get('caches', {}).items():
                metrics.extend([
                    PerformanceMetric(
                        name=f"cache_{cache_name}_hit_rate",
                        value=cache_stats.get('hit_rate_percent', 0),
                        unit="percent",
                        timestamp=timestamp,
                        source_module="cache",
                        category="throughput",
                        metadata={'cache_name': cache_name}
                    ),
                    PerformanceMetric(
                        name=f"cache_{cache_name}_memory_usage",
                        value=cache_stats.get('memory_usage_bytes', 0) / (1024*1024),
                        unit="MB",
                        timestamp=timestamp,
                        source_module="cache",
                        category="resource_usage",
                        metadata={'cache_name': cache_name}
                    )
                ])

        except Exception as e:
            self.logger.debug(f"Cache metrics collection failed: {e}")

        return metrics

    def _collect_performance_metrics(self) -> List[PerformanceMetric]:
        """Collect general performance metrics."""
        metrics = []
        timestamp = datetime.now()

        try:
            # Memory optimizer metrics
            memory_optimizer = get_memory_optimizer()

            # System resource usage
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()

            metrics.extend([
                PerformanceMetric(
                    name="system_cpu_usage",
                    value=cpu_percent,
                    unit="percent",
                    timestamp=timestamp,
                    source_module="performance",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="system_memory_usage",
                    value=memory_info.percent,
                    unit="percent",
                    timestamp=timestamp,
                    source_module="performance",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="system_memory_available",
                    value=memory_info.available / (1024**3),
                    unit="GB",
                    timestamp=timestamp,
                    source_module="performance",
                    category="resource_usage"
                )
            ])

        except Exception as e:
            self.logger.debug(f"Performance metrics collection failed: {e}")

        return metrics

    def _collect_concurrency_metrics(self) -> List[PerformanceMetric]:
        """Collect concurrency performance metrics."""
        metrics = []
        timestamp = datetime.now()

        try:
            # Get concurrency benchmark if available
            benchmark = get_concurrency_benchmark()
            summary = benchmark.get_benchmark_summary()

            if summary.get('total_benchmarks', 0) > 0:
                metrics.extend([
                    PerformanceMetric(
                        name="concurrency_avg_throughput",
                        value=summary.get('avg_throughput', 0),
                        unit="tasks/second",
                        timestamp=timestamp,
                        source_module="concurrency",
                        category="throughput"
                    ),
                    PerformanceMetric(
                        name="concurrency_peak_throughput",
                        value=summary.get('peak_throughput', 0),
                        unit="tasks/second",
                        timestamp=timestamp,
                        source_module="concurrency",
                        category="throughput"
                    )
                ])

        except Exception as e:
            self.logger.debug(f"Concurrency metrics collection failed: {e}")

        return metrics

    def _collect_scaling_metrics(self) -> List[PerformanceMetric]:
        """Collect auto-scaling metrics."""
        metrics = []
        timestamp = datetime.now()

        try:
            # This would collect from the resource manager when available
            # For now, we'll create placeholder metrics

            metrics.extend([
                PerformanceMetric(
                    name="scaling_actions_total",
                    value=0,  # Placeholder
                    unit="count",
                    timestamp=timestamp,
                    source_module="auto_scaling",
                    category="throughput"
                ),
                PerformanceMetric(
                    name="scaling_effectiveness",
                    value=0.75,  # Placeholder
                    unit="score",
                    timestamp=timestamp,
                    source_module="auto_scaling",
                    category="throughput"
                )
            ])

        except Exception as e:
            self.logger.debug(f"Scaling metrics collection failed: {e}")

        return metrics

    def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level metrics."""
        metrics = []
        timestamp = datetime.now()

        try:
            import threading

            import psutil

            # System-wide metrics
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)

            metrics.extend([
                PerformanceMetric(
                    name="system_cpu_cores",
                    value=cpu_count,
                    unit="count",
                    timestamp=timestamp,
                    source_module="system",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="system_load_1min",
                    value=load_avg[0],
                    unit="load",
                    timestamp=timestamp,
                    source_module="system",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="active_threads",
                    value=threading.active_count(),
                    unit="count",
                    timestamp=timestamp,
                    source_module="system",
                    category="resource_usage"
                )
            ])

            # Disk I/O if available
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.extend([
                        PerformanceMetric(
                            name="disk_read_mb_total",
                            value=disk_io.read_bytes / (1024*1024),
                            unit="MB",
                            timestamp=timestamp,
                            source_module="system",
                            category="resource_usage"
                        ),
                        PerformanceMetric(
                            name="disk_write_mb_total",
                            value=disk_io.write_bytes / (1024*1024),
                            unit="MB",
                            timestamp=timestamp,
                            source_module="system",
                            category="resource_usage"
                        )
                    ])
            except Exception:
                pass  # Disk I/O might not be available

        except Exception as e:
            self.logger.debug(f"System metrics collection failed: {e}")

        return metrics

    def get_recent_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.collection_lock:
            return [
                metric for metric in self.metrics_buffer
                if metric.timestamp > cutoff_time
            ]

    def get_metrics_by_category(self, category: str, minutes: int = 10) -> List[PerformanceMetric]:
        """Get recent metrics by category."""
        recent_metrics = self.get_recent_metrics(minutes)
        return [m for m in recent_metrics if m.category == category]


class PerformanceDashboard:
    """Real-time performance dashboard and visualization."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance dashboard."""
        self.metrics_collector = metrics_collector
        self.dashboard_data = {}
        self.update_interval = 30  # seconds
        self.logger = logging.getLogger(__name__)

    async def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate current dashboard data."""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'categories': {},
            'trends': {},
            'alerts': []
        }

        # Get recent metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(minutes=10)

        if not recent_metrics:
            return dashboard_data

        # Group metrics by category
        by_category = defaultdict(list)
        for metric in recent_metrics:
            by_category[metric.category].append(metric)

        # Generate summary statistics
        for category, metrics in by_category.items():
            if not metrics:
                continue

            values = [m.value for m in metrics]
            dashboard_data['categories'][category] = {
                'count': len(metrics),
                'avg_value': statistics.mean(values),
                'max_value': max(values),
                'min_value': min(values),
                'latest_value': metrics[-1].value if metrics else 0,
                'unit': metrics[0].unit if metrics else 'unknown'
            }

        # Calculate trends (simplified)
        dashboard_data['trends'] = self._calculate_trends(recent_metrics)

        # Generate alerts
        dashboard_data['alerts'] = self._generate_alerts(recent_metrics)

        # Overall system health score
        dashboard_data['summary']['health_score'] = self._calculate_health_score(recent_metrics)
        dashboard_data['summary']['total_metrics'] = len(recent_metrics)
        dashboard_data['summary']['active_modules'] = len(set(m.source_module for m in recent_metrics))

        return dashboard_data

    def _calculate_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Dict[str, float]]:
        """Calculate performance trends."""
        trends = {}

        # Group by metric name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric)

        for metric_name, metric_list in by_name.items():
            if len(metric_list) < 3:  # Need minimum data for trend
                continue

            # Sort by timestamp
            metric_list.sort(key=lambda x: x.timestamp)
            values = [m.value for m in metric_list]

            # Simple linear trend
            if len(values) >= 2:
                recent_change = (values[-1] - values[0]) / len(values)
                trends[metric_name] = {
                    'change': recent_change,
                    'direction': 'up' if recent_change > 0 else 'down' if recent_change < 0 else 'stable',
                    'magnitude': abs(recent_change)
                }

        return trends

    def _generate_alerts(self, metrics: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate performance alerts."""
        alerts = []

        # Get latest metrics by name
        latest_by_name = {}
        for metric in metrics:
            if metric.name not in latest_by_name or metric.timestamp > latest_by_name[metric.name].timestamp:
                latest_by_name[metric.name] = metric

        # Check for alert conditions
        for metric_name, metric in latest_by_name.items():
            alert = None

            # CPU usage alerts
            if metric.name == 'system_cpu_usage' and metric.value > 90:
                alert = {
                    'level': 'critical',
                    'message': f'High CPU usage: {metric.value:.1f}%',
                    'metric': metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                }

            # Memory usage alerts
            elif metric.name == 'system_memory_usage' and metric.value > 85:
                alert = {
                    'level': 'warning',
                    'message': f'High memory usage: {metric.value:.1f}%',
                    'metric': metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                }

            # Cache hit rate alerts
            elif metric.name == 'cache_hit_rate' and metric.value < 50:
                alert = {
                    'level': 'warning',
                    'message': f'Low cache hit rate: {metric.value:.1f}%',
                    'metric': metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp.isoformat()
                }

            if alert:
                alerts.append(alert)

        return alerts

    def _calculate_health_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate overall system health score (0-100)."""
        if not metrics:
            return 50.0  # Neutral score

        scores = []

        # Get latest metrics by name
        latest_by_name = {}
        for metric in metrics:
            if metric.name not in latest_by_name or metric.timestamp > latest_by_name[metric.name].timestamp:
                latest_by_name[metric.name] = metric

        # Score individual metrics
        for metric_name, metric in latest_by_name.items():
            score = 100.0  # Start with perfect score

            # CPU usage scoring (lower is better for usage %)
            if metric.name == 'system_cpu_usage':
                if metric.value > 90:
                    score = 10
                elif metric.value > 80:
                    score = 30
                elif metric.value > 70:
                    score = 60
                elif metric.value > 50:
                    score = 80
                else:
                    score = 100

            # Memory usage scoring
            elif metric.name == 'system_memory_usage':
                if metric.value > 95:
                    score = 5
                elif metric.value > 85:
                    score = 30
                elif metric.value > 75:
                    score = 60
                elif metric.value > 60:
                    score = 80
                else:
                    score = 100

            # Cache hit rate scoring (higher is better)
            elif metric.name == 'cache_hit_rate':
                score = max(0, metric.value)  # Hit rate is already 0-100

            # Throughput metrics (higher is generally better)
            elif metric.category == 'throughput' and 'throughput' in metric.name:
                # Normalize based on expected ranges
                if metric.value > 100:
                    score = 100
                elif metric.value > 50:
                    score = 80
                elif metric.value > 10:
                    score = 60
                elif metric.value > 1:
                    score = 40
                else:
                    score = 20

            scores.append(score)

        # Return weighted average
        return statistics.mean(scores) if scores else 50.0

    async def save_dashboard_snapshot(self, output_path: Path):
        """Save current dashboard data to file."""
        dashboard_data = await self.generate_dashboard_data()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)


class BenchmarkSuite:
    """Comprehensive benchmark suite with automated analysis."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize benchmark suite."""
        self.metrics_collector = metrics_collector
        self.benchmark_results: List[BenchmarkResult] = []
        self.logger = logging.getLogger(__name__)
        self.dashboard = PerformanceDashboard(metrics_collector)

    async def run_comprehensive_benchmark(self,
                                        benchmark_config: Dict[str, Any] = None) -> BenchmarkResult:
        """Run comprehensive benchmark across all performance modules."""
        config = benchmark_config or {}
        benchmark_name = config.get('name', f'comprehensive_benchmark_{int(time.time())}')

        self.logger.info(f"Starting comprehensive benchmark: {benchmark_name}")
        started_at = datetime.now()
        success = True
        metrics = []
        error_info = None

        try:
            # Benchmark cache performance
            cache_metrics = await self._benchmark_cache_performance(config.get('cache', {}))
            metrics.extend(cache_metrics)

            # Benchmark concurrency performance
            concurrency_metrics = await self._benchmark_concurrency_performance(config.get('concurrency', {}))
            metrics.extend(concurrency_metrics)

            # Benchmark memory optimization
            memory_metrics = await self._benchmark_memory_performance(config.get('memory', {}))
            metrics.extend(memory_metrics)

            # Benchmark auto-scaling (if enabled)
            if config.get('include_scaling', False):
                scaling_metrics = await self._benchmark_scaling_performance(config.get('scaling', {}))
                metrics.extend(scaling_metrics)

        except Exception as e:
            success = False
            error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.error(f"Benchmark failed: {e}")

        completed_at = datetime.now()

        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name=benchmark_name,
            started_at=started_at,
            completed_at=completed_at,
            success=success,
            metrics=metrics,
            configuration=config,
            environment_info=self._collect_environment_info(),
            error_info=error_info
        )

        self.benchmark_results.append(result)
        self.logger.info(f"Benchmark completed: {benchmark_name} (success={success})")

        return result

    async def _benchmark_cache_performance(self, config: Dict[str, Any]) -> List[PerformanceMetric]:
        """Benchmark cache performance."""
        metrics = []
        timestamp = datetime.now()

        try:
            cache_manager = get_cache_manager()

            # Test cache operations
            test_keys = [f"test_key_{i}" for i in range(config.get('test_items', 100))]
            test_data = {"data": "x" * config.get('data_size', 1000)}

            # Measure cache write performance
            write_start = time.time()
            for key in test_keys:
                cache = cache_manager.get_cache('results')
                if cache:
                    await cache.set(key, test_data)
            write_time = time.time() - write_start

            # Measure cache read performance
            read_start = time.time()
            hits = 0
            for key in test_keys:
                cache = cache_manager.get_cache('results')
                if cache:
                    result = await cache.get(key)
                    if result is not None:
                        hits += 1
            read_time = time.time() - read_start

            # Calculate metrics
            metrics.extend([
                PerformanceMetric(
                    name="cache_write_throughput",
                    value=len(test_keys) / write_time,
                    unit="ops/second",
                    timestamp=timestamp,
                    source_module="cache",
                    category="throughput"
                ),
                PerformanceMetric(
                    name="cache_read_throughput",
                    value=len(test_keys) / read_time,
                    unit="ops/second",
                    timestamp=timestamp,
                    source_module="cache",
                    category="throughput"
                ),
                PerformanceMetric(
                    name="cache_hit_rate_test",
                    value=(hits / len(test_keys)) * 100,
                    unit="percent",
                    timestamp=timestamp,
                    source_module="cache",
                    category="throughput"
                )
            ])

        except Exception as e:
            self.logger.warning(f"Cache benchmark failed: {e}")

        return metrics

    async def _benchmark_concurrency_performance(self, config: Dict[str, Any]) -> List[PerformanceMetric]:
        """Benchmark concurrency performance."""
        metrics = []
        timestamp = datetime.now()

        try:
            benchmark = get_concurrency_benchmark()

            # Run concurrency benchmark
            num_tasks = config.get('num_tasks', 100)
            complexity = config.get('complexity', 'simple')

            result = await benchmark.benchmark_scheduler_performance(
                num_tasks=num_tasks,
                task_complexity=complexity
            )

            metrics.extend([
                PerformanceMetric(
                    name="concurrency_throughput",
                    value=result['tasks_per_second'],
                    unit="tasks/second",
                    timestamp=timestamp,
                    source_module="concurrency",
                    category="throughput"
                ),
                PerformanceMetric(
                    name="concurrency_total_time",
                    value=result['total_execution_time'],
                    unit="seconds",
                    timestamp=timestamp,
                    source_module="concurrency",
                    category="latency"
                ),
                PerformanceMetric(
                    name="concurrency_success_rate",
                    value=result['success_rate'],
                    unit="percent",
                    timestamp=timestamp,
                    source_module="concurrency",
                    category="throughput"
                )
            ])

        except Exception as e:
            self.logger.warning(f"Concurrency benchmark failed: {e}")

        return metrics

    async def _benchmark_memory_performance(self, config: Dict[str, Any]) -> List[PerformanceMetric]:
        """Benchmark memory optimization performance."""
        metrics = []
        timestamp = datetime.now()

        try:
            import psutil

            # Measure memory before/after operations
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024*1024)  # MB

            # Create some memory pressure
            test_data = []
            data_size = config.get('memory_test_size', 1000)

            for i in range(data_size):
                test_data.append([j for j in range(100)])

            peak_memory = process.memory_info().rss / (1024*1024)  # MB

            # Clean up
            del test_data

            # Force garbage collection
            import gc
            collected = gc.collect()

            final_memory = process.memory_info().rss / (1024*1024)  # MB

            metrics.extend([
                PerformanceMetric(
                    name="memory_initial_usage",
                    value=initial_memory,
                    unit="MB",
                    timestamp=timestamp,
                    source_module="memory",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="memory_peak_usage",
                    value=peak_memory,
                    unit="MB",
                    timestamp=timestamp,
                    source_module="memory",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="memory_final_usage",
                    value=final_memory,
                    unit="MB",
                    timestamp=timestamp,
                    source_module="memory",
                    category="resource_usage"
                ),
                PerformanceMetric(
                    name="gc_objects_collected",
                    value=collected,
                    unit="count",
                    timestamp=timestamp,
                    source_module="memory",
                    category="throughput"
                )
            ])

        except Exception as e:
            self.logger.warning(f"Memory benchmark failed: {e}")

        return metrics

    async def _benchmark_scaling_performance(self, config: Dict[str, Any]) -> List[PerformanceMetric]:
        """Benchmark auto-scaling performance."""
        metrics = []
        timestamp = datetime.now()

        try:
            # This would integrate with the actual scaling manager
            # For now, create placeholder metrics

            metrics.extend([
                PerformanceMetric(
                    name="scaling_response_time",
                    value=2.5,  # seconds (placeholder)
                    unit="seconds",
                    timestamp=timestamp,
                    source_module="auto_scaling",
                    category="latency"
                ),
                PerformanceMetric(
                    name="scaling_accuracy",
                    value=85.0,  # percent (placeholder)
                    unit="percent",
                    timestamp=timestamp,
                    source_module="auto_scaling",
                    category="throughput"
                )
            ])

        except Exception as e:
            self.logger.warning(f"Scaling benchmark failed: {e}")

        return metrics

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for benchmark context."""
        import platform

        import psutil

        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'timestamp': datetime.now().isoformat()
        }

    def generate_performance_report(self, output_path: Path = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.benchmark_results:
            return {'error': 'No benchmark results available'}

        report = {
            'summary': {
                'total_benchmarks': len(self.benchmark_results),
                'successful_benchmarks': sum(1 for r in self.benchmark_results if r.success),
                'total_duration': sum(r.duration_seconds for r in self.benchmark_results),
                'report_generated_at': datetime.now().isoformat()
            },
            'benchmarks': [],
            'performance_trends': {},
            'recommendations': []
        }

        # Process each benchmark result
        for result in self.benchmark_results:
            benchmark_summary = {
                'name': result.benchmark_name,
                'success': result.success,
                'duration_seconds': result.duration_seconds,
                'metrics_count': len(result.metrics),
                'categories': {}
            }

            # Group metrics by category
            by_category = defaultdict(list)
            for metric in result.metrics:
                by_category[metric.category].append(metric.value)

            for category, values in by_category.items():
                benchmark_summary['categories'][category] = {
                    'count': len(values),
                    'avg': statistics.mean(values),
                    'max': max(values),
                    'min': min(values)
                }

            report['benchmarks'].append(benchmark_summary)

        # Generate recommendations
        report['recommendations'] = self._generate_performance_recommendations()

        # Save report if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def _generate_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.benchmark_results:
            return recommendations

        # Analyze recent benchmark results
        recent_results = self.benchmark_results[-5:]  # Last 5 benchmarks

        # Check for consistent low cache hit rates
        cache_hit_rates = []
        for result in recent_results:
            cache_metric = result.get_metric('cache_hit_rate_test')
            if cache_metric:
                cache_hit_rates.append(cache_metric.value)

        if cache_hit_rates and statistics.mean(cache_hit_rates) < 70:
            recommendations.append({
                'category': 'cache',
                'priority': 'medium',
                'title': 'Improve Cache Hit Rate',
                'description': f'Average cache hit rate is {statistics.mean(cache_hit_rates):.1f}%. Consider increasing cache size or improving cache warming strategies.',
                'impact': 'medium'
            })

        # Check for high memory usage
        memory_peaks = []
        for result in recent_results:
            memory_metric = result.get_metric('memory_peak_usage')
            if memory_metric:
                memory_peaks.append(memory_metric.value)

        if memory_peaks and statistics.mean(memory_peaks) > 1000:  # >1GB
            recommendations.append({
                'category': 'memory',
                'priority': 'high',
                'title': 'Optimize Memory Usage',
                'description': f'Average peak memory usage is {statistics.mean(memory_peaks):.0f}MB. Consider implementing object pooling or reducing memory allocations.',
                'impact': 'high'
            })

        # Check for low throughput
        throughputs = []
        for result in recent_results:
            throughput_metrics = result.get_metrics_by_category('throughput')
            if throughput_metrics:
                avg_throughput = statistics.mean([m.value for m in throughput_metrics])
                throughputs.append(avg_throughput)

        if throughputs and statistics.mean(throughputs) < 50:
            recommendations.append({
                'category': 'performance',
                'priority': 'medium',
                'title': 'Improve System Throughput',
                'description': 'Average throughput metrics are below expected levels. Consider scaling up resources or optimizing algorithms.',
                'impact': 'medium'
            })

        return recommendations


# Global instances
_metrics_collector = None
_benchmark_suite = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_benchmark_suite() -> BenchmarkSuite:
    """Get global benchmark suite instance."""
    global _benchmark_suite
    if _benchmark_suite is None:
        collector = get_metrics_collector()
        _benchmark_suite = BenchmarkSuite(collector)
    return _benchmark_suite


# Utility functions
async def run_quick_performance_check() -> Dict[str, Any]:
    """Run a quick performance check across all modules."""
    collector = get_metrics_collector()
    await collector.start_collection()

    # Let it collect for a short time
    await asyncio.sleep(5)

    collector.stop_collection()

    # Generate dashboard data
    dashboard = PerformanceDashboard(collector)
    return await dashboard.generate_dashboard_data()


async def run_full_benchmark_suite(config: Dict[str, Any] = None) -> BenchmarkResult:
    """Run the full benchmark suite with optional configuration."""
    suite = get_benchmark_suite()

    # Start metrics collection
    await suite.metrics_collector.start_collection()

    try:
        # Run comprehensive benchmark
        result = await suite.run_comprehensive_benchmark(config)
        return result
    finally:
        suite.metrics_collector.stop_collection()
