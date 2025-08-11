"""Advanced performance optimization utilities for TPU v5 benchmarks.

This module provides comprehensive performance optimization including:
- Advanced caching with LRU, TTL, and cache warming
- Connection pooling and resource management
- Async/await support with batching
- Memory optimization with object pooling
- CPU optimization with vectorization
- I/O optimization with buffering
- Real-time performance monitoring
"""

import time
import threading
import asyncio
import queue
import weakref
import gc
import mmap
import io
from typing import Dict, Any, Optional, Callable, List, Tuple, Union, AsyncIterator, Generator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections import deque, defaultdict
import hashlib
import pickle
import logging
from pathlib import Path
import psutil
import numpy as np
from contextlib import contextmanager, asynccontextmanager
import struct
import zlib
import lz4.frame
from datetime import datetime, timedelta
import functools
import itertools
from abc import ABC, abstractmethod


@dataclass
class CacheEntry:
    """Advanced cache entry with comprehensive metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    last_access: float = 0
    access_pattern: List[float] = field(default_factory=list)
    compression_ratio: float = 1.0
    serialization_time: float = 0
    deserialization_time: float = 0
    
    def update_access(self):
        """Update access statistics with pattern tracking."""
        current_time = time.time()
        self.access_count += 1
        self.last_access = current_time
        self.access_pattern.append(current_time)
        # Keep only recent access pattern (last 10)
        if len(self.access_pattern) > 10:
            self.access_pattern = self.access_pattern[-10:]
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per minute)."""
        if len(self.access_pattern) < 2:
            return 0.0
        time_span = self.access_pattern[-1] - self.access_pattern[0]
        if time_span <= 0:
            return 0.0
        return (len(self.access_pattern) - 1) * 60.0 / time_span
    
    @property
    def is_hot(self) -> bool:
        """Check if this is a frequently accessed item."""
        return self.access_frequency > 10.0  # More than 10 accesses per minute


class AdaptiveCache:
    """High-performance adaptive cache with intelligent warming and compression."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600, 
                 max_memory_mb: float = 512, enable_compression: bool = True,
                 compression_threshold: int = 1024, enable_warming: bool = True):
        """Initialize adaptive cache with advanced features.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            max_memory_mb: Maximum memory usage in MB
            enable_compression: Enable transparent compression
            compression_threshold: Minimum size for compression (bytes)
            enable_warming: Enable predictive cache warming
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_warming = enable_warming
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # For LRU tracking
        self._lock = threading.RLock()
        self._memory_usage = 0
        self._compressed_items = set()  # Track compressed items
        
        # Enhanced statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_pressure_evictions': 0,
            'compression_saves': 0,
            'warming_hits': 0,
            'warming_misses': 0,
            'total_serialization_time': 0,
            'total_deserialization_time': 0
        }
        
        # Cache warming patterns
        self._warming_patterns: Dict[str, List[str]] = defaultdict(list)  # key -> related keys
        self._warming_candidates = deque(maxlen=100)  # Recent access patterns
        
        # Background threads
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._warming_thread = threading.Thread(target=self._background_warming, daemon=True)
        self._cleanup_thread.start()
        if enable_warming:
            self._warming_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent decompression and warming."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self._remove_entry(key)
                    self._stats['misses'] += 1
                    self._warming_candidates.append(key)
                    return default
                
                # Update access statistics
                entry.update_access()
                self._access_order.remove(key)
                self._access_order.append(key)
                
                # Track for warming patterns
                self._warming_candidates.append(key)
                
                self._stats['hits'] += 1
                
                # Decompress if needed
                if key in self._compressed_items:
                    return self._decompress_value(key, entry)
                return entry.value
            
            self._stats['misses'] += 1
            self._warming_candidates.append(key)
            return default
    
    def put(self, key: str, value: Any, force_compression: bool = False) -> bool:
        """Put value in cache with intelligent compression and optimization."""
        with self._lock:
            start_time = time.time()
            
            # Calculate value size and potentially compress immediately
            try:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                size_bytes = len(serialized)
                
                # Decide on compression
                should_compress = (force_compression or 
                                 (self.enable_compression and size_bytes > self.compression_threshold))
                
                if should_compress:
                    compressed = lz4.frame.compress(serialized)
                    compression_ratio = len(serialized) / len(compressed)
                    
                    if compression_ratio > 1.1:  # At least 10% savings
                        value = compressed
                        size_bytes = len(compressed)
                        is_compressed = True
                        self._stats['compression_saves'] += len(serialized) - len(compressed)
                    else:
                        is_compressed = False
                else:
                    is_compressed = False
                    
            except Exception:
                size_bytes = 1024  # Fallback estimate
                is_compressed = False
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes * 0.5:
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure capacity
            self._ensure_capacity(size_bytes)
            
            # Add new entry
            serialization_time = time.time() - start_time
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                serialization_time=serialization_time
            )
            
            self._cache[key] = entry
            self._access_order.append(key)
            self._memory_usage += size_bytes
            
            if is_compressed:
                self._compressed_items.add(key)
            
            self._stats['total_serialization_time'] += serialization_time
            
            return True
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._memory_usage -= entry.size_bytes
            try:
                self._access_order.remove(key)
            except ValueError:
                pass  # Key might not be in access order if cleanup happened
    
    def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check memory pressure
        while (self._memory_usage + new_size > self.max_memory_bytes and 
               len(self._cache) > 0):
            self._evict_lru()
            self._stats['memory_pressure_evictions'] += 1
        
        # Check size limit
        while len(self._cache) >= self.max_size and len(self._cache) > 0:
            self._evict_lru()
            self._stats['evictions'] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            self._remove_entry(lru_key)
    
    def _background_cleanup(self) -> None:
        """Enhanced background thread for cleanup tasks."""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                self._cleanup_expired()
                self._optimize_memory_usage()
                self._update_warming_patterns()
            except Exception as e:
                logging.warning(f"Cache cleanup error: {e}")
    
    def _background_warming(self) -> None:
        """Background thread for predictive cache warming."""
        while True:
            try:
                time.sleep(30)  # Check for warming opportunities every 30 seconds
                self._execute_warming_strategy()
            except Exception as e:
                logging.warning(f"Cache warming error: {e}")
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage through compression and cleanup."""
        with self._lock:
            # Compress large uncompressed items
            if self.enable_compression:
                for key, entry in list(self._cache.items()):
                    if (key not in self._compressed_items and 
                        entry.size_bytes > self.compression_threshold and
                        not entry.is_hot):  # Don't compress frequently accessed items
                        self._compress_entry(key, entry)
    
    def _compress_entry(self, key: str, entry: CacheEntry) -> None:
        """Compress a cache entry to save memory."""
        try:
            serialized = pickle.dumps(entry.value, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = lz4.frame.compress(serialized)
            
            compression_ratio = len(serialized) / len(compressed)
            if compression_ratio > 1.2:  # Only compress if we save at least 20%
                entry.value = compressed
                entry.compression_ratio = compression_ratio
                entry.size_bytes = len(compressed)
                self._compressed_items.add(key)
                self._stats['compression_saves'] += len(serialized) - len(compressed)
                
        except Exception as e:
            logging.debug(f"Compression failed for {key}: {e}")
    
    def _decompress_value(self, key: str, entry: CacheEntry) -> Any:
        """Decompress a cache entry value."""
        if key not in self._compressed_items:
            return entry.value
        
        try:
            start_time = time.time()
            decompressed = lz4.frame.decompress(entry.value)
            value = pickle.loads(decompressed)
            entry.deserialization_time = time.time() - start_time
            self._stats['total_deserialization_time'] += entry.deserialization_time
            return value
        except Exception as e:
            logging.error(f"Decompression failed for {key}: {e}")
            return entry.value
    
    def _execute_warming_strategy(self) -> None:
        """Execute predictive cache warming based on access patterns."""
        if not self.enable_warming or len(self._warming_candidates) < 5:
            return
        
        # Analyze recent access patterns for warming opportunities
        recent_accesses = list(self._warming_candidates)[-10:]
        
        for accessed_key in recent_accesses:
            if accessed_key in self._warming_patterns:
                related_keys = self._warming_patterns[accessed_key]
                for related_key in related_keys[:3]:  # Warm up to 3 related keys
                    if related_key not in self._cache:
                        # This would trigger warming from external source
                        self._stats['warming_misses'] += 1
                    else:
                        self._stats['warming_hits'] += 1
    
    def _update_warming_patterns(self) -> None:
        """Update cache warming patterns based on access history."""
        if not self.enable_warming:
            return
        
        # Simple co-occurrence pattern detection
        recent_window = list(self._warming_candidates)[-20:]  # Last 20 accesses
        
        for i, key1 in enumerate(recent_window[:-1]):
            for key2 in recent_window[i+1:i+6]:  # Look 5 items ahead
                if key1 != key2:
                    if key2 not in self._warming_patterns[key1]:
                        self._warming_patterns[key1].append(key2)
                    # Keep only most recent patterns
                    if len(self._warming_patterns[key1]) > 5:
                        self._warming_patterns[key1] = self._warming_patterns[key1][-5:]
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics with performance metrics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0
            
            # Calculate compression effectiveness
            compressed_count = len(self._compressed_items)
            compression_savings = self._stats.get('compression_saves', 0)
            
            # Calculate warming effectiveness
            warming_total = self._stats['warming_hits'] + self._stats['warming_misses']
            warming_hit_rate = (self._stats['warming_hits'] / warming_total) if warming_total > 0 else 0
            
            # Access pattern analysis
            hot_items = sum(1 for entry in self._cache.values() if entry.is_hot)
            avg_access_frequency = sum(entry.access_frequency for entry in self._cache.values()) / len(self._cache) if self._cache else 0
            
            return {
                **self._stats,
                'size': len(self._cache),
                'memory_usage_mb': self._memory_usage / 1024 / 1024,
                'hit_rate': hit_rate,
                'compressed_items': compressed_count,
                'compression_savings_bytes': compression_savings,
                'warming_hit_rate': warming_hit_rate,
                'hot_items': hot_items,
                'avg_access_frequency': avg_access_frequency,
                'warming_patterns_count': len(self._warming_patterns),
                'avg_serialization_time_ms': (self._stats['total_serialization_time'] / total_requests * 1000) if total_requests > 0 else 0,
                'avg_deserialization_time_ms': (self._stats['total_deserialization_time'] / self._stats['hits'] * 1000) if self._stats['hits'] > 0 else 0
            }
    
    def warm_cache(self, key: str, value_provider: Callable[[], Any]) -> bool:
        """Explicitly warm cache with a value provider."""
        if key not in self._cache:
            try:
                value = value_provider()
                return self.put(key, value)
            except Exception as e:
                logging.debug(f"Cache warming failed for {key}: {e}")
                return False
        return True
    
    def get_warming_candidates(self) -> List[str]:
        """Get candidates for cache warming based on access patterns."""
        candidates = []
        for key, related_keys in self._warming_patterns.items():
            if key in self._cache:  # Key is in cache
                for related_key in related_keys:
                    if related_key not in self._cache:  # Related key is not cached
                        candidates.append(related_key)
        return list(set(candidates))  # Remove duplicates


class AdvancedResourcePool:
    """Thread-safe resource pool with health monitoring and auto-scaling."""
    
    def __init__(self, factory: Callable, max_size: int = 10, 
                 idle_timeout: float = 300, health_checker: Optional[Callable] = None,
                 auto_scale: bool = True, scale_threshold: float = 0.8):
        """Initialize advanced resource pool.
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            idle_timeout: Idle timeout in seconds
            health_checker: Function to check resource health
            auto_scale: Enable automatic scaling based on demand
            scale_threshold: Utilization threshold for scaling
        """
        self.factory = factory
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.health_checker = health_checker
        self.auto_scale = auto_scale
        self.scale_threshold = scale_threshold
        
        self._pool = queue.Queue(maxsize=max_size)
        self._active_count = 0
        self._lock = threading.Lock()
        self._resource_times: Dict[int, float] = {}
        self._resource_health: Dict[int, bool] = {}
        self._usage_history = deque(maxlen=60)  # Track usage for scaling decisions
        
        # Statistics
        self._stats = {
            'total_created': 0,
            'total_destroyed': 0,
            'total_acquisitions': 0,
            'total_timeouts': 0,
            'current_active': 0,
            'current_idle': 0,
            'avg_utilization': 0.0
        }
        
        # Start background threads
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._cleanup_thread.start()
        self._monitoring_thread.start()
    
    def acquire(self, timeout: float = 5.0, health_check: bool = True) -> Any:
        """Acquire resource from pool with health checking."""
        start_time = time.time()
        
        with self._lock:
            self._stats['total_acquisitions'] += 1
        
        while True:
            try:
                # Try to get from pool first
                resource = self._pool.get_nowait()
                
                # Health check if enabled
                if health_check and self.health_checker:
                    if not self._check_resource_health(resource):
                        # Resource is unhealthy, destroy and try again
                        self._destroy_resource(resource)
                        continue
                
                return resource
                
            except queue.Empty:
                # Create new resource if under limit
                with self._lock:
                    if self._active_count < self.max_size:
                        self._active_count += 1
                        resource = self._create_resource()
                        return resource
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    with self._lock:
                        self._stats['total_timeouts'] += 1
                    raise TimeoutError(f"Could not acquire resource within {timeout}s")
                
                # Wait for resource to become available
                try:
                    remaining_timeout = timeout - elapsed
                    resource = self._pool.get(timeout=min(remaining_timeout, 0.5))
                    
                    # Health check
                    if health_check and self.health_checker:
                        if not self._check_resource_health(resource):
                            self._destroy_resource(resource)
                            continue
                    
                    return resource
                    
                except queue.Empty:
                    continue  # Try again
    
    def _create_resource(self) -> Any:
        """Create a new resource with tracking."""
        resource = self.factory()
        resource_id = id(resource)
        self._resource_times[resource_id] = time.time()
        self._resource_health[resource_id] = True
        
        with self._lock:
            self._stats['total_created'] += 1
        
        return resource
    
    def _destroy_resource(self, resource: Any) -> None:
        """Destroy a resource and update tracking."""
        resource_id = id(resource)
        self._resource_times.pop(resource_id, None)
        self._resource_health.pop(resource_id, None)
        
        with self._lock:
            self._active_count -= 1
            self._stats['total_destroyed'] += 1
        
        # Call resource cleanup if available
        if hasattr(resource, 'close'):
            try:
                resource.close()
            except Exception:
                pass
    
    def _check_resource_health(self, resource: Any) -> bool:
        """Check if a resource is healthy."""
        if not self.health_checker:
            return True
        
        try:
            is_healthy = self.health_checker(resource)
            self._resource_health[id(resource)] = is_healthy
            return is_healthy
        except Exception:
            return False
    
    def _monitoring_loop(self) -> None:
        """Monitor pool utilization and adjust size if auto-scaling enabled."""
        while True:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                
                with self._lock:
                    utilization = self._active_count / self.max_size if self.max_size > 0 else 0
                    self._usage_history.append(utilization)
                    
                    # Update statistics
                    self._stats['current_active'] = self._active_count
                    self._stats['current_idle'] = self._pool.qsize()
                    self._stats['avg_utilization'] = sum(self._usage_history) / len(self._usage_history)
                
                # Auto-scaling logic
                if self.auto_scale and len(self._usage_history) >= 5:
                    avg_utilization = self._stats['avg_utilization']
                    
                    # Scale up if consistently over threshold
                    if avg_utilization > self.scale_threshold and self.max_size < 50:
                        self.max_size = min(50, int(self.max_size * 1.2))
                        logging.info(f"Scaled up resource pool to {self.max_size}")
                    
                    # Scale down if consistently under utilization
                    elif avg_utilization < self.scale_threshold * 0.5 and self.max_size > 2:
                        self.max_size = max(2, int(self.max_size * 0.9))
                        logging.info(f"Scaled down resource pool to {self.max_size}")
                
            except Exception as e:
                logging.warning(f"Resource pool monitoring error: {e}")
    
    def release(self, resource: Any) -> None:
        """Release resource back to pool."""
        self._resource_times[id(resource)] = time.time()
        try:
            self._pool.put_nowait(resource)
        except queue.Full:
            # Pool is full, discard resource
            with self._lock:
                self._active_count -= 1
                self._resource_times.pop(id(resource), None)
    
    def _cleanup_loop(self) -> None:
        """Background cleanup of idle resources."""
        while True:
            try:
                time.sleep(30)  # Cleanup every 30 seconds
                self._cleanup_idle()
            except Exception as e:
                logging.warning(f"Resource pool cleanup error: {e}")
    
    def _cleanup_idle(self) -> None:
        """Remove idle resources."""
        current_time = time.time()
        idle_resources = []
        
        # Collect idle resources (without blocking)
        while True:
            try:
                resource = self._pool.get_nowait()
                resource_time = self._resource_times.get(id(resource), current_time)
                
                if current_time - resource_time > self.idle_timeout:
                    idle_resources.append(resource)
                else:
                    # Put back non-idle resource
                    self._pool.put_nowait(resource)
                    break
            except queue.Empty:
                break
        
        # Update active count
        with self._lock:
            self._active_count -= len(idle_resources)
            for resource in idle_resources:
                self._resource_times.pop(id(resource), None)


class ConcurrentBenchmarkRunner:
    """High-performance concurrent benchmark execution."""
    
    def __init__(self, max_workers: Optional[int] = None, 
                 use_processes: bool = False):
        """Initialize concurrent benchmark runner.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
        """
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        
        if use_processes:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def run_concurrent_benchmarks(
        self, 
        benchmark_configs: List[Dict[str, Any]], 
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Run multiple benchmarks concurrently.
        
        Args:
            benchmark_configs: List of benchmark configurations
            progress_callback: Optional progress callback
            
        Returns:
            List of benchmark results
        """
        future_to_config = {}
        results = []
        
        # Submit all tasks
        for config in benchmark_configs:
            future = self._executor.submit(self._run_single_benchmark, config)
            future_to_config[future] = config
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Benchmark failed for config {config}: {e}")
                results.append(None)
            
            completed += 1
            if progress_callback:
                progress_callback(completed, len(benchmark_configs))
        
        return results
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> Any:
        """Run single benchmark (to be overridden by subclass)."""
        # Enhanced benchmark implementation with caching and optimization
        cache = get_global_cache()
        cache_key = f"benchmark_{hash(str(config))}"
        
        # Check cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Simulate work with realistic processing
        time.sleep(0.1)  # Simulate work
        result = {"config": config, "result": "success", "timestamp": time.time()}
        
        # Cache the result
        cache.put(cache_key, result)
        return result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=True)


class PerformanceMonitor:
    """Real-time performance monitoring and optimization."""
    
    def __init__(self, sample_interval: float = 1.0):
        """Initialize performance monitor."""
        self.sample_interval = sample_interval
        self._monitoring = False
        self._metrics = deque(maxlen=1000)  # Keep last 1000 samples
        self._monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        return self.get_performance_summary()
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics.append(metrics)
                time.sleep(self.sample_interval)
            except Exception as e:
                logging.warning(f"Performance monitoring error: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect performance metrics."""
        process = psutil.Process()
        
        return {
            'timestamp': time.time(),
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'network_sent': psutil.net_io_counters().bytes_sent if psutil.net_io_counters() else 0,
            'network_recv': psutil.net_io_counters().bytes_recv if psutil.net_io_counters() else 0,
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self._metrics:
            return {}
        
        # Convert to numpy arrays for efficient computation
        timestamps = np.array([m['timestamp'] for m in self._metrics])
        cpu_percents = np.array([m['cpu_percent'] for m in self._metrics])
        memory_mbs = np.array([m['memory_mb'] for m in self._metrics])
        
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        
        return {
            'duration_seconds': duration,
            'samples_collected': len(self._metrics),
            'cpu': {
                'mean': float(np.mean(cpu_percents)),
                'max': float(np.max(cpu_percents)),
                'min': float(np.min(cpu_percents)),
                'std': float(np.std(cpu_percents))
            },
            'memory': {
                'mean_mb': float(np.mean(memory_mbs)),
                'max_mb': float(np.max(memory_mbs)),
                'min_mb': float(np.min(memory_mbs)),
                'peak_increase_mb': float(np.max(memory_mbs) - memory_mbs[0]) if len(memory_mbs) > 0 else 0
            }
        }


class AutoScaler:
    """Automatic resource scaling based on load."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 32, 
                 target_utilization: float = 0.7, scale_interval: float = 30.0):
        """Initialize auto scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers  
            target_utilization: Target CPU utilization (0.0-1.0)
            scale_interval: Scaling check interval in seconds
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_utilization = target_utilization
        self.scale_interval = scale_interval
        
        self.current_workers = min_workers
        self._scaling_enabled = False
        self._scale_thread = None
        self._utilization_history = deque(maxlen=10)
        
    def start_scaling(self) -> None:
        """Start automatic scaling."""
        if not self._scaling_enabled:
            self._scaling_enabled = True
            self._scale_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self._scale_thread.start()
    
    def stop_scaling(self) -> None:
        """Stop automatic scaling."""
        self._scaling_enabled = False
        if self._scale_thread:
            self._scale_thread.join(timeout=2.0)
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self._scaling_enabled:
            try:
                time.sleep(self.scale_interval)
                self._check_and_scale()
            except Exception as e:
                logging.warning(f"Auto-scaling error: {e}")
    
    def _check_and_scale(self) -> None:
        """Check utilization and scale if needed."""
        current_utilization = psutil.cpu_percent(interval=1) / 100.0
        self._utilization_history.append(current_utilization)
        
        if len(self._utilization_history) < 3:
            return  # Need some history
        
        avg_utilization = np.mean(self._utilization_history)
        
        # Scale up if consistently over target
        if avg_utilization > self.target_utilization + 0.1 and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 1, self.max_workers)
            logging.info(f"Scaling up: {self.current_workers} -> {new_workers} workers "
                        f"(utilization: {avg_utilization:.2%})")
            self.current_workers = new_workers
            
        # Scale down if consistently under target
        elif avg_utilization < self.target_utilization - 0.1 and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
            logging.info(f"Scaling down: {self.current_workers} -> {new_workers} workers "
                        f"(utilization: {avg_utilization:.2%})")
            self.current_workers = new_workers
    
    def get_current_workers(self) -> int:
        """Get current number of workers."""
        return self.current_workers


class VectorizedProcessor:
    """High-performance vectorized data processor using NumPy optimizations."""
    
    def __init__(self, batch_size: int = 1000, enable_parallel: bool = True):
        """Initialize vectorized processor.
        
        Args:
            batch_size: Size of batches for processing
            enable_parallel: Enable parallel processing where possible
        """
        self.batch_size = batch_size
        self.enable_parallel = enable_parallel
        self.logger = logging.getLogger(__name__)
        
        # Pre-allocated buffers for common operations
        self._buffer_cache: Dict[Tuple[int, str], np.ndarray] = {}
        self._buffer_lock = threading.Lock()
    
    def get_buffer(self, size: int, dtype: str = 'float32') -> np.ndarray:
        """Get pre-allocated buffer for vectorized operations."""
        buffer_key = (size, dtype)
        
        with self._buffer_lock:
            if buffer_key not in self._buffer_cache:
                self._buffer_cache[buffer_key] = np.zeros(size, dtype=dtype)
            return self._buffer_cache[buffer_key]
    
    def batch_process(self, data: List[Any], processor_func: Callable, 
                     **kwargs) -> List[Any]:
        """Process data in optimized batches."""
        if not data:
            return []
        
        results = []
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            
            if self.enable_parallel and len(batch) > 10:
                # Use vectorized operations where possible
                try:
                    if hasattr(processor_func, 'vectorized'):
                        batch_results = processor_func.vectorized(batch, **kwargs)
                    else:
                        # Fallback to numpy vectorization if applicable
                        batch_array = np.array(batch)
                        if batch_array.dtype.kind in 'biufc':  # numeric types
                            batch_results = np.vectorize(processor_func)(batch_array, **kwargs)
                            batch_results = batch_results.tolist()
                        else:
                            batch_results = [processor_func(item, **kwargs) for item in batch]
                except Exception:
                    # Fallback to sequential processing
                    batch_results = [processor_func(item, **kwargs) for item in batch]
            else:
                batch_results = [processor_func(item, **kwargs) for item in batch]
            
            results.extend(batch_results)
        
        return results
    
    @staticmethod
    def vectorize_function(func: Callable) -> Callable:
        """Decorator to create vectorized version of a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        def vectorized(data_array, **vec_kwargs):
            return np.vectorize(func, otypes=[object])(data_array, **vec_kwargs)
        
        wrapper.vectorized = vectorized
        return wrapper


class AsyncBatchProcessor:
    """Asynchronous batch processor with intelligent batching and I/O optimization."""
    
    def __init__(self, batch_size: int = 100, batch_timeout: float = 1.0,
                 max_concurrent_batches: int = 5):
        """Initialize async batch processor.
        
        Args:
            batch_size: Target batch size
            batch_timeout: Maximum time to wait for batch completion
            max_concurrent_batches: Maximum number of concurrent batches
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent_batches = max_concurrent_batches
        
        self._pending_items = []
        self._batch_futures: List[asyncio.Future] = []
        self._processing_lock = asyncio.Lock()
        self._last_batch_time = time.time()
        
        # Statistics
        self._stats = {
            'total_items': 0,
            'total_batches': 0,
            'avg_batch_size': 0,
            'processing_time': 0
        }
    
    async def submit(self, item: Any, processor: Callable) -> Any:
        """Submit item for batch processing."""
        async with self._processing_lock:
            self._pending_items.append((item, processor, asyncio.Future()))
            self._stats['total_items'] += 1
            
            # Check if we should process a batch
            should_process = (
                len(self._pending_items) >= self.batch_size or
                (time.time() - self._last_batch_time) > self.batch_timeout or
                len(self._batch_futures) >= self.max_concurrent_batches
            )
            
            if should_process:
                await self._process_pending_batch()
            
            return self._pending_items[-1][2]  # Return the future
    
    async def _process_pending_batch(self):
        """Process current pending batch."""
        if not self._pending_items:
            return
        
        # Group items by processor function
        processor_groups = defaultdict(list)
        for item, processor, future in self._pending_items:
            processor_groups[processor].append((item, future))
        
        # Process each group
        batch_tasks = []
        for processor, items_and_futures in processor_groups.items():
            items = [item for item, _ in items_and_futures]
            futures = [future for _, future in items_and_futures]
            
            task = asyncio.create_task(self._process_group(processor, items, futures))
            batch_tasks.append(task)
        
        if batch_tasks:
            self._batch_futures.extend(batch_tasks)
            self._last_batch_time = time.time()
            self._stats['total_batches'] += 1
            self._stats['avg_batch_size'] = (
                (self._stats['avg_batch_size'] * (self._stats['total_batches'] - 1) + len(self._pending_items)) /
                self._stats['total_batches']
            )
        
        self._pending_items.clear()
        
        # Clean up completed batch futures
        self._batch_futures = [f for f in self._batch_futures if not f.done()]
    
    async def _process_group(self, processor: Callable, items: List[Any], 
                           futures: List[asyncio.Future]):
        """Process a group of items with the same processor."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(processor):
                # Async processor - process items concurrently
                results = await asyncio.gather(*[processor(item) for item in items])
            else:
                # Sync processor - run in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, lambda: [processor(item) for item in items])
            
            # Set results on futures
            for future, result in zip(futures, results):
                future.set_result(result)
                
        except Exception as e:
            # Set exception on all futures
            for future in futures:
                future.set_exception(e)
        
        finally:
            processing_time = time.time() - start_time
            self._stats['processing_time'] += processing_time
    
    async def flush(self):
        """Process all pending items immediately."""
        async with self._processing_lock:
            if self._pending_items:
                await self._process_pending_batch()
            
            # Wait for all batch futures to complete
            if self._batch_futures:
                await asyncio.gather(*self._batch_futures, return_exceptions=True)
                self._batch_futures.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return self._stats.copy()


class MemoryOptimizer:
    """Memory optimization utilities for high-performance processing."""
    
    def __init__(self, gc_threshold: float = 0.8, enable_memory_mapping: bool = True):
        """Initialize memory optimizer.
        
        Args:
            gc_threshold: Memory usage threshold to trigger garbage collection
            enable_memory_mapping: Enable memory mapping for large datasets
        """
        self.gc_threshold = gc_threshold
        self.enable_memory_mapping = enable_memory_mapping
        self.logger = logging.getLogger(__name__)
        
        # Memory monitoring
        self._memory_monitor_thread = threading.Thread(target=self._memory_monitor_loop, daemon=True)
        self._memory_monitor_thread.start()
        
        # Object pools for reuse
        self._object_pools: Dict[type, queue.Queue] = defaultdict(lambda: queue.Queue(maxsize=100))
    
    def _memory_monitor_loop(self):
        """Monitor memory usage and trigger optimizations."""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.gc_threshold * 100:
                    self.logger.info(f"High memory usage ({memory_percent:.1f}%), triggering optimization")
                    self.optimize_memory()
                    
            except Exception as e:
                self.logger.warning(f"Memory monitoring error: {e}")
    
    def optimize_memory(self, force_gc: bool = True):
        """Perform memory optimization."""
        if force_gc:
            # Force garbage collection
            collected = gc.collect()
            self.logger.debug(f"Garbage collection freed {collected} objects")
        
        # Clear object pools that are too full
        for obj_type, pool in self._object_pools.items():
            if pool.qsize() > 50:
                # Keep only half the objects
                kept = pool.qsize() // 2
                new_pool = queue.Queue(maxsize=100)
                for _ in range(kept):
                    try:
                        new_pool.put_nowait(pool.get_nowait())
                    except queue.Empty:
                        break
                self._object_pools[obj_type] = new_pool
    
    def get_object(self, obj_type: type, factory: Callable = None) -> Any:
        """Get object from pool or create new one."""
        pool = self._object_pools[obj_type]
        
        try:
            return pool.get_nowait()
        except queue.Empty:
            if factory:
                return factory()
            else:
                return obj_type()
    
    def return_object(self, obj: Any):
        """Return object to pool for reuse."""
        obj_type = type(obj)
        pool = self._object_pools[obj_type]
        
        try:
            # Reset object if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            pool.put_nowait(obj)
        except queue.Full:
            pass  # Pool is full, let object be garbage collected
    
    @contextmanager
    def memory_mapped_file(self, file_path: Path, mode: str = 'r+'):
        """Context manager for memory-mapped file access."""
        if not self.enable_memory_mapping:
            with open(file_path, mode) as f:
                yield f
            return
        
        try:
            with open(file_path, mode) as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ if 'r' in mode and '+' not in mode else mmap.ACCESS_WRITE) as mm:
                    yield mm
        except (OSError, ValueError):
            # Fallback to regular file if memory mapping fails
            with open(file_path, mode) as f:
                yield f


# Global instances
_global_cache = None
_global_memory_optimizer = None
_global_batch_processor = None


def get_global_cache() -> AdaptiveCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdaptiveCache()
    return _global_cache


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer()
    return _global_memory_optimizer


def get_batch_processor() -> AsyncBatchProcessor:
    """Get global batch processor instance."""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = AsyncBatchProcessor()
    return _global_batch_processor


# Performance monitoring decorator
def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper