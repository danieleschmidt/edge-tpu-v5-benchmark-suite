"""Performance optimization utilities for TPU v5 benchmarks."""

import time
import threading
import asyncio
import queue
import weakref
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from collections import deque
import hashlib
import pickle
import logging
from pathlib import Path
import psutil
import numpy as np


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1


class AdaptiveCache:
    """High-performance adaptive cache with LRU and TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600, 
                 max_memory_mb: float = 512):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live for entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = deque()  # For LRU tracking
        self._lock = threading.RLock()
        self._memory_usage = 0
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_pressure_evictions': 0
        }
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self._remove_entry(key)
                    self._stats['misses'] += 1
                    return default
                
                # Update access statistics
                entry.update_access()
                self._access_order.remove(key)
                self._access_order.append(key)
                
                self._stats['hits'] += 1
                return entry.value
            
            self._stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate value size
            try:
                size_bytes = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                size_bytes = 1024  # Fallback estimate
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes * 0.5:
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure capacity
            self._ensure_capacity(size_bytes)
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._access_order.append(key)
            self._memory_usage += size_bytes
            
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
        """Background thread for cleanup tasks."""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                self._cleanup_expired()
            except Exception as e:
                logging.warning(f"Cache cleanup error: {e}")
    
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
        """Get cache statistics."""
        with self._lock:
            hit_rate = (self._stats['hits'] / 
                       (self._stats['hits'] + self._stats['misses'])) if (
                           self._stats['hits'] + self._stats['misses']) > 0 else 0
            
            return {
                **self._stats,
                'size': len(self._cache),
                'memory_usage_mb': self._memory_usage / 1024 / 1024,
                'hit_rate': hit_rate
            }


class ResourcePool:
    """Thread-safe resource pool for expensive objects."""
    
    def __init__(self, factory: Callable, max_size: int = 10, 
                 idle_timeout: float = 300):
        """Initialize resource pool.
        
        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            idle_timeout: Idle timeout in seconds
        """
        self.factory = factory
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool = queue.Queue(maxsize=max_size)
        self._active_count = 0
        self._lock = threading.Lock()
        self._resource_times: Dict[int, float] = {}
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def acquire(self, timeout: float = 5.0) -> Any:
        """Acquire resource from pool."""
        try:
            # Try to get from pool first
            resource = self._pool.get_nowait()
            return resource
        except queue.Empty:
            # Create new resource if under limit
            with self._lock:
                if self._active_count < self.max_size:
                    self._active_count += 1
                    resource = self.factory()
                    self._resource_times[id(resource)] = time.time()
                    return resource
            
            # Wait for resource to become available
            try:
                resource = self._pool.get(timeout=timeout)
                return resource
            except queue.Empty:
                raise TimeoutError(f"Could not acquire resource within {timeout}s")
    
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
        # This is a placeholder - actual implementation would depend on benchmark type
        time.sleep(0.1)  # Simulate work
        return {"config": config, "result": "success"}
    
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