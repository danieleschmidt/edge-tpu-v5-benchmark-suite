"""Quantum Performance Optimization and Scaling

Advanced performance optimization for quantum task execution with TPU v5 scaling.
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import logging
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import psutil

from .quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumResource, QuantumState
from .quantum_monitoring import MetricsCollector, PerformanceMetrics

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    LATENCY_FIRST = "latency_first"
    THROUGHPUT_FIRST = "throughput_first"
    BALANCED = "balanced"
    RESOURCE_EFFICIENT = "resource_efficient"
    POWER_EFFICIENT = "power_efficient"


@dataclass
class PerformanceProfile:
    """Performance optimization profile"""
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    max_concurrent_tasks: int = 4
    prefetch_tasks: int = 2
    batch_size: int = 1
    cache_enabled: bool = True
    auto_scaling: bool = True
    load_balancing: bool = True
    
    # TPU-specific optimizations
    tpu_pipeline_depth: int = 2
    tpu_batch_parallelism: bool = True
    tpu_memory_pooling: bool = True
    
    # Resource management
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.9
    adaptive_concurrency: bool = True
    
    def __post_init__(self):
        # Adjust defaults based on strategy
        if self.strategy == OptimizationStrategy.LATENCY_FIRST:
            self.max_concurrent_tasks = min(self.max_concurrent_tasks, 2)
            self.prefetch_tasks = 1
            self.batch_size = 1
        elif self.strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            self.max_concurrent_tasks = max(self.max_concurrent_tasks, 8)
            self.prefetch_tasks = 4
            self.batch_size = 4
        elif self.strategy == OptimizationStrategy.POWER_EFFICIENT:
            self.max_concurrent_tasks = min(self.max_concurrent_tasks, 2)
            self.tpu_batch_parallelism = False


@dataclass
class CacheEntry:
    """Cache entry for task results"""
    task_signature: str
    result: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > ttl_seconds
    
    def update_access(self) -> None:
        """Update access statistics"""
        self.access_count += 1


class AdaptiveCache:
    """Intelligent caching system with adaptive policies"""
    
    def __init__(self, max_size_mb: int = 512, ttl_seconds: float = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_pattern: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive access tracking"""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired(self.ttl_seconds):
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Update access pattern
                entry.update_access()
                self.access_pattern.append((key, time.time()))
                self.hits += 1
                
                return entry.result
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, size_hint: int = 0) -> bool:
        """Put item in cache with intelligent eviction"""
        with self._lock:
            # Estimate size if not provided
            if size_hint == 0:
                size_hint = self._estimate_size(value)
            
            # Check if we need to evict
            if self._get_total_size() + size_hint > self.max_size_bytes:
                evicted = self._evict_for_space(size_hint)
                if not evicted:
                    return False  # Cannot fit even after eviction
            
            # Create cache entry
            entry = CacheEntry(
                task_signature=key,
                result=value,
                timestamp=time.time(),
                size_bytes=size_hint
            )
            
            self.cache[key] = entry
            return True
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            # Simple size estimation
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, dict):
                return sum(len(str(k)) + self._estimate_size(v) for k, v in obj.items())
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            else:
                return 1024  # Default estimate
        except:
            return 1024
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def _evict_for_space(self, required_space: int) -> bool:
        """Evict items to make space using LFU + LRU hybrid strategy"""
        # Sort by access count (ascending) and then by timestamp (ascending)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        freed_space = 0
        keys_to_remove = []
        
        for key, entry in sorted_entries:
            keys_to_remove.append(key)
            freed_space += entry.size_bytes
            self.evictions += 1
            
            if freed_space >= required_space:
                break
        
        # Remove selected keys
        for key in keys_to_remove:
            del self.cache[key]
        
        return freed_space >= required_space
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'total_size_mb': self._get_total_size() / (1024 * 1024),
            'entry_count': len(self.cache),
            'avg_access_count': sum(e.access_count for e in self.cache.values()) / max(len(self.cache), 1)
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_pattern.clear()


class ResourcePoolManager:
    """Manage resource pools for optimal allocation"""
    
    def __init__(self):
        self.resource_pools: Dict[str, List[QuantumResource]] = defaultdict(list)
        self.active_allocations: Dict[str, Set[str]] = defaultdict(set)  # resource_type -> task_ids
        self.allocation_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def add_resource_pool(self, resource_type: str, resources: List[QuantumResource]) -> None:
        """Add a pool of resources of the same type"""
        with self._lock:
            self.resource_pools[resource_type].extend(resources)
            logger.info(f"Added {len(resources)} resources to {resource_type} pool")
    
    def allocate_optimal_resources(self, task: QuantumTask) -> Optional[Dict[str, QuantumResource]]:
        """Allocate optimal resources for task using pool management"""
        with self._lock:
            allocated_resources = {}
            
            for resource_type, required_amount in task.resource_requirements.items():
                if resource_type not in self.resource_pools:
                    return None  # Resource type not available
                
                # Find best resource in pool
                best_resource = self._find_best_resource(
                    self.resource_pools[resource_type],
                    required_amount,
                    task
                )
                
                if not best_resource or not best_resource.can_allocate(required_amount):
                    # Rollback previous allocations
                    self._rollback_allocations(allocated_resources, task)
                    return None
                
                # Allocate resource
                if best_resource.allocate(required_amount):
                    allocated_resources[resource_type] = best_resource
                    self.active_allocations[resource_type].add(task.id)
                else:
                    self._rollback_allocations(allocated_resources, task)
                    return None
            
            # Record allocation
            self.allocation_history.append({
                'timestamp': time.time(),
                'task_id': task.id,
                'resources': {k: v.name for k, v in allocated_resources.items()},
                'utilization': {k: 1.0 - (v.available_capacity / v.total_capacity) 
                              for k, v in allocated_resources.items()}
            })
            
            return allocated_resources
    
    def _find_best_resource(self, resource_pool: List[QuantumResource], 
                          required_amount: float, task: QuantumTask) -> Optional[QuantumResource]:
        """Find best resource in pool using optimization heuristics"""
        available_resources = [r for r in resource_pool if r.can_allocate(required_amount)]
        
        if not available_resources:
            return None
        
        # Heuristic: prefer resource with least fragmentation
        # (closest fit to required amount)
        def fragmentation_score(resource: QuantumResource) -> float:
            utilization = 1.0 - (resource.available_capacity / resource.total_capacity)
            waste = resource.available_capacity - required_amount
            return utilization + (waste / resource.total_capacity) * 0.1
        
        return min(available_resources, key=fragmentation_score)
    
    def _rollback_allocations(self, allocated_resources: Dict[str, QuantumResource], 
                            task: QuantumTask) -> None:
        """Rollback resource allocations"""
        for resource_type, resource in allocated_resources.items():
            required_amount = task.resource_requirements[resource_type]
            resource.release(required_amount)
            self.active_allocations[resource_type].discard(task.id)
    
    def release_task_resources(self, task: QuantumTask, 
                             allocated_resources: Dict[str, QuantumResource]) -> None:
        """Release resources allocated to task"""
        with self._lock:
            for resource_type, resource in allocated_resources.items():
                required_amount = task.resource_requirements[resource_type]
                resource.release(required_amount)
                self.active_allocations[resource_type].discard(task.id)
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        with self._lock:
            stats = {}
            
            for resource_type, resources in self.resource_pools.items():
                total_capacity = sum(r.total_capacity for r in resources)
                available_capacity = sum(r.available_capacity for r in resources)
                utilization = 1.0 - (available_capacity / max(total_capacity, 0.001))
                
                stats[resource_type] = {
                    'resource_count': len(resources),
                    'total_capacity': total_capacity,
                    'available_capacity': available_capacity,
                    'utilization': utilization,
                    'active_allocations': len(self.active_allocations[resource_type])
                }
            
            return stats


class ConcurrentExecutor:
    """High-performance concurrent task executor"""
    
    def __init__(self, profile: PerformanceProfile):
        self.profile = profile
        self.executor = None
        self.execution_semaphore = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.metrics = MetricsCollector()
        self._shutdown = False
        
        self._init_executor()
    
    def _init_executor(self) -> None:
        """Initialize executor based on performance profile"""
        if self.profile.strategy == OptimizationStrategy.THROUGHPUT_FIRST:
            # Use process pool for CPU-intensive tasks
            max_workers = min(self.profile.max_concurrent_tasks, mp.cpu_count())
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            # Use thread pool for I/O-bound tasks
            max_workers = self.profile.max_concurrent_tasks
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.execution_semaphore = asyncio.Semaphore(self.profile.max_concurrent_tasks)
    
    async def execute_batch(self, tasks: List[QuantumTask], 
                          execution_func: Any) -> List[Dict[str, Any]]:
        """Execute batch of tasks with optimal concurrency"""
        if not tasks:
            return []
        
        # Adaptive batch sizing based on system load
        actual_batch_size = self._calculate_adaptive_batch_size(len(tasks))
        
        results = []
        for i in range(0, len(tasks), actual_batch_size):
            batch = tasks[i:i + actual_batch_size]
            batch_results = await self._execute_concurrent_batch(batch, execution_func)
            results.extend(batch_results)
            
            # Brief pause between batches for system breathing room
            if i + actual_batch_size < len(tasks):
                await asyncio.sleep(0.001)
        
        return results
    
    async def _execute_concurrent_batch(self, tasks: List[QuantumTask], 
                                      execution_func: Any) -> List[Dict[str, Any]]:
        """Execute concurrent batch with resource management"""
        async def execute_with_semaphore(task):
            async with self.execution_semaphore:
                start_time = time.time()
                try:
                    result = await execution_func(task)
                    duration = time.time() - start_time
                    
                    # Record metrics
                    self.metrics.record_task_execution(
                        task.id, duration, result.get('success', False)
                    )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.metrics.record_task_execution(task.id, duration, False)
                    return {
                        'task_id': task.id,
                        'success': False,
                        'error': str(e),
                        'duration': duration
                    }
        
        # Execute tasks concurrently
        concurrent_tasks = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'task_id': tasks[i].id,
                    'success': False,
                    'error': str(result),
                    'duration': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _calculate_adaptive_batch_size(self, total_tasks: int) -> int:
        """Calculate optimal batch size based on system conditions"""
        if not self.profile.adaptive_concurrency:
            return self.profile.batch_size
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        # Adjust batch size based on system load
        base_size = self.profile.batch_size
        
        if cpu_percent > self.profile.cpu_threshold * 100:
            base_size = max(1, base_size // 2)
        elif cpu_percent < 50:
            base_size = min(base_size * 2, self.profile.max_concurrent_tasks)
        
        if memory_percent > self.profile.memory_threshold:
            base_size = max(1, base_size // 2)
        
        return min(base_size, total_tasks)
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get executor performance statistics"""
        exec_stats = self.metrics.get_task_execution_stats(window_seconds=300)
        
        return {
            'max_concurrent_tasks': self.profile.max_concurrent_tasks,
            'active_tasks': len(self.active_tasks),
            'execution_stats': exec_stats,
            'system_load': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown executor gracefully"""
        self._shutdown = True
        
        # Wait for active tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Shutdown thread/process pool
        if self.executor:
            self.executor.shutdown(wait=True)


class OptimizedQuantumTaskPlanner(QuantumTaskPlanner):
    """Performance-optimized quantum task planner"""
    
    def __init__(self, resources=None, performance_profile: Optional[PerformanceProfile] = None):
        super().__init__(resources)
        self.profile = performance_profile or PerformanceProfile()
        self.cache = AdaptiveCache()
        self.resource_pool = ResourcePoolManager()
        self.executor = ConcurrentExecutor(self.profile)
        
        # Performance tracking
        self.optimization_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_executions': 0,
            'resource_pool_hits': 0,
            'adaptive_scaling_events': 0
        }
        
        # Initialize resource pools
        self._init_resource_pools()
    
    def _init_resource_pools(self) -> None:
        """Initialize resource pools from available resources"""
        pools = defaultdict(list)
        
        for resource in self.resources.values():
            # Create additional instances for pooling
            pool_size = self._get_optimal_pool_size(resource)
            for i in range(pool_size):
                pool_resource = QuantumResource(
                    name=f"{resource.name}_pool_{i}",
                    total_capacity=resource.total_capacity,
                    allocation_quantum=resource.allocation_quantum,
                    tpu_cores=getattr(resource, 'tpu_cores', 0),
                    memory_gb=getattr(resource, 'memory_gb', 0),
                    compute_tops=getattr(resource, 'compute_tops', 0)
                )
                pools[resource.name].append(pool_resource)
        
        for resource_type, resource_list in pools.items():
            self.resource_pool.add_resource_pool(resource_type, resource_list)
    
    def _get_optimal_pool_size(self, resource: QuantumResource) -> int:
        """Calculate optimal pool size for resource type"""
        if 'tpu' in resource.name.lower():
            return 1  # TPU resources are typically exclusive
        elif 'cpu' in resource.name.lower():
            return min(4, mp.cpu_count())
        elif 'memory' in resource.name.lower():
            return 2
        else:
            return 1
    
    def _get_task_cache_key(self, task: QuantumTask) -> str:
        """Generate cache key for task"""
        key_components = [
            task.name,
            str(task.complexity),
            str(sorted(task.resource_requirements.items())),
            str(sorted(task.model_requirements))
        ]
        return hashlib.sha256('|'.join(key_components).encode()).hexdigest()[:16]
    
    async def execute_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute task with performance optimizations"""
        start_time = time.time()
        
        # Check cache first
        if self.profile.cache_enabled:
            cache_key = self._get_task_cache_key(task)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                self.optimization_metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for task {task.id}")
                return {
                    'task_id': task.id,
                    'success': True,
                    'cached': True,
                    'duration': time.time() - start_time,
                    'result': cached_result
                }
            else:
                self.optimization_metrics['cache_misses'] += 1
        
        try:
            # Optimized resource allocation
            allocated_resources = None
            if self.profile.load_balancing:
                allocated_resources = self.resource_pool.allocate_optimal_resources(task)
                if allocated_resources:
                    self.optimization_metrics['resource_pool_hits'] += 1
            
            # Execute with performance optimizations
            if self.profile.strategy == OptimizationStrategy.LATENCY_FIRST:
                result = await self._execute_latency_optimized(task)
            elif self.profile.strategy == OptimizationStrategy.THROUGHPUT_FIRST:
                result = await self._execute_throughput_optimized(task)
            else:
                result = await super().execute_task(task)
            
            # Cache successful results
            if self.profile.cache_enabled and result.get('success'):
                cache_key = self._get_task_cache_key(task)
                self.cache.put(cache_key, result)
            
            # Release resources
            if allocated_resources:
                self.resource_pool.release_task_resources(task, allocated_resources)
            
            return result
            
        except Exception as e:
            if allocated_resources:
                self.resource_pool.release_task_resources(task, allocated_resources)
            raise
    
    async def _execute_latency_optimized(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute task optimized for minimum latency"""
        # Minimize overhead, single-threaded execution
        return await super().execute_task(task)
    
    async def _execute_throughput_optimized(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute task optimized for maximum throughput"""
        # Use concurrent executor for parallel processing
        results = await self.executor.execute_batch([task], super().execute_task)
        return results[0] if results else {'task_id': task.id, 'success': False, 'error': 'Execution failed'}
    
    async def run_quantum_execution_cycle(self) -> Dict[str, Any]:
        """Optimized execution cycle with batching and prefetching"""
        cycle_start = time.time()
        
        # Get ready tasks with prefetching
        ready_tasks = self.get_ready_tasks()
        prefetch_count = min(self.profile.prefetch_tasks, len(ready_tasks))
        
        if not ready_tasks:
            return {
                'cycle_start': cycle_start,
                'tasks_executed': [],
                'tasks_failed': [],
                'resource_utilization': {},
                'quantum_coherence': 0.0,
                'cycle_duration': time.time() - cycle_start,
                'optimization_stats': self._get_optimization_stats()
            }
        
        # Batch execution for throughput
        if len(ready_tasks) >= self.profile.batch_size and self.profile.batch_size > 1:
            self.optimization_metrics['batch_executions'] += 1
            batch_results = await self.executor.execute_batch(
                ready_tasks[:self.profile.batch_size],
                self.execute_task
            )
            
            # Process batch results
            executed_tasks = []
            failed_tasks = []
            
            for result in batch_results:
                if result.get('success'):
                    executed_tasks.append(result)
                    self.completed_tasks.add(result['task_id'])
                else:
                    failed_tasks.append(result)
        
        else:
            # Single task execution
            task = ready_tasks[0]
            try:
                result = await self.execute_task(task)
                if result.get('success'):
                    executed_tasks = [result]
                    failed_tasks = []
                    self.completed_tasks.add(task.id)
                else:
                    executed_tasks = []
                    failed_tasks = [result]
            except Exception as e:
                executed_tasks = []
                failed_tasks = [{'task_id': task.id, 'error': str(e), 'success': False}]
        
        # Update resource utilization
        resource_utilization = {}
        for name, resource in self.resources.items():
            utilization = 1.0 - (resource.available_capacity / resource.total_capacity)
            resource_utilization[name] = utilization
        
        # Calculate quantum coherence
        coherent_tasks = [t for t in self.tasks.values() 
                         if t.state != QuantumState.DECOHERENT]
        total_coherence = sum(abs(t.probability_amplitude)**2 for t in coherent_tasks)
        quantum_coherence = total_coherence / max(len(coherent_tasks), 1)
        
        return {
            'cycle_start': cycle_start,
            'tasks_executed': executed_tasks,
            'tasks_failed': failed_tasks,
            'resource_utilization': resource_utilization,
            'quantum_coherence': quantum_coherence,
            'cycle_duration': time.time() - cycle_start,
            'optimization_stats': self._get_optimization_stats(),
            'cache_stats': self.cache.get_statistics(),
            'resource_pool_stats': self.resource_pool.get_pool_statistics(),
            'executor_stats': self.executor.get_performance_statistics()
        }
    
    def _get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        return {
            'cache_hit_rate': self.optimization_metrics['cache_hits'] / 
                            max(self.optimization_metrics['cache_hits'] + self.optimization_metrics['cache_misses'], 1),
            'batch_executions': self.optimization_metrics['batch_executions'],
            'resource_pool_efficiency': self.optimization_metrics['resource_pool_hits'],
            'adaptive_scaling_events': self.optimization_metrics['adaptive_scaling_events']
        }
    
    def optimize_system_parameters(self) -> None:
        """Automatically optimize system parameters based on performance data"""
        # Get recent performance statistics
        recent_stats = self.executor.get_performance_statistics()
        execution_stats = recent_stats.get('execution_stats', {})
        
        # Adaptive concurrency adjustment
        if self.profile.adaptive_concurrency:
            success_rate = execution_stats.get('success_rate', 1.0)
            avg_duration = execution_stats.get('avg_duration', 0.0)
            
            if success_rate < 0.8 and self.profile.max_concurrent_tasks > 1:
                # Reduce concurrency if success rate is low
                self.profile.max_concurrent_tasks = max(1, self.profile.max_concurrent_tasks - 1)
                self.optimization_metrics['adaptive_scaling_events'] += 1
                logger.info(f"Reduced max concurrent tasks to {self.profile.max_concurrent_tasks}")
                
            elif success_rate > 0.95 and avg_duration < 1.0:
                # Increase concurrency if performance is good
                max_possible = min(mp.cpu_count(), 16)
                if self.profile.max_concurrent_tasks < max_possible:
                    self.profile.max_concurrent_tasks += 1
                    self.optimization_metrics['adaptive_scaling_events'] += 1
                    logger.info(f"Increased max concurrent tasks to {self.profile.max_concurrent_tasks}")
        
        # Cache size optimization
        cache_stats = self.cache.get_statistics()
        if cache_stats['hit_rate'] < 0.3 and cache_stats['entry_count'] > 100:
            # Clear cache if hit rate is too low
            self.cache.clear()
            logger.info("Cleared cache due to low hit rate")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            'profile': {
                'strategy': self.profile.strategy.value,
                'max_concurrent_tasks': self.profile.max_concurrent_tasks,
                'batch_size': self.profile.batch_size,
                'cache_enabled': self.profile.cache_enabled
            },
            'optimization_metrics': self.optimization_metrics,
            'cache_statistics': self.cache.get_statistics(),
            'resource_pool_statistics': self.resource_pool.get_pool_statistics(),
            'executor_statistics': self.executor.get_performance_statistics(),
            'system_state': self.get_system_state()
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown with cleanup"""
        await self.executor.shutdown()
        self.cache.clear()
        logger.info("OptimizedQuantumTaskPlanner shutdown complete")


# Import fix
import hashlib