"""Advanced concurrency and parallel processing for TPU v5 benchmark suite."""

from typing import Dict, Any, List, Optional, Callable, Union, Tuple, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import threading
import multiprocessing
import concurrent.futures
import queue
import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
import signal
import psutil
import weakref
from contextlib import contextmanager


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    worker_id: Optional[str] = None
    
    @property
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED and self.error is None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": str(self.result) if self.result is not None else None,
            "error": str(self.error) if self.error else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time": self.execution_time,
            "worker_id": self.worker_id
        }


@dataclass
class Task:
    """Represents a unit of work to be executed."""
    id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retries: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.priority.value > other.priority.value


class WorkerPool(ABC):
    """Abstract base class for worker pools."""
    
    @abstractmethod
    async def submit(self, task: Task) -> TaskResult:
        """Submit a task for execution."""
        pass
    
    @abstractmethod
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if worker pool is healthy."""
        pass


class ThreadWorkerPool(WorkerPool):
    """Thread-based worker pool for I/O bound tasks."""
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = "TPUBenchmark"):
        self.max_workers = max_workers or min(32, (psutil.cpu_count() or 1) + 4)
        self.thread_name_prefix = thread_name_prefix
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = threading.Event()
        
    async def submit(self, task: Task) -> TaskResult:
        """Submit task to thread pool."""
        if self.shutdown_event.is_set():
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=RuntimeError("Worker pool is shutting down")
            )
        
        try:
            # Wrap task execution with monitoring
            future = self.executor.submit(self._execute_task, task)
            self.running_tasks[task.id] = future
            
            # Wait for completion with timeout
            timeout = task.timeout or 300  # 5 minute default
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: future.result(timeout=timeout)
            )
            
            return result
            
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Task {task.id} timed out after {timeout}s")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=TimeoutError(f"Task timed out after {timeout}s")
            )
        except Exception as e:
            self.logger.error(f"Task {task.id} failed: {e}")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=e
            )
        finally:
            self.running_tasks.pop(task.id, None)
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with error handling and retries."""
        worker_id = threading.current_thread().name
        started_at = datetime.now()
        
        for attempt in range(task.max_retries + 1):
            try:
                self.logger.debug(f"Executing task {task.id} (attempt {attempt + 1})")
                
                # Execute the task function
                result = task.func(*task.args, **task.kwargs)
                
                completed_at = datetime.now()
                execution_time = (completed_at - started_at).total_seconds()
                
                return TaskResult(
                    task_id=task.id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time=execution_time,
                    worker_id=worker_id
                )
                
            except Exception as e:
                self.logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {e}")
                
                if attempt < task.max_retries:
                    time.sleep(task.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    completed_at = datetime.now()
                    execution_time = (completed_at - started_at).total_seconds()
                    
                    return TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=e,
                        started_at=started_at,
                        completed_at=completed_at,
                        execution_time=execution_time,
                        worker_id=worker_id
                    )
    
    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=wait)
        self.logger.info("Thread worker pool shutdown")
    
    def is_healthy(self) -> bool:
        """Check if thread pool is healthy."""
        return not self.shutdown_event.is_set() and not self.executor._shutdown


class ProcessWorkerPool(WorkerPool):
    """Process-based worker pool for CPU-intensive tasks."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or psutil.cpu_count()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        )
        self.running_tasks: Dict[str, concurrent.futures.Future] = {}
        self.logger = logging.getLogger(__name__)
        self.shutdown_event = threading.Event()
    
    async def submit(self, task: Task) -> TaskResult:
        """Submit task to process pool."""
        if self.shutdown_event.is_set():
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=RuntimeError("Worker pool is shutting down")
            )
        
        try:
            # Process-safe task wrapper
            future = self.executor.submit(_execute_task_process, task)
            self.running_tasks[task.id] = future
            
            timeout = task.timeout or 600  # 10 minute default for processes
            result_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: future.result(timeout=timeout)
            )
            
            # Reconstruct TaskResult from serializable data
            result = TaskResult(**result_data)
            return result
            
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Process task {task.id} timed out after {timeout}s")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=TimeoutError(f"Task timed out after {timeout}s")
            )
        except Exception as e:
            self.logger.error(f"Process task {task.id} failed: {e}")
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=e
            )
        finally:
            self.running_tasks.pop(task.id, None)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the process pool."""
        self.shutdown_event.set()
        self.executor.shutdown(wait=wait)
        self.logger.info("Process worker pool shutdown")
    
    def is_healthy(self) -> bool:
        """Check if process pool is healthy."""
        return not self.shutdown_event.is_set() and not self.executor._shutdown


def _execute_task_process(task: Task) -> Dict[str, Any]:
    """Process-safe task execution function."""
    import os
    
    worker_id = f"process-{os.getpid()}"
    started_at = datetime.now()
    
    for attempt in range(task.max_retries + 1):
        try:
            result = task.func(*task.args, **task.kwargs)
            
            completed_at = datetime.now()
            execution_time = (completed_at - started_at).total_seconds()
            
            return {
                "task_id": task.id,
                "status": TaskStatus.COMPLETED,
                "result": result,
                "error": None,
                "started_at": started_at,
                "completed_at": completed_at,
                "execution_time": execution_time,
                "worker_id": worker_id
            }
            
        except Exception as e:
            if attempt < task.max_retries:
                time.sleep(task.retry_delay * (2 ** attempt))
                continue
            else:
                completed_at = datetime.now()
                execution_time = (completed_at - started_at).total_seconds()
                
                return {
                    "task_id": task.id,
                    "status": TaskStatus.FAILED,
                    "result": None,
                    "error": e,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "execution_time": execution_time,
                    "worker_id": worker_id
                }


class TaskScheduler:
    """Advanced task scheduler with dependency management and load balancing."""
    
    def __init__(self, 
                 thread_pool: Optional[ThreadWorkerPool] = None,
                 process_pool: Optional[ProcessWorkerPool] = None):
        
        self.thread_pool = thread_pool or ThreadWorkerPool()
        self.process_pool = process_pool or ProcessWorkerPool()
        
        self.task_queue = asyncio.PriorityQueue()
        self.pending_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.scheduler_task = None
        
        # Statistics
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'avg_execution_time': 0.0,
            'current_queue_size': 0
        }
        self.stats_lock = asyncio.Lock()
    
    async def start(self):
        """Start the task scheduler."""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler."""
        self.running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown worker pools
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
        
        self.logger.info("Task scheduler stopped")
    
    async def submit_task(self, task: Task, use_processes: bool = False) -> str:
        """Submit a task for execution."""
        # Add to dependency graph
        if task.dependencies:
            self.dependency_graph[task.id] = task.dependencies.copy()
        
        # Mark task metadata
        task.metadata['use_processes'] = use_processes
        
        # Add to pending tasks
        self.pending_tasks[task.id] = task
        
        # Add to queue if dependencies are satisfied
        if self._dependencies_satisfied(task):
            await self.task_queue.put(task)
        
        async with self.stats_lock:
            self.stats['total_submitted'] += 1
            self.stats['current_queue_size'] = self.task_queue.qsize()
        
        self.logger.debug(f"Submitted task {task.id} (use_processes={use_processes})")
        return task.id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Get task result, waiting if necessary."""
        start_time = time.time()
        
        while task_id not in self.completed_tasks:
            if timeout and (time.time() - start_time) > timeout:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=TimeoutError(f"Timeout waiting for task {task_id}")
                )
            
            await asyncio.sleep(0.1)
        
        return self.completed_tasks[task_id]
    
    async def get_results(self, task_ids: List[str], timeout: Optional[float] = None) -> List[TaskResult]:
        """Get multiple task results."""
        results = []
        for task_id in task_ids:
            result = await self.get_result(task_id, timeout)
            results.append(result)
        return results
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            if not self.completed_tasks[dep_id].is_successful:
                return False
        
        return True
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Get next task from queue (with timeout to check for new dependencies)
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check for newly available tasks due to completed dependencies
                    await self._check_dependency_resolution()
                    continue
                
                # Execute task
                use_processes = task.metadata.get('use_processes', False)
                worker_pool = self.process_pool if use_processes else self.thread_pool
                
                # Check worker pool health
                if not worker_pool.is_healthy():
                    self.logger.error(f"Worker pool unhealthy, failing task {task.id}")
                    result = TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=RuntimeError("Worker pool unhealthy")
                    )
                else:
                    result = await worker_pool.submit(task)
                
                # Store result
                self.completed_tasks[task.id] = result
                self.pending_tasks.pop(task.id, None)
                
                # Update statistics
                async with self.stats_lock:
                    if result.is_successful:
                        self.stats['total_completed'] += 1
                    else:
                        self.stats['total_failed'] += 1
                    
                    if result.execution_time:
                        # Update rolling average
                        total_tasks = self.stats['total_completed'] + self.stats['total_failed']
                        if total_tasks > 1:
                            self.stats['avg_execution_time'] = (
                                (self.stats['avg_execution_time'] * (total_tasks - 1) + result.execution_time) 
                                / total_tasks
                            )
                        else:
                            self.stats['avg_execution_time'] = result.execution_time
                    
                    self.stats['current_queue_size'] = self.task_queue.qsize()
                
                # Check for newly available tasks due to this completion
                await self._check_dependency_resolution()
                
                self.logger.debug(f"Completed task {task.id}: {result.status.value}")
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _check_dependency_resolution(self):
        """Check for tasks whose dependencies are now satisfied."""
        ready_tasks = []
        
        for task_id, task in list(self.pending_tasks.items()):
            if task_id not in [t.id for t in ready_tasks] and self._dependencies_satisfied(task):
                ready_tasks.append(task)
        
        # Add ready tasks to queue
        for task in ready_tasks:
            await self.task_queue.put(task)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        async with self.stats_lock:
            pool_stats = {
                'thread_pool_healthy': self.thread_pool.is_healthy(),
                'process_pool_healthy': self.process_pool.is_healthy(),
                'thread_pool_workers': self.thread_pool.max_workers,
                'process_pool_workers': self.process_pool.max_workers,
                'running_thread_tasks': len(self.thread_pool.running_tasks),
                'running_process_tasks': len(self.process_pool.running_tasks)
            }
            
            return {**self.stats, **pool_stats}


class BenchmarkJobManager:
    """High-level job manager for benchmark operations."""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_results: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the job manager."""
        await self.scheduler.start()
        self.logger.info("Benchmark job manager started")
    
    async def stop(self):
        """Stop the job manager."""
        await self.scheduler.stop()
        self.logger.info("Benchmark job manager stopped")
    
    async def run_benchmark_batch(self, models: List[str], configurations: List[Dict[str, Any]]) -> str:
        """Run a batch of benchmarks across multiple models and configurations."""
        import uuid
        
        job_id = str(uuid.uuid4())
        task_ids = []
        
        self.active_jobs[job_id] = {
            'models': models,
            'configurations': configurations,
            'started_at': datetime.now(),
            'task_ids': task_ids,
            'status': 'running'
        }
        
        # Create tasks for each model-configuration combination
        for model in models:
            for config in configurations:
                task_id = await self._create_benchmark_task(model, config)
                task_ids.append(task_id)
        
        self.active_jobs[job_id]['task_ids'] = task_ids
        self.logger.info(f"Started benchmark batch job {job_id} with {len(task_ids)} tasks")
        
        return job_id
    
    async def run_pipeline_benchmark(self, pipeline_config: Dict[str, Any]) -> str:
        """Run a complex benchmark pipeline with dependencies."""
        import uuid
        
        job_id = str(uuid.uuid4())
        
        # Create dependency chain: model_load -> compile -> benchmark -> analysis
        load_task_id = await self._create_model_load_task(pipeline_config['model'])
        compile_task_id = await self._create_compile_task(pipeline_config, dependencies=[load_task_id])
        benchmark_task_id = await self._create_benchmark_task_dep(pipeline_config, dependencies=[compile_task_id])
        analysis_task_id = await self._create_analysis_task(pipeline_config, dependencies=[benchmark_task_id])
        
        task_ids = [load_task_id, compile_task_id, benchmark_task_id, analysis_task_id]
        
        self.active_jobs[job_id] = {
            'pipeline_config': pipeline_config,
            'started_at': datetime.now(),
            'task_ids': task_ids,
            'status': 'running'
        }
        
        self.logger.info(f"Started pipeline benchmark job {job_id}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a benchmark job."""
        if job_id not in self.active_jobs and job_id not in self.job_results:
            return {'error': 'Job not found'}
        
        if job_id in self.job_results:
            return self.job_results[job_id]
        
        job_info = self.active_jobs[job_id]
        task_ids = job_info['task_ids']
        
        # Check task completion
        completed_count = 0
        failed_count = 0
        results = []
        
        for task_id in task_ids:
            if task_id in self.scheduler.completed_tasks:
                result = self.scheduler.completed_tasks[task_id]
                results.append(result.to_dict())
                
                if result.is_successful:
                    completed_count += 1
                else:
                    failed_count += 1
        
        total_tasks = len(task_ids)
        is_complete = completed_count + failed_count >= total_tasks
        
        status = {
            'job_id': job_id,
            'status': 'completed' if is_complete else 'running',
            'started_at': job_info['started_at'].isoformat(),
            'progress': {
                'total_tasks': total_tasks,
                'completed': completed_count,
                'failed': failed_count,
                'pending': total_tasks - completed_count - failed_count,
                'percentage': (completed_count + failed_count) / total_tasks * 100
            },
            'results': results
        }
        
        if is_complete:
            status['completed_at'] = datetime.now().isoformat()
            status['duration'] = (datetime.now() - job_info['started_at']).total_seconds()
            
            # Move to completed jobs
            self.job_results[job_id] = status
            del self.active_jobs[job_id]
        
        return status
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in self.active_jobs:
            return False
        
        # Mark job as cancelled
        job_info = self.active_jobs[job_id]
        job_info['status'] = 'cancelled'
        
        # Note: Individual task cancellation would require more complex implementation
        self.logger.info(f"Cancelled job {job_id}")
        return True
    
    async def _create_benchmark_task(self, model: str, config: Dict[str, Any]) -> str:
        """Create a benchmark task."""
        task = Task(
            id=f"benchmark_{model}_{hash(str(config))}",
            func=self._run_benchmark,
            args=(model, config),
            priority=TaskPriority.NORMAL,
            timeout=config.get('timeout', 300),
            max_retries=config.get('retries', 1)
        )
        
        return await self.scheduler.submit_task(task, use_processes=False)
    
    async def _create_model_load_task(self, model_path: str) -> str:
        """Create a model loading task."""
        task = Task(
            id=f"load_{hash(model_path)}",
            func=self._load_model,
            args=(model_path,),
            priority=TaskPriority.HIGH,
            timeout=60
        )
        
        return await self.scheduler.submit_task(task, use_processes=False)
    
    async def _create_compile_task(self, config: Dict[str, Any], dependencies: List[str]) -> str:
        """Create a model compilation task."""
        task = Task(
            id=f"compile_{hash(str(config))}",
            func=self._compile_model,
            args=(config,),
            dependencies=dependencies,
            priority=TaskPriority.HIGH,
            timeout=120
        )
        
        return await self.scheduler.submit_task(task, use_processes=True)  # CPU intensive
    
    async def _create_benchmark_task_dep(self, config: Dict[str, Any], dependencies: List[str]) -> str:
        """Create a benchmark task with dependencies."""
        task = Task(
            id=f"benchmark_dep_{hash(str(config))}",
            func=self._run_benchmark_with_model,
            args=(config,),
            dependencies=dependencies,
            priority=TaskPriority.NORMAL,
            timeout=300
        )
        
        return await self.scheduler.submit_task(task, use_processes=False)
    
    async def _create_analysis_task(self, config: Dict[str, Any], dependencies: List[str]) -> str:
        """Create an analysis task."""
        task = Task(
            id=f"analysis_{hash(str(config))}",
            func=self._analyze_results,
            args=(config,),
            dependencies=dependencies,
            priority=TaskPriority.LOW,
            timeout=60
        )
        
        return await self.scheduler.submit_task(task, use_processes=True)  # CPU intensive
    
    def _run_benchmark(self, model: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark (mock implementation)."""
        time.sleep(1.0)  # Simulate benchmark time
        return {
            'model': model,
            'throughput': 850.5,
            'latency': 1.2,
            'power': 0.85,
            'config': config
        }
    
    def _load_model(self, model_path: str) -> Dict[str, Any]:
        """Load model (mock implementation)."""
        time.sleep(0.5)  # Simulate loading time
        return {'model_path': model_path, 'loaded': True}
    
    def _compile_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Compile model (mock implementation)."""
        time.sleep(2.0)  # Simulate compilation time
        return {'compiled': True, 'optimizations': ['fusion', 'quantization']}
    
    def _run_benchmark_with_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark with compiled model (mock implementation)."""
        time.sleep(1.5)  # Simulate benchmark time
        return {
            'throughput': 950.2,
            'latency': 1.05,
            'power': 0.78,
            'config': config
        }
    
    def _analyze_results(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results (mock implementation)."""
        time.sleep(0.3)  # Simulate analysis time
        return {
            'efficiency_score': 95.2,
            'recommendations': ['Consider higher batch size'],
            'bottlenecks': ['Memory bandwidth']
        }


# Utility functions and context managers
@contextmanager
def benchmark_job_manager():
    """Context manager for benchmark job manager."""
    manager = BenchmarkJobManager()
    
    async def run_context():
        await manager.start()
        try:
            yield manager
        finally:
            await manager.stop()
    
    return run_context()


# Global job manager instance
_job_manager = None


async def get_job_manager() -> BenchmarkJobManager:
    """Get global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = BenchmarkJobManager()
        await _job_manager.start()
    return _job_manager