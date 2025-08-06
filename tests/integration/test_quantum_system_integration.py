"""Integration tests for quantum system components."""

import asyncio
import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.quantum_planner import (
    QuantumTaskPlanner, QuantumTask, QuantumResource, QuantumState
)
from edge_tpu_v5_benchmark.quantum_validation import (
    QuantumTaskValidator, QuantumSystemValidator, ValidationSeverity
)
from edge_tpu_v5_benchmark.quantum_monitoring import (
    QuantumHealthMonitor, MetricsCollector, HealthStatus
)
from edge_tpu_v5_benchmark.quantum_security import (
    QuantumSecurityManager, SecureQuantumTaskPlanner, SecurityPolicy, SecurityLevel
)
from edge_tpu_v5_benchmark.quantum_performance import (
    OptimizedQuantumTaskPlanner, PerformanceProfile, OptimizationStrategy
)


class TestQuantumValidationIntegration:
    """Test quantum validation system integration."""
    
    def test_task_validator_integration(self):
        """Test task validator with real planner."""
        planner = QuantumTaskPlanner()
        validator = QuantumTaskValidator()
        
        # Valid task
        valid_task = QuantumTask(
            id="valid_task",
            name="Valid Task",
            priority=1.0,
            complexity=1.0,
            estimated_duration=10.0,
            resource_requirements={"cpu_cores": 2.0, "memory_gb": 4.0}
        )
        
        report = validator.validate_task(valid_task, planner.resources)
        
        assert report.total_issues >= 0
        assert report.validation_time > 0
        
        # Invalid task
        invalid_task = QuantumTask(
            id="",  # Invalid empty ID
            name="Invalid Task",
            priority=-1.0,  # Invalid negative priority
            complexity=0.0,  # Invalid zero complexity
            estimated_duration=-5.0,  # Invalid negative duration
            resource_requirements={"nonexistent": 1000.0}  # Invalid resource
        )
        
        report = validator.validate_task(invalid_task, planner.resources)
        
        assert report.total_issues > 0
        assert report.error_issues > 0
        assert not report.is_valid
    
    def test_system_validator_integration(self):
        """Test system validator with complete planner state."""
        planner = QuantumTaskPlanner()
        validator = QuantumSystemValidator()
        
        # Add valid tasks
        task1 = QuantumTask(id="task1", name="Task 1")
        task2 = QuantumTask(id="task2", name="Task 2", dependencies={"task1"})
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.entangle_tasks("task1", "task2")
        
        # Validate system
        report = validator.validate_system(planner.tasks, planner.resources)
        
        assert report.validation_time > 0
        assert report.total_issues >= 0  # May have warnings but should be functional
        
        # Test circular dependency detection
        task3 = QuantumTask(id="task3", name="Task 3", dependencies={"task4"})
        task4 = QuantumTask(id="task4", name="Task 4", dependencies={"task3"})
        
        planner.add_task(task3)
        planner.add_task(task4)
        
        report_with_cycle = validator.validate_system(planner.tasks, planner.resources)
        
        assert report_with_cycle.critical_issues > 0
        assert not report_with_cycle.is_valid
    
    def test_validation_with_execution(self):
        """Test validation during task execution."""
        planner = QuantumTaskPlanner()
        validator = QuantumTaskValidator()
        
        # Create task that should pass validation
        task = QuantumTask(
            id="execution_test",
            name="Execution Test",
            estimated_duration=0.1,
            resource_requirements={"cpu_cores": 1.0}
        )
        
        # Validate before adding
        report = validator.validate_task(task, planner.resources)
        assert report.is_valid or report.error_issues == 0
        
        planner.add_task(task)
        
        # System should remain valid
        system_report = validator.validate_system(planner.tasks, planner.resources)
        assert not system_report.has_blocking_issues()


class TestQuantumMonitoringIntegration:
    """Test quantum monitoring system integration."""
    
    @pytest.mark.asyncio
    async def test_health_monitor_integration(self):
        """Test health monitor with real planner."""
        planner = QuantumTaskPlanner()
        monitor = QuantumHealthMonitor(planner)
        
        # Add some tasks
        for i in range(3):
            task = QuantumTask(
                id=f"monitor_task_{i}",
                name=f"Monitor Task {i}",
                estimated_duration=0.1
            )
            planner.add_task(task)
        
        # Get initial health status
        health_status = monitor.get_current_health_status()
        
        assert "overall_status" in health_status
        assert "health_checks" in health_status
        assert "metrics_summary" in health_status
        assert len(health_status["health_checks"]) > 0
        
        # Run some execution cycles and monitor
        for _ in range(2):
            await planner.run_quantum_execution_cycle()
            await asyncio.sleep(0.01)
        
        # Get updated health status
        updated_health_status = monitor.get_current_health_status()
        assert "timestamp" in updated_health_status
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self):
        """Test continuous monitoring functionality."""
        planner = QuantumTaskPlanner()
        monitor = QuantumHealthMonitor(planner)
        
        # Start monitoring
        await monitor.start_monitoring(interval=0.1)
        
        # Add and execute tasks
        task = QuantumTask(id="monitored", name="Monitored Task", estimated_duration=0.05)
        planner.add_task(task)
        
        await planner.run_quantum_execution_cycle()
        
        # Wait for monitoring cycle
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Check that metrics were collected
        metrics_history = monitor.metrics_collector.get_performance_history()
        assert len(metrics_history) > 0
    
    def test_metrics_collection(self):
        """Test metrics collection during execution."""
        planner = QuantumTaskPlanner()
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_metric("test_metric", 1.5, {"source": "test"})
        collector.record_task_execution("task1", 2.0, True)
        collector.record_task_execution("task2", 1.0, False)
        
        # Get metric history
        history = collector.get_metric_history("test_metric")
        assert len(history) == 1
        assert history[0].value == 1.5
        
        # Get execution stats
        stats = collector.get_task_execution_stats()
        assert stats["total_executions"] == 2
        assert stats["successful_executions"] == 1
        assert stats["failed_executions"] == 1
        assert stats["success_rate"] == 0.5


class TestQuantumSecurityIntegration:
    """Test quantum security system integration."""
    
    def test_secure_planner_creation(self):
        """Test secure quantum planner creation."""
        security_policy = SecurityPolicy(
            max_execution_time=30.0,
            max_memory_usage=1024 * 1024 * 1024,  # 1GB
            require_signature=False,
            audit_enabled=True
        )
        
        secure_planner = SecureQuantumTaskPlanner(security_policy=security_policy)
        
        assert secure_planner.security_manager is not None
        assert secure_planner.security_manager.policy == security_policy
    
    def test_task_security_validation(self):
        """Test task security validation during addition."""
        secure_planner = SecureQuantumTaskPlanner()
        
        # Valid task
        valid_task = QuantumTask(
            id="secure_task",
            name="Secure Task",
            estimated_duration=10.0,
            complexity=2.0
        )
        
        secure_planner.add_task(valid_task, SecurityLevel.INTERNAL)
        
        assert "secure_task" in secure_planner.tasks
        assert secure_planner.security_manager.security_contexts["secure_task"] == SecurityLevel.INTERNAL
    
    def test_task_security_rejection(self):
        """Test task rejection due to security violations."""
        policy = SecurityPolicy(
            max_execution_time=1.0,  # Very low limit
            max_memory_usage=1024   # Very low limit
        )
        secure_planner = SecureQuantumTaskPlanner(security_policy=policy)
        
        # Task that violates security policy
        dangerous_task = QuantumTask(
            id="dangerous",
            name="Dangerous Task",
            estimated_duration=10.0,  # Exceeds limit
            resource_requirements={"memory_gb": 10.0}  # Exceeds limit
        )
        
        with pytest.raises(Exception):  # Should raise BenchmarkError
            secure_planner.add_task(dangerous_task, SecurityLevel.RESTRICTED)
    
    @pytest.mark.asyncio
    async def test_secure_task_execution(self):
        """Test secure task execution with monitoring."""
        secure_planner = SecureQuantumTaskPlanner()
        
        task = QuantumTask(
            id="secure_exec",
            name="Secure Execution",
            estimated_duration=0.1
        )
        
        secure_planner.add_task(task, SecurityLevel.INTERNAL)
        
        # Execute with security monitoring
        result = await secure_planner.execute_task(task)
        
        assert "task_id" in result
        assert result["task_id"] == "secure_exec"
        
        # Check that security audit was recorded
        audit_summary = secure_planner.security_manager.auditor.get_audit_summary(hours=1)
        assert audit_summary["total_events"] > 0
    
    def test_task_signing_and_verification(self):
        """Test task digital signing and verification."""
        security_manager = QuantumSecurityManager()
        
        task = QuantumTask(
            id="signed_task",
            name="Signed Task",
            priority=2.0,
            complexity=1.5
        )
        
        # Sign task
        signature = security_manager.sign_task(task)
        assert len(signature) > 0
        
        # Verify signature
        is_valid = security_manager.signer.verify_task_signature(task, signature)
        assert is_valid
        
        # Modify task and verify signature fails
        task.priority = 3.0
        is_valid_after_change = security_manager.signer.verify_task_signature(task, signature)
        assert not is_valid_after_change


class TestQuantumPerformanceIntegration:
    """Test quantum performance optimization integration."""
    
    def test_optimized_planner_creation(self):
        """Test optimized planner creation with different profiles."""
        profiles = [
            PerformanceProfile(strategy=OptimizationStrategy.LATENCY_FIRST),
            PerformanceProfile(strategy=OptimizationStrategy.THROUGHPUT_FIRST),
            PerformanceProfile(strategy=OptimizationStrategy.BALANCED)
        ]
        
        for profile in profiles:
            planner = OptimizedQuantumTaskPlanner(performance_profile=profile)
            
            assert planner.profile == profile
            assert planner.cache is not None
            assert planner.resource_pool is not None
            assert planner.executor is not None
    
    @pytest.mark.asyncio
    async def test_performance_optimization_execution(self):
        """Test performance-optimized task execution."""
        profile = PerformanceProfile(
            strategy=OptimizationStrategy.THROUGHPUT_FIRST,
            max_concurrent_tasks=2,
            cache_enabled=True,
            batch_size=2
        )
        
        planner = OptimizedQuantumTaskPlanner(performance_profile=profile)
        
        # Add multiple tasks
        tasks = []
        for i in range(4):
            task = QuantumTask(
                id=f"perf_task_{i}",
                name=f"Performance Task {i}",
                estimated_duration=0.05
            )
            tasks.append(task)
            planner.add_task(task)
        
        # Execute with performance optimizations
        results = await planner.run_quantum_execution_cycle()
        
        assert "optimization_stats" in results
        assert "cache_stats" in results
        assert "executor_stats" in results
        
        # Check that some optimization was applied
        opt_stats = results["optimization_stats"]
        assert "cache_hit_rate" in opt_stats
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self):
        """Test adaptive caching behavior."""
        profile = PerformanceProfile(cache_enabled=True)
        planner = OptimizedQuantumTaskPlanner(performance_profile=profile)
        
        # Create identical tasks for cache testing
        task1 = QuantumTask(
            id="cache_test_1",
            name="Cache Test",
            complexity=1.0,
            resource_requirements={"cpu_cores": 1.0}
        )
        
        task2 = QuantumTask(
            id="cache_test_2",
            name="Cache Test",  # Same name/config for cache hit
            complexity=1.0,
            resource_requirements={"cpu_cores": 1.0}
        )
        
        planner.add_task(task1)
        planner.add_task(task2)
        
        # Execute both tasks
        await planner.execute_task(task1)
        result2 = await planner.execute_task(task2)
        
        # Second execution might be cached
        cache_stats = planner.cache.get_statistics()
        assert cache_stats["total_size_mb"] >= 0
    
    def test_resource_pool_management(self):
        """Test resource pool optimization."""
        planner = OptimizedQuantumTaskPlanner()
        
        # Get resource pool statistics
        pool_stats = planner.resource_pool.get_pool_statistics()
        
        assert isinstance(pool_stats, dict)
        assert len(pool_stats) > 0
        
        # Each resource type should have pool info
        for resource_type, stats in pool_stats.items():
            assert "resource_count" in stats
            assert "total_capacity" in stats
            assert "utilization" in stats
    
    @pytest.mark.asyncio
    async def test_performance_report_generation(self):
        """Test performance report generation."""
        planner = OptimizedQuantumTaskPlanner()
        
        # Execute some tasks to generate data
        task = QuantumTask(id="report_test", name="Report Test", estimated_duration=0.05)
        planner.add_task(task)
        await planner.run_quantum_execution_cycle()
        
        # Generate performance report
        report = planner.get_performance_report()
        
        assert "profile" in report
        assert "optimization_metrics" in report
        assert "cache_statistics" in report
        assert "resource_pool_statistics" in report
        assert "executor_statistics" in report
        assert "system_state" in report
        
        # Verify report structure
        profile_info = report["profile"]
        assert "strategy" in profile_info
        assert "max_concurrent_tasks" in profile_info


class TestQuantumSystemEndToEnd:
    """End-to-end integration tests for complete quantum system."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_with_all_components(self):
        """Test complete workflow with validation, monitoring, security, and optimization."""
        # Create secure optimized planner
        security_policy = SecurityPolicy(audit_enabled=True)
        performance_profile = PerformanceProfile(
            strategy=OptimizationStrategy.BALANCED,
            cache_enabled=True
        )
        
        planner = SecureQuantumTaskPlanner(security_policy=security_policy)
        
        # Add health monitoring
        monitor = QuantumHealthMonitor(planner)
        
        # Create workflow tasks
        workflow_tasks = [
            {
                "id": "init",
                "name": "Initialize",
                "duration": 0.05,
                "security": SecurityLevel.PUBLIC
            },
            {
                "id": "process",
                "name": "Process Data",
                "duration": 0.1,
                "deps": ["init"],
                "security": SecurityLevel.INTERNAL
            },
            {
                "id": "finalize",
                "name": "Finalize",
                "duration": 0.05,
                "deps": ["process"],
                "security": SecurityLevel.INTERNAL
            }
        ]
        
        # Add tasks to planner
        for task_data in workflow_tasks:
            task = QuantumTask(
                id=task_data["id"],
                name=task_data["name"],
                estimated_duration=task_data["duration"],
                dependencies=set(task_data.get("deps", []))
            )
            planner.add_task(task, task_data["security"])
        
        # Start monitoring
        await monitor.start_monitoring(interval=0.1)
        
        # Execute workflow
        max_cycles = 10
        cycle_count = 0
        
        while planner.get_ready_tasks() and cycle_count < max_cycles:
            cycle_count += 1
            
            # Run execution cycle
            results = await planner.run_quantum_execution_cycle()
            
            # Verify results structure
            assert "tasks_executed" in results
            assert "cycle_duration" in results
            
            await asyncio.sleep(0.01)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Verify workflow completion
        assert len(planner.completed_tasks) == len(workflow_tasks)
        
        # Check health status
        health_status = monitor.get_current_health_status()
        assert health_status["overall_status"] in ["healthy", "warning", "degraded"]
        
        # Check security audit
        audit_summary = planner.security_manager.auditor.get_audit_summary(hours=1)
        assert audit_summary["total_events"] > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self):
        """Test system resilience with various error conditions."""
        planner = QuantumTaskPlanner()
        monitor = QuantumHealthMonitor(planner)
        
        # Add tasks with potential issues
        tasks = [
            QuantumTask(id="normal", name="Normal Task", estimated_duration=0.05),
            QuantumTask(id="resource_heavy", name="Resource Heavy", 
                       resource_requirements={"memory_gb": 1000.0}),  # Excessive requirement
            QuantumTask(id="dependent", name="Dependent Task", dependencies={"nonexistent"})
        ]
        
        for task in tasks:
            planner.add_task(task)
        
        # Execute with error handling
        results = await planner.run_quantum_execution_cycle()
        
        # System should handle errors gracefully
        assert "tasks_executed" in results
        assert "tasks_failed" in results
        total_processed = len(results["tasks_executed"]) + len(results["tasks_failed"])
        assert total_processed <= len(tasks)  # Some tasks may not be ready
        
        # Health monitor should detect issues
        health_status = monitor.get_current_health_status()
        assert health_status["overall_status"] in ["healthy", "warning", "degraded", "critical"]
    
    def test_configuration_and_persistence(self):
        """Test configuration management and state persistence."""
        planner = QuantumTaskPlanner()
        
        # Add tasks and execute some
        task1 = QuantumTask(id="persist1", name="Persist Task 1")
        task2 = QuantumTask(id="persist2", name="Persist Task 2", dependencies={"persist1"})
        
        planner.add_task(task1)
        planner.add_task(task2)
        planner.entangle_tasks("persist1", "persist2")
        planner.completed_tasks.add("persist1")
        
        # Export state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            planner.export_quantum_state(export_path)
            
            # Verify export file exists and has content
            assert Path(export_path).exists()
            
            with open(export_path) as f:
                exported_data = json.load(f)
            
            assert "system_state" in exported_data
            assert "tasks" in exported_data
            assert "entanglement_graph" in exported_data
            
            # Verify task data
            assert "persist1" in exported_data["tasks"]
            assert "persist2" in exported_data["tasks"]
            assert exported_data["tasks"]["persist1"]["name"] == "Persist Task 1"
            
            # Verify entanglement
            assert "persist1" in exported_data["entanglement_graph"]
            assert "persist2" in exported_data["entanglement_graph"]["persist1"]
            
        finally:
            # Cleanup
            Path(export_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_scalability_and_performance(self):
        """Test system scalability with larger numbers of tasks."""
        # Use optimized planner for better performance
        profile = PerformanceProfile(
            strategy=OptimizationStrategy.THROUGHPUT_FIRST,
            max_concurrent_tasks=4,
            batch_size=3
        )
        planner = OptimizedQuantumTaskPlanner(performance_profile=profile)
        
        # Create many tasks
        num_tasks = 20
        tasks = []
        
        for i in range(num_tasks):
            task = QuantumTask(
                id=f"scale_task_{i}",
                name=f"Scale Task {i}",
                priority=float(i % 5 + 1),  # Vary priorities
                complexity=float(i % 3 + 1),  # Vary complexity
                estimated_duration=0.01,  # Very fast execution
                resource_requirements={"cpu_cores": float(i % 2 + 1)}
            )
            tasks.append(task)
            planner.add_task(task)
        
        start_time = time.time()
        
        # Execute all tasks
        max_cycles = 50
        cycle_count = 0
        
        while planner.get_ready_tasks() and cycle_count < max_cycles:
            cycle_count += 1
            results = await planner.run_quantum_execution_cycle()
            
            # Brief pause to allow system breathing room
            await asyncio.sleep(0.001)
        
        execution_time = time.time() - start_time
        
        # Verify performance
        assert len(planner.completed_tasks) > 0  # Some tasks should complete
        assert execution_time < 10.0  # Should complete reasonably quickly
        assert cycle_count < max_cycles  # Should not hit cycle limit
        
        # Get final performance report
        performance_report = planner.get_performance_report()
        
        assert performance_report["optimization_metrics"]["cache_hit_rate"] >= 0.0
        assert performance_report["system_state"]["total_tasks"] == num_tasks
        
        # Cleanup
        await planner.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])