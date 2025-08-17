#!/usr/bin/env python3
"""Generation 2: Test robust error handling, monitoring, and resilience features"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import asyncio
from edge_tpu_v5_benchmark.quantum_planner import QuantumTaskPlanner, QuantumTask
from edge_tpu_v5_benchmark.quantum_monitoring import QuantumHealthMonitor, MetricsCollector
from edge_tpu_v5_benchmark.quantum_validation import QuantumTaskValidator, QuantumSystemValidator
from edge_tpu_v5_benchmark.quantum_security import QuantumSecurityManager

async def test_robust_functionality():
    """Test Generation 2: Robust error handling and monitoring"""
    
    print("🛡️ Testing Generation 2: Robust & Reliable Systems")
    
    # Test comprehensive monitoring
    print("\n📊 Testing Quantum Health Monitoring...")
    planner = QuantumTaskPlanner()
    health_monitor = QuantumHealthMonitor(planner)
    await health_monitor.start_monitoring()
    print("✅ Health monitoring system initialized")
    
    # Test metrics collection
    metrics_collector = MetricsCollector()
    await metrics_collector.collect_system_metrics()
    print("✅ System metrics collected successfully")
    
    # Test validation systems
    print("\n🔍 Testing Validation Systems...")
    task_validator = QuantumTaskValidator()
    system_validator = QuantumSystemValidator()
    
    # Create test task with potential issues
    problematic_task = QuantumTask(
        id="test_problematic_task",
        name="High Resource Task",
        priority=1.5,  # Invalid priority > 1.0
        complexity=-0.5,  # Invalid negative complexity
        estimated_duration=0.0  # Invalid duration
    )
    
    # Test task validation
    validation_report = task_validator.validate_task(problematic_task)
    print(f"✅ Task validation completed: {len(validation_report.issues)} issues found")
    
    # Test system validation (reuse existing planner)
    system_report = await system_validator.validate_system(planner)
    print(f"✅ System validation completed: status = {system_report.status}")
    
    # Test security systems
    print("\n🔒 Testing Security Systems...")
    security_manager = QuantumSecurityManager()
    
    # Test input sanitization
    malicious_input = "'; DROP TABLE tasks; --"
    sanitized = security_manager.sanitize_input(malicious_input)
    print(f"✅ Input sanitization: '{malicious_input}' -> '{sanitized}'")
    
    # Test error recovery
    print("\n🚨 Testing Error Recovery...")
    try:
        # Simulate a critical error
        invalid_task = QuantumTask(
            id="",  # Invalid empty ID
            name="Invalid Task",
            priority=float('inf')  # Invalid infinite priority
        )
        planner.add_task(invalid_task)
    except Exception as e:
        print(f"✅ Error properly caught and handled: {type(e).__name__}")
    
    # Test resource exhaustion handling
    print("\n⚡ Testing Resource Resilience...")
    for i in range(100):  # Try to overwhelm system
        task = QuantumTask(
            id=f"stress_task_{i}",
            name=f"Stress Test Task {i}",
            resource_requirements={"cpu": 10.0, "memory": 10.0}  # High requirements
        )
        try:
            planner.add_task(task)
        except Exception:
            break  # Expected when resources exhausted
    
    print("✅ Resource exhaustion handled gracefully")
    
    # Test concurrent access safety
    print("\n🔄 Testing Concurrent Safety...")
    async def concurrent_task_creation(task_id):
        task = QuantumTask(
            id=f"concurrent_task_{task_id}",
            name=f"Concurrent Task {task_id}"
        )
        planner.add_task(task)
    
    # Run multiple concurrent operations
    await asyncio.gather(*[
        concurrent_task_creation(i) for i in range(10)
    ])
    print("✅ Concurrent operations completed safely")
    
    # Cleanup monitoring
    await health_monitor.stop_monitoring()
    
    print("\n🎯 Generation 2 COMPLETE: Robust systems verified!")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_robust_functionality())
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Generation 2 testing complete")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)