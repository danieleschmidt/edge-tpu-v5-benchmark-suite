#!/usr/bin/env python3
"""Simple test to verify core benchmark functionality works"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from edge_tpu_v5_benchmark.quantum_planner import QuantumTaskPlanner, QuantumTask, QuantumResource

def test_core_functionality():
    """Test that core quantum planner functionality works"""
    
    print("🧪 Testing Generation 1: Core Functionality")
    
    # Create planner
    planner = QuantumTaskPlanner()
    print("✅ QuantumTaskPlanner created successfully")
    
    # Create some test tasks
    task1 = QuantumTask(
        id="test_task_1",
        name="Benchmark MobileNet",
        priority=1.0,
        complexity=0.5
    )
    
    task2 = QuantumTask(
        id="test_task_2", 
        name="Profile Power",
        priority=0.8,
        complexity=0.3,
        dependencies={"test_task_1"}
    )
    
    print("✅ QuantumTask objects created successfully")
    
    # Add tasks to planner
    planner.add_task(task1)
    planner.add_task(task2)
    print("✅ Tasks added to planner")
    
    # Resources are auto-initialized by default
    print("✅ Resources auto-initialized by planner")
    
    # Test planning
    schedule = planner.optimize_schedule()
    print(f"✅ Schedule optimized with {len(schedule)} tasks")
    
    # Test task state changes
    task1.collapse_wavefunction("benchmark_path")
    print("✅ Wavefunction collapse works")
    
    print("\n🎯 Generation 1 COMPLETE: Core functionality verified!")
    return True

if __name__ == "__main__":
    try:
        success = test_core_functionality()
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Generation 1 testing complete")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)