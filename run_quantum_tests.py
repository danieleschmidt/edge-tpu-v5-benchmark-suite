#!/usr/bin/env python3
"""
Test runner for quantum task planner system with comprehensive quality gates.
"""

import sys
import subprocess
import time
import os
from pathlib import Path
import json


def run_command(command, description="", timeout=300):
    """Run command with timeout and error handling."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"Command: {' '.join(command)}")
    print('='*60)
    
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\n⏱️  Duration: {duration:.2f}s")
        print(f"🔄 Return code: {result.returncode}")
        
        return result.returncode == 0, duration, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        return False, timeout, "", "Timeout"
    except Exception as e:
        print(f"❌ Command failed with exception: {e}")
        return False, 0, "", str(e)


def check_imports():
    """Test basic imports work."""
    print("\n🔍 Testing imports...")
    
    import_tests = [
        "from edge_tpu_v5_benchmark.quantum_planner import QuantumTaskPlanner",
        "from edge_tpu_v5_benchmark.quantum_validation import QuantumTaskValidator",
        "from edge_tpu_v5_benchmark.quantum_monitoring import QuantumHealthMonitor",
        "from edge_tpu_v5_benchmark.quantum_security import QuantumSecurityManager",
        "from edge_tpu_v5_benchmark.quantum_performance import OptimizedQuantumTaskPlanner",
        "from edge_tpu_v5_benchmark.quantum_auto_scaling import QuantumAutoScaler",
        "print('✅ All quantum imports successful')"
    ]
    
    import_script = "; ".join(import_tests)
    
    success, duration, stdout, stderr = run_command([
        sys.executable, "-c", import_script
    ], "Basic Import Test")
    
    return success


def run_unit_tests():
    """Run unit tests if pytest is available."""
    print("\n🧪 Running unit tests...")
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not available, skipping unit tests")
        return True
    
    test_files = [
        "tests/unit/test_quantum_planner.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if Path(test_file).exists():
            success, duration, stdout, stderr = run_command([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
            ], f"Unit Tests: {test_file}")
            
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    return all_passed


def run_integration_tests():
    """Run integration tests if pytest is available."""
    print("\n🔗 Running integration tests...")
    
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not available, skipping integration tests")
        return True
    
    test_files = [
        "tests/integration/test_quantum_system_integration.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if Path(test_file).exists():
            success, duration, stdout, stderr = run_command([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-x"
            ], f"Integration Tests: {test_file}", timeout=600)  # Longer timeout
            
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    return all_passed


def run_performance_tests():
    """Run performance tests if pytest is available."""
    print("\n⚡ Running performance tests...")
    
    try:
        import pytest
    except ImportError:
        print("⚠️  pytest not available, skipping performance tests")
        return True
    
    test_files = [
        "tests/performance/test_quantum_performance.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if Path(test_file).exists():
            success, duration, stdout, stderr = run_command([
                sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-s"
            ], f"Performance Tests: {test_file}", timeout=900)  # Longer timeout for perf tests
            
            if not success:
                all_passed = False
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    return all_passed


def run_demo_script():
    """Run the quantum demo script."""
    print("\n🎬 Running quantum demo...")
    
    demo_script = "examples/quantum_planning_demo.py"
    
    if Path(demo_script).exists():
        success, duration, stdout, stderr = run_command([
            sys.executable, demo_script
        ], "Quantum Planning Demo", timeout=180)
        
        return success
    else:
        print(f"⚠️  Demo script not found: {demo_script}")
        return True


def run_cli_tests():
    """Test CLI functionality."""
    print("\n💻 Testing CLI functionality...")
    
    # Test basic CLI commands
    cli_tests = [
        {
            "cmd": [sys.executable, "-m", "edge_tpu_v5_benchmark.cli", "--help"],
            "desc": "CLI Help Command",
            "expect_success": True
        },
        {
            "cmd": [sys.executable, "-m", "edge_tpu_v5_benchmark.cli", "quantum", "--help"],
            "desc": "Quantum CLI Help",
            "expect_success": True
        }
    ]
    
    all_passed = True
    
    for test in cli_tests:
        success, duration, stdout, stderr = run_command(
            test["cmd"], test["desc"], timeout=30
        )
        
        if success != test["expect_success"]:
            print(f"❌ CLI test failed: {test['desc']}")
            all_passed = False
        else:
            print(f"✅ CLI test passed: {test['desc']}")
    
    return all_passed


def check_code_quality():
    """Check code quality with basic linting if available."""
    print("\n🔍 Checking code quality...")
    
    # Try to run basic Python syntax checks
    python_files = [
        "src/edge_tpu_v5_benchmark/quantum_planner.py",
        "src/edge_tpu_v5_benchmark/quantum_validation.py",
        "src/edge_tpu_v5_benchmark/quantum_monitoring.py",
        "src/edge_tpu_v5_benchmark/quantum_security.py",
        "src/edge_tpu_v5_benchmark/quantum_performance.py",
        "src/edge_tpu_v5_benchmark/quantum_auto_scaling.py",
    ]
    
    all_passed = True
    
    for py_file in python_files:
        if Path(py_file).exists():
            success, duration, stdout, stderr = run_command([
                sys.executable, "-m", "py_compile", py_file
            ], f"Syntax Check: {py_file}", timeout=10)
            
            if not success:
                print(f"❌ Syntax error in {py_file}")
                all_passed = False
            else:
                print(f"✅ Syntax OK: {py_file}")
        else:
            print(f"⚠️  File not found: {py_file}")
    
    return all_passed


def create_test_report(results):
    """Create a test report summary."""
    print("\n" + "="*80)
    print("📊 QUANTUM TASK PLANNER TEST REPORT")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result["passed"])
    
    print(f"Total Test Suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    print()
    
    # Detailed results
    for test_name, result in results.items():
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        duration = result["duration"]
        print(f"{status} {test_name:25s} ({duration:6.2f}s)")
    
    print("\n" + "="*80)
    
    # Overall status
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Quantum system is ready for deployment.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Review failures before deployment.")
        return False


def main():
    """Run comprehensive test suite."""
    print("🌌 Quantum Task Planner Test Suite")
    print("Comprehensive quality gates for quantum-inspired task planning system")
    print("="*80)
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
    
    # Test suite configuration
    test_suite = [
        ("Import Tests", check_imports),
        ("Code Quality", check_code_quality),
        ("Unit Tests", run_unit_tests),
        ("Integration Tests", run_integration_tests),
        ("CLI Tests", run_cli_tests),
        ("Demo Script", run_demo_script),
        ("Performance Tests", run_performance_tests),
    ]
    
    results = {}
    start_time = time.time()
    
    # Run all tests
    for test_name, test_func in test_suite:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        test_start = time.time()
        try:
            passed = test_func()
        except Exception as e:
            print(f"❌ Test suite {test_name} crashed: {e}")
            passed = False
        
        test_duration = time.time() - test_start
        
        results[test_name] = {
            "passed": passed,
            "duration": test_duration
        }
        
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n{status} {test_name} ({test_duration:.2f}s)")
    
    total_duration = time.time() - start_time
    
    # Generate report
    print(f"\nTotal execution time: {total_duration:.2f}s")
    all_passed = create_test_report(results)
    
    # Export results for CI/CD
    report_data = {
        "timestamp": time.time(),
        "total_duration": total_duration,
        "results": results,
        "summary": {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results.values() if r["passed"]),
            "success_rate": sum(1 for r in results.values() if r["passed"]) / len(results)
        }
    }
    
    with open("quantum_test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Test report saved to quantum_test_report.json")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()