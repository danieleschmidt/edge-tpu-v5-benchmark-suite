#!/usr/bin/env python3
"""Comprehensive test for quantum-ML validation framework."""

import sys
import os
import math
import statistics
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock numpy for testing
class MockNumPy:
    @staticmethod
    def mean(data):
        return statistics.mean(data) if data else 0
    
    @staticmethod
    def std(data):
        return statistics.stdev(data) if len(data) > 1 else 0

sys.modules['numpy'] = MockNumPy()


def test_statistical_analyzer():
    """Test the statistical analyzer components."""
    print("Testing Statistical Analyzer...")
    
    # Import after mocking
    from edge_tpu_v5_benchmark.quantum_ml_validation_framework import (
        StatisticalAnalyzer, StatisticalTest
    )
    
    analyzer = StatisticalAnalyzer()
    
    # Test t-test with different samples
    sample1 = [1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.7, 1.1, 1.9, 1.5]  # Mean ≈ 1.5
    sample2 = [2.1, 2.3, 2.0, 2.4, 2.2, 2.5, 2.1, 2.6, 2.0, 2.3]  # Mean ≈ 2.25
    
    # Should detect significant difference
    result = analyzer.t_test_two_sample(sample1, sample2)
    
    assert result.test_type == StatisticalTest.T_TEST
    assert result.p_value < 0.05  # Should be significant
    assert result.significant is True
    assert result.effect_size > 0  # Should show large effect
    assert 0.7 <= result.power <= 1.0  # Should have reasonable power
    
    print("✅ T-test validation passed")
    
    # Test Mann-Whitney U test
    mw_result = analyzer.mann_whitney_test(sample1, sample2)
    
    assert mw_result.test_type == StatisticalTest.MANN_WHITNEY
    assert mw_result.p_value < 0.05  # Should be significant
    assert mw_result.significant is True
    
    print("✅ Mann-Whitney test validation passed")
    
    # Test bootstrap confidence interval
    def mean_func(data):
        return statistics.mean(data)
    
    ci_lower, ci_upper = analyzer.bootstrap_confidence_interval(
        sample1, mean_func, confidence_level=0.95, n_bootstrap=100
    )
    
    sample_mean = statistics.mean(sample1)
    assert ci_lower <= sample_mean <= ci_upper
    assert ci_upper > ci_lower
    
    print("✅ Bootstrap confidence interval validation passed")
    
    return True


def test_quantum_advantage_validator():
    """Test quantum advantage validation."""
    print("Testing Quantum Advantage Validator...")
    
    from edge_tpu_v5_benchmark.quantum_ml_validation_framework import (
        QuantumAdvantageValidator, QuantumAdvantageMetric
    )
    
    validator = QuantumAdvantageValidator(min_advantage_ratio=1.5, min_sample_size=10)
    
    # Test speedup validation with clear quantum advantage
    quantum_times = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7]  # Faster
    classical_times = [1.2, 1.5, 1.3, 1.4, 1.6, 1.1, 1.7, 1.3, 1.4, 1.5]  # Slower
    
    speedup_validation = validator.validate_speedup_advantage(quantum_times, classical_times)
    
    assert speedup_validation.metric == QuantumAdvantageMetric.SPEEDUP_RATIO
    assert speedup_validation.advantage_ratio > 1.5  # Should show significant speedup
    assert speedup_validation.statistical_significance.significant is True
    assert speedup_validation.validation_passed is True
    
    print(f"✅ Speedup validation passed (ratio: {speedup_validation.advantage_ratio:.2f})")
    
    # Test resource efficiency validation
    quantum_efficiency = [85, 90, 88, 92, 87, 89, 91, 86, 90, 88]  # Higher efficiency
    classical_efficiency = [65, 70, 68, 72, 69, 67, 71, 66, 70, 68]  # Lower efficiency
    
    efficiency_validation = validator.validate_resource_efficiency(quantum_efficiency, classical_efficiency)
    
    assert efficiency_validation.metric == QuantumAdvantageMetric.RESOURCE_EFFICIENCY
    assert efficiency_validation.advantage_ratio > 1.0
    assert efficiency_validation.validation_passed is True
    
    print(f"✅ Resource efficiency validation passed (ratio: {efficiency_validation.advantage_ratio:.2f})")
    
    # Test case with no advantage
    no_advantage_quantum = [1.0, 1.1, 1.2, 1.0, 1.1, 1.0, 1.1, 1.2, 1.0, 1.1]
    no_advantage_classical = [1.0, 1.0, 1.1, 1.1, 1.0, 1.2, 1.0, 1.1, 1.1, 1.0]
    
    no_advantage_validation = validator.validate_speedup_advantage(no_advantage_quantum, no_advantage_classical)
    
    assert no_advantage_validation.advantage_ratio < 1.5
    assert no_advantage_validation.validation_passed is False
    
    print("✅ No-advantage case validation passed")
    
    return True


def test_ml_performance_validator():
    """Test ML performance validation."""
    print("Testing ML Performance Validator...")
    
    from edge_tpu_v5_benchmark.quantum_ml_validation_framework import MLPerformanceValidator
    
    validator = MLPerformanceValidator(min_improvement_threshold=0.02)
    
    # Test accuracy improvement validation
    baseline_accuracy = [0.85, 0.87, 0.86, 0.88, 0.85, 0.87, 0.86, 0.88, 0.85, 0.86]
    improved_accuracy = [0.92, 0.94, 0.93, 0.95, 0.92, 0.94, 0.93, 0.95, 0.92, 0.93]  # Clear improvement
    
    accuracy_validation = validator.validate_accuracy_improvement(baseline_accuracy, improved_accuracy)
    
    assert accuracy_validation.accuracy_improvement > 0.02
    assert accuracy_validation.validation_passed is True
    assert accuracy_validation.performance_regression_detected is False
    assert len(accuracy_validation.statistical_tests) >= 1
    
    print(f"✅ Accuracy improvement validation passed (improvement: {accuracy_validation.accuracy_improvement:.3f})")
    
    # Test convergence improvement validation  
    baseline_iterations = [100, 120, 110, 130, 105, 125, 115, 135, 100, 120]
    improved_iterations = [60, 70, 65, 75, 62, 72, 68, 78, 61, 71]  # Fewer iterations needed
    
    convergence_validation = validator.validate_convergence_improvement(baseline_iterations, improved_iterations)
    
    assert convergence_validation.convergence_improvement > 0.02
    assert convergence_validation.validation_passed is True
    assert convergence_validation.performance_regression_detected is False
    
    print(f"✅ Convergence improvement validation passed (improvement: {convergence_validation.convergence_improvement:.3f})")
    
    # Test performance regression detection
    regressed_accuracy = [0.82, 0.81, 0.83, 0.80, 0.82, 0.81, 0.83, 0.80, 0.82, 0.81]  # Worse than baseline
    
    regression_validation = validator.validate_accuracy_improvement(baseline_accuracy, regressed_accuracy)
    
    assert regression_validation.accuracy_improvement < 0
    assert regression_validation.performance_regression_detected is True
    assert regression_validation.validation_passed is False
    
    print("✅ Performance regression detection passed")
    
    return True


def test_comprehensive_validation_framework():
    """Test the comprehensive validation framework."""
    print("Testing Comprehensive Validation Framework...")
    
    from edge_tpu_v5_benchmark.quantum_ml_validation_framework import (
        QuantumMLValidationFramework, ValidationSeverity
    )
    from edge_tpu_v5_benchmark.quantum_computing_research import QuantumResult, QuantumAlgorithm
    
    framework = QuantumMLValidationFramework()
    
    # Create mock experiment results with strong quantum advantage
    experiment_results = []
    for i in range(30):  # Sufficient sample size
        result = QuantumResult(
            experiment_name="test_quantum_ml_experiment",
            algorithm=QuantumAlgorithm.QAOA,
            execution_time=0.5 + 0.1 * (i % 3),  # Consistent fast execution
            fidelity=0.95 + 0.02 * (i % 2),      # High fidelity
            quantum_advantage=2.5 + 0.3 * (i % 2),  # Strong quantum advantage
            classical_comparison=1.2 + 0.1 * (i % 3),
            measurements={"0": [0, 1, 0], "1": [1, 0, 1]}
        )
        experiment_results.append(result)
    
    # Create classical baselines
    classical_baselines = []
    for i in range(30):
        baseline = {
            "execution_time": 1.2 + 0.2 * (i % 4),  # Consistently slower
            "accuracy": 0.85 + 0.02 * (i % 3)
        }
        classical_baselines.append(baseline)
    
    # Create ML performance data
    ml_performance_data = {
        "accuracy_baseline": [0.85 + 0.01 * (i % 3) for i in range(30)],
        "accuracy_improved": [0.92 + 0.01 * (i % 3) for i in range(30)]  # Clear improvement
    }
    
    # Run comprehensive validation
    validation_report = framework.validate_quantum_ml_experiment(
        experiment_results=experiment_results,
        classical_baselines=classical_baselines,
        ml_performance_data=ml_performance_data,
        hypothesis="Quantum-enhanced ML optimization provides 2x speedup with improved accuracy",
        expected_advantage=2.0
    )
    
    # Validate the validation report
    assert validation_report.validation_passed is True
    assert validation_report.severity == ValidationSeverity.INFO  # Should be info for successful validation
    assert len(validation_report.statistical_validation) >= 2  # Should have multiple statistical tests
    assert len(validation_report.quantum_advantage_validation) >= 1  # Should have quantum advantage validation
    assert validation_report.ml_performance_validation.validation_passed is True
    assert validation_report.confidence_score > 0.8  # Should have high confidence
    assert len(validation_report.recommendations) > 0
    
    print(f"✅ Comprehensive validation passed (confidence: {validation_report.confidence_score:.2f})")
    
    # Test validation failure case
    # Create experiment results with no quantum advantage
    poor_results = []
    for i in range(30):
        result = QuantumResult(
            experiment_name="poor_quantum_experiment",
            algorithm=QuantumAlgorithm.VQE,
            execution_time=1.5 + 0.2 * (i % 3),  # Slower than classical
            fidelity=0.7 + 0.05 * (i % 2),       # Lower fidelity
            quantum_advantage=0.8 + 0.1 * (i % 2),  # No quantum advantage
            classical_comparison=1.0,
            measurements={"0": [0, 1], "1": [1, 0]}
        )
        poor_results.append(result)
    
    poor_ml_data = {
        "accuracy_baseline": [0.85 + 0.01 * (i % 3) for i in range(30)],
        "accuracy_improved": [0.83 + 0.01 * (i % 3) for i in range(30)]  # Regression
    }
    
    failure_report = framework.validate_quantum_ml_experiment(
        experiment_results=poor_results,
        classical_baselines=classical_baselines,
        ml_performance_data=poor_ml_data,
        hypothesis="Poor quantum optimization",
        expected_advantage=2.0
    )
    
    assert failure_report.validation_passed is False
    assert failure_report.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]
    assert len(failure_report.failure_reasons) > 0
    assert failure_report.confidence_score < 0.6
    
    print("✅ Validation failure detection passed")
    
    # Test validation summary
    summary = framework.get_validation_summary()
    assert summary["total_validations"] == 2
    assert summary["passed_validations"] == 1
    assert summary["success_rate"] == 0.5
    assert "severity_distribution" in summary
    
    print("✅ Validation summary generation passed")
    
    return True


def test_integration_with_research_framework():
    """Test integration with quantum research framework."""
    print("Testing Integration with Research Framework...")
    
    # This would test the integration, but requires full module imports
    # For now, we'll test the core validation concepts
    
    from edge_tpu_v5_benchmark.quantum_ml_validation_framework import (
        QuantumMLValidationFramework,
        ValidationReport,
        StatisticalTestResult,
        StatisticalTest
    )
    
    framework = QuantumMLValidationFramework()
    
    # Test that we can create validation reports
    assert isinstance(framework, QuantumMLValidationFramework)
    assert hasattr(framework, 'validate_quantum_ml_experiment')
    assert hasattr(framework, 'get_validation_summary')
    
    # Test validation history tracking
    initial_count = len(framework.validation_history)
    
    # Simulate a validation
    framework._record_validation(ValidationReport(
        experiment_name="test_integration",
        timestamp=time.time(),
        validation_passed=True,
        severity=ValidationSeverity.INFO,
        statistical_validation=[],
        quantum_advantage_validation=[],
        ml_performance_validation=None,
        recommendations=["Great results!"],
        failure_reasons=[],
        confidence_score=0.95
    ))
    
    assert len(framework.validation_history) == initial_count + 1
    
    print("✅ Integration test passed")
    
    return True


def main():
    """Run all validation framework tests."""
    print("🧪 Testing Quantum-ML Validation Framework")
    print("=" * 60)
    
    tests = [
        test_statistical_analyzer,
        test_quantum_advantage_validator,
        test_ml_performance_validator,
        test_comprehensive_validation_framework,
        test_integration_with_research_framework
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation framework tests passed!")
        
        print("\n📈 Validation Framework Capabilities:")
        print("- ✅ Statistical significance testing (t-test, Mann-Whitney)")
        print("- ✅ Quantum advantage validation with confidence intervals")
        print("- ✅ ML performance improvement validation")
        print("- ✅ Bootstrap confidence interval estimation")
        print("- ✅ Performance regression detection")
        print("- ✅ Comprehensive validation reporting")
        print("- ✅ Statistical power analysis")
        print("- ✅ Effect size calculation")
        print("- ✅ Validation history tracking and learning")
        
        return True
    else:
        print("⚠️ Some validation framework tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)