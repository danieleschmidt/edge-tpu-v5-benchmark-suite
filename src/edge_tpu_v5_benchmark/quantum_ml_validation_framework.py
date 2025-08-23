"""Quantum-ML Validation Framework with Statistical Analysis

This module provides comprehensive validation capabilities for quantum-enhanced
machine learning optimization, including statistical significance testing,
quantum advantage validation, and performance regression analysis.
"""

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .adaptive_quantum_error_mitigation import (
    ErrorMitigationType,
    MLWorkloadType,
    MitigationStrategy,
    WorkloadCharacteristics
)

# Import types to avoid circular dependency
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .quantum_computing_research import QuantumResult, QuantumExperiment


class ValidationSeverity(Enum):
    """Severity levels for validation results."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class StatisticalTest(Enum):
    """Statistical tests for validation."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    CHI_SQUARE = "chi_square"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    BOOTSTRAP = "bootstrap"


class QuantumAdvantageMetric(Enum):
    """Metrics for measuring quantum advantage."""
    SPEEDUP_RATIO = "speedup_ratio"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SOLUTION_QUALITY = "solution_quality"
    ERROR_REDUCTION = "error_reduction"
    CONVERGENCE_RATE = "convergence_rate"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_type: StatisticalTest
    test_statistic: float
    p_value: float
    degrees_of_freedom: Optional[int]
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    interpretation: str
    significant: bool
    
    
@dataclass
class QuantumAdvantageValidation:
    """Validation of quantum advantage claims."""
    metric: QuantumAdvantageMetric
    quantum_performance: float
    classical_baseline: float
    advantage_ratio: float
    confidence_interval: Tuple[float, float]
    statistical_significance: StatisticalTestResult
    sample_size: int
    validation_passed: bool
    

@dataclass
class MLPerformanceValidation:
    """Validation of ML performance metrics."""
    accuracy_improvement: Optional[float]
    convergence_improvement: Optional[float]
    resource_efficiency_gain: Optional[float]
    error_mitigation_effectiveness: Optional[float]
    statistical_tests: List[StatisticalTestResult]
    performance_regression_detected: bool
    validation_passed: bool
    

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    experiment_name: str
    timestamp: float
    validation_passed: bool
    severity: ValidationSeverity
    statistical_validation: List[StatisticalTestResult]
    quantum_advantage_validation: List[QuantumAdvantageValidation]
    ml_performance_validation: MLPerformanceValidation
    recommendations: List[str]
    failure_reasons: List[str]
    confidence_score: float
    

class StatisticalAnalyzer:
    """Advanced statistical analysis for quantum-ML validation."""
    
    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.logger = logging.getLogger(__name__)
        
    def t_test_two_sample(self, sample1: List[float], sample2: List[float], 
                         equal_var: bool = True) -> StatisticalTestResult:
        """Perform two-sample t-test."""
        n1, n2 = len(sample1), len(sample2)
        
        if n1 < 2 or n2 < 2:
            raise ValueError("Insufficient sample sizes for t-test")
        
        mean1 = statistics.mean(sample1)
        mean2 = statistics.mean(sample2)
        var1 = statistics.variance(sample1) if n1 > 1 else 0
        var2 = statistics.variance(sample2) if n2 > 1 else 0
        
        if equal_var:
            # Pooled variance t-test
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = math.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = math.sqrt(var1/n1 + var2/n2)
            df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
            df = int(df)
        
        if se == 0:
            t_stat = float('inf') if mean1 != mean2 else 0
        else:
            t_stat = (mean1 - mean2) / se
        
        # Approximate p-value using normal distribution for large samples
        # For proper implementation, use scipy.stats.t.cdf
        if n1 + n2 > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        else:
            # Simplified approximation for small samples
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        # Calculate effect size (Cohen's d)
        if equal_var:
            effect_size = (mean1 - mean2) / math.sqrt(pooled_var) if pooled_var > 0 else 0
        else:
            pooled_std = math.sqrt((var1 + var2) / 2)
            effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for difference in means
        critical_t = 1.96 if n1 + n2 > 30 else 2.0  # Approximation
        margin_error = critical_t * se
        mean_diff = mean1 - mean2
        ci = (mean_diff - margin_error, mean_diff + margin_error)
        
        # Calculate statistical power (approximation)
        power = self._calculate_power(effect_size, n1, n2, self.alpha)
        
        # Interpret results
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Significant difference detected (p={p_value:.4f} < α={self.alpha})"
        else:
            interpretation = f"No significant difference (p={p_value:.4f} ≥ α={self.alpha})"
        
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            confidence_interval=ci,
            effect_size=effect_size,
            power=power,
            interpretation=interpretation,
            significant=significant
        )
    
    def mann_whitney_test(self, sample1: List[float], sample2: List[float]) -> StatisticalTestResult:
        """Perform Mann-Whitney U test (non-parametric)."""
        n1, n2 = len(sample1), len(sample2)
        
        if n1 == 0 or n2 == 0:
            raise ValueError("Empty samples provided")
        
        # Combine and rank all observations
        combined = [(val, 1) for val in sample1] + [(val, 2) for val in sample2]
        combined.sort(key=lambda x: x[0])
        
        # Assign ranks
        ranks = {}
        current_rank = 1
        i = 0
        while i < len(combined):
            val = combined[i][0]
            tied_count = 1
            
            # Count tied values
            while i + tied_count < len(combined) and combined[i + tied_count][0] == val:
                tied_count += 1
            
            # Assign average rank to tied values
            avg_rank = current_rank + (tied_count - 1) / 2
            for j in range(i, i + tied_count):
                ranks[j] = avg_rank
            
            current_rank += tied_count
            i += tied_count
        
        # Calculate rank sums
        R1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 1)
        R2 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 2)
        
        # Calculate U statistics
        U1 = R1 - n1 * (n1 + 1) / 2
        U2 = R2 - n2 * (n2 + 1) / 2
        
        # Use smaller U as test statistic
        U = min(U1, U2)
        
        # Calculate approximate p-value using normal approximation
        if n1 > 20 and n2 > 20:
            mean_U = n1 * n2 / 2
            var_U = n1 * n2 * (n1 + n2 + 1) / 12
            z = (U - mean_U) / math.sqrt(var_U) if var_U > 0 else 0
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            # For small samples, use approximation
            p_value = 0.05 if U < n1 * n2 / 4 else 0.5
        
        # Effect size (r = Z / sqrt(N))
        N = n1 + n2
        z_approx = (U - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) if n1 * n2 > 0 else 0
        effect_size = abs(z_approx) / math.sqrt(N) if N > 0 else 0
        
        # Confidence interval for difference in medians (approximation)
        median1 = statistics.median(sample1)
        median2 = statistics.median(sample2)
        diff = median1 - median2
        se_approx = math.sqrt((n1 + n2) * (n1 + n2 + 1) / (12 * n1 * n2)) if n1 * n2 > 0 else 1
        margin = 1.96 * se_approx
        ci = (diff - margin, diff + margin)
        
        significant = p_value < self.alpha
        interpretation = f"Mann-Whitney U test: {'Significant' if significant else 'No significant'} difference in distributions"
        
        return StatisticalTestResult(
            test_type=StatisticalTest.MANN_WHITNEY,
            test_statistic=U,
            p_value=p_value,
            degrees_of_freedom=None,
            confidence_interval=ci,
            effect_size=effect_size,
            power=0.8,  # Default approximation
            interpretation=interpretation,
            significant=significant
        )
    
    def bootstrap_confidence_interval(self, sample: List[float], 
                                    statistic_func: Callable[[List[float]], float],
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        import random
        
        bootstrap_stats = []
        n = len(sample)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample (sample with replacement)
            bootstrap_sample = [random.choice(sample) for _ in range(n)]
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats.sort()
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2
        
        lower_idx = int(lower_percentile * n_bootstrap)
        upper_idx = int(upper_percentile * n_bootstrap)
        
        return bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function for standard normal."""
        # Abramowitz and Stegun approximation
        if x < 0:
            return 1 - self._normal_cdf(-x)
        
        # Constants
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
        
        return y
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Approximate cumulative distribution function for t-distribution."""
        # Simple approximation - for proper implementation use scipy
        if df >= 30:
            return self._normal_cdf(t)
        
        # Crude approximation for small degrees of freedom
        return 0.5 + 0.5 * math.tanh(t / math.sqrt(df))
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """Calculate statistical power approximation."""
        # Cohen's approximation for power calculation
        delta = effect_size * math.sqrt(n1 * n2 / (n1 + n2))
        critical_z = 1.96  # For alpha = 0.05
        power_z = delta - critical_z
        power = self._normal_cdf(power_z)
        return max(0.0, min(1.0, power))


class QuantumAdvantageValidator:
    """Validator for quantum advantage claims."""
    
    def __init__(self, min_advantage_ratio: float = 1.1, min_sample_size: int = 30):
        self.min_advantage_ratio = min_advantage_ratio
        self.min_sample_size = min_sample_size
        self.statistical_analyzer = StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def validate_speedup_advantage(self, quantum_times: List[float], 
                                 classical_times: List[float]) -> QuantumAdvantageValidation:
        """Validate quantum speedup advantage."""
        if len(quantum_times) < self.min_sample_size or len(classical_times) < self.min_sample_size:
            self.logger.warning(f"Insufficient samples for robust validation: "
                              f"quantum={len(quantum_times)}, classical={len(classical_times)}")
        
        quantum_mean = statistics.mean(quantum_times)
        classical_mean = statistics.mean(classical_times)
        
        # Speedup ratio (classical time / quantum time)
        advantage_ratio = classical_mean / quantum_mean if quantum_mean > 0 else 0
        
        # Statistical test for difference in means
        stat_test = self.statistical_analyzer.t_test_two_sample(
            classical_times, quantum_times, equal_var=False
        )
        
        # Bootstrap confidence interval for speedup ratio
        def speedup_statistic(combined_sample):
            mid = len(combined_sample) // 2
            c_times = combined_sample[:mid]
            q_times = combined_sample[mid:]
            c_mean = statistics.mean(c_times)
            q_mean = statistics.mean(q_times)
            return c_mean / q_mean if q_mean > 0 else 0
        
        combined_sample = classical_times + quantum_times
        try:
            ci_lower, ci_upper = self.statistical_analyzer.bootstrap_confidence_interval(
                combined_sample, speedup_statistic, confidence_level=0.95
            )
        except:
            ci_lower, ci_upper = advantage_ratio * 0.9, advantage_ratio * 1.1
        
        validation_passed = (
            advantage_ratio >= self.min_advantage_ratio and
            stat_test.significant and
            ci_lower > 1.0  # Confidence interval doesn't include 1.0 (no advantage)
        )
        
        return QuantumAdvantageValidation(
            metric=QuantumAdvantageMetric.SPEEDUP_RATIO,
            quantum_performance=quantum_mean,
            classical_baseline=classical_mean,
            advantage_ratio=advantage_ratio,
            confidence_interval=(ci_lower, ci_upper),
            statistical_significance=stat_test,
            sample_size=len(quantum_times),
            validation_passed=validation_passed
        )
    
    def validate_resource_efficiency(self, quantum_efficiency: List[float],
                                   classical_efficiency: List[float]) -> QuantumAdvantageValidation:
        """Validate quantum resource efficiency advantage."""
        quantum_mean = statistics.mean(quantum_efficiency)
        classical_mean = statistics.mean(classical_efficiency)
        
        # Efficiency ratio (quantum efficiency / classical efficiency)
        advantage_ratio = quantum_mean / classical_mean if classical_mean > 0 else 0
        
        # Statistical test
        stat_test = self.statistical_analyzer.mann_whitney_test(
            quantum_efficiency, classical_efficiency
        )
        
        # Simple confidence interval approximation
        quantum_std = statistics.stdev(quantum_efficiency) if len(quantum_efficiency) > 1 else 0
        margin = 1.96 * quantum_std / math.sqrt(len(quantum_efficiency)) if len(quantum_efficiency) > 0 else 0
        ci = (max(0, advantage_ratio - margin), advantage_ratio + margin)
        
        validation_passed = (
            advantage_ratio >= self.min_advantage_ratio and
            stat_test.significant
        )
        
        return QuantumAdvantageValidation(
            metric=QuantumAdvantageMetric.RESOURCE_EFFICIENCY,
            quantum_performance=quantum_mean,
            classical_baseline=classical_mean,
            advantage_ratio=advantage_ratio,
            confidence_interval=ci,
            statistical_significance=stat_test,
            sample_size=len(quantum_efficiency),
            validation_passed=validation_passed
        )


class MLPerformanceValidator:
    """Validator for ML performance improvements."""
    
    def __init__(self, min_improvement_threshold: float = 0.02):
        self.min_improvement_threshold = min_improvement_threshold
        self.statistical_analyzer = StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def validate_accuracy_improvement(self, baseline_accuracy: List[float],
                                    improved_accuracy: List[float]) -> MLPerformanceValidation:
        """Validate ML accuracy improvements."""
        statistical_tests = []
        
        # Perform statistical tests
        t_test = self.statistical_analyzer.t_test_two_sample(
            improved_accuracy, baseline_accuracy
        )
        statistical_tests.append(t_test)
        
        # Mann-Whitney test as non-parametric alternative
        try:
            mw_test = self.statistical_analyzer.mann_whitney_test(
                improved_accuracy, baseline_accuracy
            )
            statistical_tests.append(mw_test)
        except Exception as e:
            self.logger.warning(f"Mann-Whitney test failed: {e}")
        
        # Calculate improvement metrics
        baseline_mean = statistics.mean(baseline_accuracy)
        improved_mean = statistics.mean(improved_accuracy)
        accuracy_improvement = improved_mean - baseline_mean
        
        # Check for performance regression
        performance_regression_detected = accuracy_improvement < -self.min_improvement_threshold
        
        # Validation passes if improvement is significant and above threshold
        validation_passed = (
            accuracy_improvement >= self.min_improvement_threshold and
            t_test.significant and
            not performance_regression_detected
        )
        
        return MLPerformanceValidation(
            accuracy_improvement=accuracy_improvement,
            convergence_improvement=None,
            resource_efficiency_gain=None,
            error_mitigation_effectiveness=None,
            statistical_tests=statistical_tests,
            performance_regression_detected=performance_regression_detected,
            validation_passed=validation_passed
        )
    
    def validate_convergence_improvement(self, baseline_iterations: List[int],
                                       improved_iterations: List[int]) -> MLPerformanceValidation:
        """Validate convergence rate improvements."""
        statistical_tests = []
        
        # Convert to convergence rates (1 / iterations)
        baseline_rates = [1.0 / max(1, iters) for iters in baseline_iterations]
        improved_rates = [1.0 / max(1, iters) for iters in improved_iterations]
        
        # Statistical tests
        t_test = self.statistical_analyzer.t_test_two_sample(
            improved_rates, baseline_rates
        )
        statistical_tests.append(t_test)
        
        # Calculate improvement
        baseline_mean_iters = statistics.mean(baseline_iterations)
        improved_mean_iters = statistics.mean(improved_iterations)
        
        # Convergence improvement as reduction in iterations needed
        convergence_improvement = (baseline_mean_iters - improved_mean_iters) / baseline_mean_iters
        
        # Check for regression (more iterations needed)
        performance_regression_detected = convergence_improvement < -0.1  # 10% worse
        
        validation_passed = (
            convergence_improvement >= self.min_improvement_threshold and
            t_test.significant and
            not performance_regression_detected
        )
        
        return MLPerformanceValidation(
            accuracy_improvement=None,
            convergence_improvement=convergence_improvement,
            resource_efficiency_gain=None,
            error_mitigation_effectiveness=None,
            statistical_tests=statistical_tests,
            performance_regression_detected=performance_regression_detected,
            validation_passed=validation_passed
        )


class QuantumMLValidationFramework:
    """Comprehensive validation framework for quantum-ML optimization."""
    
    def __init__(self, alpha: float = 0.05, min_sample_size: int = 30):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.quantum_advantage_validator = QuantumAdvantageValidator()
        self.ml_performance_validator = MLPerformanceValidator()
        self.statistical_analyzer = StatisticalAnalyzer(alpha=alpha)
        self.logger = logging.getLogger(__name__)
        
        # Validation history for learning
        self.validation_history = []
    
    def validate_quantum_ml_experiment(self, 
                                     experiment_results: List['QuantumResult'],
                                     classical_baselines: List[Dict[str, float]],
                                     ml_performance_data: Dict[str, List[float]],
                                     hypothesis: str = "",
                                     expected_advantage: float = 1.5) -> ValidationReport:
        """Comprehensive validation of quantum-ML experiment results."""
        
        if not experiment_results:
            return self._create_failed_validation_report("No experiment results provided")
        
        experiment_name = experiment_results[0].experiment_name
        self.logger.info(f"Validating quantum-ML experiment: {experiment_name}")
        
        # Extract performance data
        quantum_times = [result.execution_time for result in experiment_results]
        classical_times = [baseline.get('execution_time', 1.0) for baseline in classical_baselines]
        
        quantum_advantages = [result.quantum_advantage for result in experiment_results]
        quantum_fidelities = [result.fidelity for result in experiment_results]
        
        # Statistical validation tests
        statistical_tests = []
        
        # Test for consistent quantum advantage
        if len(quantum_advantages) >= 2:
            advantage_test = self._test_quantum_advantage_consistency(quantum_advantages, expected_advantage)
            statistical_tests.append(advantage_test)
        
        # Test for fidelity requirements
        if quantum_fidelities:
            fidelity_test = self._test_fidelity_requirements(quantum_fidelities, min_fidelity=0.9)
            statistical_tests.append(fidelity_test)
        
        # Quantum advantage validation
        quantum_advantage_validations = []
        
        if quantum_times and classical_times and len(quantum_times) >= self.min_sample_size:
            speedup_validation = self.quantum_advantage_validator.validate_speedup_advantage(
                quantum_times, classical_times
            )
            quantum_advantage_validations.append(speedup_validation)
        
        # ML performance validation
        ml_validation = None
        if 'accuracy_baseline' in ml_performance_data and 'accuracy_improved' in ml_performance_data:
            ml_validation = self.ml_performance_validator.validate_accuracy_improvement(
                ml_performance_data['accuracy_baseline'],
                ml_performance_data['accuracy_improved']
            )
        
        # Overall validation assessment
        validation_passed = self._assess_overall_validation(
            statistical_tests, quantum_advantage_validations, ml_validation
        )
        
        # Determine severity
        severity = self._determine_severity(validation_passed, statistical_tests)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            statistical_tests, quantum_advantage_validations, ml_validation
        )
        
        # Generate failure reasons if validation failed
        failure_reasons = []
        if not validation_passed:
            failure_reasons = self._identify_failure_reasons(
                statistical_tests, quantum_advantage_validations, ml_validation
            )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            statistical_tests, quantum_advantage_validations, len(experiment_results)
        )
        
        validation_report = ValidationReport(
            experiment_name=experiment_name,
            timestamp=time.time(),
            validation_passed=validation_passed,
            severity=severity,
            statistical_validation=statistical_tests,
            quantum_advantage_validation=quantum_advantage_validations,
            ml_performance_validation=ml_validation or MLPerformanceValidation(
                accuracy_improvement=None,
                convergence_improvement=None,
                resource_efficiency_gain=None,
                error_mitigation_effectiveness=None,
                statistical_tests=[],
                performance_regression_detected=False,
                validation_passed=True
            ),
            recommendations=recommendations,
            failure_reasons=failure_reasons,
            confidence_score=confidence_score
        )
        
        # Record validation for learning
        self._record_validation(validation_report)
        
        return validation_report
    
    def _test_quantum_advantage_consistency(self, advantages: List[float], 
                                          expected: float) -> StatisticalTestResult:
        """Test if quantum advantages are consistently above expected threshold."""
        # One-sample t-test against expected advantage
        n = len(advantages)
        mean_advantage = statistics.mean(advantages)
        std_advantage = statistics.stdev(advantages) if n > 1 else 0
        
        if std_advantage == 0:
            t_stat = float('inf') if mean_advantage > expected else 0
            p_value = 0.0 if mean_advantage > expected else 1.0
        else:
            t_stat = (mean_advantage - expected) / (std_advantage / math.sqrt(n))
            # Approximate p-value
            p_value = 1 - self.statistical_analyzer._normal_cdf(t_stat)
        
        significant = p_value < self.alpha and mean_advantage > expected
        
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=n-1,
            confidence_interval=(mean_advantage - 1.96*std_advantage/math.sqrt(n),
                               mean_advantage + 1.96*std_advantage/math.sqrt(n)),
            effect_size=(mean_advantage - expected) / std_advantage if std_advantage > 0 else 0,
            power=0.8,  # Approximation
            interpretation=f"Quantum advantage consistency: {'Consistent' if significant else 'Inconsistent'}",
            significant=significant
        )
    
    def _test_fidelity_requirements(self, fidelities: List[float], 
                                   min_fidelity: float) -> StatisticalTestResult:
        """Test if fidelities meet minimum requirements."""
        n = len(fidelities)
        mean_fidelity = statistics.mean(fidelities)
        std_fidelity = statistics.stdev(fidelities) if n > 1 else 0
        
        # One-sample t-test
        if std_fidelity == 0:
            t_stat = float('inf') if mean_fidelity >= min_fidelity else -float('inf')
        else:
            t_stat = (mean_fidelity - min_fidelity) / (std_fidelity / math.sqrt(n))
        
        p_value = 1 - self.statistical_analyzer._normal_cdf(t_stat)
        significant = p_value < self.alpha and mean_fidelity >= min_fidelity
        
        return StatisticalTestResult(
            test_type=StatisticalTest.T_TEST,
            test_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=n-1,
            confidence_interval=(mean_fidelity - 1.96*std_fidelity/math.sqrt(n),
                               mean_fidelity + 1.96*std_fidelity/math.sqrt(n)),
            effect_size=(mean_fidelity - min_fidelity) / std_fidelity if std_fidelity > 0 else 0,
            power=0.8,
            interpretation=f"Fidelity requirement: {'Met' if significant else 'Not met'}",
            significant=significant
        )
    
    def _assess_overall_validation(self, statistical_tests: List[StatisticalTestResult],
                                 advantage_validations: List[QuantumAdvantageValidation],
                                 ml_validation: Optional[MLPerformanceValidation]) -> bool:
        """Assess overall validation success."""
        # Statistical tests must pass
        statistical_passed = all(test.significant for test in statistical_tests) if statistical_tests else True
        
        # Quantum advantage validations must pass
        advantage_passed = all(val.validation_passed for val in advantage_validations) if advantage_validations else True
        
        # ML validation must pass if present
        ml_passed = ml_validation.validation_passed if ml_validation else True
        
        return statistical_passed and advantage_passed and ml_passed
    
    def _determine_severity(self, validation_passed: bool, 
                          statistical_tests: List[StatisticalTestResult]) -> ValidationSeverity:
        """Determine validation severity level."""
        if validation_passed:
            return ValidationSeverity.INFO
        
        # Check for critical failures
        critical_failures = sum(1 for test in statistical_tests if not test.significant and test.power < 0.5)
        
        if critical_failures > len(statistical_tests) / 2:
            return ValidationSeverity.CRITICAL
        elif critical_failures > 0:
            return ValidationSeverity.HIGH
        else:
            return ValidationSeverity.MEDIUM
    
    def _generate_recommendations(self, statistical_tests: List[StatisticalTestResult],
                                advantage_validations: List[QuantumAdvantageValidation],
                                ml_validation: Optional[MLPerformanceValidation]) -> List[str]:
        """Generate recommendations for improving validation results."""
        recommendations = []
        
        # Statistical test recommendations
        for test in statistical_tests:
            if not test.significant:
                if test.power < 0.8:
                    recommendations.append(f"Increase sample size to improve statistical power for {test.test_type.value}")
                if abs(test.effect_size) < 0.2:
                    recommendations.append(f"Effect size is small for {test.test_type.value}; consider practical significance")
        
        # Quantum advantage recommendations
        for validation in advantage_validations:
            if not validation.validation_passed:
                if validation.advantage_ratio < 1.1:
                    recommendations.append(f"Quantum advantage ratio ({validation.advantage_ratio:.2f}) is below minimum threshold")
                if not validation.statistical_significance.significant:
                    recommendations.append(f"Quantum advantage not statistically significant for {validation.metric.value}")
        
        # ML performance recommendations
        if ml_validation and not ml_validation.validation_passed:
            if ml_validation.performance_regression_detected:
                recommendations.append("Performance regression detected; review optimization strategy")
            recommendations.append("Consider additional error mitigation techniques")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Validation passed successfully; consider scaling to larger problem sizes")
        
        return recommendations
    
    def _identify_failure_reasons(self, statistical_tests: List[StatisticalTestResult],
                                advantage_validations: List[QuantumAdvantageValidation],
                                ml_validation: Optional[MLPerformanceValidation]) -> List[str]:
        """Identify specific reasons for validation failure."""
        reasons = []
        
        for test in statistical_tests:
            if not test.significant:
                reasons.append(f"Statistical test {test.test_type.value} failed (p={test.p_value:.4f})")
        
        for validation in advantage_validations:
            if not validation.validation_passed:
                reasons.append(f"Quantum advantage validation failed for {validation.metric.value}")
        
        if ml_validation and not ml_validation.validation_passed:
            reasons.append("ML performance validation failed")
            if ml_validation.performance_regression_detected:
                reasons.append("Performance regression detected")
        
        return reasons
    
    def _calculate_confidence_score(self, statistical_tests: List[StatisticalTestResult],
                                  advantage_validations: List[QuantumAdvantageValidation],
                                  sample_size: int) -> float:
        """Calculate overall confidence score for validation."""
        scores = []
        
        # Statistical test confidence
        for test in statistical_tests:
            if test.significant:
                scores.append(1 - test.p_value)  # Higher confidence for lower p-values
            else:
                scores.append(test.p_value)  # Lower confidence for non-significant results
        
        # Sample size factor
        sample_score = min(1.0, sample_size / self.min_sample_size)
        scores.append(sample_score)
        
        # Power factor
        avg_power = statistics.mean([test.power for test in statistical_tests]) if statistical_tests else 0.8
        scores.append(avg_power)
        
        return statistics.mean(scores) if scores else 0.5
    
    def _create_failed_validation_report(self, reason: str) -> ValidationReport:
        """Create a validation report for failed validation."""
        return ValidationReport(
            experiment_name="failed_validation",
            timestamp=time.time(),
            validation_passed=False,
            severity=ValidationSeverity.CRITICAL,
            statistical_validation=[],
            quantum_advantage_validation=[],
            ml_performance_validation=MLPerformanceValidation(
                accuracy_improvement=None,
                convergence_improvement=None,
                resource_efficiency_gain=None,
                error_mitigation_effectiveness=None,
                statistical_tests=[],
                performance_regression_detected=False,
                validation_passed=False
            ),
            recommendations=[f"Address validation failure: {reason}"],
            failure_reasons=[reason],
            confidence_score=0.0
        )
    
    def _record_validation(self, report: ValidationReport):
        """Record validation for learning and improvement."""
        validation_record = {
            'timestamp': report.timestamp,
            'experiment_name': report.experiment_name,
            'validation_passed': report.validation_passed,
            'severity': report.severity.value,
            'confidence_score': report.confidence_score,
            'num_statistical_tests': len(report.statistical_validation),
            'num_advantage_validations': len(report.quantum_advantage_validation)
        }
        
        self.validation_history.append(validation_record)
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history and performance."""
        if not self.validation_history:
            return {"total_validations": 0}
        
        total = len(self.validation_history)
        passed = sum(1 for record in self.validation_history if record['validation_passed'])
        avg_confidence = statistics.mean([record['confidence_score'] for record in self.validation_history])
        
        severity_counts = {}
        for record in self.validation_history:
            severity = record['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_validations": total,
            "passed_validations": passed,
            "success_rate": passed / total,
            "average_confidence": avg_confidence,
            "severity_distribution": severity_counts,
            "recent_validations": self.validation_history[-10:]  # Last 10 validations
        }


# Export main classes
__all__ = [
    'QuantumMLValidationFramework',
    'ValidationReport',
    'StatisticalTestResult',
    'QuantumAdvantageValidation',
    'MLPerformanceValidation',
    'ValidationSeverity',
    'StatisticalTest',
    'QuantumAdvantageMetric'
]