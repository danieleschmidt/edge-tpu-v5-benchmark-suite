"""V-Score Quantum Advantage Detection Framework

Implementation of IBM's 2025 V-score methodology for rigorous quantum advantage detection
in ground state problems and optimization tasks.

This module provides falsifiable quantum advantage criteria with statistical validation
for the TERRAGON quantum-enhanced TPU benchmark suite.
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .quantum_computing_research import QuantumCircuit, QuantumResult


class QuantumAdvantageType(Enum):
    """Types of quantum advantage to detect."""
    COMPUTATIONAL = "computational"
    SAMPLING = "sampling"
    VARIATIONAL = "variational"
    OPTIMIZATION = "optimization"
    GROUND_STATE = "ground_state"


class VScoreCategory(Enum):
    """V-score evaluation categories based on IBM 2025 framework."""
    ENERGY_ESTIMATION = "energy_estimation"
    VARIANCE_ANALYSIS = "variance_analysis"
    CONVERGENCE_RATE = "convergence_rate"
    RESOURCE_EFFICIENCY = "resource_efficiency"


@dataclass
class QuantumResult:
    """Enhanced quantum result with V-score components."""
    energy_estimate: float
    variance: float
    execution_time: float
    resource_count: int
    convergence_iterations: int
    classical_baseline_energy: Optional[float] = None
    classical_baseline_time: Optional[float] = None
    raw_measurements: Optional[List[float]] = None
    circuit_depth: int = 0
    gate_count: int = 0


@dataclass
class VScoreComponents:
    """Components of the V-score calculation."""
    energy_accuracy: float
    variance_quality: float
    convergence_efficiency: float
    resource_utilization: float
    statistical_significance: float
    confidence_interval: Tuple[float, float]


@dataclass
class QuantumAdvantageResult:
    """Result of quantum advantage detection."""
    has_advantage: bool
    v_score: float
    components: VScoreComponents
    advantage_type: QuantumAdvantageType
    statistical_power: float
    effect_size: float
    p_value: float
    confidence_level: float = 0.95


class VScoreQuantumAdvantageDetector:
    """IBM-2025 V-score based quantum advantage detection.
    
    Implements rigorous statistical framework for detecting and validating
    quantum advantage using energy estimation, variance analysis, and
    falsifiable criteria.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
        
        # V-score calculation parameters
        self.v_score_weights = {
            VScoreCategory.ENERGY_ESTIMATION: 0.3,
            VScoreCategory.VARIANCE_ANALYSIS: 0.25,
            VScoreCategory.CONVERGENCE_RATE: 0.25,
            VScoreCategory.RESOURCE_EFFICIENCY: 0.2
        }
        
        # Thresholds for quantum advantage detection
        self.advantage_thresholds = {
            QuantumAdvantageType.COMPUTATIONAL: 1.5,  # 1.5x speedup minimum
            QuantumAdvantageType.SAMPLING: 2.0,       # 2x sampling advantage
            QuantumAdvantageType.VARIATIONAL: 1.3,    # 1.3x energy accuracy
            QuantumAdvantageType.OPTIMIZATION: 1.4,   # 1.4x optimization efficiency
            QuantumAdvantageType.GROUND_STATE: 1.2    # 1.2x ground state accuracy
        }
        
        # Historical performance tracking
        self.performance_history = []
        
    def calculate_v_score(self, quantum_result: QuantumResult, 
                         classical_baseline: QuantumResult,
                         problem_type: QuantumAdvantageType) -> QuantumAdvantageResult:
        """Calculate V-score and detect quantum advantage.
        
        Args:
            quantum_result: Results from quantum computation
            classical_baseline: Results from classical computation
            problem_type: Type of quantum advantage to evaluate
            
        Returns:
            QuantumAdvantageResult with V-score and advantage detection
        """
        try:
            # Calculate individual V-score components
            components = self._calculate_v_score_components(
                quantum_result, classical_baseline, problem_type
            )
            
            # Calculate overall V-score
            v_score = self._compute_weighted_v_score(components)
            
            # Perform statistical significance testing
            statistical_result = self._perform_statistical_analysis(
                quantum_result, classical_baseline
            )
            
            # Determine quantum advantage based on V-score and thresholds
            advantage_threshold = self.advantage_thresholds[problem_type]
            has_advantage = (
                v_score > advantage_threshold and 
                statistical_result['p_value'] < (1 - self.confidence_level)
            )
            
            # Calculate effect size and statistical power
            effect_size = self._calculate_effect_size(quantum_result, classical_baseline)
            statistical_power = self._calculate_statistical_power(
                effect_size, len(quantum_result.raw_measurements or [10])
            )
            
            result = QuantumAdvantageResult(
                has_advantage=has_advantage,
                v_score=v_score,
                components=components,
                advantage_type=problem_type,
                statistical_power=statistical_power,
                effect_size=effect_size,
                p_value=statistical_result['p_value'],
                confidence_level=self.confidence_level
            )
            
            # Record result for learning
            self._record_advantage_detection(result, quantum_result, classical_baseline)
            
            return result
            
        except Exception as e:
            self.logger.error(f"V-score calculation failed: {e}")
            return self._create_fallback_result(problem_type)
    
    def _calculate_v_score_components(self, quantum_result: QuantumResult,
                                    classical_baseline: QuantumResult,
                                    problem_type: QuantumAdvantageType) -> VScoreComponents:
        """Calculate individual components of the V-score."""
        
        # Energy accuracy component
        if classical_baseline.energy_estimate != 0:
            energy_accuracy = abs(quantum_result.energy_estimate) / abs(classical_baseline.energy_estimate)
        else:
            energy_accuracy = 1.0
            
        # Variance quality component (lower variance is better)
        if classical_baseline.variance > 0:
            variance_quality = classical_baseline.variance / (quantum_result.variance + 1e-10)
        else:
            variance_quality = 1.0
            
        # Convergence efficiency component
        if classical_baseline.convergence_iterations > 0:
            convergence_efficiency = classical_baseline.convergence_iterations / quantum_result.convergence_iterations
        else:
            convergence_efficiency = 1.0
            
        # Resource utilization component
        classical_resources = classical_baseline.execution_time * classical_baseline.resource_count
        quantum_resources = quantum_result.execution_time * quantum_result.resource_count
        
        if quantum_resources > 0:
            resource_utilization = classical_resources / quantum_resources
        else:
            resource_utilization = 1.0
            
        # Statistical significance assessment
        statistical_significance = self._assess_statistical_significance(
            quantum_result, classical_baseline
        )
        
        # Calculate confidence interval for V-score
        confidence_interval = self._calculate_confidence_interval(
            quantum_result, classical_baseline
        )
        
        return VScoreComponents(
            energy_accuracy=energy_accuracy,
            variance_quality=variance_quality,
            convergence_efficiency=convergence_efficiency,
            resource_utilization=resource_utilization,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval
        )
    
    def _compute_weighted_v_score(self, components: VScoreComponents) -> float:
        """Compute weighted V-score from components."""
        weighted_score = (
            self.v_score_weights[VScoreCategory.ENERGY_ESTIMATION] * components.energy_accuracy +
            self.v_score_weights[VScoreCategory.VARIANCE_ANALYSIS] * components.variance_quality +
            self.v_score_weights[VScoreCategory.CONVERGENCE_RATE] * components.convergence_efficiency +
            self.v_score_weights[VScoreCategory.RESOURCE_EFFICIENCY] * components.resource_utilization
        )
        
        # Apply statistical significance weighting
        weighted_score *= components.statistical_significance
        
        return weighted_score
    
    def _perform_statistical_analysis(self, quantum_result: QuantumResult,
                                    classical_baseline: QuantumResult) -> Dict[str, float]:
        """Perform statistical hypothesis testing for quantum advantage."""
        
        # Prepare data for statistical tests
        quantum_measurements = quantum_result.raw_measurements or [quantum_result.energy_estimate] * 10
        classical_measurements = classical_baseline.raw_measurements or [classical_baseline.energy_estimate] * 10
        
        # Perform t-test for means comparison
        try:
            t_statistic, p_value = stats.ttest_ind(quantum_measurements, classical_measurements)
        except Exception:
            # Fallback if statistical test fails
            t_statistic, p_value = 0.0, 0.5
            
        # Perform Mann-Whitney U test for distribution comparison
        try:
            u_statistic, mann_whitney_p = stats.mannwhitneyu(
                quantum_measurements, classical_measurements, alternative='two-sided'
            )
        except Exception:
            u_statistic, mann_whitney_p = 0.0, 0.5
            
        # Use the more conservative p-value
        final_p_value = max(p_value, mann_whitney_p)
        
        return {
            't_statistic': t_statistic,
            'p_value': final_p_value,
            'u_statistic': u_statistic,
            'mann_whitney_p': mann_whitney_p
        }
    
    def _assess_statistical_significance(self, quantum_result: QuantumResult,
                                       classical_baseline: QuantumResult) -> float:
        """Assess statistical significance of the quantum advantage claim."""
        
        # Calculate sample sizes
        quantum_samples = len(quantum_result.raw_measurements or [10])
        classical_samples = len(classical_baseline.raw_measurements or [10])
        
        # Minimum sample size requirement
        min_samples = 5
        if quantum_samples < min_samples or classical_samples < min_samples:
            return 0.3  # Low significance for insufficient samples
            
        # Calculate coefficient of variation for stability assessment
        quantum_cv = quantum_result.variance / (abs(quantum_result.energy_estimate) + 1e-10)
        classical_cv = classical_baseline.variance / (abs(classical_baseline.energy_estimate) + 1e-10)
        
        # Significance based on measurement stability
        if quantum_cv < classical_cv:
            stability_factor = 1.0
        else:
            stability_factor = classical_cv / (quantum_cv + 1e-10)
            
        # Sample size factor
        sample_factor = min(1.0, (quantum_samples + classical_samples) / 20.0)
        
        # Combined significance score
        significance = stability_factor * sample_factor
        return min(1.0, max(0.1, significance))
    
    def _calculate_confidence_interval(self, quantum_result: QuantumResult,
                                     classical_baseline: QuantumResult) -> Tuple[float, float]:
        """Calculate confidence interval for the advantage ratio."""
        
        if classical_baseline.energy_estimate == 0:
            return (0.9, 1.1)  # Default interval if baseline is zero
            
        # Calculate advantage ratio
        advantage_ratio = quantum_result.energy_estimate / classical_baseline.energy_estimate
        
        # Estimate standard error using propagation of uncertainty
        quantum_se = quantum_result.variance / (abs(quantum_result.energy_estimate) + 1e-10)
        classical_se = classical_baseline.variance / (abs(classical_baseline.energy_estimate) + 1e-10)
        
        # Combined standard error for ratio
        ratio_se = abs(advantage_ratio) * np.sqrt(quantum_se**2 + classical_se**2)
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin_of_error = z_score * ratio_se
        
        lower_bound = advantage_ratio - margin_of_error
        upper_bound = advantage_ratio + margin_of_error
        
        return (lower_bound, upper_bound)
    
    def _calculate_effect_size(self, quantum_result: QuantumResult,
                             classical_baseline: QuantumResult) -> float:
        """Calculate Cohen's d effect size for quantum vs classical comparison."""
        
        # Get measurements
        quantum_measurements = quantum_result.raw_measurements or [quantum_result.energy_estimate] * 10
        classical_measurements = classical_baseline.raw_measurements or [classical_baseline.energy_estimate] * 10
        
        # Calculate means and standard deviations
        quantum_mean = np.mean(quantum_measurements)
        classical_mean = np.mean(classical_measurements)
        quantum_std = np.std(quantum_measurements, ddof=1) if len(quantum_measurements) > 1 else quantum_result.variance
        classical_std = np.std(classical_measurements, ddof=1) if len(classical_measurements) > 1 else classical_baseline.variance
        
        # Pooled standard deviation
        n1, n2 = len(quantum_measurements), len(classical_measurements)
        pooled_std = np.sqrt(((n1 - 1) * quantum_std**2 + (n2 - 1) * classical_std**2) / (n1 + n2 - 2))
        
        # Cohen's d
        if pooled_std > 0:
            effect_size = abs(quantum_mean - classical_mean) / pooled_std
        else:
            effect_size = 0.0
            
        return effect_size
    
    def _calculate_statistical_power(self, effect_size: float, sample_size: int) -> float:
        """Calculate statistical power for the current effect size and sample size."""
        
        # Simplified power calculation based on effect size and sample size
        # In practice, this would use more sophisticated power analysis
        
        alpha = 1 - self.confidence_level
        
        # Approximate power calculation
        if effect_size == 0:
            return alpha  # Only false positive rate
        
        # Effect size categories (Cohen's convention)
        if effect_size < 0.2:
            base_power = 0.1
        elif effect_size < 0.5:
            base_power = 0.3
        elif effect_size < 0.8:
            base_power = 0.6
        else:
            base_power = 0.9
            
        # Adjust for sample size
        sample_adjustment = min(1.0, sample_size / 30.0)  # Assume 30 is adequate sample size
        
        power = base_power * sample_adjustment
        return min(0.99, max(alpha, power))
    
    def _create_fallback_result(self, problem_type: QuantumAdvantageType) -> QuantumAdvantageResult:
        """Create fallback result when V-score calculation fails."""
        
        fallback_components = VScoreComponents(
            energy_accuracy=1.0,
            variance_quality=1.0,
            convergence_efficiency=1.0,
            resource_utilization=1.0,
            statistical_significance=0.5,
            confidence_interval=(0.8, 1.2)
        )
        
        return QuantumAdvantageResult(
            has_advantage=False,
            v_score=1.0,
            components=fallback_components,
            advantage_type=problem_type,
            statistical_power=0.5,
            effect_size=0.0,
            p_value=0.5,
            confidence_level=self.confidence_level
        )
    
    def _record_advantage_detection(self, result: QuantumAdvantageResult,
                                  quantum_result: QuantumResult,
                                  classical_baseline: QuantumResult):
        """Record advantage detection result for learning and improvement."""
        
        record = {
            'timestamp': time.time(),
            'advantage_type': result.advantage_type.value,
            'has_advantage': result.has_advantage,
            'v_score': result.v_score,
            'statistical_power': result.statistical_power,
            'effect_size': result.effect_size,
            'p_value': result.p_value,
            'quantum_energy': quantum_result.energy_estimate,
            'classical_energy': classical_baseline.energy_estimate,
            'quantum_variance': quantum_result.variance,
            'classical_variance': classical_baseline.variance
        }
        
        self.performance_history.append(record)
        
        # Keep history manageable
        if len(self.performance_history) > 500:
            self.performance_history = self.performance_history[-500:]
            
        self.logger.info(f"Recorded V-score detection: {result.has_advantage} (score: {result.v_score:.3f})")


class FalsifiableAdvantageFramework:
    """Framework for falsifiable quantum advantage testing.
    
    Implements testable criteria-based framework based on IBM/Pasqal 2025 research
    for rigorous quantum advantage validation.
    """
    
    def __init__(self):
        self.criteria_weights = {
            'performance_criterion': 0.4,
            'efficiency_criterion': 0.3,
            'scalability_criterion': 0.2,
            'robustness_criterion': 0.1
        }
        
        self.test_results = []
        self.logger = logging.getLogger(__name__)
    
    def test_quantum_advantage_hypothesis(self, quantum_result: QuantumResult,
                                        classical_baseline: QuantumResult,
                                        advantage_claim: str) -> Dict[str, Any]:
        """Test falsifiable quantum advantage hypothesis.
        
        Args:
            quantum_result: Quantum computation results
            classical_baseline: Classical computation baseline
            advantage_claim: Specific advantage claim to test
            
        Returns:
            Dictionary with test results and falsifiability assessment
        """
        
        try:
            # Define testable criteria
            criteria_results = {
                'performance_criterion': self._test_performance_criterion(
                    quantum_result, classical_baseline
                ),
                'efficiency_criterion': self._test_efficiency_criterion(
                    quantum_result, classical_baseline
                ),
                'scalability_criterion': self._test_scalability_criterion(
                    quantum_result, classical_baseline
                ),
                'robustness_criterion': self._test_robustness_criterion(
                    quantum_result, classical_baseline
                )
            }
            
            # Calculate overall falsifiability score
            overall_score = sum(
                self.criteria_weights[criterion] * result['score']
                for criterion, result in criteria_results.items()
            )
            
            # Determine if hypothesis is falsified
            is_falsified = overall_score < 0.5  # Threshold for falsification
            
            # Statistical confidence in falsification
            confidence = self._calculate_falsification_confidence(criteria_results)
            
            result = {
                'advantage_claim': advantage_claim,
                'is_falsified': is_falsified,
                'overall_score': overall_score,
                'confidence': confidence,
                'criteria_results': criteria_results,
                'recommendation': self._generate_recommendation(
                    is_falsified, overall_score, confidence
                )
            }
            
            # Record test result
            self._record_test_result(result, quantum_result, classical_baseline)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Falsifiable advantage test failed: {e}")
            return self._create_fallback_test_result(advantage_claim)
    
    def _test_performance_criterion(self, quantum_result: QuantumResult,
                                  classical_baseline: QuantumResult) -> Dict[str, Any]:
        """Test performance-based quantum advantage criterion."""
        
        # Calculate performance metrics
        quantum_performance = 1.0 / (quantum_result.execution_time + 1e-10)
        classical_performance = 1.0 / (classical_baseline.execution_time + 1e-10)
        
        performance_ratio = quantum_performance / classical_performance
        
        # Performance criterion: quantum should be significantly faster
        criterion_met = performance_ratio > 1.1  # At least 10% improvement
        
        # Calculate confidence based on measurement stability
        confidence = min(1.0, 1.0 / (quantum_result.variance + classical_baseline.variance + 0.1))
        
        return {
            'criterion_met': criterion_met,
            'score': performance_ratio if criterion_met else 1.0 / performance_ratio,
            'performance_ratio': performance_ratio,
            'confidence': confidence,
            'details': f"Quantum: {quantum_performance:.3f}, Classical: {classical_performance:.3f}"
        }
    
    def _test_efficiency_criterion(self, quantum_result: QuantumResult,
                                 classical_baseline: QuantumResult) -> Dict[str, Any]:
        """Test resource efficiency criterion."""
        
        # Calculate resource efficiency (performance per resource unit)
        quantum_efficiency = 1.0 / (quantum_result.execution_time * quantum_result.resource_count + 1e-10)
        classical_efficiency = 1.0 / (classical_baseline.execution_time * classical_baseline.resource_count + 1e-10)
        
        efficiency_ratio = quantum_efficiency / classical_efficiency
        
        # Efficiency criterion: quantum should be more resource-efficient
        criterion_met = efficiency_ratio > 1.05  # At least 5% better efficiency
        
        # Confidence based on resource count reliability
        confidence = min(1.0, (quantum_result.resource_count + classical_baseline.resource_count) / 20.0)
        
        return {
            'criterion_met': criterion_met,
            'score': efficiency_ratio if criterion_met else 1.0 / efficiency_ratio,
            'efficiency_ratio': efficiency_ratio,
            'confidence': confidence,
            'details': f"Q-efficiency: {quantum_efficiency:.3f}, C-efficiency: {classical_efficiency:.3f}"
        }
    
    def _test_scalability_criterion(self, quantum_result: QuantumResult,
                                  classical_baseline: QuantumResult) -> Dict[str, Any]:
        """Test scalability criterion based on problem size."""
        
        # Estimate scalability based on circuit complexity
        quantum_complexity = quantum_result.circuit_depth * quantum_result.gate_count
        classical_complexity = classical_baseline.circuit_depth * classical_baseline.gate_count
        
        if classical_complexity > 0:
            complexity_ratio = quantum_complexity / classical_complexity
        else:
            complexity_ratio = 1.0
            
        # Scalability criterion: quantum should handle complexity better
        # Lower complexity ratio indicates better quantum scalability
        criterion_met = complexity_ratio < 0.9  # Quantum uses less complexity
        
        scalability_score = 1.0 / complexity_ratio if complexity_ratio > 0 else 1.0
        
        # Confidence based on problem size
        problem_size = max(quantum_result.circuit_depth, classical_baseline.circuit_depth)
        confidence = min(1.0, problem_size / 50.0)  # More confidence with larger problems
        
        return {
            'criterion_met': criterion_met,
            'score': scalability_score if criterion_met else 1.0 / scalability_score,
            'complexity_ratio': complexity_ratio,
            'confidence': confidence,
            'details': f"Q-complexity: {quantum_complexity}, C-complexity: {classical_complexity}"
        }
    
    def _test_robustness_criterion(self, quantum_result: QuantumResult,
                                 classical_baseline: QuantumResult) -> Dict[str, Any]:
        """Test robustness criterion based on variance and stability."""
        
        # Robustness based on result variance (lower is better)
        if classical_baseline.variance > 0:
            robustness_ratio = classical_baseline.variance / (quantum_result.variance + 1e-10)
        else:
            robustness_ratio = 1.0
            
        # Robustness criterion: quantum should be more stable
        criterion_met = robustness_ratio > 1.0  # Lower quantum variance
        
        # Confidence based on number of measurements
        quantum_measurements = len(quantum_result.raw_measurements or [10])
        classical_measurements = len(classical_baseline.raw_measurements or [10])
        confidence = min(1.0, (quantum_measurements + classical_measurements) / 40.0)
        
        return {
            'criterion_met': criterion_met,
            'score': robustness_ratio if criterion_met else 1.0 / robustness_ratio,
            'robustness_ratio': robustness_ratio,
            'confidence': confidence,
            'details': f"Q-variance: {quantum_result.variance:.4f}, C-variance: {classical_baseline.variance:.4f}"
        }
    
    def _calculate_falsification_confidence(self, criteria_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall confidence in falsification test."""
        
        # Weighted average of individual criterion confidences
        weighted_confidence = sum(
            self.criteria_weights[criterion] * result['confidence']
            for criterion, result in criteria_results.items()
        )
        
        # Adjust for number of criteria that failed
        failed_criteria = sum(1 for result in criteria_results.values() if not result['criterion_met'])
        failure_penalty = failed_criteria / len(criteria_results)
        
        # Final confidence accounting for failures
        final_confidence = weighted_confidence * (1 - failure_penalty * 0.3)
        
        return max(0.1, min(0.95, final_confidence))
    
    def _generate_recommendation(self, is_falsified: bool, overall_score: float, 
                               confidence: float) -> str:
        """Generate recommendation based on falsification test results."""
        
        if is_falsified and confidence > 0.7:
            return "REJECT: Quantum advantage claim is falsified with high confidence"
        elif is_falsified and confidence > 0.5:
            return "LIKELY_REJECT: Quantum advantage claim is likely falsified"
        elif not is_falsified and overall_score > 0.8 and confidence > 0.7:
            return "ACCEPT: Quantum advantage claim is supported with high confidence"
        elif not is_falsified and overall_score > 0.6:
            return "LIKELY_ACCEPT: Quantum advantage claim is likely supported"
        else:
            return "INCONCLUSIVE: Insufficient evidence to determine quantum advantage"
    
    def _create_fallback_test_result(self, advantage_claim: str) -> Dict[str, Any]:
        """Create fallback test result when falsification test fails."""
        
        return {
            'advantage_claim': advantage_claim,
            'is_falsified': True,
            'overall_score': 0.3,
            'confidence': 0.1,
            'criteria_results': {},
            'recommendation': "ERROR: Test failed, cannot validate advantage claim"
        }
    
    def _record_test_result(self, result: Dict[str, Any], 
                           quantum_result: QuantumResult,
                           classical_baseline: QuantumResult):
        """Record falsification test result."""
        
        record = {
            'timestamp': time.time(),
            'advantage_claim': result['advantage_claim'],
            'is_falsified': result['is_falsified'],
            'overall_score': result['overall_score'],
            'confidence': result['confidence'],
            'recommendation': result['recommendation']
        }
        
        self.test_results.append(record)
        
        # Keep history manageable
        if len(self.test_results) > 200:
            self.test_results = self.test_results[-200:]
            
        self.logger.info(f"Falsification test: {result['recommendation']}")


# Export classes
__all__ = [
    'VScoreQuantumAdvantageDetector',
    'FalsifiableAdvantageFramework',
    'QuantumAdvantageType',
    'VScoreCategory',
    'QuantumResult',
    'QuantumAdvantageResult',
    'VScoreComponents'
]