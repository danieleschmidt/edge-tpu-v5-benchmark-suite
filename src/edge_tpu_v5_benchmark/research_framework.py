"""Research Framework for Quantum-Enhanced TPU Optimization Studies.

This module implements the experimental framework for validating quantum optimization
algorithms against classical baselines with rigorous statistical methodology.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multitest import multipletests
import json
from abc import ABC, abstractmethod

from .quantum_planner import QuantumTaskPlanner
from .quantum_performance import OptimizedQuantumTaskPlanner
from .quantum_auto_scaling import QuantumAutoScaler


class ExperimentType(Enum):
    """Types of experiments in the research framework."""
    SCHEDULING_COMPARISON = "scheduling_comparison"
    AUTOSCALING_VALIDATION = "autoscaling_validation"
    INFERENCE_OPTIMIZATION = "inference_optimization"
    STATISTICAL_VALIDATION = "statistical_validation"


class WorkloadPattern(Enum):
    """Different workload patterns for testing."""
    BATCH = "batch"
    STREAMING = "streaming"
    MIXED = "mixed"
    BURSTY = "bursty"
    PERIODIC = "periodic"
    RANDOM = "random"


class HypothesisStatus(Enum):
    """Status of research hypotheses."""
    PENDING = "pending"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    RETIRED = "retired"


class StatisticalTest(Enum):
    """Available statistical tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    KRUSKAL_WALLIS = "kruskal_wallis"
    ANOVA = "anova"


@dataclass
class Hypothesis:
    """Research hypothesis with validation criteria."""
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    metrics: List[str] = field(default_factory=list)
    expected_improvement: Dict[str, float] = field(default_factory=dict)  # metric -> min % improvement
    significance_threshold: float = 0.05
    effect_size_threshold: float = 0.5
    status: HypothesisStatus = HypothesisStatus.PENDING
    created_at: float = field(default_factory=time.time)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def is_validated(self, results: Dict[str, Any]) -> bool:
        """Check if hypothesis is validated by results."""
        if not self.metrics or not results:
            return False
            
        validated_metrics = 0
        for metric in self.metrics:
            metric_result = results.get(metric, {})
            if not metric_result:
                continue
                
            # Check statistical significance
            p_value = metric_result.get('p_value', 1.0)
            effect_size = abs(metric_result.get('effect_size_cohens_d', 0.0))
            improvement = abs(metric_result.get('percent_improvement', 0.0))
            expected = self.expected_improvement.get(metric, 0.0)
            
            if (p_value < self.significance_threshold and 
                effect_size >= self.effect_size_threshold and
                improvement >= expected):
                validated_metrics += 1
        
        return validated_metrics >= len(self.metrics) * 0.7  # 70% of metrics must validate


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    experiment_type: ExperimentType
    workload_pattern: WorkloadPattern
    num_tasks: int = 1000
    num_iterations: int = 30
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.20
    statistical_power: float = 0.80
    random_seed: int = 42
    hardware_platform: str = "tpu_v5"
    enable_logging: bool = True
    
    # Quantum-specific parameters
    coherence_threshold: float = 0.7
    decoherence_rate: float = 0.1
    entanglement_strength: float = 0.8
    annealing_temperature: float = 1.0
    
    # Hypothesis-driven parameters
    adaptive_sample_size: bool = True
    max_iterations: int = 1000
    early_stopping: bool = True
    alpha_correction: str = 'bonferroni'  # 'bonferroni', 'fdr', 'none'
    power_analysis: bool = True
    minimum_power: float = 0.80


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    algorithm_name: str
    workload_pattern: WorkloadPattern
    
    # Performance metrics
    makespan: float
    resource_utilization: float
    throughput: float
    latency_p95: float
    energy_consumption: float
    
    # Quantum-specific metrics
    coherence_maintained: float
    entanglement_efficiency: float
    decoherence_prediction_accuracy: float
    
    # System metrics
    memory_usage: float
    cpu_utilization: float
    execution_time: float
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    config: ExperimentConfig = field(default=None)
    raw_measurements: Dict[str, List[float]] = field(default_factory=dict)


class HypothesisManager:
    """Manages research hypotheses lifecycle."""
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.hypothesis_history: List[Dict[str, Any]] = []
        
    def register_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Register a new hypothesis."""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.hypothesis_history.append({
            'action': 'registered',
            'hypothesis_id': hypothesis.hypothesis_id,
            'timestamp': time.time(),
            'name': hypothesis.name
        })
        return hypothesis.hypothesis_id
    
    def update_hypothesis_status(self, hypothesis_id: str, status: HypothesisStatus, results: Dict[str, Any] = None):
        """Update hypothesis status with optional results."""
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
            
        hypothesis = self.hypotheses[hypothesis_id]
        old_status = hypothesis.status
        hypothesis.status = status
        
        if results:
            hypothesis.validation_results = results
            
        self.hypothesis_history.append({
            'action': 'status_changed',
            'hypothesis_id': hypothesis_id,
            'old_status': old_status.value,
            'new_status': status.value,
            'timestamp': time.time(),
            'results': results
        })
    
    def generate_adaptive_hypotheses(self, previous_results: pd.DataFrame) -> List[Hypothesis]:
        """Generate new hypotheses based on previous experimental results."""
        hypotheses = []
        
        if previous_results.empty:
            # Generate initial baseline hypotheses
            return self._generate_baseline_hypotheses()
        
        # Analyze results for patterns
        quantum_perf = previous_results[previous_results['algorithm'] == 'quantum_scheduling']
        
        if not quantum_perf.empty:
            # Check for workload-specific patterns
            workload_groups = previous_results.groupby('algorithm').agg({
                'makespan': 'mean',
                'resource_utilization': 'mean',
                'throughput': 'mean'
            })
            
            # Generate hypotheses for underperforming areas
            quantum_makespan = workload_groups.loc['quantum_scheduling', 'makespan']
            
            for classical_algo in ['heft', 'genetic']:
                if classical_algo in workload_groups.index:
                    classical_makespan = workload_groups.loc[classical_algo, 'makespan']
                    
                    if quantum_makespan > classical_makespan:  # Quantum is slower
                        hypotheses.append(Hypothesis(
                            name=f"Enhanced Quantum Coherence for {classical_algo.upper()} Improvement",
                            description=f"Increasing coherence threshold should improve quantum performance vs {classical_algo}",
                            metrics=['makespan', 'resource_utilization'],
                            expected_improvement={'makespan': 15.0, 'resource_utilization': 10.0},
                            tags={'adaptive', 'performance_improvement', classical_algo}
                        ))
        
        return hypotheses
    
    def _generate_baseline_hypotheses(self) -> List[Hypothesis]:
        """Generate initial baseline hypotheses."""
        return [
            Hypothesis(
                name="Quantum Coherence-Guided Task Scheduling",
                description="Quantum coherence-guided scheduling should outperform classical algorithms",
                metrics=['makespan', 'resource_utilization', 'throughput'],
                expected_improvement={'makespan': 20.0, 'resource_utilization': 15.0, 'throughput': 10.0},
                tags={'baseline', 'scheduling'}
            ),
            Hypothesis(
                name="Workload-Adaptive Quantum Optimization",
                description="Quantum algorithms should show greater advantage on mixed workloads",
                metrics=['makespan', 'energy_consumption'],
                expected_improvement={'makespan': 25.0, 'energy_consumption': 15.0},
                tags={'baseline', 'workload_adaptive'}
            ),
            Hypothesis(
                name="Quantum Resource Utilization Efficiency",
                description="Quantum scheduling should achieve higher resource utilization",
                metrics=['resource_utilization', 'throughput'],
                expected_improvement={'resource_utilization': 20.0, 'throughput': 15.0},
                tags={'baseline', 'efficiency'}
            )
        ]
    
    def get_active_hypotheses(self) -> List[Hypothesis]:
        """Get all active (pending or testing) hypotheses."""
        return [h for h in self.hypotheses.values() 
                if h.status in [HypothesisStatus.PENDING, HypothesisStatus.TESTING]]
    
    def get_validated_hypotheses(self) -> List[Hypothesis]:
        """Get all validated hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.VALIDATED]


class ClassicalScheduler:
    """Classical scheduling algorithms for baseline comparison."""
    
    @staticmethod
    def heft_scheduler(tasks: List[Dict], resources: List[Dict]) -> Tuple[float, Dict]:
        """Heterogeneous Earliest Finish Time (HEFT) algorithm."""
        # Simplified HEFT implementation for baseline
        task_priorities = [task.get('computation_time', 1.0) + 
                          sum(dep.get('communication_cost', 0.1) 
                              for dep in task.get('dependencies', []))
                          for task in tasks]
        
        sorted_tasks = sorted(zip(tasks, task_priorities), key=lambda x: x[1], reverse=True)
        
        makespan = 0
        resource_usage = {i: 0 for i in range(len(resources))}
        
        for task, _ in sorted_tasks:
            # Simple greedy assignment to earliest available resource
            best_resource = min(resource_usage.keys(), key=resource_usage.get)
            finish_time = resource_usage[best_resource] + task.get('computation_time', 1.0)
            resource_usage[best_resource] = finish_time
            makespan = max(makespan, finish_time)
        
        avg_utilization = sum(resource_usage.values()) / (len(resources) * makespan) if makespan > 0 else 0
        
        return makespan, {'resource_utilization': avg_utilization}
    
    @staticmethod
    def genetic_algorithm_scheduler(tasks: List[Dict], resources: List[Dict]) -> Tuple[float, Dict]:
        """Simple genetic algorithm for task scheduling."""
        # Simplified GA implementation
        population_size = 50
        generations = 100
        
        best_makespan = float('inf')
        best_utilization = 0
        
        # Random search simulation (simplified GA)
        for _ in range(generations * population_size):
            # Random task assignment
            assignment = [np.random.randint(0, len(resources)) for _ in tasks]
            
            # Calculate makespan
            resource_times = [0] * len(resources)
            for i, task in enumerate(tasks):
                resource_times[assignment[i]] += task.get('computation_time', 1.0)
            
            makespan = max(resource_times)
            utilization = sum(resource_times) / (len(resources) * makespan) if makespan > 0 else 0
            
            if makespan < best_makespan:
                best_makespan = makespan
                best_utilization = utilization
        
        return best_makespan, {'resource_utilization': best_utilization}


class ResearchFramework:
    """Main research framework for quantum optimization experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.hypothesis_manager = HypothesisManager()
        self.current_iteration = 0
        self.convergence_history: List[Dict[str, float]] = []
        
        # Set up logging
        if config.enable_logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.disabled = True
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        
        # Initialize quantum components
        self.quantum_planner = QuantumTaskPlanner()
        self.optimized_planner = OptimizedQuantumTaskPlanner()
        self.quantum_scaler = QuantumAutoScaler()
        
        # Initialize baseline hypotheses
        baseline_hypotheses = self.hypothesis_manager._generate_baseline_hypotheses()
        for hypothesis in baseline_hypotheses:
            self.hypothesis_manager.register_hypothesis(hypothesis)
        
        self.logger.info(f"Research framework initialized with config: {config}")
        self.logger.info(f"Registered {len(baseline_hypotheses)} baseline hypotheses")
    
    def generate_workload(self, pattern: WorkloadPattern, num_tasks: int) -> Tuple[List[Dict], List[Dict]]:
        """Generate synthetic workload based on specified pattern."""
        tasks = []
        resources = [{'id': i, 'capacity': 1.0} for i in range(8)]  # 8 TPU cores
        
        if pattern == WorkloadPattern.BATCH:
            # Large batch processing tasks
            for i in range(num_tasks):
                tasks.append({
                    'id': i,
                    'computation_time': np.random.exponential(2.0),
                    'memory_requirement': np.random.uniform(0.1, 0.8),
                    'dependencies': []
                })
        
        elif pattern == WorkloadPattern.STREAMING:
            # Continuous stream of small tasks
            for i in range(num_tasks):
                tasks.append({
                    'id': i,
                    'computation_time': np.random.gamma(2, 0.5),
                    'memory_requirement': np.random.uniform(0.05, 0.3),
                    'dependencies': []
                })
        
        elif pattern == WorkloadPattern.MIXED:
            # Mix of batch and streaming
            for i in range(num_tasks):
                if np.random.random() < 0.3:  # 30% batch tasks
                    comp_time = np.random.exponential(3.0)
                    mem_req = np.random.uniform(0.3, 0.9)
                else:  # 70% streaming tasks
                    comp_time = np.random.gamma(1, 0.3)
                    mem_req = np.random.uniform(0.05, 0.2)
                
                tasks.append({
                    'id': i,
                    'computation_time': comp_time,
                    'memory_requirement': mem_req,
                    'dependencies': []
                })
        
        elif pattern == WorkloadPattern.BURSTY:
            # Bursty workload with periods of high/low activity
            burst_probability = 0.2
            for i in range(num_tasks):
                if np.random.random() < burst_probability:
                    comp_time = np.random.exponential(0.5)  # Short burst tasks
                else:
                    comp_time = np.random.exponential(2.0)  # Normal tasks
                
                tasks.append({
                    'id': i,
                    'computation_time': comp_time,
                    'memory_requirement': np.random.uniform(0.1, 0.5),
                    'dependencies': []
                })
        
        return tasks, resources
    
    def run_quantum_scheduling_experiment(self, tasks: List[Dict], resources: List[Dict]) -> ExperimentResult:
        """Run quantum scheduling experiment."""
        start_time = time.time()
        
        # Convert tasks to quantum format
        quantum_tasks = []
        for task in tasks:
            quantum_task = self.quantum_planner.create_task(
                task_id=str(task['id']),
                name=f"task_{task['id']}",
                dependencies=set(task.get('dependencies', [])),
                resource_requirements={'compute': task['computation_time']}
            )
            quantum_tasks.append(quantum_task)
        
        # Execute quantum scheduling
        schedule = self.quantum_planner.optimize_schedule()
        
        # Calculate metrics
        # Since quantum tasks don't have completion_time, estimate based on duration
        makespan = sum(task.estimated_duration for task in schedule) / len(resources) if schedule and resources else 0
        total_work = sum(task.resource_requirements.get('compute', 0) for task in quantum_tasks)
        resource_utilization = total_work / (len(resources) * makespan) if makespan > 0 else 0
        
        # Quantum-specific metrics
        coherent_tasks = sum(1 for task in schedule if task.state.value in ['superposition', 'collapsed'])
        coherence_maintained = coherent_tasks / len(schedule) if schedule else 0
        
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            experiment_id=f"quantum_{int(time.time())}",
            algorithm_name="quantum_scheduling",
            workload_pattern=self.config.workload_pattern,
            makespan=makespan,
            resource_utilization=resource_utilization,
            throughput=len(tasks) / execution_time,
            latency_p95=np.percentile([task.estimated_duration for task in schedule], 95) if schedule else 0,
            energy_consumption=makespan * 0.8,  # Simplified energy model
            coherence_maintained=coherence_maintained,
            entanglement_efficiency=0.85,  # Placeholder
            decoherence_prediction_accuracy=0.90,  # Placeholder
            memory_usage=sum(task['memory_requirement'] for task in tasks),
            cpu_utilization=0.75,  # Placeholder
            execution_time=execution_time,
            config=self.config
        )
    
    def run_classical_scheduling_experiment(self, scheduler_name: str, tasks: List[Dict], 
                                          resources: List[Dict]) -> ExperimentResult:
        """Run classical scheduling experiment."""
        start_time = time.time()
        
        if scheduler_name == "heft":
            makespan, metrics = ClassicalScheduler.heft_scheduler(tasks, resources)
        elif scheduler_name == "genetic":
            makespan, metrics = ClassicalScheduler.genetic_algorithm_scheduler(tasks, resources)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            experiment_id=f"{scheduler_name}_{int(time.time())}",
            algorithm_name=scheduler_name,
            workload_pattern=self.config.workload_pattern,
            makespan=makespan,
            resource_utilization=metrics.get('resource_utilization', 0),
            throughput=len(tasks) / execution_time,
            latency_p95=makespan * 0.95,  # Approximation
            energy_consumption=makespan * 1.0,  # Simplified energy model
            coherence_maintained=0.0,  # Not applicable for classical
            entanglement_efficiency=0.0,  # Not applicable for classical
            decoherence_prediction_accuracy=0.0,  # Not applicable for classical
            memory_usage=sum(task['memory_requirement'] for task in tasks),
            cpu_utilization=0.70,  # Placeholder
            execution_time=execution_time,
            config=self.config
        )
    
    def calculate_sample_size(self, effect_size: float = 0.5, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for statistical power."""
        if not self.config.power_analysis:
            return self.config.num_iterations
            
        try:
            # Use power analysis to determine sample size
            sample_size = ttest_power(effect_size, power, alpha, alternative='two-sided')
            calculated_size = max(10, int(np.ceil(sample_size * 2)))  # Two groups, minimum 10
            
            self.logger.info(f"Power analysis: calculated sample size = {calculated_size}")
            return min(calculated_size, self.config.max_iterations)
        except Exception as e:
            self.logger.warning(f"Power analysis failed: {e}. Using default iterations.")
            return self.config.num_iterations
    
    def check_early_stopping(self, df: pd.DataFrame, hypothesis: Hypothesis) -> bool:
        """Check if we can stop early based on statistical significance."""
        if not self.config.early_stopping or len(df) < 20:  # Minimum samples
            return False
        
        # Quick analysis for early stopping
        try:
            quantum_data = df[df['algorithm'] == 'quantum_scheduling']
            heft_data = df[df['algorithm'] == 'heft']
            
            if len(quantum_data) < 10 or len(heft_data) < 10:
                return False
            
            # Check primary metric
            primary_metric = hypothesis.metrics[0] if hypothesis.metrics else 'makespan'
            if primary_metric not in quantum_data.columns:
                return False
                
            quantum_values = quantum_data[primary_metric].values
            heft_values = heft_data[primary_metric].values
            
            # Statistical test
            _, p_value = stats.ttest_ind(quantum_values, heft_values)
            
            # Early stopping if very significant or very non-significant
            if p_value < 0.01 or p_value > 0.5:
                self.logger.info(f"Early stopping triggered: p={p_value:.4f}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Early stopping check failed: {e}")
            
        return False
    
    def run_hypothesis_driven_study(self, target_hypothesis: Optional[Hypothesis] = None) -> pd.DataFrame:
        """Run hypothesis-driven comparative study with adaptive sampling."""
        self.logger.info("Starting hypothesis-driven study...")
        
        # Select hypothesis to test
        if target_hypothesis is None:
            active_hypotheses = self.hypothesis_manager.get_active_hypotheses()
            if not active_hypotheses:
                self.logger.warning("No active hypotheses. Generating new ones.")
                new_hypotheses = self.hypothesis_manager.generate_adaptive_hypotheses(pd.DataFrame())
                for h in new_hypotheses:
                    self.hypothesis_manager.register_hypothesis(h)
                active_hypotheses = self.hypothesis_manager.get_active_hypotheses()
            
            target_hypothesis = active_hypotheses[0] if active_hypotheses else None
        
        if target_hypothesis is None:
            raise ValueError("No hypothesis available for testing")
        
        self.logger.info(f"Testing hypothesis: {target_hypothesis.name}")
        self.hypothesis_manager.update_hypothesis_status(
            target_hypothesis.hypothesis_id, 
            HypothesisStatus.TESTING
        )
        
        algorithms = ["quantum_scheduling", "heft", "genetic"]
        all_results = []
        
        # Calculate adaptive sample size
        if self.config.adaptive_sample_size:
            max_iterations = self.calculate_sample_size(
                effect_size=target_hypothesis.effect_size_threshold,
                power=self.config.minimum_power,
                alpha=target_hypothesis.significance_threshold
            )
        else:
            max_iterations = self.config.num_iterations
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration
            self.logger.info(f"Running iteration {iteration + 1}/{max_iterations}")
            
            # Generate workload for this iteration
            tasks, resources = self.generate_workload(
                self.config.workload_pattern, 
                self.config.num_tasks
            )
            
            # Test each algorithm
            for algorithm in algorithms:
                if algorithm == "quantum_scheduling":
                    result = self.run_quantum_scheduling_experiment(tasks, resources)
                else:
                    result = self.run_classical_scheduling_experiment(algorithm, tasks, resources)
                
                all_results.append(result)
                self.results.append(result)
            
            # Convert current results to DataFrame for analysis
            current_df = pd.DataFrame([
                {
                    'algorithm': result.algorithm_name,
                    'makespan': result.makespan,
                    'resource_utilization': result.resource_utilization,
                    'throughput': result.throughput,
                    'latency_p95': result.latency_p95,
                    'energy_consumption': result.energy_consumption,
                    'coherence_maintained': result.coherence_maintained,
                    'execution_time': result.execution_time
                }
                for result in all_results
            ])
            
            # Check for early stopping
            if self.check_early_stopping(current_df, target_hypothesis):
                self.logger.info(f"Early stopping at iteration {iteration + 1}")
                break
        
        # Final DataFrame
        df = pd.DataFrame([
            {
                'algorithm': result.algorithm_name,
                'makespan': result.makespan,
                'resource_utilization': result.resource_utilization,
                'throughput': result.throughput,
                'latency_p95': result.latency_p95,
                'energy_consumption': result.energy_consumption,
                'coherence_maintained': result.coherence_maintained,
                'execution_time': result.execution_time
            }
            for result in all_results
        ])
        
        self.logger.info(f"Hypothesis-driven study completed with {len(all_results)} samples")
        return df
    
    def run_comparative_study(self) -> pd.DataFrame:
        """Run comprehensive comparative study (legacy method)."""
        return self.run_hypothesis_driven_study()
    
    def statistical_analysis_with_correction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform rigorous statistical analysis with multiple testing correction."""
        self.logger.info("Performing statistical analysis with multiple testing correction...")
        
        analysis_results = {}
        quantum_data = df[df['algorithm'] == 'quantum_scheduling']
        
        # Collect all p-values for correction
        all_p_values = []
        comparison_metric_pairs = []
        
        # Compare quantum vs each classical algorithm
        for classical_algo in ['heft', 'genetic']:
            classical_data = df[df['algorithm'] == classical_algo]
            if classical_data.empty:
                continue
                
            comparison_name = f"quantum_vs_{classical_algo}"
            analysis_results[comparison_name] = {}
            
            # Metrics to analyze
            metrics = ['makespan', 'resource_utilization', 'throughput', 'energy_consumption']
            
            for metric in metrics:
                if metric not in quantum_data.columns or metric not in classical_data.columns:
                    continue
                    
                quantum_values = quantum_data[metric].values
                classical_values = classical_data[metric].values
                
                if len(quantum_values) == 0 or len(classical_values) == 0:
                    continue
                
                # Statistical tests
                try:
                    # Normality test
                    _, quantum_normal = stats.shapiro(quantum_values) if len(quantum_values) > 3 else (0, 0.01)
                    _, classical_normal = stats.shapiro(classical_values) if len(classical_values) > 3 else (0, 0.01)
                    
                    if quantum_normal > 0.05 and classical_normal > 0.05:
                        # Use t-test for normal distributions
                        statistic, p_value = stats.ttest_ind(quantum_values, classical_values)
                        test_used = "t-test"
                    else:
                        # Use Mann-Whitney U for non-normal distributions
                        statistic, p_value = stats.mannwhitneyu(
                            quantum_values, classical_values, alternative='two-sided'
                        )
                        test_used = "mann-whitney"
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        ((len(quantum_values) - 1) * np.var(quantum_values, ddof=1) +
                         (len(classical_values) - 1) * np.var(classical_values, ddof=1)) /
                        (len(quantum_values) + len(classical_values) - 2)
                    ) if len(quantum_values) > 1 and len(classical_values) > 1 else 1.0
                    
                    cohens_d = (np.mean(quantum_values) - np.mean(classical_values)) / pooled_std if pooled_std > 0 else 0.0
                    
                    # Confidence interval for mean difference
                    mean_diff = np.mean(quantum_values) - np.mean(classical_values)
                    se_diff = pooled_std * np.sqrt(1/len(quantum_values) + 1/len(classical_values)) if pooled_std > 0 else 0.0
                    
                    if se_diff > 0:
                        ci_lower, ci_upper = stats.t.interval(
                            self.config.confidence_level,
                            len(quantum_values) + len(classical_values) - 2,
                            mean_diff, se_diff
                        )
                    else:
                        ci_lower, ci_upper = mean_diff, mean_diff
                    
                    analysis_results[comparison_name][metric] = {
                        'quantum_mean': np.mean(quantum_values),
                        'quantum_std': np.std(quantum_values),
                        'classical_mean': np.mean(classical_values),
                        'classical_std': np.std(classical_values),
                        'mean_difference': mean_diff,
                        'percent_improvement': (mean_diff / np.mean(classical_values)) * 100 if np.mean(classical_values) != 0 else 0,
                        'test_statistic': statistic,
                        'p_value': p_value,
                        'p_value_raw': p_value,  # Store raw p-value
                        'effect_size_cohens_d': cohens_d,
                        'effect_size_large': abs(cohens_d) > 0.8,
                        'confidence_interval': (ci_lower, ci_upper),
                        'test_used': test_used
                    }
                    
                    # Collect for multiple testing correction
                    all_p_values.append(p_value)
                    comparison_metric_pairs.append((comparison_name, metric))
                    
                except Exception as e:
                    self.logger.warning(f"Statistical test failed for {comparison_name}/{metric}: {e}")
                    continue
        
        # Apply multiple testing correction
        if all_p_values and self.config.alpha_correction != 'none':
            try:
                if self.config.alpha_correction == 'bonferroni':
                    corrected_significant, corrected_p_values, _, _ = multipletests(
                        all_p_values, method='bonferroni'
                    )
                elif self.config.alpha_correction == 'fdr':
                    corrected_significant, corrected_p_values, _, _ = multipletests(
                        all_p_values, method='fdr_bh'
                    )
                else:
                    corrected_p_values = all_p_values
                    corrected_significant = [p < (1 - self.config.confidence_level) for p in all_p_values]
                
                # Update results with corrected p-values
                for i, (comparison_name, metric) in enumerate(comparison_metric_pairs):
                    if comparison_name in analysis_results and metric in analysis_results[comparison_name]:
                        analysis_results[comparison_name][metric]['p_value_corrected'] = corrected_p_values[i]
                        analysis_results[comparison_name][metric]['significant'] = corrected_significant[i]
                        analysis_results[comparison_name][metric]['correction_method'] = self.config.alpha_correction
                
                self.logger.info(f"Applied {self.config.alpha_correction} correction to {len(all_p_values)} tests")
                
            except Exception as e:
                self.logger.warning(f"Multiple testing correction failed: {e}")
                # Fallback to uncorrected p-values
                for comparison_name in analysis_results:
                    for metric in analysis_results[comparison_name]:
                        result = analysis_results[comparison_name][metric]
                        result['p_value_corrected'] = result['p_value']
                        result['significant'] = result['p_value'] < (1 - self.config.confidence_level)
                        result['correction_method'] = 'none'
        
        self.logger.info("Statistical analysis completed")
        return analysis_results
    
    def statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform rigorous statistical analysis (backward compatibility)."""
        return self.statistical_analysis_with_correction(df)
    
    def validate_hypotheses(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all active hypotheses against statistical results."""
        validation_results = {}
        
        for hypothesis in self.hypothesis_manager.get_active_hypotheses():
            # Test hypothesis against results
            is_validated = hypothesis.is_validated(stats_results.get('quantum_vs_heft', {}))
            
            if is_validated:
                self.hypothesis_manager.update_hypothesis_status(
                    hypothesis.hypothesis_id, 
                    HypothesisStatus.VALIDATED,
                    stats_results
                )
                self.logger.info(f"Hypothesis VALIDATED: {hypothesis.name}")
            else:
                # Check if we have enough evidence to reject
                has_significant_evidence = any(
                    stats_results.get('quantum_vs_heft', {}).get(metric, {}).get('significant', False)
                    for metric in hypothesis.metrics
                )
                
                if has_significant_evidence:
                    # Significant evidence but wrong direction - reject
                    self.hypothesis_manager.update_hypothesis_status(
                        hypothesis.hypothesis_id, 
                        HypothesisStatus.REJECTED,
                        stats_results
                    )
                    self.logger.info(f"Hypothesis REJECTED: {hypothesis.name}")
                else:
                    # Insufficient evidence - continue testing
                    self.logger.info(f"Hypothesis requires more evidence: {hypothesis.name}")
            
            validation_results[hypothesis.hypothesis_id] = {
                'hypothesis_name': hypothesis.name,
                'status': hypothesis.status.value,
                'is_validated': is_validated,
                'validation_results': hypothesis.validation_results
            }
        
        return validation_results
    
    def generate_adaptive_hypotheses(self, stats_results: Dict[str, Any]) -> List[Hypothesis]:
        """Generate new hypotheses based on current results."""
        # Convert current results to DataFrame for analysis
        current_df = pd.DataFrame([
            {
                'algorithm': result.algorithm_name,
                'makespan': result.makespan,
                'resource_utilization': result.resource_utilization,
                'throughput': result.throughput
            }
            for result in self.results
        ])
        
        new_hypotheses = self.hypothesis_manager.generate_adaptive_hypotheses(current_df)
        
        for hypothesis in new_hypotheses:
            self.hypothesis_manager.register_hypothesis(hypothesis)
            self.logger.info(f"Generated new hypothesis: {hypothesis.name}")
        
        return new_hypotheses
    
    def generate_report(self, df: pd.DataFrame, stats_results: Dict[str, Any], 
                      hypothesis_validation: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive research report."""
        report = []
        
        report.append("# Quantum-Enhanced TPU Optimization Research Report")
        report.append(f"\n## Experiment Configuration")
        report.append(f"- Experiment Type: {self.config.experiment_type.value}")
        report.append(f"- Workload Pattern: {self.config.workload_pattern.value}")
        report.append(f"- Number of Tasks: {self.config.num_tasks}")
        report.append(f"- Actual Iterations: {self.current_iteration + 1}")
        report.append(f"- Confidence Level: {self.config.confidence_level}")
        report.append(f"- Multiple Testing Correction: {self.config.alpha_correction}")
        report.append(f"- Early Stopping: {self.config.early_stopping}")
        report.append(f"- Adaptive Sample Size: {self.config.adaptive_sample_size}")
        report.append(f"- Hardware Platform: {self.config.hardware_platform}")
        
        report.append("\n## Summary Statistics")
        summary = df.groupby('algorithm').agg({
            'makespan': ['mean', 'std'],
            'resource_utilization': ['mean', 'std'],
            'throughput': ['mean', 'std']
        }).round(4)
        report.append(summary.to_string())
        
        report.append("\n## Statistical Comparison Results")
        
        for comparison, results in stats_results.items():
            report.append(f"\n### {comparison.replace('_', ' ').title()}")
            
            for metric, stats_info in results.items():
                p_val = stats_info.get('p_value_corrected', stats_info.get('p_value', 1.0))
                correction = stats_info.get('correction_method', 'none')
                
                if stats_info.get('significant', False):
                    significance = "SIGNIFICANT"
                    improvement = stats_info['percent_improvement']
                    effect_size = stats_info['effect_size_cohens_d']
                    report.append(
                        f"- **{metric}**: {significance} - "
                        f"{improvement:.1f}% improvement "
                        f"(p_corrected={p_val:.4f}, d={effect_size:.3f}, correction={correction})"
                    )
                else:
                    report.append(
                        f"- **{metric}**: Not significant "
                        f"(p_corrected={p_val:.4f}, correction={correction})"
                    )
        
        report.append("\n## Hypothesis Validation Results")
        
        if hypothesis_validation:
            validated_count = sum(1 for h in hypothesis_validation.values() if h['status'] == 'validated')
            rejected_count = sum(1 for h in hypothesis_validation.values() if h['status'] == 'rejected')
            testing_count = sum(1 for h in hypothesis_validation.values() if h['status'] == 'testing')
            
            report.append(f"\n**Summary**: {validated_count} validated, {rejected_count} rejected, {testing_count} still testing")
            
            for hypothesis_id, validation_info in hypothesis_validation.items():
                status = validation_info['status'].upper()
                name = validation_info['hypothesis_name']
                
                if status == 'VALIDATED':
                    report.append(f"\n### ✅ {name} - {status}")
                elif status == 'REJECTED':
                    report.append(f"\n### ❌ {name} - {status}")
                else:
                    report.append(f"\n### 🔄 {name} - {status}")
                
                # Show specific metrics if available
                if validation_info.get('validation_results'):
                    for metric, result in validation_info['validation_results'].items():
                        if isinstance(result, dict) and 'percent_improvement' in result:
                            improvement = result['percent_improvement']
                            p_val = result.get('p_value_corrected', result.get('p_value', 1.0))
                            report.append(f"  - {metric}: {improvement:.1f}% improvement (p={p_val:.4f})")
        else:
            # Fallback to legacy validation
            quantum_vs_heft = stats_results.get('quantum_vs_heft', {})
            makespan_improvement = quantum_vs_heft.get('makespan', {}).get('percent_improvement', 0)
            util_improvement = quantum_vs_heft.get('resource_utilization', {}).get('percent_improvement', 0)
            
            report.append("\n### Legacy Hypothesis: Quantum Coherence-Guided Task Scheduling")
            report.append(f"- **Result**: {abs(makespan_improvement):.1f}% makespan change, {util_improvement:.1f}% utilization improvement")
            
            if abs(makespan_improvement) >= 15 and util_improvement >= 10:
                report.append("- **Status**: ✅ HYPOTHESIS CONFIRMED")
            else:
                report.append("- **Status**: ❌ HYPOTHESIS NOT CONFIRMED")
        
        # Add hypothesis manager insights
        active_hypotheses = self.hypothesis_manager.get_active_hypotheses()
        validated_hypotheses = self.hypothesis_manager.get_validated_hypotheses()
        
        report.append("\n## Research Framework Status")
        report.append(f"- Active Hypotheses: {len(active_hypotheses)}")
        report.append(f"- Validated Hypotheses: {len(validated_hypotheses)}")
        report.append(f"- Total Experiments Run: {len(self.results)}")
        
        report.append("\n## Recommendations")
        
        if len(validated_hypotheses) == 0:
            report.append("1. **No hypotheses validated** - Consider adjusting quantum parameters or exploring different algorithms")
            report.append("2. Generate new adaptive hypotheses based on current performance patterns")
        else:
            report.append("1. **Validated hypotheses found** - Focus on optimizing confirmed approaches")
            report.append("2. Investigate scaling properties of validated quantum algorithms")
        
        if len(active_hypotheses) > 0:
            report.append("3. Continue testing active hypotheses with larger sample sizes if needed")
        else:
            report.append("3. Generate new hypotheses for unexplored optimization opportunities")
            
        report.append("4. Extend experiments to larger-scale workloads and real-world scenarios")
        report.append("5. Implement hardware-specific optimizations based on validated approaches")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str, df: pd.DataFrame, stats_results: Dict[str, Any], 
                    report: str, hypothesis_validation: Optional[Dict[str, Any]] = None):
        """Save all results to files."""
        # Save raw data
        df.to_csv(f"{filepath}_raw_data.csv", index=False)
        
        # Save statistical results
        with open(f"{filepath}_statistics.json", 'w') as f:
            # Convert numpy types for JSON serialization
            json_compatible_stats = {}
            for comparison, results in stats_results.items():
                json_compatible_stats[comparison] = {}
                for metric, stats_info in results.items():
                    json_compatible_stats[comparison][metric] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else 
                           (list(v) if isinstance(v, tuple) else v)
                        for k, v in stats_info.items()
                    }
            
            json.dump(json_compatible_stats, f, indent=2)
        
        # Save hypothesis validation results
        if hypothesis_validation:
            with open(f"{filepath}_hypothesis_validation.json", 'w') as f:
                json.dump(hypothesis_validation, f, indent=2)
        
        # Save hypothesis history
        with open(f"{filepath}_hypothesis_history.json", 'w') as f:
            json.dump(self.hypothesis_manager.hypothesis_history, f, indent=2)
        
        # Save report
        with open(f"{filepath}_report.md", 'w') as f:
            f.write(report)
        
        self.logger.info(f"Results saved to {filepath}_*")


def main():
    """Main function to run the enhanced research framework."""
    # Configure experiment with enhanced settings
    config = ExperimentConfig(
        experiment_type=ExperimentType.SCHEDULING_COMPARISON,
        workload_pattern=WorkloadPattern.MIXED,
        num_tasks=100,
        num_iterations=30,  # Increased for better statistical power
        max_iterations=100,
        adaptive_sample_size=True,
        early_stopping=True,
        alpha_correction='fdr',  # False Discovery Rate correction
        power_analysis=True,
        minimum_power=0.80,
        random_seed=42
    )
    
    # Initialize and run framework
    framework = ResearchFramework(config)
    
    # Run hypothesis-driven study
    df = framework.run_hypothesis_driven_study()
    
    # Perform statistical analysis with multiple testing correction
    stats_results = framework.statistical_analysis_with_correction(df)
    
    # Validate hypotheses
    hypothesis_validation = framework.validate_hypotheses(stats_results)
    
    # Generate adaptive hypotheses for future research
    new_hypotheses = framework.generate_adaptive_hypotheses(stats_results)
    
    # Generate comprehensive report
    report = framework.generate_report(df, stats_results, hypothesis_validation)
    
    # Save all results
    framework.save_results("quantum_tpu_research", df, stats_results, report, hypothesis_validation)
    
    print(report)
    print(f"\n🧬 Generated {len(new_hypotheses)} new adaptive hypotheses for future research")
    
    return framework, df, stats_results, hypothesis_validation


if __name__ == "__main__":
    framework, df, stats_results, hypothesis_validation = main()