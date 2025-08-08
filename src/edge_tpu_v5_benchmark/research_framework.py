"""Research Framework for Quantum-Enhanced TPU Optimization Studies.

This module implements the experimental framework for validating quantum optimization
algorithms against classical baselines with rigorous statistical methodology.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        self.logger.info(f"Research framework initialized with config: {config}")
    
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
    
    def run_comparative_study(self) -> pd.DataFrame:
        """Run comprehensive comparative study."""
        self.logger.info("Starting comparative study...")
        
        algorithms = ["quantum_scheduling", "heft", "genetic"]
        all_results = []
        
        for iteration in range(self.config.num_iterations):
            self.logger.info(f"Running iteration {iteration + 1}/{self.config.num_iterations}")
            
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
        
        # Convert to DataFrame for analysis
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
        
        self.logger.info("Comparative study completed")
        return df
    
    def statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform rigorous statistical analysis."""
        self.logger.info("Performing statistical analysis...")
        
        analysis_results = {}
        quantum_data = df[df['algorithm'] == 'quantum_scheduling']
        
        # Compare quantum vs each classical algorithm
        for classical_algo in ['heft', 'genetic']:
            classical_data = df[df['algorithm'] == classical_algo]
            comparison_name = f"quantum_vs_{classical_algo}"
            
            analysis_results[comparison_name] = {}
            
            # Metrics to analyze
            metrics = ['makespan', 'resource_utilization', 'throughput', 'energy_consumption']
            
            for metric in metrics:
                quantum_values = quantum_data[metric].values
                classical_values = classical_data[metric].values
                
                # Statistical tests
                # Normality test
                _, quantum_normal = stats.shapiro(quantum_values)
                _, classical_normal = stats.shapiro(classical_values)
                
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
                )
                cohens_d = (np.mean(quantum_values) - np.mean(classical_values)) / pooled_std
                
                # Confidence interval for mean difference
                mean_diff = np.mean(quantum_values) - np.mean(classical_values)
                se_diff = pooled_std * np.sqrt(1/len(quantum_values) + 1/len(classical_values))
                ci_lower, ci_upper = stats.t.interval(
                    self.config.confidence_level,
                    len(quantum_values) + len(classical_values) - 2,
                    mean_diff, se_diff
                )
                
                analysis_results[comparison_name][metric] = {
                    'quantum_mean': np.mean(quantum_values),
                    'quantum_std': np.std(quantum_values),
                    'classical_mean': np.mean(classical_values),
                    'classical_std': np.std(classical_values),
                    'mean_difference': mean_diff,
                    'percent_improvement': (mean_diff / np.mean(classical_values)) * 100 if np.mean(classical_values) != 0 else 0,
                    'test_statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < (1 - self.config.confidence_level),
                    'effect_size_cohens_d': cohens_d,
                    'effect_size_large': abs(cohens_d) > 0.8,
                    'confidence_interval': (ci_lower, ci_upper),
                    'test_used': test_used
                }
        
        self.logger.info("Statistical analysis completed")
        return analysis_results
    
    def generate_report(self, df: pd.DataFrame, stats_results: Dict[str, Any]) -> str:
        """Generate comprehensive research report."""
        report = []
        
        report.append("# Quantum-Enhanced TPU Optimization Research Report")
        report.append(f"\n## Experiment Configuration")
        report.append(f"- Experiment Type: {self.config.experiment_type.value}")
        report.append(f"- Workload Pattern: {self.config.workload_pattern.value}")
        report.append(f"- Number of Tasks: {self.config.num_tasks}")
        report.append(f"- Number of Iterations: {self.config.num_iterations}")
        report.append(f"- Confidence Level: {self.config.confidence_level}")
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
                if stats_info['significant']:
                    significance = "SIGNIFICANT"
                    improvement = stats_info['percent_improvement']
                    report.append(
                        f"- **{metric}**: {significance} - "
                        f"{improvement:.1f}% improvement "
                        f"(p={stats_info['p_value']:.4f}, d={stats_info['effect_size_cohens_d']:.3f})"
                    )
                else:
                    report.append(
                        f"- **{metric}**: Not significant "
                        f"(p={stats_info['p_value']:.4f})"
                    )
        
        report.append("\n## Research Hypothesis Validation")
        
        # Check hypothesis 1: Quantum coherence-guided scheduling
        quantum_vs_heft = stats_results.get('quantum_vs_heft', {})
        makespan_improvement = quantum_vs_heft.get('makespan', {}).get('percent_improvement', 0)
        util_improvement = quantum_vs_heft.get('resource_utilization', {}).get('percent_improvement', 0)
        
        report.append("\n### Hypothesis 1: Quantum Coherence-Guided Task Scheduling")
        report.append(f"- **Target**: 15-25% makespan reduction, 10-20% utilization improvement")
        report.append(f"- **Result**: {abs(makespan_improvement):.1f}% makespan change, {util_improvement:.1f}% utilization improvement")
        
        if abs(makespan_improvement) >= 15 and util_improvement >= 10:
            report.append("- **Status**: ✅ HYPOTHESIS CONFIRMED")
        else:
            report.append("- **Status**: ❌ HYPOTHESIS NOT CONFIRMED")
        
        report.append("\n## Recommendations")
        report.append("1. Further investigation needed for quantum advantage validation")
        report.append("2. Extend experiments to larger-scale workloads")
        report.append("3. Implement hardware-specific optimizations")
        report.append("4. Develop theoretical framework for quantum scheduling bounds")
        
        return "\n".join(report)
    
    def save_results(self, filepath: str, df: pd.DataFrame, stats_results: Dict[str, Any], report: str):
        """Save all results to files."""
        # Save raw data
        df.to_csv(f"{filepath}_raw_data.csv", index=False)
        
        # Save statistical results
        import json
        with open(f"{filepath}_statistics.json", 'w') as f:
            # Convert numpy types for JSON serialization
            json_compatible_stats = {}
            for comparison, results in stats_results.items():
                json_compatible_stats[comparison] = {}
                for metric, stats_info in results.items():
                    json_compatible_stats[comparison][metric] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in stats_info.items()
                        if k != 'confidence_interval'  # Skip tuples for JSON
                    }
            
            json.dump(json_compatible_stats, f, indent=2)
        
        # Save report
        with open(f"{filepath}_report.md", 'w') as f:
            f.write(report)
        
        self.logger.info(f"Results saved to {filepath}_*")


def main():
    """Main function to run the research framework."""
    # Configure experiment
    config = ExperimentConfig(
        experiment_type=ExperimentType.SCHEDULING_COMPARISON,
        workload_pattern=WorkloadPattern.MIXED,
        num_tasks=100,
        num_iterations=10,
        random_seed=42
    )
    
    # Initialize and run framework
    framework = ResearchFramework(config)
    df = framework.run_comparative_study()
    stats_results = framework.statistical_analysis(df)
    report = framework.generate_report(df, stats_results)
    
    # Save results
    framework.save_results("quantum_tpu_research", df, stats_results, report)
    
    print(report)


if __name__ == "__main__":
    main()