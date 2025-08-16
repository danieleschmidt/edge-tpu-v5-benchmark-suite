"""Adaptive Optimization Engine for TPU v5 Benchmark Suite

This module implements autonomous optimization strategies that adapt
to workload patterns and system performance characteristics.
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json
from pathlib import Path
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from enum import Enum
import statistics

from .security import SecurityContext, InputValidator
from .config import get_config


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    LATENCY_FOCUSED = "latency_focused"
    THROUGHPUT_FOCUSED = "throughput_focused"
    POWER_EFFICIENT = "power_efficient"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class OptimizationProfile:
    """Profile containing optimization parameters."""
    name: str
    strategy: OptimizationStrategy
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    thread_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    memory_pool_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    cache_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    prefetch_depths: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    optimization_weights: Dict[str, float] = field(default_factory=lambda: {
        "latency": 0.4,
        "throughput": 0.3,
        "power": 0.2,
        "memory": 0.1
    })


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    latency_ms: float
    throughput_ops_sec: float
    power_watts: float
    memory_mb: float
    cpu_utilization: float
    gpu_utilization: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite performance score."""
        normalized_latency = 1.0 / (1.0 + self.latency_ms / 100.0)
        normalized_throughput = min(self.throughput_ops_sec / 1000.0, 1.0)
        normalized_power = 1.0 / (1.0 + self.power_watts / 10.0)
        normalized_memory = 1.0 / (1.0 + self.memory_mb / 1000.0)
        
        return (
            weights.get("latency", 0.25) * normalized_latency +
            weights.get("throughput", 0.25) * normalized_throughput +
            weights.get("power", 0.25) * normalized_power +
            weights.get("memory", 0.25) * normalized_memory
        )


class AdaptiveParameterTuner:
    """Bayesian optimization-inspired parameter tuning."""
    
    def __init__(self):
        self.parameter_history: Dict[str, List[Tuple[Any, float]]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    def suggest_next_parameter(self, param_name: str, candidates: List[Any]) -> Any:
        """Suggest next parameter value using acquisition function."""
        if not self.parameter_history[param_name]:
            return np.random.choice(candidates)
        
        history = self.parameter_history[param_name]
        scores = [score for _, score in history]
        
        if len(scores) < 3:
            return np.random.choice(candidates)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-6
        
        acquisition_scores = []
        for candidate in candidates:
            tested_scores = [score for param, score in history if param == candidate]
            if tested_scores:
                exploitation = np.mean(tested_scores)
                exploration = std_score / np.sqrt(len(tested_scores))
            else:
                exploitation = mean_score
                exploration = std_score
            
            acquisition = exploitation + 2.0 * exploration
            acquisition_scores.append(acquisition)
        
        best_idx = np.argmax(acquisition_scores)
        return candidates[best_idx]
    
    def record_result(self, param_name: str, value: Any, score: float):
        """Record parameter evaluation result."""
        self.parameter_history[param_name].append((value, score))
        if len(self.parameter_history[param_name]) > 100:
            self.parameter_history[param_name] = self.parameter_history[param_name][-50:]


class WorkloadPatternAnalyzer:
    """Analyzes workload patterns for adaptive optimization."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.pattern_cache = {}
        self.logger = logging.getLogger(__name__)
        
    def analyze_patterns(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze current workload patterns."""
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) < 10:
            return {"pattern": "initializing", "confidence": 0.0}
        
        recent_metrics = list(self.metrics_history)[-10:]
        
        latencies = [m.latency_ms for m in recent_metrics]
        throughputs = [m.throughput_ops_sec for m in recent_metrics]
        powers = [m.power_watts for m in recent_metrics]
        
        patterns = {
            "latency_trend": self._analyze_trend(latencies),
            "throughput_trend": self._analyze_trend(throughputs),
            "power_trend": self._analyze_trend(powers),
            "variability": {
                "latency_cv": np.std(latencies) / (np.mean(latencies) + 1e-6),
                "throughput_cv": np.std(throughputs) / (np.mean(throughputs) + 1e-6),
                "power_cv": np.std(powers) / (np.mean(powers) + 1e-6),
            },
            "workload_type": self._classify_workload(recent_metrics)
        }
        
        return patterns
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in metric values."""
        if len(values) < 5:
            return "stable"
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        relative_slope = slope / (np.mean(values) + 1e-6)
        
        if relative_slope > 0.05:
            return "increasing"
        elif relative_slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def _classify_workload(self, metrics: List[PerformanceMetrics]) -> str:
        """Classify workload type based on patterns."""
        latencies = [m.latency_ms for m in metrics]
        throughputs = [m.throughput_ops_sec for m in metrics]
        
        avg_latency = np.mean(latencies)
        avg_throughput = np.mean(throughputs)
        
        if avg_latency < 5.0 and avg_throughput > 500:
            return "high_frequency_low_latency"
        elif avg_latency > 20.0 and avg_throughput < 100:
            return "complex_computation"
        elif np.std(latencies) / (avg_latency + 1e-6) > 0.3:
            return "variable_workload"
        else:
            return "steady_state"


class AdaptiveOptimizer:
    """Main adaptive optimization engine."""
    
    def __init__(self, 
                 security_context: Optional[SecurityContext] = None,
                 optimization_profile: Optional[OptimizationProfile] = None):
        self.security_context = security_context or SecurityContext()
        self.profile = optimization_profile or self._create_default_profile()
        self.parameter_tuner = AdaptiveParameterTuner()
        self.pattern_analyzer = WorkloadPatternAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        self.current_config = self._initialize_config()
        self.optimization_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        
        self._running = False
        self._optimization_thread = None
        
    def _create_default_profile(self) -> OptimizationProfile:
        """Create default optimization profile."""
        return OptimizationProfile(
            name="adaptive_default",
            strategy=OptimizationStrategy.ADAPTIVE
        )
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize optimization configuration."""
        return {
            "batch_size": 4,
            "thread_count": 2,
            "memory_pool_size": 128,
            "cache_size": 64,
            "prefetch_depth": 2,
            "enable_async": True,
            "enable_caching": True,
            "enable_prefetch": True
        }
    
    def start_optimization(self):
        """Start autonomous optimization process."""
        with self.lock:
            if self._running:
                return
            
            self._running = True
            self._optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self._optimization_thread.start()
            self.logger.info("Adaptive optimizer started")
    
    def stop_optimization(self):
        """Stop optimization process."""
        with self.lock:
            self._running = False
            if self._optimization_thread:
                self._optimization_thread.join(timeout=5.0)
            self.logger.info("Adaptive optimizer stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self._running:
            try:
                self._perform_optimization_step()
                time.sleep(10.0)  # Optimization interval
            except Exception as e:
                self.logger.error(f"Optimization step failed: {e}")
                time.sleep(30.0)  # Back off on error
    
    def _perform_optimization_step(self):
        """Perform single optimization step."""
        patterns = self.pattern_analyzer.analyze_patterns(
            self._collect_current_metrics()
        )
        
        if patterns["pattern"] == "initializing":
            return
        
        workload_type = patterns["workload_type"]
        self.logger.debug(f"Detected workload type: {workload_type}")
        
        optimization_strategy = self._select_strategy(patterns)
        
        if optimization_strategy != self.profile.strategy:
            self._adapt_optimization_profile(optimization_strategy, patterns)
        
        self._tune_parameters(patterns)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        return PerformanceMetrics(
            latency_ms=np.random.uniform(1.0, 10.0),  # Mock data
            throughput_ops_sec=np.random.uniform(100, 1000),  # Mock data
            power_watts=np.random.uniform(1.0, 5.0),  # Mock data
            memory_mb=memory_info.used / (1024 * 1024),
            cpu_utilization=cpu_percent
        )
    
    def _select_strategy(self, patterns: Dict[str, Any]) -> OptimizationStrategy:
        """Select optimization strategy based on patterns."""
        workload_type = patterns["workload_type"]
        variability = patterns["variability"]
        
        if workload_type == "high_frequency_low_latency":
            return OptimizationStrategy.LATENCY_FOCUSED
        elif workload_type == "complex_computation":
            return OptimizationStrategy.THROUGHPUT_FOCUSED
        elif variability["power_cv"] > 0.3:
            return OptimizationStrategy.POWER_EFFICIENT
        else:
            return OptimizationStrategy.ADAPTIVE
    
    def _adapt_optimization_profile(self, 
                                   strategy: OptimizationStrategy, 
                                   patterns: Dict[str, Any]):
        """Adapt optimization profile based on strategy."""
        self.profile.strategy = strategy
        
        if strategy == OptimizationStrategy.LATENCY_FOCUSED:
            self.profile.optimization_weights = {
                "latency": 0.6, "throughput": 0.3, "power": 0.05, "memory": 0.05
            }
        elif strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            self.profile.optimization_weights = {
                "latency": 0.2, "throughput": 0.6, "power": 0.1, "memory": 0.1
            }
        elif strategy == OptimizationStrategy.POWER_EFFICIENT:
            self.profile.optimization_weights = {
                "latency": 0.25, "throughput": 0.25, "power": 0.4, "memory": 0.1
            }
        else:  # ADAPTIVE or BALANCED
            self.profile.optimization_weights = {
                "latency": 0.3, "throughput": 0.3, "power": 0.2, "memory": 0.2
            }
        
        self.logger.info(f"Adapted to {strategy.value} strategy")
    
    def _tune_parameters(self, patterns: Dict[str, Any]):
        """Tune optimization parameters."""
        current_metrics = self._collect_current_metrics()
        current_score = current_metrics.composite_score(
            self.profile.optimization_weights
        )
        
        parameter_suggestions = {}
        
        for param_name, candidates in [
            ("batch_size", self.profile.batch_sizes),
            ("thread_count", self.profile.thread_counts),
            ("cache_size", self.profile.cache_sizes),
            ("prefetch_depth", self.profile.prefetch_depths)
        ]:
            suggested = self.parameter_tuner.suggest_next_parameter(
                param_name, candidates
            )
            parameter_suggestions[param_name] = suggested
        
        for param_name, value in parameter_suggestions.items():
            if param_name in self.current_config:
                old_value = self.current_config[param_name]
                self.current_config[param_name] = value
                
                test_metrics = self._evaluate_configuration()
                test_score = test_metrics.composite_score(
                    self.profile.optimization_weights
                )
                
                if test_score > current_score:
                    self.parameter_tuner.record_result(param_name, value, test_score)
                    self.logger.info(f"Improved {param_name}: {old_value} -> {value}")
                else:
                    self.current_config[param_name] = old_value
                    self.parameter_tuner.record_result(param_name, value, test_score)
    
    def _evaluate_configuration(self) -> PerformanceMetrics:
        """Evaluate current configuration performance."""
        return self._collect_current_metrics()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current optimization configuration."""
        with self.lock:
            return self.current_config.copy()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        with self.lock:
            return list(self.optimization_history)
    
    def force_reoptimization(self):
        """Force immediate reoptimization."""
        if self._running:
            self._perform_optimization_step()
    
    def export_optimization_data(self, filepath: Path):
        """Export optimization data for analysis."""
        data = {
            "current_config": self.get_current_config(),
            "optimization_profile": {
                "name": self.profile.name,
                "strategy": self.profile.strategy.value,
                "weights": self.profile.optimization_weights
            },
            "parameter_history": dict(self.parameter_tuner.parameter_history),
            "optimization_history": self.get_optimization_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Optimization data exported to {filepath}")


class QuantumInspiredOptimizer(AdaptiveOptimizer):
    """Quantum-inspired optimization using superposition and entanglement concepts."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantum_state = np.random.random(8) + 1j * np.random.random(8)
        self.quantum_state /= np.linalg.norm(self.quantum_state)
        
    def _quantum_parameter_selection(self, candidates: List[Any]) -> Any:
        """Use quantum-inspired selection for parameters."""
        n_candidates = len(candidates)
        if n_candidates == 0:
            return None
        
        quantum_probs = np.abs(self.quantum_state[:n_candidates]) ** 2
        quantum_probs /= np.sum(quantum_probs)
        
        selected_idx = np.random.choice(n_candidates, p=quantum_probs)
        
        self._evolve_quantum_state()
        
        return candidates[selected_idx]
    
    def _evolve_quantum_state(self):
        """Evolve quantum state based on performance feedback."""
        rotation_angle = 0.1 * np.random.randn()
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
        
        for i in range(0, len(self.quantum_state), 2):
            if i + 1 < len(self.quantum_state):
                state_pair = np.array([
                    self.quantum_state[i],
                    self.quantum_state[i + 1]
                ])
                evolved_pair = rotation_matrix @ state_pair
                self.quantum_state[i] = evolved_pair[0]
                self.quantum_state[i + 1] = evolved_pair[1]
        
        self.quantum_state /= np.linalg.norm(self.quantum_state)


def create_adaptive_optimizer(strategy: str = "adaptive",
                            security_context: Optional[SecurityContext] = None) -> AdaptiveOptimizer:
    """Factory function to create adaptive optimizer."""
    strategy_enum = OptimizationStrategy(strategy)
    
    profile = OptimizationProfile(
        name=f"auto_{strategy}",
        strategy=strategy_enum
    )
    
    if strategy == "quantum":
        return QuantumInspiredOptimizer(security_context, profile)
    else:
        return AdaptiveOptimizer(security_context, profile)