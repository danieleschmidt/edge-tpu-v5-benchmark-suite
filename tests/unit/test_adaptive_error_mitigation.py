"""Tests for adaptive quantum error mitigation framework."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
    AdaptiveErrorMitigationFramework,
    MLWorkloadProfiler,
    QuantumErrorPatternClassifier,
    AdaptiveMitigationSelector,
    ErrorMitigationType,
    MLWorkloadType,
    WorkloadCharacteristics,
    ErrorProfile,
    MitigationStrategy
)
from edge_tpu_v5_benchmark.quantum_computing_research import (
    QuantumCircuit,
    QuantumResearchFramework
)
from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
    SimpleQuantumCircuit
)


class TestMLWorkloadProfiler:
    """Test ML workload profiling functionality."""
    
    def setup_method(self):
        self.profiler = MLWorkloadProfiler()
        
    def test_profile_inference_workload(self):
        """Test profiling of inference workloads."""
        circuit = QuantumCircuit(n_qubits=4, name="test_circuit")
        circuit.add_gate("hadamard", [0])
        circuit.add_gate("cnot", [0, 1])
        
        ml_context = {
            "workload_type": "inference",
            "fidelity_threshold": 0.95,
            "error_budget": 0.01,
            "quantum_advantage_target": 2.0,
            "tpu_utilization": {"compute": 0.8, "memory": 0.6}
        }
        
        characteristics = self.profiler.profile_workload(circuit, ml_context)
        
        assert characteristics.workload_type == MLWorkloadType.INFERENCE
        assert characteristics.fidelity_threshold == 0.95
        assert characteristics.error_budget == 0.01
        assert characteristics.quantum_advantage_target == 2.0
        assert characteristics.circuit_depth == 2
        assert characteristics.gate_count == 2
        assert characteristics.two_qubit_gate_count == 1
        
    def test_profile_training_workload(self):
        """Test profiling of training workloads."""
        circuit = QuantumCircuit(n_qubits=6, name="training_circuit")
        for i in range(5):
            circuit.add_gate("rotation_y", [i], angle=0.5)
            if i < 4:
                circuit.add_gate("cnot", [i, i+1])
        
        ml_context = {
            "workload_type": "training",
            "fidelity_threshold": 0.9,
            "error_budget": 0.05
        }
        
        characteristics = self.profiler.profile_workload(circuit, ml_context)
        
        assert characteristics.workload_type == MLWorkloadType.TRAINING
        assert characteristics.fidelity_threshold == 0.9
        assert characteristics.error_budget == 0.05
        assert characteristics.circuit_depth == 9  # 5 single qubit + 4 two qubit gates
        assert characteristics.two_qubit_gate_count == 4

    def test_invalid_workload_type_fallback(self):
        """Test fallback for invalid workload types."""
        circuit = QuantumCircuit(n_qubits=2, name="test")
        ml_context = {"workload_type": "invalid_type"}
        
        characteristics = self.profiler.profile_workload(circuit, ml_context)
        
        # Should fallback to INFERENCE
        assert characteristics.workload_type == MLWorkloadType.INFERENCE


class TestQuantumErrorPatternClassifier:
    """Test quantum error pattern classification."""
    
    def setup_method(self):
        self.classifier = QuantumErrorPatternClassifier()
        
    def test_classify_simple_circuit_errors(self):
        """Test error classification for simple circuits."""
        circuit = QuantumCircuit(n_qubits=3, name="simple")
        circuit.add_gate("hadamard", [0])
        circuit.add_gate("cnot", [0, 1])
        circuit.add_gate("rotation_z", [1], angle=0.5)
        
        workload = WorkloadCharacteristics(
            workload_type=MLWorkloadType.INFERENCE,
            circuit_depth=3,
            gate_count=3,
            two_qubit_gate_count=1,
            coherence_time_required=30.0,
            fidelity_threshold=0.95,
            error_budget=0.01,
            tpu_utilization_pattern={"compute": 0.5, "memory": 0.3}
        )
        
        error_profile = self.classifier.classify_errors(circuit, workload)
        
        assert isinstance(error_profile, ErrorProfile)
        assert len(error_profile.dominant_error_types) >= 2  # At least depolarizing and measurement
        assert "depolarizing_error" in error_profile.dominant_error_types
        assert "measurement_error" in error_profile.dominant_error_types
        assert all(rate > 0 for rate in error_profile.error_rates.values())
        assert len(error_profile.predicted_mitigation_effectiveness) > 0
        
    def test_deep_circuit_error_classification(self):
        """Test error classification for deep circuits prone to decoherence."""
        circuit = QuantumCircuit(n_qubits=4, name="deep")
        # Create a deep circuit
        for layer in range(20):  # Deep circuit with many layers
            for qubit in range(4):
                circuit.add_gate("rotation_y", [qubit], angle=0.1)
            for qubit in range(3):
                circuit.add_gate("cnot", [qubit, qubit+1])
        
        workload = WorkloadCharacteristics(
            workload_type=MLWorkloadType.TRAINING,
            circuit_depth=circuit.depth(),
            gate_count=len(circuit.gates),
            two_qubit_gate_count=60,  # 3 CNOTs * 20 layers
            coherence_time_required=200.0,
            fidelity_threshold=0.85,
            error_budget=0.1
        )
        
        error_profile = self.classifier.classify_errors(circuit, workload)
        
        # Deep circuits should be flagged for decoherence errors
        assert "decoherence_error" in error_profile.dominant_error_types
        # High two-qubit gate ratio should flag crosstalk
        assert "crosstalk_error" in error_profile.dominant_error_types
        
        # Decoherence mitigation should be highly effective
        assert error_profile.predicted_mitigation_effectiveness[ErrorMitigationType.ADAPTIVE_DYNAMICAL_DECOUPLING] > 0.8


class TestAdaptiveMitigationSelector:
    """Test adaptive mitigation strategy selection."""
    
    def setup_method(self):
        self.selector = AdaptiveMitigationSelector()
        
    def test_select_strategy_for_decoherence_errors(self):
        """Test strategy selection for decoherence-dominated errors."""
        error_profile = ErrorProfile(
            dominant_error_types=["decoherence_error", "depolarizing_error"],
            error_rates={"decoherence_error": 0.02, "depolarizing_error": 0.005},
            correlation_patterns={"depth_decoherence_correlation": 0.8},
            temporal_variations=[1.0, 1.1, 1.2, 1.0],
            circuit_specific_errors={"consecutive_gate_error": 0.01},
            predicted_mitigation_effectiveness={
                ErrorMitigationType.ADAPTIVE_DYNAMICAL_DECOUPLING: 0.85,
                ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION: 0.3,
                ErrorMitigationType.ML_ASSISTED_ERROR_CORRECTION: 0.9
            }
        )
        
        workload = WorkloadCharacteristics(
            workload_type=MLWorkloadType.INFERENCE,
            circuit_depth=50,
            gate_count=100,
            two_qubit_gate_count=25,
            coherence_time_required=100.0,
            fidelity_threshold=0.9,
            error_budget=0.02
        )
        
        strategy = self.selector.select_strategy(error_profile, workload)
        
        # Should select ML-assisted correction as primary (highest effectiveness)
        assert strategy.primary_method == ErrorMitigationType.ML_ASSISTED_ERROR_CORRECTION
        assert strategy.expected_improvement > 0
        assert strategy.expected_overhead > 1.0  # Should have overhead
        assert 0.0 <= strategy.confidence_score <= 1.0
        
    def test_select_strategy_for_crosstalk_errors(self):
        """Test strategy selection for crosstalk-dominated errors."""
        error_profile = ErrorProfile(
            dominant_error_types=["crosstalk_error", "depolarizing_error"],
            error_rates={"crosstalk_error": 0.01, "depolarizing_error": 0.003},
            correlation_patterns={"connectivity_crosstalk_correlation": 0.9},
            temporal_variations=[1.0] * 8,
            circuit_specific_errors={},
            predicted_mitigation_effectiveness={
                ErrorMitigationType.PROBABILISTIC_ERROR_CANCELLATION: 0.9,
                ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION: 0.4,
                ErrorMitigationType.ML_ASSISTED_ERROR_CORRECTION: 0.9
            }
        )
        
        workload = WorkloadCharacteristics(
            workload_type=MLWorkloadType.TRAINING,
            circuit_depth=20,
            gate_count=50,
            two_qubit_gate_count=30,
            coherence_time_required=50.0,
            fidelity_threshold=0.85,
            error_budget=0.05
        )
        
        strategy = self.selector.select_strategy(error_profile, workload)
        
        # Should select one of the highly effective methods
        assert strategy.primary_method in [
            ErrorMitigationType.PROBABILISTIC_ERROR_CANCELLATION,
            ErrorMitigationType.ML_ASSISTED_ERROR_CORRECTION
        ]


class TestAdaptiveErrorMitigationFramework:
    """Test the main adaptive error mitigation framework."""
    
    def setup_method(self):
        self.framework = AdaptiveErrorMitigationFramework()
        
    def test_optimize_error_mitigation_basic(self):
        """Test basic error mitigation optimization."""
        circuit = QuantumCircuit(n_qubits=3, name="test")
        circuit.add_gate("hadamard", [0])
        circuit.add_gate("cnot", [0, 1])
        circuit.add_gate("rotation_z", [2], angle=0.5)
        
        ml_context = {
            "workload_type": "inference",
            "fidelity_threshold": 0.95,
            "error_budget": 0.01,
            "tpu_utilization": {"compute": 0.7}
        }
        
        mitigated_circuit, strategy = self.framework.optimize_error_mitigation(
            circuit, ml_context
        )
        
        assert isinstance(mitigated_circuit, (QuantumCircuit, SimpleQuantumCircuit))
        assert isinstance(strategy, MitigationStrategy)
        assert mitigated_circuit.n_qubits == circuit.n_qubits
        # Mitigated circuit should generally have same or more gates
        assert len(mitigated_circuit.gates) >= len(circuit.gates)
        
    def test_error_mitigation_with_fallback(self):
        """Test error mitigation with fallback on errors."""
        circuit = QuantumCircuit(n_qubits=2, name="test")
        circuit.add_gate("hadamard", [0])
        
        # Create invalid ML context to trigger fallback
        ml_context = None
        
        # Should not raise exception, should return fallback strategy
        mitigated_circuit, strategy = self.framework.optimize_error_mitigation(
            circuit, ml_context
        )
        
        assert isinstance(mitigated_circuit, (QuantumCircuit, SimpleQuantumCircuit))
        assert isinstance(strategy, MitigationStrategy)
        
    def test_performance_feedback_update(self):
        """Test performance feedback learning mechanism."""
        strategy = MitigationStrategy(
            primary_method=ErrorMitigationType.ZERO_NOISE_EXTRAPOLATION,
            secondary_methods=[],
            parameters={"noise_factors": [1.0, 1.1, 1.2]},
            expected_overhead=3.0,
            expected_improvement=0.3,
            confidence_score=0.7
        )
        
        # Update with positive feedback
        self.framework.update_performance_feedback(strategy, 0.35, 2.8)
        
        assert len(self.framework.mitigation_results) == 1
        result = self.framework.mitigation_results[0]
        assert result["actual_improvement"] == 0.35
        assert result["success"] is True  # 0.35 >= 0.3 * 0.8
        
        # Update with negative feedback
        self.framework.update_performance_feedback(strategy, 0.1, 4.0)
        
        assert len(self.framework.mitigation_results) == 2
        result = self.framework.mitigation_results[1]
        assert result["actual_improvement"] == 0.1
        assert result["success"] is False  # 0.1 < 0.3 * 0.8


class TestQuantumResearchFrameworkIntegration:
    """Test integration with quantum research framework."""
    
    def test_research_framework_with_error_mitigation(self):
        """Test quantum research framework with error mitigation enabled."""
        framework = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=True)
        
        assert framework.enable_error_mitigation is True
        assert framework.error_mitigation_framework is not None
        assert framework.workload_profiler is not None
        assert isinstance(framework.mitigation_performance_history, list)
        
    def test_research_framework_without_error_mitigation(self):
        """Test quantum research framework with error mitigation disabled."""
        framework = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=False)
        
        assert framework.enable_error_mitigation is False
        assert framework.error_mitigation_framework is None
        assert framework.workload_profiler is None

    @pytest.mark.asyncio
    async def test_run_experiment_with_mitigation(self):
        """Test running experiments with error mitigation."""
        framework = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=True)
        
        # Create a simple experiment
        from edge_tpu_v5_benchmark.quantum_computing_research import QuantumAlgorithm
        
        experiment = framework.design_experiment(
            "test_grover",
            QuantumAlgorithm.GROVER_SEARCH,
            {"n_qubits": 3, "marked_items": [2]},
            "Test Grover search with error mitigation"
        )
        
        ml_context = {
            "workload_type": "inference",
            "fidelity_threshold": 0.9,
            "error_budget": 0.02
        }
        
        # Run with mitigation
        result = await framework.run_experiment_with_mitigation(experiment, ml_context)
        
        assert result.experiment_name == "test_grover"
        assert result.algorithm == QuantumAlgorithm.GROVER_SEARCH
        assert result.execution_time > 0
        assert 0 <= result.fidelity <= 1
        assert len(framework.mitigation_performance_history) == 1
        
        # Check mitigation performance tracking
        perf_record = framework.mitigation_performance_history[0]
        assert perf_record["experiment_name"] == "test_grover"
        assert perf_record["algorithm"] == "grover_search"
        assert "mitigation_strategy" in perf_record

    @pytest.mark.asyncio 
    async def test_run_experiment_without_mitigation_fallback(self):
        """Test fallback to standard execution when mitigation is disabled."""
        framework = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=False)
        
        from edge_tpu_v5_benchmark.quantum_computing_research import QuantumAlgorithm
        
        experiment = framework.design_experiment(
            "test_qaoa", 
            QuantumAlgorithm.QAOA,
            {"problem_graph": {(0, 1): 1.0}, "p_layers": 1}
        )
        
        ml_context = {"workload_type": "training"}
        
        # Should run without mitigation
        result = await framework.run_experiment_with_mitigation(experiment, ml_context)
        
        assert result.experiment_name == "test_qaoa"
        assert len(framework.mitigation_performance_history) == 0  # No mitigation tracking


if __name__ == "__main__":
    pytest.main([__file__])