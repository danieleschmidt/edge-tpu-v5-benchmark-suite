#!/usr/bin/env python3
"""Simple validation script for adaptive error mitigation framework."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock numpy and scikit-learn for basic validation
class MockNumPy:
    @staticmethod
    def zeros(shape, dtype=float):
        if isinstance(shape, (list, tuple)):
            return [[0] * shape[1] for _ in range(shape[0])] if len(shape) == 2 else [0] * shape[0]
        return [0] * shape
    
    @staticmethod
    def array(data):
        return data
    
    @staticmethod
    def mean(data):
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def std(data):
        if not data:
            return 0
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return variance ** 0.5
    
    @staticmethod
    def percentile(data, percentile):
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    @staticmethod
    def prod(data):
        result = 1
        for item in data:
            result *= item
        return result
    
    @staticmethod
    def var(data):
        if not data:
            return 0
        mean_val = sum(data) / len(data)
        return sum((x - mean_val) ** 2 for x in data) / len(data)
    
    @staticmethod
    def max(data):
        return max(data) if data else 0
    
    @staticmethod
    def min(data):
        return min(data) if data else 0
    
    class random:
        @staticmethod
        def uniform(low, high, size=None):
            import random
            if size is None:
                return random.uniform(low, high)
            return [random.uniform(low, high) for _ in range(size)]
        
        @staticmethod
        def normal(mean, std, size=None):
            import random
            if size is None:
                return random.gauss(mean, std)
            return [random.gauss(mean, std) for _ in range(size)]
        
        @staticmethod
        def choice(options):
            import random
            return random.choice(options)
        
        @staticmethod
        def random():
            import random
            return random.random()

class MockSklearn:
    class ensemble:
        class IsolationForest:
            def __init__(self, **kwargs):
                self.contamination = kwargs.get('contamination', 0.1)
            
            def fit(self, data):
                return self
            
            def predict(self, data):
                return [1] * len(data)  # All normal
    
    class cluster:
        class KMeans:
            def __init__(self, **kwargs):
                self.n_clusters = kwargs.get('n_clusters', 5)
            
            def fit(self, data):
                return self
            
            def predict(self, data):
                import random
                return [random.randint(0, self.n_clusters - 1) for _ in data]

# Mock the modules
sys.modules['numpy'] = MockNumPy()
sys.modules['sklearn'] = MockSklearn()
sys.modules['sklearn.ensemble'] = MockSklearn.ensemble()
sys.modules['sklearn.cluster'] = MockSklearn.cluster()

# Now test imports and basic functionality
def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
            AdaptiveErrorMitigationFramework,
            MLWorkloadProfiler,
            ErrorMitigationType,
            MLWorkloadType
        )
        print("✅ Adaptive error mitigation imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from edge_tpu_v5_benchmark.quantum_computing_research import (
            QuantumCircuit,
            QuantumResearchFramework
        )
        print("✅ Quantum research framework imports successful")
    except Exception as e:
        print(f"❌ Quantum framework import error: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the error mitigation framework."""
    print("\nTesting basic functionality...")
    
    try:
        from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
            AdaptiveErrorMitigationFramework,
            MLWorkloadProfiler,
            MLWorkloadType
        )
        from edge_tpu_v5_benchmark.quantum_computing_research import QuantumCircuit
        
        # Create error mitigation framework
        framework = AdaptiveErrorMitigationFramework()
        print("✅ Error mitigation framework created")
        
        # Create a simple quantum circuit
        circuit = QuantumCircuit(n_qubits=3, name="test_circuit")
        circuit.add_gate("hadamard", [0])
        circuit.add_gate("cnot", [0, 1])
        circuit.add_gate("rotation_z", [2], angle=0.5)
        print("✅ Quantum circuit created")
        
        # Create ML context
        ml_context = {
            "workload_type": "inference",
            "fidelity_threshold": 0.95,
            "error_budget": 0.01,
            "quantum_advantage_target": 2.0
        }
        
        # Test error mitigation optimization
        mitigated_circuit, strategy = framework.optimize_error_mitigation(circuit, ml_context)
        print("✅ Error mitigation optimization completed")
        
        # Validate results
        assert mitigated_circuit.n_qubits == circuit.n_qubits
        assert strategy.primary_method is not None
        assert strategy.expected_improvement >= 0
        print("✅ Error mitigation results validated")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workload_profiling():
    """Test ML workload profiling."""
    print("\nTesting workload profiling...")
    
    try:
        from edge_tpu_v5_benchmark.adaptive_quantum_error_mitigation import (
            MLWorkloadProfiler,
            MLWorkloadType
        )
        from edge_tpu_v5_benchmark.quantum_computing_research import QuantumCircuit
        
        profiler = MLWorkloadProfiler()
        
        # Test inference workload
        circuit = QuantumCircuit(n_qubits=4, name="inference_test")
        circuit.add_gate("hadamard", [0])
        circuit.add_gate("cnot", [0, 1])
        
        ml_context = {
            "workload_type": "inference",
            "fidelity_threshold": 0.9,
            "error_budget": 0.02
        }
        
        characteristics = profiler.profile_workload(circuit, ml_context)
        
        assert characteristics.workload_type == MLWorkloadType.INFERENCE
        assert characteristics.fidelity_threshold == 0.9
        assert characteristics.error_budget == 0.02
        assert characteristics.circuit_depth == 2
        assert characteristics.two_qubit_gate_count == 1
        
        print("✅ Workload profiling successful")
        return True
        
    except Exception as e:
        print(f"❌ Workload profiling test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration with quantum research framework."""
    print("\nTesting framework integration...")
    
    try:
        from edge_tpu_v5_benchmark.quantum_computing_research import QuantumResearchFramework
        
        # Test with error mitigation enabled
        framework = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=True)
        assert framework.enable_error_mitigation is True
        assert framework.error_mitigation_framework is not None
        assert framework.workload_profiler is not None
        print("✅ Error mitigation integration successful")
        
        # Test with error mitigation disabled
        framework_disabled = QuantumResearchFramework(max_qubits=4, enable_error_mitigation=False)
        assert framework_disabled.enable_error_mitigation is False
        assert framework_disabled.error_mitigation_framework is None
        print("✅ Disabled error mitigation configuration successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("🧪 Validating Adaptive Quantum Error Mitigation Framework")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_workload_profiling,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Adaptive error mitigation framework is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)