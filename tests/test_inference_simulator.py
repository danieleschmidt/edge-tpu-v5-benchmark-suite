"""Tests for InferenceSimulator."""
import pytest
from edge_tpu_v5_bench import ModelBenchmark, InferenceSimulator


@pytest.fixture
def model():
    return ModelBenchmark("test_model", [
        {"type": "linear", "in_features": 1000, "out_features": 1000},
    ])


@pytest.fixture
def sim():
    return InferenceSimulator(tops=4.0, memory_bandwidth_gbps=8.0)


def test_simulate_returns_dict(sim, model):
    result = sim.simulate(model)
    assert isinstance(result, dict)
    assert "latency_ms" in result
    assert "throughput_qps" in result
    assert "memory_util" in result


def test_latency_positive(sim, model):
    result = sim.simulate(model)
    assert result["latency_ms"] > 0


def test_throughput_positive(sim, model):
    result = sim.simulate(model)
    assert result["throughput_qps"] > 0


def test_memory_util_range(sim, model):
    result = sim.simulate(model)
    assert 0.0 <= result["memory_util"] <= 1.0


def test_int4_faster_than_int8(sim, model):
    r_int8 = sim.simulate(model, quantization="INT8")
    r_int4 = sim.simulate(model, quantization="INT4")
    assert r_int4["latency_ms"] < r_int8["latency_ms"]


def test_int4_higher_throughput(sim, model):
    r_int8 = sim.simulate(model, quantization="INT8")
    r_int4 = sim.simulate(model, quantization="INT4")
    assert r_int4["throughput_qps"] > r_int8["throughput_qps"]


def test_batch_size_scales_latency(sim, model):
    r1 = sim.simulate(model, batch_size=1)
    r4 = sim.simulate(model, batch_size=4)
    assert abs(r4["latency_ms"] - 4 * r1["latency_ms"]) < 1e-6


def test_repr():
    s = InferenceSimulator()
    assert "InferenceSimulator" in repr(s)


def test_latency_formula():
    """Verify: latency = ops / (tops * 1e12) * 1000."""
    m = ModelBenchmark("m", [{"type": "linear", "in_features": 100, "out_features": 100}])
    ops = m.op_count()  # 10000
    sim = InferenceSimulator(tops=4.0)
    result = sim.simulate(m, quantization="INT8")
    expected_ms = (ops / (4.0 * 1e12)) * 1000.0
    assert abs(result["latency_ms"] - expected_ms) < 1e-12
