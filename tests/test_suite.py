"""Tests for BenchmarkSuite."""
import json
import os
import tempfile
import pytest
from edge_tpu_v5_bench import ModelBenchmark, InferenceSimulator, PowerEstimator, BenchmarkSuite


@pytest.fixture
def suite_with_models():
    suite = BenchmarkSuite()
    suite.add_model(ModelBenchmark("model_a", [
        {"type": "linear", "in_features": 64, "out_features": 32},
    ]))
    suite.add_model(ModelBenchmark("model_b", [
        {"type": "conv", "in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ]))
    return suite


@pytest.fixture
def sim():
    return InferenceSimulator()


@pytest.fixture
def est():
    return PowerEstimator()


def test_add_model_increments_len():
    suite = BenchmarkSuite()
    assert len(suite) == 0
    suite.add_model(ModelBenchmark("m", []))
    assert len(suite) == 1


def test_run_all_returns_list(suite_with_models, sim, est):
    results = suite_with_models.run_all(sim, est)
    assert isinstance(results, list)
    assert len(results) == 2


def test_run_all_result_keys(suite_with_models, sim, est):
    results = suite_with_models.run_all(sim, est)
    required_keys = {
        "model_name", "param_count", "op_count", "memory_mb_int8",
        "latency_ms", "throughput_qps", "memory_util",
        "power_w", "tops_utilized", "tops_per_watt", "within_budget",
    }
    for r in results:
        assert required_keys.issubset(r.keys())


def test_run_all_model_names(suite_with_models, sim, est):
    results = suite_with_models.run_all(sim, est)
    names = [r["model_name"] for r in results]
    assert "model_a" in names
    assert "model_b" in names


def test_save_csv(suite_with_models, sim, est):
    results = suite_with_models.run_all(sim, est)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        suite_with_models.save_csv(results, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_save_json(suite_with_models, sim, est):
    results = suite_with_models.run_all(sim, est)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        suite_with_models.save_json(results, path)
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
    finally:
        os.unlink(path)


def test_empty_suite_run_all(sim, est):
    suite = BenchmarkSuite()
    results = suite.run_all(sim, est)
    assert results == []


def test_repr():
    suite = BenchmarkSuite()
    assert "BenchmarkSuite" in repr(suite)
