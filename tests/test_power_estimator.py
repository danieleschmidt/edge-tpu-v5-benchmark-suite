"""Tests for PowerEstimator."""
import pytest
from edge_tpu_v5_bench import ModelBenchmark, PowerEstimator


@pytest.fixture
def model():
    return ModelBenchmark("power_test", [
        {"type": "linear", "in_features": 512, "out_features": 256},
    ])


@pytest.fixture
def estimator():
    return PowerEstimator(power_budget_w=2.0, efficiency_tops_per_w=50.0)


def test_estimate_returns_dict(estimator, model):
    result = estimator.estimate(model)
    assert isinstance(result, dict)
    assert "power_w" in result
    assert "tops_utilized" in result
    assert "tops_per_watt" in result
    assert "within_budget" in result


def test_power_positive(estimator, model):
    result = estimator.estimate(model)
    assert result["power_w"] >= 0


def test_tops_utilized_positive(estimator, model):
    result = estimator.estimate(model)
    assert result["tops_utilized"] >= 0


def test_tops_per_watt_equals_efficiency(estimator, model):
    result = estimator.estimate(model)
    # For a model, tops_per_watt = tops_utilized / power_w = efficiency
    assert abs(result["tops_per_watt"] - 50.0) < 1e-6


def test_within_budget_small_model():
    """A tiny model should always be within 2W budget."""
    m = ModelBenchmark("tiny", [{"type": "linear", "in_features": 10, "out_features": 5}])
    est = PowerEstimator(power_budget_w=2.0, efficiency_tops_per_w=50.0)
    result = est.estimate(m)
    assert result["within_budget"] is True


def test_within_budget_huge_model():
    """A model with trillions of ops may exceed the power budget."""
    # We need ops > power_budget * efficiency * 1e12
    # power_budget=2.0, efficiency=50 → need >100 TOPS → >100e12 MACs
    # Use a large linear layer: 15M * 10M = 150e12 MACs → 150 TOPS → 3.0W > 2.0W
    m = ModelBenchmark("huge", [
        {"type": "linear", "in_features": 15_000_000, "out_features": 10_000_000},
    ])
    est = PowerEstimator(power_budget_w=2.0, efficiency_tops_per_w=50.0)
    result = est.estimate(m)
    assert result["within_budget"] is False


def test_repr():
    e = PowerEstimator()
    assert "PowerEstimator" in repr(e)
