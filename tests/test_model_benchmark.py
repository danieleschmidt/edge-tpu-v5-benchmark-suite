"""Tests for ModelBenchmark."""
import pytest
from edge_tpu_v5_bench import ModelBenchmark


# --- Fixtures ---

@pytest.fixture
def simple_conv_model():
    return ModelBenchmark("simple_conv", [
        {"type": "conv", "in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ])


@pytest.fixture
def linear_model():
    return ModelBenchmark("linear", [
        {"type": "linear", "in_features": 128, "out_features": 64},
    ])


# --- param_count tests ---

def test_conv_param_count(simple_conv_model):
    # 3 * 16 * 9 (kernel) + 16 (bias) = 432 + 16 = 448
    assert simple_conv_model.param_count() == 448


def test_linear_param_count(linear_model):
    # 128 * 64 + 64 = 8192 + 64 = 8256
    assert linear_model.param_count() == 8256


def test_depthwise_param_count():
    m = ModelBenchmark("dw", [
        {"type": "depthwise_conv", "channels": 32, "kernel_size": 3},
    ])
    # 32 * 9 + 32 = 288 + 32 = 320
    assert m.param_count() == 320


def test_batchnorm_param_count():
    m = ModelBenchmark("bn", [
        {"type": "batchnorm", "num_features": 64},
    ])
    # 2 * 64 = 128
    assert m.param_count() == 128


def test_multi_layer_param_count():
    m = ModelBenchmark("multi", [
        {"type": "linear", "in_features": 10, "out_features": 5},  # 10*5+5=55
        {"type": "linear", "in_features": 5, "out_features": 2},   # 5*2+2=12
    ])
    assert m.param_count() == 67


def test_unknown_layer_zero_params():
    m = ModelBenchmark("unknown", [
        {"type": "relu"},
        {"type": "dropout"},
    ])
    assert m.param_count() == 0


# --- op_count tests ---

def test_conv_op_count_default_output():
    m = ModelBenchmark("conv", [
        {"type": "conv", "in_channels": 3, "out_channels": 16, "kernel_size": 3},
    ])
    # Default output_size [1,1] → 3*9 * 16 * 1 * 1 = 432
    assert m.op_count() == 432


def test_conv_op_count_with_output_size():
    m = ModelBenchmark("conv_spatial", [
        {"type": "conv", "in_channels": 3, "out_channels": 16, "kernel_size": 3, "output_size": [4, 4]},
    ])
    # 3*9 * 16 * 4 * 4 = 27 * 16 * 16 = 6912
    assert m.op_count() == 6912


def test_linear_op_count(linear_model):
    # 128 * 64 = 8192
    assert linear_model.op_count() == 8192


# --- memory_mb tests ---

def test_memory_mb_int8(simple_conv_model):
    params = simple_conv_model.param_count()  # 448
    expected = params / (1024 ** 2)
    assert abs(simple_conv_model.memory_mb(dtype_bytes=1) - expected) < 1e-9


def test_memory_mb_fp32(simple_conv_model):
    params = simple_conv_model.param_count()
    expected = params * 4 / (1024 ** 2)
    assert abs(simple_conv_model.memory_mb(dtype_bytes=4) - expected) < 1e-9


# --- repr ---

def test_repr(simple_conv_model):
    r = repr(simple_conv_model)
    assert "ModelBenchmark" in r
    assert "simple_conv" in r
