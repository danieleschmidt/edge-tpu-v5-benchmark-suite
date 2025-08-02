"""Tests for models module."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from edge_tpu_v5_benchmark.models import ModelLoader, CompiledTPUModel


class TestModelLoader:
    """Test cases for ModelLoader class."""
    
    @patch('edge_tpu_v5_benchmark.models.Path.exists')
    def test_from_onnx_success(self, mock_exists):
        """Test successful ONNX model loading."""
        mock_exists.return_value = True
        
        model = ModelLoader.from_onnx(
            "test_model.onnx",
            optimization_level=2,
            target="tpu_v5_edge"
        )
        
        assert isinstance(model, CompiledTPUModel)
        assert model.optimization_level == 2
        assert model.target == "tpu_v5_edge"
        assert model.format == "onnx"
    
    @patch('edge_tpu_v5_benchmark.models.Path.exists')
    def test_from_onnx_file_not_found(self, mock_exists):
        """Test ONNX model loading with missing file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            ModelLoader.from_onnx("nonexistent.onnx")
    
    @patch('edge_tpu_v5_benchmark.models.Path.exists')
    def test_from_tflite_success(self, mock_exists):
        """Test successful TFLite model loading."""
        mock_exists.return_value = True
        
        model = ModelLoader.from_tflite("test_model.tflite")
        
        assert isinstance(model, CompiledTPUModel)
        assert model.format == "tflite"
    
    @patch('edge_tpu_v5_benchmark.models.Path.exists')
    def test_from_tflite_file_not_found(self, mock_exists):
        """Test TFLite model loading with missing file."""
        mock_exists.return_value = False
        
        with pytest.raises(FileNotFoundError):
            ModelLoader.from_tflite("nonexistent.tflite")


class TestCompiledTPUModel:
    """Test cases for CompiledTPUModel class."""
    
    def test_init_default_values(self):
        """Test model initialization with default values."""
        model = CompiledTPUModel("test_model.onnx")
        
        assert model.path == "test_model.onnx"
        assert model.optimization_level == 3
        assert model.target == "tpu_v5_edge"
        assert model.format == "onnx"
        assert not model._compiled
    
    def test_init_custom_values(self):
        """Test model initialization with custom values."""
        model = CompiledTPUModel(
            path="custom_model.tflite",
            optimization_level=2,
            target="tpu_v4",
            format="tflite"
        )
        
        assert model.path == "custom_model.tflite"
        assert model.optimization_level == 2
        assert model.target == "tpu_v4"
        assert model.format == "tflite"
    
    def test_run_inference(self):
        """Test model inference execution."""
        model = CompiledTPUModel("test_model.onnx")
        mock_input = Mock()
        
        result = model.run(mock_input)
        
        assert result is not None
        assert "output" in result
        assert model._compiled
    
    def test_get_info(self):
        """Test model information retrieval."""
        model = CompiledTPUModel(
            path="test_model.onnx",
            optimization_level=2,
            target="tpu_v5_edge",
            format="onnx"
        )
        
        info = model.get_info()
        
        assert info["path"] == "test_model.onnx"
        assert info["optimization_level"] == 2
        assert info["target"] == "tpu_v5_edge"
        assert info["format"] == "onnx"
        assert "compiled" in info