"""Integration tests for model conversion pipeline."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from edge_tpu_v5_benchmark.models import ModelConverter, ModelLoader


class TestModelConversionIntegration:
    """Integration tests for model conversion workflow."""
    
    @pytest.fixture
    def converter(self):
        """Create a model converter instance."""
        return ModelConverter()
    
    @pytest.fixture
    def model_loader(self):
        """Create a model loader instance."""
        return ModelLoader()
    
    @pytest.mark.integration
    def test_onnx_to_tflite_conversion(self, converter, sample_onnx_model_path, temp_dir):
        """Test ONNX to TensorFlow Lite conversion."""
        output_path = temp_dir / "converted_model.tflite"
        
        # Mock the actual conversion process
        with patch.object(converter, '_convert_onnx_to_tensorflow') as mock_tf_convert:
            with patch.object(converter, '_convert_tensorflow_to_tflite') as mock_tflite_convert:
                mock_tf_convert.return_value = temp_dir / "temp_tf_model"
                mock_tflite_convert.return_value = output_path
                
                result_path = converter.convert_onnx_to_tflite(
                    onnx_path=sample_onnx_model_path,
                    output_path=output_path,
                    optimization_level=2
                )
                
                assert result_path == output_path
                mock_tf_convert.assert_called_once_with(
                    sample_onnx_model_path, 
                    temp_dir / "temp_tf_model"
                )
                mock_tflite_convert.assert_called_once()
    
    @pytest.mark.integration
    def test_model_loading_and_validation(self, model_loader, sample_tflite_model_path):
        """Test model loading and validation."""
        model = model_loader.load_model(sample_tflite_model_path)
        
        assert model is not None
        assert model.model_path == str(sample_tflite_model_path)
        assert hasattr(model, 'input_shape')
        assert hasattr(model, 'output_shape')
        assert callable(model.run_inference)
    
    @pytest.mark.integration
    def test_model_optimization_pipeline(self, converter, sample_onnx_model_path, temp_dir):
        """Test complete model optimization pipeline."""
        output_path = temp_dir / "optimized_model.tflite"
        
        with patch.object(converter, 'optimize_for_tpu_v5') as mock_optimize:
            mock_optimize.return_value = output_path
            
            result = converter.convert_and_optimize(
                input_path=sample_onnx_model_path,
                output_path=output_path,
                target_device="tpu_v5_edge",
                optimization_profile="balanced"
            )
            
            assert result == output_path
            mock_optimize.assert_called_once()
    
    @pytest.mark.integration
    def test_conversion_with_quantization(self, converter, sample_onnx_model_path, temp_dir):
        """Test model conversion with quantization."""
        output_path = temp_dir / "quantized_model.tflite"
        
        # Mock calibration dataset
        mock_calibration_data = [
            Mock() for _ in range(100)  # 100 calibration samples
        ]
        
        with patch.object(converter, '_apply_quantization') as mock_quantize:
            mock_quantize.return_value = output_path
            
            result = converter.convert_with_quantization(
                input_path=sample_onnx_model_path,
                output_path=output_path,
                quantization_method="post_training",
                calibration_data=mock_calibration_data
            )
            
            assert result == output_path
            mock_quantize.assert_called_once_with(
                model_path=sample_onnx_model_path,
                output_path=output_path,
                method="post_training",
                calibration_data=mock_calibration_data
            )
    
    @pytest.mark.integration
    def test_conversion_accuracy_validation(self, converter, model_loader, 
                                          sample_onnx_model_path, temp_dir):
        """Test conversion accuracy validation."""
        tflite_path = temp_dir / "converted_model.tflite"
        
        # Mock original and converted models
        with patch.object(model_loader, 'load_onnx_model') as mock_load_onnx:
            with patch.object(model_loader, 'load_tflite_model') as mock_load_tflite:
                mock_original = Mock()
                mock_converted = Mock()
                mock_load_onnx.return_value = mock_original
                mock_load_tflite.return_value = mock_converted
                
                # Mock inference outputs for comparison
                import numpy as np
                original_output = np.random.random((1, 1000))
                converted_output = original_output + np.random.normal(0, 0.01, (1, 1000))  # Small difference
                
                mock_original.run_inference.return_value = original_output
                mock_converted.run_inference.return_value = converted_output
                
                with patch.object(converter, 'convert_onnx_to_tflite') as mock_convert:
                    mock_convert.return_value = tflite_path
                    
                    validation_result = converter.convert_and_validate(
                        input_path=sample_onnx_model_path,
                        output_path=tflite_path,
                        test_samples=10,
                        tolerance=0.05
                    )
                    
                    assert validation_result['conversion_successful'] is True
                    assert validation_result['accuracy_preserved'] is True
                    assert 'mean_absolute_error' in validation_result
                    assert 'max_difference' in validation_result
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_model_conversion(self, converter, temp_dir):
        """Test batch conversion of multiple models."""
        # Create multiple mock ONNX models
        model_paths = []
        for i in range(3):
            model_path = temp_dir / f"model_{i}.onnx"
            with open(model_path, 'wb') as f:
                f.write(b"mock_onnx_data")  # Mock ONNX file
            model_paths.append(model_path)
        
        output_dir = temp_dir / "converted_models"
        output_dir.mkdir()
        
        with patch.object(converter, 'convert_onnx_to_tflite') as mock_convert:
            def mock_convert_func(input_path, output_path, **kwargs):
                # Create mock output file
                with open(output_path, 'wb') as f:
                    f.write(b"mock_tflite_data")
                return output_path
            
            mock_convert.side_effect = mock_convert_func
            
            results = converter.batch_convert(
                input_paths=model_paths,
                output_dir=output_dir,
                conversion_type="onnx_to_tflite"
            )
            
            assert len(results) == 3
            assert all(result['status'] == 'success' for result in results)
            assert mock_convert.call_count == 3
    
    @pytest.mark.integration
    def test_conversion_error_handling(self, converter, temp_dir):
        """Test error handling in model conversion."""
        invalid_model_path = temp_dir / "invalid_model.onnx"
        output_path = temp_dir / "output.tflite"
        
        # Create invalid ONNX file
        with open(invalid_model_path, 'wb') as f:
            f.write(b"invalid_onnx_data")
        
        with patch.object(converter, '_convert_onnx_to_tensorflow') as mock_convert:
            mock_convert.side_effect = ValueError("Invalid ONNX model")
            
            with pytest.raises(ValueError, match="Invalid ONNX model"):
                converter.convert_onnx_to_tflite(
                    onnx_path=invalid_model_path,
                    output_path=output_path
                )
    
    @pytest.mark.integration
    def test_model_size_optimization(self, converter, sample_onnx_model_path, temp_dir):
        """Test model size optimization during conversion."""
        output_path = temp_dir / "optimized_model.tflite"
        
        with patch.object(converter, '_optimize_model_size') as mock_optimize:
            mock_optimize.return_value = {
                'original_size': 10_000_000,  # 10 MB
                'optimized_size': 2_500_000,  # 2.5 MB
                'compression_ratio': 0.25
            }
            
            result = converter.convert_with_size_optimization(
                input_path=sample_onnx_model_path,
                output_path=output_path,
                target_size_mb=5,
                optimization_techniques=['pruning', 'quantization']
            )
            
            assert result['compression_ratio'] < 1.0
            assert result['optimized_size'] < result['original_size']
            mock_optimize.assert_called_once()
    
    @pytest.mark.integration
    def test_conversion_metadata_preservation(self, converter, sample_onnx_model_path, temp_dir):
        """Test preservation of model metadata during conversion."""
        output_path = temp_dir / "converted_with_metadata.tflite"
        
        original_metadata = {
            'model_name': 'test_model',
            'version': '1.0.0',
            'author': 'test_author',
            'description': 'Test model for benchmarking',
            'input_shape': [1, 3, 224, 224],
            'output_shape': [1, 1000]
        }
        
        with patch.object(converter, '_extract_metadata') as mock_extract:
            with patch.object(converter, '_embed_metadata') as mock_embed:
                mock_extract.return_value = original_metadata
                
                converter.convert_with_metadata_preservation(
                    input_path=sample_onnx_model_path,
                    output_path=output_path,
                    preserve_metadata=True
                )
                
                mock_extract.assert_called_once_with(sample_onnx_model_path)
                mock_embed.assert_called_once_with(output_path, original_metadata)