"""Sample model generators and mock model data for testing."""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple


def create_mock_onnx_model(output_path: Path) -> None:
    """Create a mock ONNX model file for testing.
    
    Args:
        output_path: Path where the mock model file should be created
    """
    # Create minimal ONNX-like file structure
    mock_onnx_content = b"""
    MOCK ONNX MODEL FILE
    
    This is a mock ONNX model file created for testing purposes.
    In a real implementation, this would contain actual ONNX protobuf data.
    
    Model: MobileNet V3
    Input: [1, 224, 224, 3]
    Output: [1, 1000]
    """
    
    output_path.write_bytes(mock_onnx_content)


def create_mock_tflite_model(output_path: Path) -> None:
    """Create a mock TensorFlow Lite model file for testing.
    
    Args:
        output_path: Path where the mock model file should be created
    """
    # Create minimal TFLite-like file structure
    mock_tflite_content = b"""
    MOCK TFLITE MODEL FILE
    
    This is a mock TensorFlow Lite model file created for testing purposes.
    In a real implementation, this would contain actual TFLite flatbuffer data.
    
    Model: MobileNet V3 (Edge TPU optimized)
    Input: [1, 224, 224, 3]
    Output: [1, 1000]
    Quantization: INT8
    """
    
    output_path.write_bytes(mock_tflite_content)


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get mock model information for testing.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing mock model information
    """
    model_configs = {
        "mobilenet_v3": {
            "name": "MobileNet V3",
            "input_shape": (1, 224, 224, 3),
            "output_shape": (1, 1000),
            "size_mb": 5.2,
            "params_millions": 4.2,
            "flops_millions": 216,
            "expected_latency_ms": 8.5,
            "expected_throughput_fps": 117.6,
            "optimization_level": 2
        },
        "efficientnet_lite": {
            "name": "EfficientNet Lite",
            "input_shape": (1, 224, 224, 3),
            "output_shape": (1, 1000),
            "size_mb": 6.8,
            "params_millions": 5.3,
            "flops_millions": 307,
            "expected_latency_ms": 12.1,
            "expected_throughput_fps": 82.6,
            "optimization_level": 2
        },
        "resnet50": {
            "name": "ResNet-50",
            "input_shape": (1, 224, 224, 3),
            "output_shape": (1, 1000),
            "size_mb": 25.6,
            "params_millions": 25.6,
            "flops_millions": 4090,
            "expected_latency_ms": 15.3,
            "expected_throughput_fps": 65.4,
            "optimization_level": 1
        },
        "yolov8n": {
            "name": "YOLOv8 Nano",
            "input_shape": (1, 640, 640, 3),
            "output_shape": (1, 8400, 84),
            "size_mb": 6.2,
            "params_millions": 3.2,
            "flops_millions": 8150,
            "expected_latency_ms": 22.8,
            "expected_throughput_fps": 43.9,
            "optimization_level": 2
        },
        "bert_base": {
            "name": "BERT Base",
            "input_shape": (1, 128),
            "output_shape": (1, 768),
            "size_mb": 417.0,
            "params_millions": 110.0,
            "flops_millions": 22300,
            "expected_latency_ms": 45.2,
            "expected_throughput_fps": 22.1,
            "optimization_level": 1
        }
    }
    
    return model_configs.get(model_name, {
        "name": model_name,
        "input_shape": (1, 224, 224, 3),
        "output_shape": (1, 1000),
        "size_mb": 10.0,
        "params_millions": 10.0,
        "flops_millions": 1000,
        "expected_latency_ms": 20.0,
        "expected_throughput_fps": 50.0,
        "optimization_level": 1
    })


def generate_mock_inference_output(output_shape: Tuple[int, ...]) -> np.ndarray:
    """Generate mock inference output with realistic values.
    
    Args:
        output_shape: Shape of the expected output tensor
        
    Returns:
        Mock output tensor with realistic values
    """
    if len(output_shape) == 2 and output_shape[1] == 1000:
        # Classification output - softmax-like distribution
        output = np.random.exponential(1.0, output_shape).astype(np.float32)
        output = output / np.sum(output, axis=1, keepdims=True)
        return output
    elif len(output_shape) == 2 and output_shape[1] == 768:
        # BERT-like embedding output
        return np.random.normal(0.0, 0.5, output_shape).astype(np.float32)
    elif len(output_shape) == 3 and output_shape[2] == 84:
        # YOLO-like detection output
        output = np.random.uniform(0.0, 1.0, output_shape).astype(np.float32)
        # Set confidence scores to more realistic values
        output[:, :, 4] = np.random.beta(0.5, 2.0, (output_shape[0], output_shape[1]))
        return output
    else:
        # Generic output
        return np.random.normal(0.0, 1.0, output_shape).astype(np.float32)


def create_test_dataset(
    batch_size: int = 8,
    input_shape: Tuple[int, ...] = (224, 224, 3),
    num_samples: int = 100
) -> np.ndarray:
    """Create a test dataset for benchmarking.
    
    Args:
        batch_size: Number of samples per batch
        input_shape: Shape of individual input samples
        num_samples: Total number of samples to generate
        
    Returns:
        Test dataset as numpy array
    """
    if len(input_shape) == 3:
        # Image data
        full_shape = (num_samples,) + input_shape
        dataset = np.random.randint(0, 255, full_shape, dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Create simple patterns
        for i in range(num_samples):
            # Add some gradients and patterns
            x = np.linspace(0, 1, input_shape[1])
            y = np.linspace(0, 1, input_shape[0])
            xx, yy = np.meshgrid(x, y)
            
            pattern = (np.sin(xx * 4 + i * 0.1) * np.cos(yy * 4 + i * 0.1) + 1) * 127
            pattern = pattern.astype(np.uint8)
            
            for c in range(input_shape[2]):
                dataset[i, :, :, c] = np.clip(
                    dataset[i, :, :, c].astype(np.float32) * 0.7 + pattern * 0.3,
                    0, 255
                ).astype(np.uint8)
                
    elif len(input_shape) == 1:
        # Text/token data
        full_shape = (num_samples,) + input_shape
        dataset = np.random.randint(0, 30522, full_shape, dtype=np.int32)  # BERT vocab size
        
    else:
        # Generic data
        full_shape = (num_samples,) + input_shape
        dataset = np.random.normal(0.0, 1.0, full_shape).astype(np.float32)
        
    return dataset


def get_model_benchmark_expectations(model_name: str) -> Dict[str, Dict[str, float]]:
    """Get expected benchmark results for different models.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary containing expected benchmark ranges
    """
    expectations = {
        "mobilenet_v3": {
            "latency_ms": {"min": 7.0, "max": 10.0, "target": 8.5},
            "throughput_fps": {"min": 100.0, "max": 140.0, "target": 117.6},
            "power_w": {"min": 2.0, "max": 3.5, "target": 2.7},
            "accuracy_top1": {"min": 0.75, "max": 0.78, "target": 0.764}
        },
        "efficientnet_lite": {
            "latency_ms": {"min": 10.0, "max": 15.0, "target": 12.1},
            "throughput_fps": {"min": 65.0, "max": 100.0, "target": 82.6},
            "power_w": {"min": 2.2, "max": 3.8, "target": 3.0},
            "accuracy_top1": {"min": 0.77, "max": 0.80, "target": 0.785}
        },
        "resnet50": {
            "latency_ms": {"min": 12.0, "max": 18.0, "target": 15.3},
            "throughput_fps": {"min": 55.0, "max": 80.0, "target": 65.4},
            "power_w": {"min": 2.5, "max": 4.0, "target": 3.2},
            "accuracy_top1": {"min": 0.75, "max": 0.78, "target": 0.764}
        }
    }
    
    return expectations.get(model_name, {
        "latency_ms": {"min": 10.0, "max": 30.0, "target": 20.0},
        "throughput_fps": {"min": 30.0, "max": 100.0, "target": 50.0},
        "power_w": {"min": 2.0, "max": 4.0, "target": 3.0},
        "accuracy_top1": {"min": 0.70, "max": 0.80, "target": 0.75}
    })


# Pre-defined test configurations
TEST_MODEL_CONFIGS = {
    "quick_test": {
        "models": ["mobilenet_v3"],
        "iterations": 10,
        "warmup_iterations": 2,
        "timeout_seconds": 30
    },
    "comprehensive_test": {
        "models": ["mobilenet_v3", "efficientnet_lite", "resnet50"],
        "iterations": 100,
        "warmup_iterations": 10,
        "timeout_seconds": 300
    },
    "performance_test": {
        "models": ["mobilenet_v3", "efficientnet_lite"],
        "iterations": 1000,
        "warmup_iterations": 100,
        "timeout_seconds": 600
    }
}