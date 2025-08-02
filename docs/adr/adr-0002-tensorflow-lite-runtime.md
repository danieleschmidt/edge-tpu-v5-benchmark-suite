# ADR-0002: TensorFlow Lite for Model Runtime

## Status

Accepted

## Context

The Edge TPU v5 requires a model runtime that can efficiently execute machine learning models on the specialized hardware. Key considerations include:

- TPU v5 native support and optimization
- Model format compatibility across frameworks
- Performance and latency requirements
- Memory efficiency on edge devices
- Quantization and optimization support

## Decision

We will use TensorFlow Lite as the primary model runtime for Edge TPU v5 inference, with ONNX as an intermediate format for multi-framework support.

## Consequences

### Positive
- **Native TPU Support**: TensorFlow Lite provides first-class Edge TPU delegate support
- **Optimized Performance**: Hardware-specific optimizations for TPU v5 architecture
- **Quantization Support**: Built-in INT8/INT16 quantization with accuracy preservation
- **Multi-Framework**: Can convert models from TensorFlow, PyTorch, ONNX
- **Active Development**: Continuous improvements and TPU v5 feature support

### Negative
- **Framework Lock-in**: Primary dependence on TensorFlow ecosystem
- **Conversion Complexity**: Multi-step conversion process for non-TensorFlow models
- **Limited Ops**: Some operators may not be supported on Edge TPU
- **Debugging Challenges**: Limited visibility into TPU execution details

### Implementation Details
- Use TensorFlow Lite Converter for model optimization
- Implement ONNX → TensorFlow → TensorFlow Lite conversion pipeline
- Provide fallback to CPU execution for unsupported operations
- Include model validation and accuracy verification steps

## Alternatives Considered

### ONNX Runtime
- **Pros**: Framework-agnostic, good performance, growing ecosystem
- **Cons**: Limited Edge TPU support, less mature TPU delegation
- **Decision**: Rejected due to limited TPU v5 optimization support

### PyTorch Mobile
- **Pros**: Native PyTorch integration, good mobile performance
- **Cons**: No Edge TPU delegate support, limited quantization options
- **Decision**: Rejected due to lack of TPU integration

### TensorRT
- **Pros**: Excellent NVIDIA GPU performance, advanced optimizations
- **Cons**: No Edge TPU support, NVIDIA-specific ecosystem
- **Decision**: Rejected due to incompatibility with Edge TPU

### Custom Runtime
- **Pros**: Maximum control, TPU-specific optimizations
- **Cons**: Massive development effort, maintenance burden, ecosystem fragmentation
- **Decision**: Rejected due to development complexity and time constraints

## Related Decisions

- ADR-0003: Modular Plugin Architecture (allows for future runtime alternatives)
- Future ADR on model conversion pipeline architecture
- Future ADR on quantization strategy and accuracy validation