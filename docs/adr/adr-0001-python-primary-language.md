# ADR-0001: Python as Primary Language

## Status

Accepted

## Context

The Edge TPU v5 Benchmark Suite requires a programming language that balances performance, ecosystem compatibility, and developer productivity. Key considerations include:

- TPU runtime libraries are primarily available in Python and C++
- Machine learning ecosystem is predominantly Python-based
- Need for rapid prototyping and community contributions
- Performance requirements for benchmark accuracy
- Integration with existing ML workflows and tools

## Decision

We will use Python 3.8+ as the primary programming language for the Edge TPU v5 Benchmark Suite.

## Consequences

### Positive
- **Ecosystem Integration**: Seamless integration with TensorFlow, ONNX, PyTorch, and other ML frameworks
- **Community Adoption**: Lower barrier to entry for ML researchers and practitioners
- **Rich Libraries**: Access to extensive scientific computing libraries (NumPy, SciPy, Pandas)
- **Rapid Development**: Faster iteration cycles for feature development and bug fixes
- **TPU Support**: First-class support for Edge TPU Python APIs

### Negative
- **Performance Overhead**: Interpreter overhead compared to compiled languages
- **GIL Limitations**: Global Interpreter Lock may limit multi-threading performance
- **Deployment Size**: Larger deployment artifacts compared to native binaries

### Mitigation Strategies
- Use NumPy and native extensions for performance-critical operations
- Leverage async/await for I/O-bound operations to work around GIL
- Implement optional C++ extensions for ultra-low-latency measurements
- Use PyInstaller or similar tools for deployment optimization

## Alternatives Considered

### C++
- **Pros**: Maximum performance, direct TPU driver access, minimal overhead
- **Cons**: Higher development complexity, smaller contributor pool, longer development cycles
- **Decision**: Rejected due to developer productivity concerns and ecosystem fragmentation

### Rust
- **Pros**: Memory safety, excellent performance, growing ML ecosystem
- **Cons**: Limited TPU ecosystem support, smaller community, learning curve
- **Decision**: Rejected due to limited TPU runtime support and ecosystem maturity

### Go
- **Pros**: Good performance, simple deployment, built-in concurrency
- **Cons**: Limited ML ecosystem, no direct TPU support, smaller community
- **Decision**: Rejected due to lack of ML ecosystem integration

## Related Decisions

- Future ADR on C++ extension strategy for performance-critical components
- Future ADR on deployment and distribution strategy