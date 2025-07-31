---
name: Performance Issue
about: Report performance problems or benchmarking issues
title: '[PERFORMANCE] '
labels: ['performance', 'needs-triage']
assignees: ['danielschmidt']
---

## Performance Issue Description
<!-- Describe the performance problem you're experiencing -->

## Environment Details
- **OS**: [e.g., Ubuntu 22.04]
- **Python Version**: [e.g., 3.11.2]
- **Package Version**: [e.g., 0.1.0]
- **TPU Hardware**: [e.g., TPU v5 Edge]
- **System Memory**: [e.g., 16GB]
- **CPU**: [e.g., Intel i7-12700K]

## Benchmark Configuration
<!-- Provide details about the benchmark configuration -->
```python
benchmark_config = {
    "model": "model_name",
    "batch_size": 1,
    "iterations": 1000,
    "input_shape": (1, 3, 224, 224),
    # Add other relevant configuration
}
```

## Performance Metrics
<!-- Provide current performance measurements -->
| Metric | Current | Expected | Difference |
|--------|---------|----------|------------|
| Throughput (FPS) | | | |
| Latency (ms) | | | |
| Memory Usage (MB) | | | |
| Power Consumption (W) | | | |

## Reproduction Steps
1. Run benchmark with: `command here`
2. Observe performance metrics
3. Compare with expected values

## System Resource Usage
<!-- If available, provide system resource usage during benchmark -->
- **CPU Usage**: X%
- **Memory Usage**: X GB
- **GPU/TPU Usage**: X%
- **Disk I/O**: X MB/s

## Performance Comparison
<!-- Compare with other models, versions, or expected performance -->
**Expected Performance**: Based on [source/documentation/previous version]
**Actual Performance**: Current measurements
**Performance Gap**: X% slower than expected

## Profiling Information
<!-- If you've done any profiling, include relevant information -->
```
Add profiling output here if available
```

## Hardware Monitoring
<!-- If available, include hardware monitoring data -->
- **Temperature**: X°C
- **Clock Speeds**: X MHz
- **Power Draw**: X W
- **Thermal Throttling**: Yes/No

## Additional Context
<!-- Any other information that might help diagnose the performance issue -->

## Potential Causes
<!-- If you have theories about what might be causing the issue -->
- [ ] Model compilation issue
- [ ] Memory bottleneck  
- [ ] I/O bottleneck
- [ ] Thermal throttling
- [ ] Driver/runtime issue
- [ ] Configuration problem

---

### Checklist
- [ ] I have provided detailed benchmark configuration
- [ ] I have included performance measurements
- [ ] I have compared against expected performance
- [ ] I have checked for system resource constraints
- [ ] I have searched existing issues to avoid duplicates