#!/usr/bin/env python3
"""
Basic benchmark example for Edge TPU v5.

This example demonstrates how to:
1. Load a model for TPU v5
2. Run a basic benchmark
3. Analyze results
"""

from edge_tpu_v5_benchmark import TPUv5Benchmark, ModelLoader
from edge_tpu_v5_benchmark.power import PowerProfiler

def main():
    """Run a basic benchmark example."""
    print("🚀 Edge TPU v5 Basic Benchmark Example")
    print("=" * 50)
    
    # Initialize benchmark
    benchmark = TPUv5Benchmark(device_path="/dev/apex_0")
    
    # Load model (you'll need to provide a real model file)
    print("📥 Loading model...")
    try:
        model = ModelLoader.from_onnx(
            "mobilenet_v3.onnx",  # Replace with actual model path
            optimization_level=3,
            target="tpu_v5_edge"
        )
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please provide a valid ONNX model.")
        print("   You can download MobileNetV3 from:")
        print("   https://github.com/onnx/models/tree/main/vision/classification/mobilenet")
        return
    
    # Run benchmark
    print("\n🏃 Running benchmark...")
    results = benchmark.run(
        model=model,
        input_shape=(1, 3, 224, 224),
        iterations=1000,
        warmup=100
    )
    
    # Display results
    print("\n📊 Benchmark Results:")
    print(f"   Throughput: {results.throughput:.1f} FPS")
    print(f"   Latency p99: {results.latency_p99:.2f} ms")
    print(f"   Average Power: {results.avg_power:.2f} W")
    print(f"   Efficiency: {results.inferences_per_watt:.0f} inferences/W")
    print(f"   Total Iterations: {results.total_iterations}")
    print(f"   Duration: {results.duration_seconds:.2f} seconds")
    
    # System information
    print("\n🖥️  System Information:")
    system_info = benchmark.get_system_info()
    for key, value in system_info.items():
        print(f"   {key}: {value}")

def power_profiling_example():
    """Example of power profiling during benchmarks."""
    print("\n⚡ Power Profiling Example")
    print("=" * 30)
    
    # Initialize power profiler
    profiler = PowerProfiler(
        device="/dev/apex_0",
        sample_rate=1000  # 1kHz sampling
    )
    
    # Example model (placeholder)
    model = ModelLoader.from_onnx("mobilenet_v3.onnx")
    
    # Profile power during benchmark
    print("🔋 Starting power measurement...")
    with profiler.measure() as measurement:
        # Simulate running inference
        for i in range(100):
            result = model.run([1, 3, 224, 224])  # Dummy input
            if i % 20 == 0:
                print(f"   Iteration {i}/100")
    
    # Analyze power consumption
    stats = measurement.get_statistics()
    print(f"\n📈 Power Statistics:")
    print(f"   Average Power: {stats.mean:.3f} W")
    print(f"   Peak Power: {stats.max:.3f} W")
    print(f"   Min Power: {stats.min:.3f} W")
    print(f"   Std Deviation: {stats.std:.3f} W")
    print(f"   Total Energy: {stats.total_energy:.3f} J")
    
    # Generate power timeline plot
    profiler.plot_timeline(
        measurement,
        save_path="power_timeline.png",
        show_events=True
    )
    print("   Power timeline saved to: power_timeline.png")

if __name__ == "__main__":
    try:
        main()
        
        # Uncomment to run power profiling example
        # power_profiling_example()
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you have:")
        print("   1. A TPU v5 edge device connected")
        print("   2. Proper permissions for /dev/apex_0")
        print("   3. A valid ONNX model file")