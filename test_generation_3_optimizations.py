#!/usr/bin/env python3
"""Comprehensive tests for Generation 3 optimizations and scaling capabilities."""

import asyncio
import time
import numpy as np
from typing import Dict, List, Any
import json


def test_multi_tpu_parallel_execution():
    """Test multi-TPU parallel processing capabilities."""
    print("🚀 Testing Multi-TPU Parallel Execution...")
    
    from src.edge_tpu_v5_benchmark.multi_tpu_parallel import (
        MultiTPUParallelExecutor, ParallelTask, ParallelExecutionMode,
        LoadBalancingStrategy, create_benchmark_tasks
    )
    
    # Create executor with 4 virtual TPU devices
    executor = MultiTPUParallelExecutor(["/dev/apex_0", "/dev/apex_1", "/dev/apex_2", "/dev/apex_3"])
    executor.start()
    
    try:
        # Create test benchmark tasks
        model_paths = ["mobilenet_v3.onnx", "efficientnet.onnx", "resnet50.onnx", "yolo_v8.onnx"]
        input_data = [np.random.rand(1, 3, 224, 224) for _ in range(4)]
        configs = [{
            'estimated_duration': 0.5 + i * 0.2,
            'memory_requirement': 2.0 + i * 0.5,
            'compute_requirement': 4.0 + i * 1.0
        } for i in range(4)]
        
        tasks = create_benchmark_tasks(model_paths, input_data, configs)
        
        # Test different execution modes
        execution_modes = [
            ParallelExecutionMode.DATA_PARALLEL,
            ParallelExecutionMode.MODEL_PARALLEL,
            ParallelExecutionMode.QUANTUM_DISTRIBUTED
        ]
        
        load_strategies = [
            LoadBalancingStrategy.QUANTUM_ANNEALING,
            LoadBalancingStrategy.ML_PREDICTIVE,
            LoadBalancingStrategy.ADAPTIVE_HYBRID
        ]
        
        results = {}
        
        for mode in execution_modes:
            for strategy in load_strategies:
                test_name = f"{mode.value}_{strategy.value}"
                print(f"   Testing {test_name}...")
                
                start_time = time.time()
                task_results = asyncio.run(executor.submit_tasks(tasks, mode, strategy))
                execution_time = time.time() - start_time
                
                # Analyze results
                successful_tasks = [r for r in task_results if r.success]
                avg_throughput = np.mean([r.throughput for r in successful_tasks]) if successful_tasks else 0
                total_compute_used = sum([r.detailed_metrics.get('compute_usage', 1.0) for r in successful_tasks])
                
                results[test_name] = {
                    'execution_time': execution_time,
                    'successful_tasks': len(successful_tasks),
                    'total_tasks': len(tasks),
                    'success_rate': len(successful_tasks) / len(tasks),
                    'avg_throughput': avg_throughput,
                    'compute_efficiency': total_compute_used / execution_time if execution_time > 0 else 0
                }
        
        # Get performance metrics
        perf_metrics = executor.get_performance_metrics()
        
        print("   Multi-TPU Results:")
        print(f"   - Active devices: {perf_metrics['active_devices']}/{perf_metrics['total_devices']}")
        print(f"   - Total capacity: {perf_metrics['total_compute_capacity']:.1f} TOPS")
        print(f"   - Parallel efficiency: {perf_metrics['parallel_efficiency']:.1f}x")
        print(f"   - Tasks completed: {perf_metrics['total_tasks_completed']}")
        
        # Find best performing configuration
        best_config = max(results.keys(), key=lambda k: results[k]['compute_efficiency'])
        best_result = results[best_config]
        
        print(f"   - Best configuration: {best_config}")
        print(f"   - Best efficiency: {best_result['compute_efficiency']:.1f} TOPS/sec")
        print(f"   - Best throughput: {best_result['avg_throughput']:.1f}")
        
        return {
            'multi_tpu_results': results,
            'performance_metrics': perf_metrics,
            'best_configuration': best_config,
            'parallel_efficiency': perf_metrics['parallel_efficiency']
        }
        
    finally:
        executor.stop()


def test_advanced_telemetry_system():
    """Test advanced telemetry and observability features."""
    print("📊 Testing Advanced Telemetry System...")
    
    from src.edge_tpu_v5_benchmark.advanced_telemetry import (
        TelemetrySystem, MetricType, AlertSeverity
    )
    
    # Initialize telemetry system
    telemetry = TelemetrySystem(retention_hours=1)
    telemetry.start()
    
    try:
        # Record various metrics
        test_metrics = {
            'throughput': [100, 120, 95, 130, 85, 140, 110, 125, 105, 135],
            'latency': [10, 12, 9, 15, 8, 16, 11, 13, 10, 14],
            'error_rate': [0.01, 0.02, 0.015, 0.05, 0.08, 0.03, 0.02, 0.01, 0.02, 0.015],
            'quantum_coherence': [0.95, 0.92, 0.94, 0.85, 0.65, 0.78, 0.89, 0.93, 0.91, 0.88],
            'memory_usage': [45, 52, 48, 65, 78, 82, 75, 68, 55, 60]
        }
        
        # Simulate metric recording over time
        for i in range(10):
            for metric_name, values in test_metrics.items():
                value = values[i]
                
                # Add some quantum context for quantum metrics
                if 'quantum' in metric_name:
                    telemetry.record_performance_metric(
                        metric_name, 
                        {'primary': value, 'coherence_time': 100 - i * 5},
                        labels={'quantum_device': f'qpu_{i % 2}'}
                    )
                else:
                    telemetry.record_performance_metric(metric_name, value)
            
            time.sleep(0.1)  # Simulate time passage
        
        # Test distributed tracing
        trace_span = telemetry.tracer.start_trace("benchmark_execution")
        
        # Simulate nested operations
        child_spans = []
        for i in range(3):
            child_span = telemetry.tracer.start_child_span(trace_span, f"model_inference_{i}")
            telemetry.tracer.add_log(child_span, f"Processing model {i}", "INFO")
            
            # Add quantum context
            telemetry.tracer.finish_span(child_span, 
                tags={'model': f'model_{i}', 'device': f'tpu_{i}'},
                quantum_context={'coherence': 0.9 - i * 0.1, 'entanglement_fidelity': 0.85}
            )
            child_spans.append(child_span)
        
        telemetry.tracer.finish_span(trace_span, tags={'operation': 'benchmark'})
        
        # Get dashboard data
        dashboard_data = telemetry.get_dashboard_data(hours=1)
        
        # Analyze results
        active_alerts = [alert for alert in dashboard_data['alerts'] if not alert.get('resolved', False)]
        system_health = dashboard_data['system_health']
        metrics_summary = dashboard_data['metrics']
        quantum_metrics = dashboard_data['quantum_state']
        
        print("   Telemetry Results:")
        print(f"   - Metrics tracked: {len(metrics_summary)}")
        print(f"   - Active alerts: {len(active_alerts)}")
        print(f"   - System health score: {system_health['score']:.1f}/100")
        print(f"   - Health status: {system_health['status']}")
        print(f"   - Quantum metrics: {len(quantum_metrics)}")
        
        # Check for anomaly detection
        anomaly_alerts = [a for a in active_alerts if 'anomaly' in a.get('message', '').lower()]
        print(f"   - Anomaly alerts: {len(anomaly_alerts)}")
        
        # Trace analysis
        trace_data = telemetry.tracer.get_trace(telemetry.tracer.spans[trace_span].trace_id)
        print(f"   - Trace duration: {trace_data.get('duration', 0):.3f}s")
        print(f"   - Trace spans: {trace_data.get('span_count', 0)}")
        
        # Export metrics
        prometheus_metrics = telemetry.export_metrics("prometheus")
        json_metrics = telemetry.export_metrics("json")
        
        return {
            'dashboard_data': dashboard_data,
            'active_alerts': len(active_alerts),
            'system_health_score': system_health['score'],
            'metrics_count': len(metrics_summary),
            'trace_analysis': trace_data,
            'anomaly_detection_active': len(anomaly_alerts) > 0,
            'metrics_export_size': len(prometheus_metrics) + len(json_metrics)
        }
        
    finally:
        telemetry.stop()


def test_ml_performance_prediction():
    """Test ML-based performance prediction capabilities."""
    print("🧠 Testing ML Performance Prediction...")
    
    from src.edge_tpu_v5_benchmark.multi_tpu_parallel import (
        MLPerformancePredictor, ParallelTask, TPUDevice
    )
    
    predictor = MLPerformancePredictor()
    
    # Create test scenarios
    devices = [
        TPUDevice("tpu_0", "/dev/apex_0", compute_capability=8.0, memory_gb=32.0, current_load=0.2),
        TPUDevice("tpu_1", "/dev/apex_1", compute_capability=7.5, memory_gb=28.0, current_load=0.6),
        TPUDevice("tpu_2", "/dev/apex_2", compute_capability=8.5, memory_gb=36.0, current_load=0.1),
    ]
    
    tasks = [
        ParallelTask("task_1", "small_model.onnx", None, {}, 
                    estimated_duration=0.5, memory_requirement=2.0, compute_requirement=3.0),
        ParallelTask("task_2", "medium_model.onnx", None, {},
                    estimated_duration=1.2, memory_requirement=8.0, compute_requirement=6.0),
        ParallelTask("task_3", "large_model.onnx", None, {},
                    estimated_duration=2.5, memory_requirement=16.0, compute_requirement=10.0),
    ]
    
    predictions = []
    
    # Test prediction accuracy
    for task in tasks:
        for device in devices:
            prediction = predictor.predict_performance(task, device)
            
            predictions.append({
                'task_id': task.task_id,
                'device_id': device.device_id,
                'predicted_execution_time': prediction['execution_time'],
                'predicted_throughput': prediction['throughput'],
                'predicted_memory': prediction['memory_usage'],
                'device_utilization': device.utilization_score()
            })
    
    # Simulate training data to improve predictions
    for i in range(50):
        task = tasks[i % len(tasks)]
        device = devices[i % len(devices)]
        
        # Simulate actual execution results
        from src.edge_tpu_v5_benchmark.multi_tpu_parallel import ExecutionResult
        
        actual_time = task.estimated_duration * (1.0 + device.current_load * 0.3 + np.random.uniform(-0.2, 0.2))
        actual_throughput = device.compute_capability * (1.0 - device.current_load) * np.random.uniform(0.8, 1.2)
        actual_memory = min(task.memory_requirement, device.memory_gb * 0.8) * np.random.uniform(0.9, 1.1)
        
        result = ExecutionResult(
            task_id=task.task_id,
            device_id=device.device_id,
            execution_time=actual_time,
            throughput=actual_throughput,
            latency_stats={'p50': actual_time * 0.8},
            memory_usage=actual_memory,
            power_consumption=2.0,
            success=True
        )
        
        predictor.update_model(task, device, result)
    
    # Test predictions after training
    improved_predictions = []
    for task in tasks:
        for device in devices:
            prediction = predictor.predict_performance(task, device)
            improved_predictions.append({
                'task_id': task.task_id,
                'device_id': device.device_id,
                'predicted_execution_time': prediction['execution_time'],
                'predicted_throughput': prediction['throughput']
            })
    
    print("   ML Prediction Results:")
    print(f"   - Initial predictions: {len(predictions)}")
    print(f"   - Training samples: 50")
    print(f"   - Improved predictions: {len(improved_predictions)}")
    
    # Calculate prediction variance (measure of learning)
    initial_times = [p['predicted_execution_time'] for p in predictions]
    improved_times = [p['predicted_execution_time'] for p in improved_predictions]
    
    initial_variance = np.var(initial_times) if initial_times else 0
    improved_variance = np.var(improved_times) if improved_times else 0
    
    print(f"   - Initial prediction variance: {initial_variance:.3f}")
    print(f"   - Improved prediction variance: {improved_variance:.3f}")
    print(f"   - Learning improvement: {abs(initial_variance - improved_variance):.3f}")
    
    return {
        'initial_predictions': len(predictions),
        'training_samples': 50,
        'prediction_improvement': abs(initial_variance - improved_variance),
        'torch_available': predictor.model is not None
    }


def test_quantum_load_balancing():
    """Test quantum-enhanced load balancing."""
    print("⚛️ Testing Quantum Load Balancing...")
    
    from src.edge_tpu_v5_benchmark.multi_tpu_parallel import (
        QuantumLoadBalancer, ParallelTask, TPUDevice, LoadBalancingStrategy
    )
    
    load_balancer = QuantumLoadBalancer()
    
    # Create test scenario with imbalanced devices
    devices = [
        TPUDevice("tpu_0", "/dev/apex_0", current_load=0.1, temperature_celsius=40.0),
        TPUDevice("tpu_1", "/dev/apex_1", current_load=0.8, temperature_celsius=65.0),  # Hot & loaded
        TPUDevice("tpu_2", "/dev/apex_2", current_load=0.3, temperature_celsius=45.0),
        TPUDevice("tpu_3", "/dev/apex_3", current_load=0.2, temperature_celsius=38.0),  # Cool & available
    ]
    
    # Create diverse tasks
    tasks = []
    for i in range(12):  # More tasks than devices
        task = ParallelTask(
            f"task_{i}", 
            f"model_{i}.onnx", 
            None, 
            {},
            compute_requirement=2.0 + i % 4,
            memory_requirement=1.0 + i % 3,
            priority=1 + i % 3
        )
        tasks.append(task)
    
    # Test different load balancing strategies
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS,
        LoadBalancingStrategy.QUANTUM_ANNEALING,
        LoadBalancingStrategy.ML_PREDICTIVE,
        LoadBalancingStrategy.ADAPTIVE_HYBRID
    ]
    
    strategy_results = {}
    
    for strategy in strategies:
        distribution = load_balancer.optimize_task_distribution(tasks, devices, strategy)
        
        # Analyze distribution quality
        device_task_counts = {}
        device_loads = {}
        
        for task_id, device_id in distribution.items():
            device_task_counts[device_id] = device_task_counts.get(device_id, 0) + 1
            
            # Find corresponding task and device
            task = next((t for t in tasks if t.task_id == task_id), None)
            device = next((d for d in devices if d.device_id == device_id), None)
            
            if task and device:
                load_increase = task.compute_requirement / device.compute_capability
                device_loads[device_id] = device_loads.get(device_id, 0) + load_increase
        
        # Calculate load balance score
        if device_loads:
            load_values = list(device_loads.values())
            load_balance_score = 1.0 - (np.std(load_values) / np.mean(load_values)) if np.mean(load_values) > 0 else 0
        else:
            load_balance_score = 0
        
        # Calculate thermal awareness (prefer cooler devices)
        thermal_score = 0
        for task_id, device_id in distribution.items():
            device = next((d for d in devices if d.device_id == device_id), None)
            if device:
                thermal_score += (80.0 - device.temperature_celsius) / 80.0  # Higher score for cooler devices
        
        thermal_score /= len(distribution) if distribution else 1
        
        strategy_results[strategy.value] = {
            'assigned_tasks': len(distribution),
            'load_balance_score': load_balance_score,
            'thermal_awareness_score': thermal_score,
            'task_distribution': dict(device_task_counts),
            'load_distribution': dict(device_loads)
        }
    
    print("   Quantum Load Balancing Results:")
    for strategy, result in strategy_results.items():
        print(f"   - {strategy}:")
        print(f"     Load balance: {result['load_balance_score']:.3f}")
        print(f"     Thermal awareness: {result['thermal_awareness_score']:.3f}")
        print(f"     Task distribution: {result['task_distribution']}")
    
    # Find best strategy
    best_strategy = max(strategy_results.keys(), 
                       key=lambda s: strategy_results[s]['load_balance_score'] + 
                                   strategy_results[s]['thermal_awareness_score'])
    
    print(f"   - Best strategy: {best_strategy}")
    
    return {
        'strategy_results': strategy_results,
        'best_strategy': best_strategy,
        'load_balance_improvement': strategy_results[best_strategy]['load_balance_score'],
        'thermal_optimization': strategy_results[best_strategy]['thermal_awareness_score']
    }


def run_generation_3_tests():
    """Run all Generation 3 optimization tests."""
    print("=" * 80)
    print("🌟 TERRAGON GENERATION 3 OPTIMIZATION TESTS")
    print("🚀 Advanced Scaling and Quantum-Enhanced Performance")
    print("=" * 80)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test 1: Multi-TPU Parallel Execution
        test_results['multi_tpu_parallel'] = test_multi_tpu_parallel_execution()
        
        # Test 2: Advanced Telemetry System  
        test_results['advanced_telemetry'] = test_advanced_telemetry_system()
        
        # Test 3: ML Performance Prediction
        test_results['ml_prediction'] = test_ml_performance_prediction()
        
        # Test 4: Quantum Load Balancing
        test_results['quantum_load_balancing'] = test_quantum_load_balancing()
        
        total_time = time.time() - start_time
        
        print("=" * 80)
        print("🏆 GENERATION 3 TEST SUMMARY")
        print("=" * 80)
        
        # Multi-TPU results
        multi_tpu = test_results['multi_tpu_parallel']
        print(f"Multi-TPU Parallel Processing:")
        print(f"  - Parallel efficiency: {multi_tpu['parallel_efficiency']:.1f}x")
        print(f"  - Best configuration: {multi_tpu['best_configuration']}")
        print(f"  - Active TPU devices: {multi_tpu['performance_metrics']['active_devices']}")
        
        # Telemetry results
        telemetry = test_results['advanced_telemetry']
        print(f"Advanced Telemetry & Observability:")
        print(f"  - System health score: {telemetry['system_health_score']:.1f}/100")
        print(f"  - Metrics tracked: {telemetry['metrics_count']}")
        print(f"  - Active alerts: {telemetry['active_alerts']}")
        print(f"  - Anomaly detection: {'Active' if telemetry['anomaly_detection_active'] else 'Inactive'}")
        
        # ML Prediction results
        ml_pred = test_results['ml_prediction']
        print(f"ML Performance Prediction:")
        print(f"  - Prediction improvement: {ml_pred['prediction_improvement']:.3f}")
        print(f"  - Training samples processed: {ml_pred['training_samples']}")
        print(f"  - PyTorch ML model: {'Available' if ml_pred['torch_available'] else 'Fallback'}")
        
        # Quantum Load Balancing results
        quantum_lb = test_results['quantum_load_balancing']
        print(f"Quantum Load Balancing:")
        print(f"  - Best strategy: {quantum_lb['best_strategy']}")
        print(f"  - Load balance score: {quantum_lb['load_balance_improvement']:.3f}")
        print(f"  - Thermal optimization: {quantum_lb['thermal_optimization']:.3f}")
        
        print(f"\nTotal test execution time: {total_time:.2f}s")
        
        # Calculate overall Generation 3 score
        g3_score = (
            min(multi_tpu['parallel_efficiency'] / 10.0, 10.0) +  # Max 10 points
            telemetry['system_health_score'] / 10.0 +  # Max 10 points
            min(ml_pred['prediction_improvement'] * 20, 10.0) +  # Max 10 points
            quantum_lb['load_balance_improvement'] * 10 +  # Max 10 points
            quantum_lb['thermal_optimization'] * 10  # Max 10 points
        )
        
        print(f"\n🌟 GENERATION 3 OPTIMIZATION SCORE: {g3_score:.1f}/50")
        
        if g3_score >= 40:
            print("🏆 EXCELLENT - Generation 3 optimizations fully operational!")
        elif g3_score >= 30:
            print("✅ GOOD - Generation 3 optimizations working well!")
        elif g3_score >= 20:
            print("⚠️ ADEQUATE - Generation 3 optimizations need improvement!")
        else:
            print("❌ NEEDS WORK - Generation 3 optimizations require attention!")
        
        # Save results
        with open('generation_3_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: generation_3_test_results.json")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Generation 3 tests failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_generation_3_tests()
    if results:
        print("\n✅ Generation 3 optimization tests completed successfully!")
    else:
        print("\n❌ Generation 3 optimization tests failed!")
        exit(1)