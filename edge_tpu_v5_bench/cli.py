"""
CLI entry point for edge-tpu-v5-bench.
"""
from __future__ import annotations
import argparse
import json
import sys

from .model_benchmark import ModelBenchmark
from .inference_simulator import InferenceSimulator
from .power_estimator import PowerEstimator
from .suite import BenchmarkSuite


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="edge-tpu-v5-bench",
        description="Benchmark a neural network model on the Google Edge TPU v5.",
    )
    parser.add_argument("--model-name", required=True, help="Model name/identifier")
    parser.add_argument(
        "--layers",
        required=True,
        help='JSON string describing layer specs, e.g. \'[{"type":"conv","in_channels":3,"out_channels":32,"kernel_size":3}]\'',
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument(
        "--quantization",
        default="INT8",
        choices=["INT8", "INT4", "FP16", "FP32"],
        help="Quantization mode (default: INT8)",
    )
    parser.add_argument("--tops", type=float, default=4.0, help="TPU TOPS rating (default: 4.0)")
    parser.add_argument(
        "--memory-bandwidth", type=float, default=8.0,
        help="TPU memory bandwidth GB/s (default: 8.0)"
    )
    parser.add_argument("--power-budget", type=float, default=2.0, help="Power budget W (default: 2.0)")
    parser.add_argument("--efficiency", type=float, default=50.0, help="TOPS/W efficiency (default: 50.0)")
    parser.add_argument("--output-csv", default=None, help="Path to save CSV results")
    parser.add_argument("--output-json", default=None, help="Path to save JSON results")

    args = parser.parse_args(argv)

    try:
        layer_specs = json.loads(args.layers)
    except json.JSONDecodeError as exc:
        print(f"Error: --layers must be valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    model = ModelBenchmark(model_name=args.model_name, layer_specs=layer_specs)
    simulator = InferenceSimulator(tops=args.tops, memory_bandwidth_gbps=args.memory_bandwidth)
    estimator = PowerEstimator(power_budget_w=args.power_budget, efficiency_tops_per_w=args.efficiency)

    suite = BenchmarkSuite()
    suite.add_model(model)
    results = suite.run_all(
        simulator, estimator,
        batch_size=args.batch_size,
        quantization=args.quantization,
    )

    # Pretty print
    for r in results:
        print(f"\n{'='*50}")
        print(f"Model:          {r['model_name']}")
        print(f"Params:         {r['param_count']:,}")
        print(f"MACs:           {r['op_count']:,}")
        print(f"Memory (INT8):  {r['memory_mb_int8']:.4f} MB")
        print(f"Latency:        {r['latency_ms']:.4f} ms")
        print(f"Throughput:     {r['throughput_qps']:.2f} QPS")
        print(f"Memory util:    {r['memory_util']*100:.2f}%")
        print(f"Power:          {r['power_w']:.4f} W")
        print(f"TOPS/W:         {r['tops_per_watt']:.2f}")
        print(f"Within budget:  {r['within_budget']}")
        print(f"{'='*50}")

    if args.output_csv:
        suite.save_csv(results, args.output_csv)
        print(f"\nCSV saved to: {args.output_csv}")

    if args.output_json:
        suite.save_json(results, args.output_json)
        print(f"JSON saved to: {args.output_json}")


if __name__ == "__main__":
    main()
