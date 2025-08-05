"""Command-line interface for edge-tpu-v5-benchmark."""

import click
import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
import numpy as np

from . import __version__
from .benchmark import TPUv5Benchmark
from .models import ModelLoader, ModelRegistry

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """Edge TPU v5 Benchmark Suite CLI."""
    pass


@main.command()
def detect():
    """Detect available TPU v5 edge devices."""
    console.print("[yellow]Detecting TPU v5 edge devices...[/yellow]")
    
    # Check for TPU devices
    devices_found = []
    potential_paths = ["/dev/apex_0", "/dev/apex_1", "/dev/apex_2"]
    
    for device_path in potential_paths:
        if Path(device_path).exists():
            devices_found.append(device_path)
    
    if devices_found:
        for device in devices_found:
            console.print(f"[green]✓ Found TPU v5 edge device at {device}[/green]")
        
        # Show device information
        benchmark = TPUv5Benchmark(devices_found[0])
        system_info = benchmark.get_system_info()
        
        info_table = Table(title="Device Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("TPU Version", system_info['tpu_version'])
        info_table.add_row("Compiler Version", system_info['compiler_version']) 
        info_table.add_row("Runtime Version", system_info['runtime_version'])
        info_table.add_row("Simulation Mode", str(system_info['simulation_mode']))
        
        console.print(info_table)
    else:
        console.print("[red]✗ No TPU v5 edge devices found[/red]")
        console.print("[yellow]Will run in simulation mode for development[/yellow]")


@main.command()
@click.option("--workload", default="all", help="Benchmark workload to run")
@click.option("--iterations", default=100, help="Number of iterations")
@click.option("--model", help="Specific model to benchmark")
@click.option("--profile-power", is_flag=True, help="Enable power profiling")
@click.option("--output", help="Output file for results (JSON)")
@click.option("--warmup", default=50, help="Number of warmup iterations")
def run(workload, iterations, model, profile_power, output, warmup):
    """Run benchmark suite."""
    registry = ModelRegistry()
    benchmark = TPUv5Benchmark(enable_power_monitoring=profile_power)
    
    # Determine models to run
    if model:
        models_to_run = [model] if registry.get_model_info(model) else []
        if not models_to_run:
            console.print(f"[red]Error: Model '{model}' not found[/red]")
            return
    elif workload == "all":
        models_to_run = list(registry.list_models().keys())
    else:
        models_to_run = list(registry.list_models(workload).keys())
    
    if not models_to_run:
        console.print(f"[red]No models found for workload: {workload}[/red]")
        return
    
    console.print(f"[cyan]Running benchmark suite[/cyan]")
    console.print(f"Models: {', '.join(models_to_run)}")
    console.print(f"Iterations: {iterations} (warmup: {warmup})")
    
    if profile_power:
        console.print("[yellow]Power profiling enabled[/yellow]")
    
    all_results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("Overall Progress", total=len(models_to_run))
        
        for model_id in models_to_run:
            model_info = registry.get_model_info(model_id)
            progress.update(main_task, description=f"Benchmarking {model_info['name']}")
            
            # Create dummy model for demonstration
            dummy_model = ModelLoader.from_onnx(
                f"models/{model_id}.onnx",  # This would be a real path
                optimization_level=3
            )
            
            # Run benchmark
            try:
                results = benchmark.run(
                    model=dummy_model,
                    input_shape=model_info['input_shape'],
                    iterations=iterations,
                    warmup=warmup,
                    measure_power=profile_power
                )
                
                all_results[model_id] = {
                    "model_info": model_info,
                    "results": results.to_dict()
                }
                
                # Display quick results
                console.print(
                    f"[green]✓ {model_info['name']}: "
                    f"{results.throughput:.1f} inferences/sec, "
                    f"{results.latency_p99:.2f}ms p99 latency[/green]"
                )
                
            except Exception as e:
                console.print(f"[red]✗ Failed to benchmark {model_info['name']}: {e}[/red]")
                all_results[model_id] = {"error": str(e)}
            
            progress.advance(main_task)
    
    # Display summary table
    _display_results_table(all_results)
    
    # Save results if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output_path}[/green]")
    
    console.print("[green]✓ Benchmark suite completed[/green]")


@main.command()
@click.option("--category", default="vision", help="Leaderboard category")
@click.option("--metric", default="throughput", help="Metric to display")
def leaderboard(category, metric):
    """View benchmark leaderboard."""
    registry = ModelRegistry()
    models = registry.list_models(category if category != "all" else None)
    
    if not models:
        console.print(f"[red]No models found for category: {category}[/red]")
        return
    
    table = Table(title=f"TPU v5 Edge {category.title()} Leaderboard - {metric.title()}")
    
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Category", style="blue")
    table.add_column("Score", style="green")
    table.add_column("Efficiency", style="yellow")
    table.add_column("Description", style="white")
    
    # Generate simulated leaderboard data
    leaderboard_data = []
    for i, (model_id, info) in enumerate(models.items()):
        # Simulate performance metrics
        base_score = 1000 - (i * 100) + np.random.randint(-50, 50)
        efficiency = base_score * (0.8 + np.random.random() * 0.4)
        
        leaderboard_data.append({
            "rank": i + 1,
            "model": info["name"],
            "category": info["category"],
            "score": f"{base_score} FPS",
            "efficiency": f"{efficiency:.0f} FPS/W",
            "description": info["description"]
        })
    
    # Sort by score (extract numeric value)
    leaderboard_data.sort(key=lambda x: int(x["score"].split()[0]), reverse=True)
    
    # Update ranks after sorting
    for i, item in enumerate(leaderboard_data):
        item["rank"] = i + 1
        table.add_row(
            str(item["rank"]),
            item["model"],
            item["category"],
            item["score"],
            item["efficiency"],
            item["description"]
        )
    
    console.print(table)


def _display_results_table(results: dict):
    """Display benchmark results in a formatted table."""
    if not results:
        return
    
    table = Table(title="Benchmark Results Summary")
    table.add_column("Model", style="magenta")
    table.add_column("Throughput", style="green")
    table.add_column("Latency (p99)", style="cyan")
    table.add_column("Power", style="yellow")
    table.add_column("Efficiency", style="blue")
    table.add_column("Success Rate", style="white")
    
    for model_id, data in results.items():
        if "error" in data:
            table.add_row(
                model_id,
                "[red]FAILED[/red]",
                "[red]FAILED[/red]",
                "[red]FAILED[/red]",
                "[red]FAILED[/red]",
                "[red]0%[/red]"
            )
        else:
            results_data = data["results"]
            table.add_row(
                data["model_info"]["name"],
                f"{results_data['throughput']:.1f} FPS",
                f"{results_data['latency_metrics']['p99']:.2f} ms",
                f"{results_data['power_metrics']['avg_power']:.2f} W",
                f"{results_data['power_metrics']['efficiency']:.0f} FPS/W",
                f"{results_data['execution_metrics']['success_rate']*100:.1f}%"
            )
    
    console.print(table)


@main.command()
@click.option("--model-path", required=True, help="Path to model file")
@click.option("--format", default="onnx", type=click.Choice(["onnx", "tflite"]), help="Model format")
@click.option("--iterations", default=1000, help="Number of iterations")
@click.option("--input-shape", help="Input shape (e.g., '1,3,224,224')")
def quick_benchmark(model_path, format, iterations, input_shape):
    """Run a quick benchmark on a single model file."""
    model_path = Path(model_path)
    if not model_path.exists():
        console.print(f"[red]Error: Model file not found: {model_path}[/red]")
        return
    
    # Parse input shape
    if input_shape:
        try:
            shape = tuple(int(x) for x in input_shape.split(','))
        except ValueError:
            console.print(f"[red]Error: Invalid input shape format: {input_shape}[/red]")
            console.print("[yellow]Use format: '1,3,224,224'[/yellow]")
            return
    else:
        # Default shape for vision models
        shape = (1, 3, 224, 224)
    
    console.print(f"[cyan]Quick benchmarking: {model_path.name}[/cyan]")
    console.print(f"Format: {format}, Input shape: {shape}, Iterations: {iterations}")
    
    try:
        # Load model
        if format == "onnx":
            model = ModelLoader.from_onnx(str(model_path))
        else:
            model = ModelLoader.from_tflite(str(model_path))
        
        # Run benchmark
        benchmark = TPUv5Benchmark()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running benchmark...", total=iterations)
            
            results = benchmark.run(
                model=model,
                input_shape=shape,
                iterations=iterations,
                warmup=min(100, iterations // 10)
            )
            
            progress.update(task, completed=iterations)
        
        # Display results
        panel_content = f"""
[green]Throughput:[/green] {results.throughput:.1f} inferences/sec
[cyan]Latency (mean):[/cyan] {results.latency_mean:.2f} ms
[cyan]Latency (p99):[/cyan] {results.latency_p99:.2f} ms
[yellow]Power (avg):[/yellow] {results.avg_power:.2f} W
[blue]Efficiency:[/blue] {results.inferences_per_watt:.0f} inferences/W
[white]Success Rate:[/white] {results.success_rate*100:.1f}%
        """
        
        console.print(Panel(panel_content, title="Benchmark Results", expand=False))
        
    except Exception as e:
        console.print(f"[red]Error running benchmark: {e}[/red]")


if __name__ == "__main__":
    main()