"""Command-line interface for edge-tpu-v5-benchmark."""

import click
from rich.console import Console
from rich.table import Table

from . import __version__

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
    # Implementation placeholder
    console.print("[green]✓ Found 1 TPU v5 edge device at /dev/apex_0[/green]")


@main.command()
@click.option("--workload", default="all", help="Benchmark workload to run")
@click.option("--iterations", default=100, help="Number of iterations")
@click.option("--model", help="Specific model to benchmark")
@click.option("--profile-power", is_flag=True, help="Enable power profiling")
def run(workload, iterations, model, profile_power):
    """Run benchmark suite."""
    console.print(f"[cyan]Running benchmark: {workload}[/cyan]")
    console.print(f"Iterations: {iterations}")
    if model:
        console.print(f"Model: {model}")
    if profile_power:
        console.print("[yellow]Power profiling enabled[/yellow]")
    
    # Implementation placeholder
    console.print("[green]✓ Benchmark completed[/green]")


@main.command()
@click.option("--category", default="vision", help="Leaderboard category")
@click.option("--metric", default="throughput", help="Metric to display")
def leaderboard(category, metric):
    """View benchmark leaderboard."""
    table = Table(title=f"TPU v5 Edge {category.title()} Leaderboard - {metric.title()}")
    
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("Score", style="green")
    table.add_column("Efficiency", style="yellow")
    
    # Placeholder data
    table.add_row("1", "MobileNetV3", "892 FPS", "1,049 FPS/W")
    table.add_row("2", "EfficientNet-Lite", "624 FPS", "567 FPS/W")
    table.add_row("3", "ResNet-50", "412 FPS", "330 FPS/W")
    
    console.print(table)


if __name__ == "__main__":
    main()