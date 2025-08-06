"""Quantum-Inspired Task Planner CLI

Command-line interface for the quantum task planning system integrated with TPU benchmarking.
"""

import asyncio
import click
import json
import time
from typing import Optional, List
from pathlib import Path

from .quantum_planner import (
    QuantumTaskPlanner,
    QuantumTask,
    QuantumResource,
    QuantumState
)
from .config import get_config
from .logging_config import setup_logging


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config-file', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.pass_context
def quantum_cli(ctx, verbose: bool, config_file: Optional[str]):
    """Quantum-Inspired Task Planner for TPU v5 Optimization"""
    ctx.ensure_object(dict)
    
    # Setup logging
    log_level = 'DEBUG' if verbose else 'INFO'
    setup_logging(level=log_level)
    
    # Load configuration
    if config_file:
        ctx.obj['config'] = get_config(config_file)
    else:
        ctx.obj['config'] = get_config()
    
    # Initialize planner
    ctx.obj['planner'] = QuantumTaskPlanner()


@quantum_cli.command()
@click.option('--task-id', '-i', required=True, help='Unique task identifier')
@click.option('--name', '-n', required=True, help='Task name')
@click.option('--priority', '-p', type=float, default=1.0, help='Task priority (higher = more important)')
@click.option('--complexity', '-x', type=float, default=1.0, help='Task complexity factor')
@click.option('--duration', '-d', type=float, default=1.0, help='Estimated duration in seconds')
@click.option('--dependencies', '-dep', multiple=True, help='Task dependencies (can specify multiple)')
@click.option('--tpu-affinity', '-tpu', help='Preferred TPU device')
@click.option('--memory-gb', '-m', type=float, default=0.0, help='Memory requirement in GB')
@click.option('--cpu-cores', type=float, default=0.0, help='CPU cores requirement')
@click.pass_context
def add_task(ctx, task_id: str, name: str, priority: float, complexity: float, 
             duration: float, dependencies: List[str], tpu_affinity: Optional[str], 
             memory_gb: float, cpu_cores: float):
    """Add a new quantum task to the planner"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    # Build resource requirements
    resource_requirements = {}
    if memory_gb > 0:
        resource_requirements['memory_gb'] = memory_gb
    if cpu_cores > 0:
        resource_requirements['cpu_cores'] = cpu_cores
    if tpu_affinity:
        resource_requirements['tpu_v5_primary'] = 1.0
    
    # Create task
    task = QuantumTask(
        id=task_id,
        name=name,
        priority=priority,
        complexity=complexity,
        estimated_duration=duration,
        dependencies=set(dependencies),
        tpu_affinity=tpu_affinity,
        resource_requirements=resource_requirements
    )
    
    planner.add_task(task)
    
    click.echo(f"✅ Added quantum task '{name}' (ID: {task_id})")
    click.echo(f"   Priority: {priority}, Complexity: {complexity}")
    click.echo(f"   Dependencies: {list(dependencies) if dependencies else 'None'}")
    click.echo(f"   Resource requirements: {resource_requirements}")


@quantum_cli.command()
@click.option('--task1', '-t1', required=True, help='First task ID')
@click.option('--task2', '-t2', required=True, help='Second task ID')
@click.pass_context
def entangle(ctx, task1: str, task2: str):
    """Create quantum entanglement between two tasks"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    try:
        planner.entangle_tasks(task1, task2)
        click.echo(f"🔗 Quantum entanglement created between {task1} and {task2}")
    except Exception as e:
        click.echo(f"❌ Failed to create entanglement: {e}")


@quantum_cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file for schedule')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='text', help='Output format')
@click.pass_context
def optimize(ctx, output: Optional[str], format: str):
    """Optimize task schedule using quantum annealing"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    click.echo("🔄 Running quantum annealing optimization...")
    
    optimized_schedule = planner.optimize_schedule()
    
    if not optimized_schedule:
        click.echo("📭 No tasks ready for optimization")
        return
    
    if format == 'json':
        schedule_data = []
        for i, task in enumerate(optimized_schedule):
            schedule_data.append({
                'position': i,
                'task_id': task.id,
                'name': task.name,
                'priority': planner.calculate_quantum_priority(task),
                'complexity': task.complexity,
                'state': task.state.value,
                'decoherence': task.measure_decoherence()
            })
        
        if output:
            with open(output, 'w') as f:
                json.dump(schedule_data, f, indent=2)
            click.echo(f"📄 Schedule saved to {output}")
        else:
            click.echo(json.dumps(schedule_data, indent=2))
    
    else:  # text format
        click.echo("\n🎯 Optimized Quantum Schedule:")
        click.echo("=" * 80)
        
        for i, task in enumerate(optimized_schedule):
            quantum_priority = planner.calculate_quantum_priority(task)
            decoherence = task.measure_decoherence()
            
            click.echo(f"{i+1:2d}. {task.name} (ID: {task.id})")
            click.echo(f"    Quantum Priority: {quantum_priority:.2f}")
            click.echo(f"    Complexity: {task.complexity:.1f}")
            click.echo(f"    State: {task.state.value}")
            click.echo(f"    Decoherence: {decoherence:.1%}")
            click.echo(f"    Dependencies: {list(task.dependencies) if task.dependencies else 'None'}")
            click.echo()
        
        if output:
            with open(output, 'w') as f:
                f.write("Optimized Quantum Schedule\n")
                f.write("=" * 30 + "\n\n")
                for i, task in enumerate(optimized_schedule):
                    f.write(f"{i+1}. {task.name} (ID: {task.id})\n")
                    f.write(f"   Priority: {planner.calculate_quantum_priority(task):.2f}\n")
                    f.write(f"   State: {task.state.value}\n\n")
            click.echo(f"📄 Schedule saved to {output}")


@quantum_cli.command()
@click.option('--cycles', '-c', type=int, default=1, help='Number of execution cycles')
@click.option('--interval', '-i', type=float, default=1.0, help='Interval between cycles in seconds')
@click.option('--export', '-e', type=click.Path(), help='Export final state to file')
@click.pass_context
def execute(ctx, cycles: int, interval: float, export: Optional[str]):
    """Execute quantum tasks using the optimized scheduler"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    async def run_execution():
        click.echo(f"🚀 Starting quantum task execution ({cycles} cycles)")
        click.echo("=" * 60)
        
        total_executed = 0
        total_failed = 0
        
        for cycle in range(cycles):
            click.echo(f"\n🔄 Quantum Execution Cycle {cycle + 1}/{cycles}")
            
            # Run execution cycle
            results = await planner.run_quantum_execution_cycle()
            
            # Display results
            executed_count = len(results['tasks_executed'])
            failed_count = len(results['tasks_failed'])
            
            total_executed += executed_count
            total_failed += failed_count
            
            click.echo(f"   ✅ Tasks executed: {executed_count}")
            click.echo(f"   ❌ Tasks failed: {failed_count}")
            click.echo(f"   ⏱️  Cycle duration: {results['cycle_duration']:.2f}s")
            click.echo(f"   🌊 Quantum coherence: {results['quantum_coherence']:.2%}")
            
            # Show resource utilization
            click.echo("   📊 Resource utilization:")
            for resource, utilization in results['resource_utilization'].items():
                bar_length = 20
                filled = int(utilization * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                click.echo(f"      {resource}: {bar} {utilization:.1%}")
            
            # Wait between cycles
            if cycle < cycles - 1 and interval > 0:
                await asyncio.sleep(interval)
        
        click.echo("\n" + "=" * 60)
        click.echo(f"🏁 Execution complete:")
        click.echo(f"   Total executed: {total_executed}")
        click.echo(f"   Total failed: {total_failed}")
        click.echo(f"   Success rate: {total_executed/(total_executed+total_failed)*100:.1f}%" 
                   if total_executed + total_failed > 0 else "No tasks processed")
        
        # Export final state if requested
        if export:
            planner.export_quantum_state(export)
            click.echo(f"📤 Final quantum state exported to {export}")
    
    # Run the async execution
    try:
        asyncio.run(run_execution())
    except KeyboardInterrupt:
        click.echo("\n⏸️  Execution interrupted by user")
    except Exception as e:
        click.echo(f"\n💥 Execution failed: {e}")


@quantum_cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'summary']), default='summary', help='Output format')
@click.option('--export', '-e', type=click.Path(), help='Export to file')
@click.pass_context
def status(ctx, format: str, export: Optional[str]):
    """Show current quantum system status"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    system_state = planner.get_system_state()
    
    if format == 'json':
        output = json.dumps(system_state, indent=2)
        if export:
            with open(export, 'w') as f:
                f.write(output)
            click.echo(f"📤 Status exported to {export}")
        else:
            click.echo(output)
    
    else:  # summary format
        click.echo("\n🌌 Quantum System Status")
        click.echo("=" * 50)
        
        # Task overview
        click.echo(f"📋 Tasks:")
        click.echo(f"   Total: {system_state['total_tasks']}")
        click.echo(f"   Completed: {system_state['completed_tasks']}")
        click.echo(f"   Ready: {system_state['ready_tasks']}")
        click.echo(f"   Remaining: {system_state['total_tasks'] - system_state['completed_tasks']}")
        
        # Quantum metrics
        qm = system_state['quantum_metrics']
        click.echo(f"\n🔬 Quantum Metrics:")
        click.echo(f"   Average coherence: {qm['average_coherence']:.2%}")
        click.echo(f"   Entanglement pairs: {qm['entanglement_pairs']}")
        click.echo(f"   Superposition tasks: {qm['superposition_tasks']}")
        click.echo(f"   Collapsed tasks: {qm['collapsed_tasks']}")
        click.echo(f"   Entangled tasks: {qm['entangled_tasks']}")
        click.echo(f"   Decoherent tasks: {qm['decoherent_tasks']}")
        
        # Resource utilization
        click.echo(f"\n📊 Resource Utilization:")
        for resource, utilization in system_state['resource_utilization'].items():
            bar_length = 30
            filled = int(utilization * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            click.echo(f"   {resource:15s}: {bar} {utilization:.1%}")
        
        if export:
            # Export summary to text file
            with open(export, 'w') as f:
                f.write("Quantum System Status\n")
                f.write("=" * 25 + "\n\n")
                f.write(f"Total tasks: {system_state['total_tasks']}\n")
                f.write(f"Completed: {system_state['completed_tasks']}\n")
                f.write(f"Ready: {system_state['ready_tasks']}\n")
                f.write(f"Average coherence: {qm['average_coherence']:.2%}\n")
                f.write(f"Entanglement pairs: {qm['entanglement_pairs']}\n")
            click.echo(f"📤 Status exported to {export}")


@quantum_cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), required=True, help='Task definition file (JSON)')
@click.pass_context
def load_tasks(ctx, file: str):
    """Load tasks from JSON file"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    try:
        with open(file, 'r') as f:
            tasks_data = json.load(f)
        
        loaded_count = 0
        
        for task_data in tasks_data.get('tasks', []):
            task = QuantumTask(
                id=task_data['id'],
                name=task_data['name'],
                priority=task_data.get('priority', 1.0),
                complexity=task_data.get('complexity', 1.0),
                estimated_duration=task_data.get('duration', 1.0),
                dependencies=set(task_data.get('dependencies', [])),
                resource_requirements=task_data.get('resource_requirements', {}),
                tpu_affinity=task_data.get('tpu_affinity')
            )
            planner.add_task(task)
            loaded_count += 1
        
        # Create entanglements if specified
        for entanglement in tasks_data.get('entanglements', []):
            planner.entangle_tasks(entanglement['task1'], entanglement['task2'])
        
        click.echo(f"✅ Loaded {loaded_count} quantum tasks from {file}")
        
        entanglement_count = len(tasks_data.get('entanglements', []))
        if entanglement_count > 0:
            click.echo(f"🔗 Created {entanglement_count} quantum entanglements")
    
    except Exception as e:
        click.echo(f"❌ Failed to load tasks: {e}")


@quantum_cli.command()
@click.option('--output', '-o', type=click.Path(), required=True, help='Output file path')
@click.pass_context
def export_state(ctx, output: str):
    """Export complete quantum state to file"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    try:
        planner.export_quantum_state(output)
        click.echo(f"✅ Quantum state exported to {output}")
    except Exception as e:
        click.echo(f"❌ Export failed: {e}")


@quantum_cli.command()
@click.argument('task_ids', nargs=-1, required=True)
@click.pass_context
def remove_tasks(ctx, task_ids: List[str]):
    """Remove tasks from the quantum planner"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    removed_count = 0
    for task_id in task_ids:
        if task_id in planner.tasks:
            # Remove from entanglement graph
            if task_id in planner.entanglement_graph:
                for entangled_id in planner.entanglement_graph[task_id]:
                    if entangled_id in planner.tasks:
                        planner.tasks[entangled_id].entangled_tasks.discard(task_id)
                del planner.entanglement_graph[task_id]
            
            # Remove task
            del planner.tasks[task_id]
            planner.completed_tasks.discard(task_id)
            removed_count += 1
            click.echo(f"🗑️  Removed task: {task_id}")
        else:
            click.echo(f"⚠️  Task not found: {task_id}")
    
    if removed_count > 0:
        click.echo(f"✅ Removed {removed_count} quantum task(s)")
    
    # Rebuild coherence matrix
    planner._expand_coherence_matrix()


@quantum_cli.command()
@click.pass_context  
def clear_all(ctx):
    """Clear all tasks and reset quantum system"""
    planner: QuantumTaskPlanner = ctx.obj['planner']
    
    if click.confirm('⚠️  This will remove all tasks and reset the quantum system. Continue?'):
        planner.tasks.clear()
        planner.completed_tasks.clear()
        planner.entanglement_graph.clear()
        planner.coherence_matrix = planner.coherence_matrix[:0, :0]
        
        # Reset resource utilization
        for resource in planner.resources.values():
            resource.available_capacity = resource.total_capacity
        
        click.echo("🧹 Quantum system cleared and reset")
    else:
        click.echo("Operation cancelled")


if __name__ == '__main__':
    quantum_cli()