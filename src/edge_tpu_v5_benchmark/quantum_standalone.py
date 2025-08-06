"""Standalone Quantum Task Planner

Self-contained implementation that doesn't depend on the TPU benchmark infrastructure.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import only quantum-specific modules to avoid TPU dependencies
from .quantum_planner import (
    QuantumTaskPlanner, QuantumTask, QuantumResource, QuantumState, QuantumAnnealer
)
from .quantum_validation import QuantumTaskValidator, QuantumSystemValidator
from .quantum_monitoring import QuantumHealthMonitor, MetricsCollector
from .quantum_security import QuantumSecurityManager, SecurityPolicy, SecurityLevel
from .quantum_performance import OptimizedQuantumTaskPlanner, PerformanceProfile
from .quantum_i18n import QuantumLocalizer, LocalizationConfig, SupportedLanguage
from .quantum_compliance import QuantumComplianceManager, DataCategory, ProcessingPurpose

logger = logging.getLogger(__name__)


@dataclass
class StandaloneConfig:
    """Configuration for standalone quantum planner."""
    # Localization
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    timezone: str = "UTC"
    
    # Performance
    max_concurrent_tasks: int = 4
    enable_caching: bool = True
    enable_optimization: bool = True
    
    # Security
    enable_security: bool = True
    require_signatures: bool = False
    audit_logging: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 30.0
    
    # Compliance
    enable_compliance: bool = True
    gdpr_compliance: bool = False
    ccpa_compliance: bool = False
    
    # Storage
    state_file: Optional[str] = None
    export_reports: bool = True


class StandaloneQuantumPlanner:
    """Self-contained quantum task planner with all features."""
    
    def __init__(self, config: Optional[StandaloneConfig] = None):
        self.config = config or StandaloneConfig()
        
        # Initialize localization
        localization_config = LocalizationConfig(
            language=self.config.language,
            gdpr_compliance=self.config.gdpr_compliance,
            ccpa_compliance=self.config.ccpa_compliance
        )
        self.localizer = QuantumLocalizer(localization_config)
        
        # Initialize core planner
        if self.config.enable_optimization:
            performance_profile = PerformanceProfile(
                max_concurrent_tasks=self.config.max_concurrent_tasks,
                cache_enabled=self.config.enable_caching
            )
            self.planner = OptimizedQuantumTaskPlanner(performance_profile=performance_profile)
        else:
            self.planner = QuantumTaskPlanner()
        
        # Initialize optional components
        self.validator = QuantumTaskValidator()
        self.system_validator = QuantumSystemValidator()
        
        self.security_manager = None
        if self.config.enable_security:
            security_policy = SecurityPolicy(
                require_signature=self.config.require_signatures,
                audit_enabled=self.config.audit_logging
            )
            self.security_manager = QuantumSecurityManager(security_policy)
        
        self.health_monitor = None
        if self.config.enable_monitoring:
            self.health_monitor = QuantumHealthMonitor(self.planner)
        
        self.compliance_manager = None
        if self.config.enable_compliance:
            self.compliance_manager = QuantumComplianceManager(localization_config)
        
        # Runtime state
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Standalone quantum planner initialized")
    
    async def start(self) -> None:
        """Start the quantum planner system."""
        if self.is_running:
            logger.warning("Planner is already running")
            return
        
        self.is_running = True
        logger.info(self.localizer.t("quantum.system.starting"))
        
        # Start monitoring if enabled
        if self.health_monitor:
            await self.health_monitor.start_monitoring(self.config.monitoring_interval)
        
        # Load previous state if configured
        if self.config.state_file and Path(self.config.state_file).exists():
            await self._load_state()
        
        logger.info("Standalone quantum planner started successfully")
    
    async def stop(self) -> None:
        """Stop the quantum planner system."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping standalone quantum planner")
        
        # Stop monitoring
        if self.health_monitor:
            await self.health_monitor.stop_monitoring()
        
        # Save state if configured
        if self.config.state_file:
            await self._save_state()
        
        # Shutdown optimized planner if applicable
        if hasattr(self.planner, 'shutdown'):
            await self.planner.shutdown()
        
        logger.info("Standalone quantum planner stopped")
    
    def create_task(self, 
                   task_id: str,
                   name: str,
                   priority: float = 1.0,
                   complexity: float = 1.0,
                   estimated_duration: float = 1.0,
                   dependencies: Optional[Set[str]] = None,
                   resource_requirements: Optional[Dict[str, float]] = None,
                   security_level: SecurityLevel = SecurityLevel.PUBLIC) -> str:
        """Create a new quantum task with comprehensive features."""
        
        # Create basic task
        task = QuantumTask(
            id=task_id,
            name=name,
            priority=priority,
            complexity=complexity,
            estimated_duration=estimated_duration,
            dependencies=dependencies or set(),
            resource_requirements=resource_requirements or {}
        )
        
        # Validate task
        validation_report = self.validator.validate_task(task, self.planner.resources)
        if validation_report.has_blocking_issues():
            error_msg = self.localizer.t("error.validation.failed", 
                                       details=", ".join(str(issue) for issue in validation_report.issues))
            raise ValueError(error_msg)
        
        # Add with security if enabled
        if self.security_manager:
            # Set security level
            self.security_manager.set_task_security_level(task_id, security_level)
            
            # Sign task if required
            if self.security_manager.policy.require_signature:
                self.security_manager.sign_task(task)
            
            # Validate security
            if not self.security_manager.validate_task_for_execution(task):
                raise ValueError(self.localizer.t("security.access.denied"))
        
        # Record compliance data if enabled
        if self.compliance_manager:
            self.compliance_manager.record_data_processing(
                subject_id=None,  # No personal data in task creation
                data_category=DataCategory.TASK_EXECUTION,
                purpose=ProcessingPurpose.TASK_EXECUTION,
                data_fields=["task_id", "name", "priority", "complexity"]
            )
        
        # Add to planner
        self.planner.add_task(task)
        
        logger.info(self.localizer.t("quantum.task.created", name=name))
        return task_id
    
    def create_entanglement(self, task1_id: str, task2_id: str) -> None:
        """Create quantum entanglement between two tasks."""
        self.planner.entangle_tasks(task1_id, task2_id)
        logger.info(f"Created quantum entanglement: {task1_id} <-> {task2_id}")
    
    async def execute_cycle(self) -> Dict[str, Any]:
        """Execute one quantum processing cycle."""
        if not self.is_running:
            raise RuntimeError("Planner is not running")
        
        logger.debug("Starting quantum execution cycle")
        
        # Run system validation
        validation_report = self.system_validator.validate_system(
            self.planner.tasks, self.planner.resources
        )
        
        if validation_report.has_blocking_issues():
            logger.warning(f"System validation found {validation_report.critical_issues} critical issues")
        
        # Execute cycle with monitoring
        results = await self.planner.run_quantum_execution_cycle()
        
        # Add localized status messages
        for executed_task in results["tasks_executed"]:
            task_id = executed_task["task_id"]
            if task_id in self.planner.tasks:
                task_name = self.planner.tasks[task_id].name
                logger.info(self.localizer.t("quantum.task.completed", name=task_name))
        
        for failed_task in results["tasks_failed"]:
            task_id = failed_task["task_id"]
            error = failed_task.get("error", "Unknown error")
            if task_id in self.planner.tasks:
                task_name = self.planner.tasks[task_id].name
                logger.error(self.localizer.t("quantum.task.failed", name=task_name, error=error))
        
        return results
    
    async def execute_until_complete(self, max_cycles: int = 100) -> Dict[str, Any]:
        """Execute cycles until all tasks are complete or max cycles reached."""
        if not self.is_running:
            await self.start()
        
        execution_summary = {
            "start_time": time.time(),
            "cycles_executed": 0,
            "total_tasks_executed": 0,
            "total_tasks_failed": 0,
            "final_state": {}
        }
        
        for cycle in range(max_cycles):
            if not self.planner.get_ready_tasks():
                logger.info(f"All tasks completed after {cycle} cycles")
                break
            
            try:
                cycle_results = await self.execute_cycle()
                execution_summary["cycles_executed"] += 1
                execution_summary["total_tasks_executed"] += len(cycle_results["tasks_executed"])
                execution_summary["total_tasks_failed"] += len(cycle_results["tasks_failed"])
                
                # Brief pause between cycles
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Cycle {cycle} failed: {e}")
                break
        
        execution_summary["end_time"] = time.time()
        execution_summary["total_duration"] = execution_summary["end_time"] - execution_summary["start_time"]
        execution_summary["final_state"] = self.get_system_state()
        
        return execution_summary
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get comprehensive system state."""
        state = self.planner.get_system_state()
        
        # Add additional standalone information
        state["standalone_config"] = {
            "language": self.config.language.value,
            "security_enabled": self.config.enable_security,
            "monitoring_enabled": self.config.enable_monitoring,
            "compliance_enabled": self.config.enable_compliance
        }
        
        # Add health information if available
        if self.health_monitor:
            state["health_status"] = self.health_monitor.get_current_health_status()
        
        # Add security information if available
        if self.security_manager:
            state["security_status"] = self.security_manager.get_security_status()
        
        return state
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "system_state": self.get_system_state(),
            "localization": {
                "language": self.config.language.value,
                "supported_languages": [lang.value for lang in SupportedLanguage]
            }
        }
        
        # Add performance metrics if available
        if hasattr(self.planner, 'get_performance_report'):
            report["performance_metrics"] = self.planner.get_performance_report()
        
        # Add compliance audit if enabled
        if self.compliance_manager:
            report["compliance_audit"] = self.compliance_manager.perform_compliance_audit()
        
        return report
    
    def export_state(self, filename: str) -> None:
        """Export complete system state to file."""
        export_data = {
            "export_timestamp": time.time(),
            "config": {
                "language": self.config.language.value,
                "max_concurrent_tasks": self.config.max_concurrent_tasks,
                "enable_caching": self.config.enable_caching,
                "enable_security": self.config.enable_security
            },
            "system_state": self.get_system_state(),
            "performance_report": self.get_performance_report()
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"System state exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export state to {filename}: {e}")
    
    async def _save_state(self) -> None:
        """Save current state to configured file."""
        if self.config.state_file:
            self.export_state(self.config.state_file)
    
    async def _load_state(self) -> None:
        """Load state from configured file."""
        if not self.config.state_file or not Path(self.config.state_file).exists():
            return
        
        try:
            with open(self.config.state_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            # Restore tasks from saved state
            if "system_state" in saved_data and "tasks" in saved_data["system_state"]:
                tasks_data = saved_data["system_state"]["tasks"]
                
                for task_id, task_info in tasks_data.items():
                    # Recreate task if not already present
                    if task_id not in self.planner.tasks:
                        task = QuantumTask(
                            id=task_id,
                            name=task_info.get("name", f"Restored Task {task_id}"),
                            priority=task_info.get("priority", 1.0),
                            complexity=task_info.get("complexity", 1.0),
                            dependencies=set(task_info.get("dependencies", []))
                        )
                        self.planner.add_task(task)
                
                logger.info(f"Restored {len(tasks_data)} tasks from {self.config.state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load state from {self.config.state_file}: {e}")


# Convenience functions for quick usage
def create_simple_planner(language: SupportedLanguage = SupportedLanguage.ENGLISH) -> StandaloneQuantumPlanner:
    """Create a simple quantum planner with minimal configuration."""
    config = StandaloneConfig(
        language=language,
        enable_security=False,
        enable_monitoring=False,
        enable_compliance=False
    )
    return StandaloneQuantumPlanner(config)


def create_secure_planner(language: SupportedLanguage = SupportedLanguage.ENGLISH) -> StandaloneQuantumPlanner:
    """Create a quantum planner with full security features."""
    config = StandaloneConfig(
        language=language,
        enable_security=True,
        require_signatures=True,
        enable_monitoring=True,
        enable_compliance=True,
        audit_logging=True
    )
    return StandaloneQuantumPlanner(config)


def create_performance_planner(language: SupportedLanguage = SupportedLanguage.ENGLISH) -> StandaloneQuantumPlanner:
    """Create a quantum planner optimized for performance."""
    config = StandaloneConfig(
        language=language,
        max_concurrent_tasks=8,
        enable_caching=True,
        enable_optimization=True,
        enable_monitoring=True
    )
    return StandaloneQuantumPlanner(config)


async def run_simple_workflow(tasks_config: List[Dict[str, Any]], 
                             language: SupportedLanguage = SupportedLanguage.ENGLISH) -> Dict[str, Any]:
    """Run a simple workflow with quantum task planning."""
    
    planner = create_simple_planner(language)
    
    try:
        await planner.start()
        
        # Create tasks from configuration
        for task_config in tasks_config:
            planner.create_task(
                task_id=task_config["id"],
                name=task_config["name"],
                priority=task_config.get("priority", 1.0),
                complexity=task_config.get("complexity", 1.0),
                estimated_duration=task_config.get("duration", 1.0),
                dependencies=set(task_config.get("dependencies", [])),
                resource_requirements=task_config.get("resources", {})
            )
        
        # Create entanglements if specified
        for task_config in tasks_config:
            for entangled_id in task_config.get("entangled_with", []):
                if entangled_id in [t["id"] for t in tasks_config]:
                    planner.create_entanglement(task_config["id"], entangled_id)
        
        # Execute workflow
        results = await planner.execute_until_complete()
        
        return results
        
    finally:
        await planner.stop()


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create a simple workflow
        workflow = [
            {"id": "init", "name": "Initialize", "duration": 0.5},
            {"id": "process", "name": "Process Data", "duration": 1.0, "dependencies": ["init"]},
            {"id": "finalize", "name": "Finalize", "duration": 0.3, "dependencies": ["process"]}
        ]
        
        print("🌌 Running Quantum Task Planner Workflow")
        results = await run_simple_workflow(workflow, SupportedLanguage.ENGLISH)
        
        print(f"✅ Workflow completed in {results['total_duration']:.2f} seconds")
        print(f"📊 Executed {results['total_tasks_executed']} tasks in {results['cycles_executed']} cycles")
    
    asyncio.run(main())