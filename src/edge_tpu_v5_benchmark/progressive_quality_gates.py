"""Progressive Quality Gates for Autonomous SDLC

This module implements progressive quality gates that enforce increasingly strict
quality standards as the project evolves through generations:
- Generation 1: Basic functionality and safety
- Generation 2: Robustness and reliability 
- Generation 3: Performance and scalability optimization
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import sys
import importlib.util

from .quantum_validation import ValidationReport, ValidationIssue, ValidationSeverity
from .security import SecurityScanner
from .monitoring import MetricsCollector
from .performance import PerformanceProfiler
from .exceptions import BenchmarkError, QuantumError

logger = logging.getLogger(__name__)


class Generation(Enum):
    """Development generations for progressive enhancement"""
    GENERATION_1_BASIC = "gen1_basic"
    GENERATION_2_ROBUST = "gen2_robust" 
    GENERATION_3_OPTIMIZED = "gen3_optimized"


class GateStatus(Enum):
    """Quality gate execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class QualityGateResult:
    """Result of a single quality gate execution"""
    gate_name: str
    status: GateStatus
    execution_time: float = 0.0
    message: str = ""
    score: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def is_success(self) -> bool:
        """Check if gate passed successfully"""
        return self.status == GateStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "message": self.message,
            "score": self.score,
            "threshold": self.threshold,
            "details": self.details,
            "error_details": self.error_details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class GenerationReport:
    """Complete report for a generation's quality gates"""
    generation: Generation
    total_gates: int = 0
    passed_gates: int = 0
    failed_gates: int = 0
    skipped_gates: int = 0
    error_gates: int = 0
    execution_time: float = 0.0
    gate_results: List[QualityGateResult] = field(default_factory=list)
    overall_score: Optional[float] = None
    is_passed: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def add_result(self, result: QualityGateResult) -> None:
        """Add a gate result and update counters"""
        self.gate_results.append(result)
        self.total_gates += 1
        
        if result.status == GateStatus.PASSED:
            self.passed_gates += 1
        elif result.status == GateStatus.FAILED:
            self.failed_gates += 1
        elif result.status == GateStatus.SKIPPED:
            self.skipped_gates += 1
        elif result.status == GateStatus.ERROR:
            self.error_gates += 1
    
    def calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_gates == 0:
            return 0.0
        return (self.passed_gates / self.total_gates) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "generation": self.generation.value,
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "skipped_gates": self.skipped_gates,
            "error_gates": self.error_gates,
            "execution_time": self.execution_time,
            "success_rate": self.calculate_success_rate(),
            "gate_results": [result.to_dict() for result in self.gate_results],
            "overall_score": self.overall_score,
            "is_passed": self.is_passed,
            "timestamp": self.timestamp.isoformat()
        }


class QualityGate:
    """Base class for all quality gates"""
    
    def __init__(self, name: str, description: str, threshold: Optional[float] = None):
        self.name = name
        self.description = description
        self.threshold = threshold
    
    async def execute(self) -> QualityGateResult:
        """Execute the quality gate and return result"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing quality gate: {self.name}")
            result = await self._execute_gate()
            result.execution_time = time.time() - start_time
            
            # Determine status based on score and threshold
            if result.status == GateStatus.RUNNING:
                if result.score is not None and self.threshold is not None:
                    result.status = GateStatus.PASSED if result.score >= self.threshold else GateStatus.FAILED
                else:
                    result.status = GateStatus.PASSED
            
            logger.info(f"Gate {self.name} completed: {result.status.value} ({result.execution_time:.2f}s)")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Gate {self.name} failed with error: {e}")
            return QualityGateResult(
                gate_name=self.name,
                status=GateStatus.ERROR,
                execution_time=execution_time,
                message=f"Gate execution failed: {str(e)}",
                error_details=str(e)
            )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _execute_gate")


class BasicSyntaxGate(QualityGate):
    """Gate 1.1: Python syntax validation"""
    
    def __init__(self):
        super().__init__(
            name="basic_syntax",
            description="Validate Python syntax across all source files",
            threshold=100.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check Python syntax for all source files"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            # Find all Python files
            src_files = list(Path("src").rglob("*.py"))
            test_files = list(Path("tests").rglob("*.py"))
            all_files = src_files + test_files
            
            syntax_errors = []
            valid_files = 0
            
            for py_file in all_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(py_file), 'exec')
                    valid_files += 1
                except SyntaxError as e:
                    error_msg = f"{py_file}:{e.lineno}: {e.msg}"
                    syntax_errors.append(error_msg)
                except UnicodeDecodeError as e:
                    error_msg = f"{py_file}: Unicode decode error: {e}"
                    syntax_errors.append(error_msg)
            
            total_files = len(all_files)
            score = (valid_files / total_files * 100) if total_files > 0 else 0
            
            if syntax_errors:
                result.message = f"Found {len(syntax_errors)} syntax errors in {total_files} files"
                result.status = GateStatus.FAILED
            else:
                result.message = f"All {total_files} Python files have valid syntax"
                result.status = GateStatus.PASSED
            
            result.score = score
            result.details = {
                "total_files": total_files,
                "valid_files": valid_files,
                "syntax_errors": syntax_errors[:10],  # Limit to first 10 errors
                "error_count": len(syntax_errors)
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during syntax check: {str(e)}"
            result.error_details = str(e)
            return result


class CriticalImportsGate(QualityGate):
    """Gate 1.2: Critical module imports validation"""
    
    def __init__(self):
        super().__init__(
            name="critical_imports", 
            description="Validate critical modules can be imported",
            threshold=95.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check that critical imports work"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            critical_modules = [
                "edge_tpu_v5_benchmark",
                "edge_tpu_v5_benchmark.benchmark",
                "edge_tpu_v5_benchmark.models",
                "edge_tpu_v5_benchmark.cache",
                "edge_tpu_v5_benchmark.validation",
                "edge_tpu_v5_benchmark.monitoring",
                "edge_tpu_v5_benchmark.security",
                "edge_tpu_v5_benchmark.performance",
                "edge_tpu_v5_benchmark.progressive_quality_gates"
            ]
            
            import_failures = []
            successful_imports = 0
            
            # Add src to path for imports
            src_path = Path("src").absolute()
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            for module_name in critical_modules:
                try:
                    importlib.import_module(module_name)
                    successful_imports += 1
                    logger.debug(f"✓ Successfully imported {module_name}")
                except ImportError as e:
                    import_failures.append(f"{module_name}: {str(e)}")
                    logger.warning(f"✗ Failed to import {module_name}: {e}")
                except Exception as e:
                    import_failures.append(f"{module_name}: Unexpected error: {str(e)}")
                    logger.warning(f"✗ Unexpected error importing {module_name}: {e}")
            
            total_modules = len(critical_modules)
            score = (successful_imports / total_modules * 100) if total_modules > 0 else 0
            
            if import_failures:
                result.message = f"{len(import_failures)} critical imports failed out of {total_modules}"
                result.status = GateStatus.FAILED
            else:
                result.message = f"All {total_modules} critical modules imported successfully"
                result.status = GateStatus.PASSED
            
            result.score = score
            result.details = {
                "total_modules": total_modules,
                "successful_imports": successful_imports,
                "import_failures": import_failures,
                "failure_count": len(import_failures)
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during import check: {str(e)}"
            result.error_details = str(e)
            return result


class BasicTestsGate(QualityGate):
    """Gate 1.3: Basic test execution"""
    
    def __init__(self):
        super().__init__(
            name="basic_tests",
            description="Execute basic unit tests",
            threshold=80.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Run basic tests"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            # Find test files
            test_files = list(Path("tests/unit").glob("test_*.py"))
            
            if not test_files:
                result.message = "No unit test files found"
                result.status = GateStatus.SKIPPED
                result.score = 0.0
                return result
            
            # Try to load test modules
            sys.path.insert(0, str(Path("tests").absolute()))
            sys.path.insert(0, str(Path("src").absolute()))
            
            loadable_tests = 0
            test_errors = []
            
            for test_file in test_files[:5]:  # Limit to first 5 test files
                module_name = test_file.stem
                try:
                    spec = importlib.util.spec_from_file_location(module_name, test_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        loadable_tests += 1
                        logger.debug(f"✓ Successfully loaded test module {module_name}")
                    else:
                        test_errors.append(f"{module_name}: Could not create module spec")
                except Exception as e:
                    test_errors.append(f"{module_name}: {str(e)}")
                    logger.warning(f"✗ Failed to load test {test_file}: {e}")
            
            tested_files = min(len(test_files), 5)
            score = (loadable_tests / tested_files * 100) if tested_files > 0 else 0
            
            if test_errors:
                result.message = f"{len(test_errors)} test modules failed to load out of {tested_files}"
                if score < self.threshold:
                    result.status = GateStatus.FAILED
                else:
                    result.status = GateStatus.PASSED
            else:
                result.message = f"All {tested_files} test modules loaded successfully"
                result.status = GateStatus.PASSED
            
            result.score = score
            result.details = {
                "total_test_files": len(test_files),
                "tested_files": tested_files,
                "loadable_tests": loadable_tests,
                "test_errors": test_errors[:5],  # Limit errors shown
                "error_count": len(test_errors)
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during test execution: {str(e)}"
            result.error_details = str(e)
            return result


class ProjectStructureGate(QualityGate):
    """Gate 1.4: Project structure validation"""
    
    def __init__(self):
        super().__init__(
            name="project_structure",
            description="Validate project directory structure",
            threshold=90.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check project structure"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            required_paths = [
                "src/edge_tpu_v5_benchmark",
                "tests/unit",
                "tests/integration",
                "README.md",
                "pyproject.toml",
                "LICENSE"
            ]
            
            missing_paths = []
            existing_paths = 0
            
            for path_str in required_paths:
                path = Path(path_str)
                if path.exists():
                    existing_paths += 1
                    logger.debug(f"✓ Found required path: {path_str}")
                else:
                    missing_paths.append(path_str)
                    logger.warning(f"✗ Missing required path: {path_str}")
            
            total_paths = len(required_paths)
            score = (existing_paths / total_paths * 100) if total_paths > 0 else 0
            
            if missing_paths:
                result.message = f"{len(missing_paths)} required paths missing out of {total_paths}"
                if score < self.threshold:
                    result.status = GateStatus.FAILED
                else:
                    result.status = GateStatus.PASSED
            else:
                result.message = f"All {total_paths} required paths exist"
                result.status = GateStatus.PASSED
            
            result.score = score
            result.details = {
                "total_paths": total_paths,
                "existing_paths": existing_paths,
                "missing_paths": missing_paths,
                "missing_count": len(missing_paths)
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during structure check: {str(e)}"
            result.error_details = str(e)
            return result


class ProgressiveQualityGateRunner:
    """Main runner for progressive quality gates"""
    
    def __init__(self):
        self.gates_by_generation = {
            Generation.GENERATION_1_BASIC: [
                BasicSyntaxGate(),
                CriticalImportsGate(), 
                BasicTestsGate(),
                ProjectStructureGate()
            ],
            Generation.GENERATION_2_ROBUST: [
                # Will be implemented in Generation 2
            ],
            Generation.GENERATION_3_OPTIMIZED: [
                # Will be implemented in Generation 3
            ]
        }
        self.execution_history: List[GenerationReport] = []
    
    async def run_generation(self, generation: Generation) -> GenerationReport:
        """Run all quality gates for a specific generation"""
        logger.info(f"Starting quality gates for {generation.value}")
        start_time = time.time()
        
        report = GenerationReport(generation=generation)
        gates = self.gates_by_generation.get(generation, [])
        
        if not gates:
            logger.warning(f"No gates defined for {generation.value}")
            report.is_passed = True
            return report
        
        # Execute gates sequentially for now (can parallelize later)
        for gate in gates:
            result = await gate.execute()
            report.add_result(result)
        
        # Calculate overall results
        report.execution_time = time.time() - start_time
        success_rate = report.calculate_success_rate()
        
        # Determine if generation passed (need at least 85% success rate)
        report.is_passed = success_rate >= 85.0 and report.failed_gates == 0
        report.overall_score = success_rate
        
        # Log summary
        logger.info(f"Generation {generation.value} completed:")
        logger.info(f"  Total gates: {report.total_gates}")
        logger.info(f"  Passed: {report.passed_gates}")
        logger.info(f"  Failed: {report.failed_gates}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Overall result: {'PASSED' if report.is_passed else 'FAILED'}")
        
        self.execution_history.append(report)
        return report
    
    async def run_all_generations(self) -> List[GenerationReport]:
        """Run all generations sequentially"""
        reports = []
        
        for generation in [Generation.GENERATION_1_BASIC, Generation.GENERATION_2_ROBUST, Generation.GENERATION_3_OPTIMIZED]:
            report = await self.run_generation(generation)
            reports.append(report)
            
            # Stop if generation failed (unless it's gen 2 or 3 with no gates)
            if not report.is_passed and report.total_gates > 0:
                logger.error(f"Generation {generation.value} failed - stopping progression")
                break
        
        return reports
    
    def save_report(self, reports: List[GenerationReport], output_path: Path) -> None:
        """Save execution reports to JSON file"""
        try:
            report_data = {
                "execution_timestamp": datetime.now().isoformat(),
                "total_generations": len(reports),
                "generation_reports": [report.to_dict() for report in reports]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Quality gate reports saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save reports: {e}")


async def run_progressive_quality_gates() -> List[GenerationReport]:
    """Main entry point for running progressive quality gates"""
    runner = ProgressiveQualityGateRunner()
    reports = await runner.run_all_generations()
    
    # Save reports
    output_path = Path("quality_gate_reports.json")
    runner.save_report(reports, output_path)
    
    return reports


if __name__ == "__main__":
    asyncio.run(run_progressive_quality_gates())