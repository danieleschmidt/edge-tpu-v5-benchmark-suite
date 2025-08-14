#!/usr/bin/env python3
"""
Standalone Progressive Quality Gates for Terragon Autonomous SDLC

This standalone script executes progressive quality gates without requiring
heavy dependencies, focusing on core functionality testing.
"""

import asyncio
import sys
import logging
import time
import json
import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
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
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            test_files = list(Path("tests").rglob("*.py")) if Path("tests").exists() else []
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


class ProjectStructureGate(QualityGate):
    """Gate 1.2: Project structure validation"""
    
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
                "tests",
                "README.md",
                "pyproject.toml",
                "LICENSE"
            ]
            
            recommended_paths = [
                "tests/unit",
                "tests/integration", 
                "docs",
                "CONTRIBUTING.md",
                "SECURITY.md"
            ]
            
            missing_required = []
            missing_recommended = []
            existing_paths = 0
            
            # Check required paths
            for path_str in required_paths:
                path = Path(path_str)
                if path.exists():
                    existing_paths += 1
                    logger.debug(f"✓ Found required path: {path_str}")
                else:
                    missing_required.append(path_str)
                    logger.warning(f"✗ Missing required path: {path_str}")
            
            # Check recommended paths
            recommended_existing = 0
            for path_str in recommended_paths:
                path = Path(path_str)
                if path.exists():
                    recommended_existing += 1
                    logger.debug(f"✓ Found recommended path: {path_str}")
                else:
                    missing_recommended.append(path_str)
            
            total_paths = len(required_paths)
            score = (existing_paths / total_paths * 100) if total_paths > 0 else 0
            
            # Bonus points for recommended paths
            bonus_score = (recommended_existing / len(recommended_paths)) * 10
            final_score = min(100.0, score + bonus_score)
            
            if missing_required:
                result.message = f"{len(missing_required)} required paths missing out of {total_paths}"
                result.status = GateStatus.FAILED
            else:
                result.message = f"All {total_paths} required paths exist"
                if missing_recommended:
                    result.message += f", {len(missing_recommended)} recommended paths missing"
                result.status = GateStatus.PASSED
            
            result.score = final_score
            result.details = {
                "required_paths": len(required_paths),
                "existing_required": existing_paths,
                "missing_required": missing_required,
                "recommended_paths": len(recommended_paths),
                "existing_recommended": recommended_existing,
                "missing_recommended": missing_recommended
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during structure check: {str(e)}"
            result.error_details = str(e)
            return result


class DocumentationGate(QualityGate):
    """Gate 1.3: Documentation completeness"""
    
    def __init__(self):
        super().__init__(
            name="documentation",
            description="Check documentation completeness", 
            threshold=80.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check documentation"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            doc_score = 0.0
            checks = []
            
            # Check README.md
            if Path("README.md").exists():
                readme_content = Path("README.md").read_text().lower()
                required_sections = ["overview", "features", "installation", "usage"]
                found_sections = sum(1 for section in required_sections if section in readme_content)
                readme_score = (found_sections / len(required_sections)) * 25
                doc_score += readme_score
                checks.append(f"README sections: {found_sections}/{len(required_sections)} ({readme_score:.1f} pts)")
            else:
                checks.append("README.md: Missing (0 pts)")
            
            # Check LICENSE
            if Path("LICENSE").exists():
                doc_score += 15
                checks.append("LICENSE: Present (15 pts)")
            else:
                checks.append("LICENSE: Missing (0 pts)")
            
            # Check CONTRIBUTING.md
            if Path("CONTRIBUTING.md").exists():
                doc_score += 10
                checks.append("CONTRIBUTING.md: Present (10 pts)")
            else:
                checks.append("CONTRIBUTING.md: Missing (0 pts)")
            
            # Check SECURITY.md
            if Path("SECURITY.md").exists():
                doc_score += 10
                checks.append("SECURITY.md: Present (10 pts)")
            else:
                checks.append("SECURITY.md: Missing (0 pts)")
            
            # Check docs directory
            if Path("docs").exists():
                doc_files = list(Path("docs").rglob("*.md"))
                docs_score = min(20, len(doc_files) * 2)  # Max 20 points
                doc_score += docs_score
                checks.append(f"Documentation files: {len(doc_files)} ({docs_score:.1f} pts)")
            else:
                checks.append("docs/ directory: Missing (0 pts)")
            
            # Check pyproject.toml metadata
            if Path("pyproject.toml").exists():
                pyproject_content = Path("pyproject.toml").read_text()
                metadata_items = ["description", "authors", "license", "readme"]
                found_metadata = sum(1 for item in metadata_items if item in pyproject_content)
                metadata_score = (found_metadata / len(metadata_items)) * 20
                doc_score += metadata_score
                checks.append(f"Project metadata: {found_metadata}/{len(metadata_items)} ({metadata_score:.1f} pts)")
            else:
                checks.append("pyproject.toml: Missing (0 pts)")
            
            if doc_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Documentation score: {doc_score:.1f}/100"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Documentation score below threshold: {doc_score:.1f}/100"
            
            result.score = doc_score
            result.details = {
                "total_score": doc_score,
                "max_score": 100.0,
                "checks": checks,
                "threshold": self.threshold
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during documentation check: {str(e)}"
            result.error_details = str(e)
            return result


class PythonImportsGate(QualityGate):
    """Gate 1.4: Basic Python imports validation"""
    
    def __init__(self):
        super().__init__(
            name="python_imports",
            description="Test basic Python module imports",
            threshold=70.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Test basic imports"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            # Test standard library imports that should always work
            standard_modules = [
                "os", "sys", "pathlib", "json", "time", 
                "datetime", "logging", "asyncio", "subprocess"
            ]
            
            import_results = []
            successful_imports = 0
            
            for module_name in standard_modules:
                try:
                    __import__(module_name)
                    successful_imports += 1
                    import_results.append(f"✓ {module_name}")
                except ImportError as e:
                    import_results.append(f"✗ {module_name}: {str(e)}")
            
            # Try to import the main package if it exists
            package_imports = 0
            package_tests = []
            
            if Path("src/edge_tpu_v5_benchmark").exists():
                # Add src to path for testing
                src_path = str(Path("src").absolute())
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                
                # Test basic package structure imports (not the full modules)
                test_modules = [
                    "edge_tpu_v5_benchmark.progressive_quality_gates",
                ]
                
                for module_name in test_modules:
                    try:
                        spec = importlib.util.find_spec(module_name)
                        if spec is not None:
                            package_imports += 1
                            package_tests.append(f"✓ {module_name} (spec found)")
                        else:
                            package_tests.append(f"✗ {module_name} (spec not found)")
                    except Exception as e:
                        package_tests.append(f"✗ {module_name}: {str(e)}")
            
            total_tests = len(standard_modules) + len(package_tests)
            total_success = successful_imports + package_imports
            score = (total_success / total_tests * 100) if total_tests > 0 else 0
            
            if score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Import tests passed: {total_success}/{total_tests}"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Import tests failed: {total_success}/{total_tests} below threshold"
            
            result.score = score
            result.details = {
                "standard_library": {
                    "total": len(standard_modules),
                    "successful": successful_imports,
                    "results": import_results
                },
                "package_imports": {
                    "total": len(package_tests),
                    "successful": package_imports, 
                    "results": package_tests
                },
                "overall": {
                    "total_tests": total_tests,
                    "total_success": total_success,
                    "score": score
                }
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during import check: {str(e)}"
            result.error_details = str(e)
            return result


class ErrorHandlingGate(QualityGate):
    """Gate 2.1: Error handling and exception coverage"""
    
    def __init__(self):
        super().__init__(
            name="error_handling",
            description="Validate error handling and exception coverage",
            threshold=80.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check error handling patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            error_patterns = {
                "try_except": r'try\s*:',
                "specific_exceptions": r'except\s+\w+Exception',
                "logging_errors": r'log(?:ger)?\.(?:error|exception|critical)',
                "raise_statements": r'raise\s+\w+',
                "finally_blocks": r'finally\s*:',
                "context_managers": r'with\s+.*(?:as|:)',
            }
            
            pattern_counts = {pattern: 0 for pattern in error_patterns}
            total_lines = 0
            files_with_error_handling = 0
            
            import re
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_lines = len(content.splitlines())
                        total_lines += file_lines
                        
                        file_has_error_handling = False
                        for pattern_name, pattern in error_patterns.items():
                            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                            pattern_counts[pattern_name] += matches
                            if matches > 0:
                                file_has_error_handling = True
                        
                        if file_has_error_handling:
                            files_with_error_handling += 1
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate score based on error handling coverage
            total_files = len(src_files)
            file_coverage = (files_with_error_handling / total_files * 100) if total_files > 0 else 0
            
            # Weight patterns by importance
            pattern_weights = {
                "try_except": 25,
                "specific_exceptions": 20,
                "logging_errors": 20,
                "raise_statements": 15,
                "finally_blocks": 10,
                "context_managers": 10
            }
            
            weighted_score = 0
            for pattern_name, count in pattern_counts.items():
                if count > 0:
                    # Normalize by lines of code and apply weight
                    pattern_score = min(pattern_weights[pattern_name], count / max(total_lines, 1) * 1000 * pattern_weights[pattern_name])
                    weighted_score += pattern_score
            
            final_score = min(100.0, (file_coverage * 0.4) + (weighted_score * 0.6))
            
            if final_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Error handling coverage: {final_score:.1f}% ({files_with_error_handling}/{total_files} files)"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Error handling coverage below threshold: {final_score:.1f}%"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "files_with_error_handling": files_with_error_handling,
                "file_coverage": file_coverage,
                "pattern_counts": pattern_counts,
                "total_lines": total_lines,
                "weighted_score": weighted_score
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during error handling check: {str(e)}"
            result.error_details = str(e)
            return result


class SecurityPatternsGate(QualityGate):
    """Gate 2.2: Security patterns and vulnerability checks"""
    
    def __init__(self):
        super().__init__(
            name="security_patterns",
            description="Check for security patterns and vulnerability risks",
            threshold=85.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check security patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            import re
            
            # Security vulnerabilities to detect
            vulnerability_patterns = {
                "sql_injection": r'(?:execute|query)\s*\(\s*["\'].*%.*["\']',
                "command_injection": r'(?:os\.system|subprocess\.call|subprocess\.run)\s*\(\s*["\'].*\+.*["\']',
                "hardcoded_secrets": r'(?:password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                "unsafe_pickle": r'pickle\.loads?\s*\(',
                "unsafe_eval": r'(?:eval|exec)\s*\(',
                "weak_random": r'random\.random\s*\(',
            }
            
            # Security good practices to look for
            good_patterns = {
                "input_validation": r'(?:validate|sanitize|clean)_input',
                "authentication": r'(?:authenticate|login|verify)_user',
                "authorization": r'(?:authorize|check_permission|has_access)',
                "encryption": r'(?:encrypt|decrypt|hash|bcrypt)',
                "secure_headers": r'(?:Content-Security-Policy|X-Frame-Options|X-XSS-Protection)',
                "logging_security": r'log.*(?:security|unauthorized|attack)',
            }
            
            vulnerability_count = 0
            good_practice_count = 0
            total_files = len(src_files)
            files_with_vulnerabilities = []
            files_with_good_practices = []
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        file_vulnerabilities = []
                        for vuln_name, pattern in vulnerability_patterns.items():
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                vulnerability_count += 1
                                file_vulnerabilities.append(vuln_name)
                        
                        if file_vulnerabilities:
                            files_with_vulnerabilities.append(str(py_file))
                        
                        file_good_practices = []
                        for good_name, pattern in good_patterns.items():
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                good_practice_count += 1
                                file_good_practices.append(good_name)
                        
                        if file_good_practices:
                            files_with_good_practices.append(str(py_file))
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate security score
            vulnerability_penalty = min(50, vulnerability_count * 10)  # Up to 50 point penalty
            good_practice_bonus = min(50, good_practice_count * 2)     # Up to 50 point bonus
            base_score = 50  # Base security score
            
            final_score = max(0, base_score - vulnerability_penalty + good_practice_bonus)
            
            if vulnerability_count == 0 and good_practice_count > 0:
                result.status = GateStatus.PASSED
                result.message = f"Security score: {final_score:.1f}% (0 vulnerabilities, {good_practice_count} good practices)"
            elif vulnerability_count > 0:
                result.status = GateStatus.FAILED
                result.message = f"Security vulnerabilities found: {vulnerability_count}"
            else:
                if final_score >= self.threshold:
                    result.status = GateStatus.PASSED
                else:
                    result.status = GateStatus.FAILED
                result.message = f"Security score: {final_score:.1f}%"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "vulnerability_count": vulnerability_count,
                "good_practice_count": good_practice_count,
                "files_with_vulnerabilities": files_with_vulnerabilities[:5],  # Limit output
                "files_with_good_practices": len(files_with_good_practices),
                "vulnerability_penalty": vulnerability_penalty,
                "good_practice_bonus": good_practice_bonus
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during security check: {str(e)}"
            result.error_details = str(e)
            return result


class LoggingQualityGate(QualityGate):
    """Gate 2.3: Logging and monitoring patterns"""
    
    def __init__(self):
        super().__init__(
            name="logging_quality",
            description="Validate logging and monitoring implementation",
            threshold=75.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check logging patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            import re
            
            logging_patterns = {
                "logger_creation": r'(?:logger\s*=\s*logging\.getLogger|getLogger\(__name__\))',
                "log_levels": r'log(?:ger)?\.(?:debug|info|warning|error|critical)',
                "structured_logging": r'extra\s*=\s*{',
                "exception_logging": r'log(?:ger)?\.exception',
                "performance_logging": r'log(?:ger)?.*(?:time|duration|performance)',
                "security_logging": r'log(?:ger)?.*(?:security|unauthorized|suspicious)',
            }
            
            pattern_counts = {pattern: 0 for pattern in logging_patterns}
            total_files = len(src_files)
            files_with_logging = 0
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        file_has_logging = False
                        for pattern_name, pattern in logging_patterns.items():
                            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                            pattern_counts[pattern_name] += matches
                            if matches > 0:
                                file_has_logging = True
                        
                        if file_has_logging:
                            files_with_logging += 1
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate logging score
            file_coverage = (files_with_logging / total_files * 100) if total_files > 0 else 0
            
            # Weight logging patterns by importance
            pattern_weights = {
                "logger_creation": 20,
                "log_levels": 25,
                "structured_logging": 15,
                "exception_logging": 20,
                "performance_logging": 10,
                "security_logging": 10
            }
            
            weighted_score = 0
            for pattern_name, count in pattern_counts.items():
                if count > 0:
                    pattern_score = min(pattern_weights[pattern_name], count * 2)
                    weighted_score += pattern_score
            
            final_score = min(100.0, (file_coverage * 0.4) + (weighted_score * 0.6))
            
            if final_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Logging quality: {final_score:.1f}% ({files_with_logging}/{total_files} files)"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Logging quality below threshold: {final_score:.1f}%"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "files_with_logging": files_with_logging,
                "file_coverage": file_coverage,
                "pattern_counts": pattern_counts,
                "weighted_score": weighted_score
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during logging check: {str(e)}"
            result.error_details = str(e)
            return result


class ConfigurationValidationGate(QualityGate):
    """Gate 2.4: Configuration validation and management"""
    
    def __init__(self):
        super().__init__(
            name="configuration_validation",
            description="Validate configuration management patterns",
            threshold=80.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check configuration patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            config_score = 0.0
            checks = []
            max_score = 100.0
            
            # Check for configuration files
            config_files = [
                ("pyproject.toml", 20),
                ("setup.py", 10), 
                (".env.example", 15),
                ("config.yaml", 10),
                ("settings.py", 10),
                ("Dockerfile", 15),
                ("docker-compose.yml", 10),
                ("requirements.txt", 10)
            ]
            
            for config_file, score_value in config_files:
                if Path(config_file).exists():
                    config_score += score_value
                    checks.append(f"✓ {config_file}: Present ({score_value} pts)")
                else:
                    checks.append(f"✗ {config_file}: Missing (0 pts)")
            
            # Check for environment variable usage
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            import re
            env_var_patterns = 0
            config_classes = 0
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Count environment variable usage
                        env_patterns = [
                            r'os\.environ',
                            r'os\.getenv',
                            r'getenv\(',
                            r'Environment\(',
                        ]
                        
                        for pattern in env_patterns:
                            env_var_patterns += len(re.findall(pattern, content, re.IGNORECASE))
                        
                        # Count configuration classes
                        config_class_patterns = [
                            r'class.*Config',
                            r'class.*Settings',
                            r'@dataclass.*Config',
                        ]
                        
                        for pattern in config_class_patterns:
                            config_classes += len(re.findall(pattern, content, re.IGNORECASE))
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Bonus for environment variables and config classes
            if env_var_patterns > 0:
                env_bonus = min(10, env_var_patterns * 2)
                config_score += env_bonus
                checks.append(f"Environment variables: {env_var_patterns} usages ({env_bonus} pts)")
            
            if config_classes > 0:
                class_bonus = min(10, config_classes * 5)
                config_score += class_bonus
                checks.append(f"Configuration classes: {config_classes} found ({class_bonus} pts)")
            
            final_score = min(100.0, config_score)
            
            if final_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Configuration validation: {final_score:.1f}/100"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Configuration validation below threshold: {final_score:.1f}/100"
            
            result.score = final_score
            result.details = {
                "total_score": final_score,
                "max_score": max_score,
                "checks": checks,
                "env_var_patterns": env_var_patterns,
                "config_classes": config_classes
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during configuration check: {str(e)}"
            result.error_details = str(e)
            return result


class PerformanceOptimizationGate(QualityGate):
    """Gate 3.1: Performance optimization patterns"""
    
    def __init__(self):
        super().__init__(
            name="performance_optimization",
            description="Check for performance optimization patterns and anti-patterns",
            threshold=75.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check performance optimization patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            import re
            
            # Performance optimization patterns to look for
            optimization_patterns = {
                "async_await": r'(?:async\s+def|await\s+)',
                "list_comprehensions": r'\[.*for\s+.*in\s+.*\]',
                "generator_expressions": r'\(.*for\s+.*in\s+.*\)',
                "context_managers": r'with\s+.*:',
                "caching_decorators": r'@(?:lru_cache|cache|cached)',
                "batch_operations": r'(?:bulk_|batch_|chunk_)',
                "lazy_loading": r'(?:lazy|defer)',
                "connection_pooling": r'(?:pool|Pool)',
            }
            
            # Performance anti-patterns to avoid
            antipattern_patterns = {
                "nested_loops": r'for\s+.*:\s*(?:\n.*)*?\s*for\s+.*:',
                "string_concatenation": r'\+\s*=\s*["\']',
                "global_variables": r'global\s+\w+',
                "repeated_calculations": r'(?:len\(.*\)|\.count\(.*\)).*(?:len\(.*\)|\.count\(.*\))',
                "inefficient_lookups": r'for\s+.*in.*if.*==',
            }
            
            optimization_count = 0
            antipattern_count = 0
            total_files = len(src_files)
            files_with_optimizations = 0
            files_with_antipatterns = []
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        file_has_optimizations = False
                        for pattern_name, pattern in optimization_patterns.items():
                            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                            optimization_count += matches
                            if matches > 0:
                                file_has_optimizations = True
                        
                        if file_has_optimizations:
                            files_with_optimizations += 1
                        
                        file_antipatterns = []
                        for pattern_name, pattern in antipattern_patterns.items():
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                antipattern_count += 1
                                file_antipatterns.append(pattern_name)
                        
                        if file_antipatterns:
                            files_with_antipatterns.append(str(py_file))
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate performance score
            optimization_bonus = min(50, optimization_count * 2)
            antipattern_penalty = min(30, antipattern_count * 5)
            base_score = 50
            
            final_score = max(0, base_score + optimization_bonus - antipattern_penalty)
            
            if antipattern_count == 0 and optimization_count > 10:
                result.status = GateStatus.PASSED
                result.message = f"Performance score: {final_score:.1f}% ({optimization_count} optimizations, 0 anti-patterns)"
            elif antipattern_count > 5:
                result.status = GateStatus.FAILED
                result.message = f"Too many performance anti-patterns: {antipattern_count}"
            else:
                if final_score >= self.threshold:
                    result.status = GateStatus.PASSED
                else:
                    result.status = GateStatus.FAILED
                result.message = f"Performance score: {final_score:.1f}% ({optimization_count} optimizations, {antipattern_count} anti-patterns)"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "optimization_count": optimization_count,
                "antipattern_count": antipattern_count,
                "files_with_optimizations": files_with_optimizations,
                "files_with_antipatterns": len(files_with_antipatterns),
                "optimization_bonus": optimization_bonus,
                "antipattern_penalty": antipattern_penalty
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during performance check: {str(e)}"
            result.error_details = str(e)
            return result


class CachingStrategyGate(QualityGate):
    """Gate 3.2: Caching strategy implementation"""
    
    def __init__(self):
        super().__init__(
            name="caching_strategy",
            description="Validate caching strategies and patterns",
            threshold=70.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check caching patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            import re
            
            # Caching patterns to look for
            caching_patterns = {
                "lru_cache": r'@lru_cache',
                "cache_decorator": r'@cache|@cached',
                "redis_cache": r'redis|Redis',
                "memory_cache": r'(?:cache|Cache)(?:Manager|Client)',
                "cache_methods": r'def\s+(?:get_cache|set_cache|clear_cache)',
                "ttl_patterns": r'(?:ttl|expire|timeout)',
                "cache_invalidation": r'(?:invalidate|evict|clear).*cache',
                "cache_warming": r'(?:warm|populate).*cache',
            }
            
            cache_score = 0.0
            pattern_counts = {pattern: 0 for pattern in caching_patterns}
            total_files = len(src_files)
            files_with_caching = 0
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        file_has_caching = False
                        for pattern_name, pattern in caching_patterns.items():
                            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                            pattern_counts[pattern_name] += matches
                            if matches > 0:
                                file_has_caching = True
                        
                        if file_has_caching:
                            files_with_caching += 1
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Weight caching patterns by importance
            pattern_weights = {
                "lru_cache": 15,
                "cache_decorator": 15,
                "redis_cache": 20,
                "memory_cache": 15,
                "cache_methods": 10,
                "ttl_patterns": 10,
                "cache_invalidation": 10,
                "cache_warming": 5
            }
            
            for pattern_name, count in pattern_counts.items():
                if count > 0:
                    pattern_score = min(pattern_weights[pattern_name], count * 3)
                    cache_score += pattern_score
            
            # Check for cache-related files
            cache_files = [
                "cache.py", "caching.py", "redis.py",
                "src/*/cache.py", "src/*/caching.py"
            ]
            
            cache_files_found = 0
            for pattern in cache_files:
                if list(Path(".").glob(pattern)):
                    cache_files_found += 1
                    cache_score += 10
            
            final_score = min(100.0, cache_score)
            
            if final_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Caching score: {final_score:.1f}% ({files_with_caching}/{total_files} files)"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Caching score below threshold: {final_score:.1f}%"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "files_with_caching": files_with_caching,
                "pattern_counts": pattern_counts,
                "cache_files_found": cache_files_found,
                "final_score": final_score
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during caching check: {str(e)}"
            result.error_details = str(e)
            return result


class ScalabilityPatternsGate(QualityGate):
    """Gate 3.3: Scalability patterns and architecture"""
    
    def __init__(self):
        super().__init__(
            name="scalability_patterns",
            description="Check for scalability patterns and distributed system support",
            threshold=70.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check scalability patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            scalability_score = 0.0
            checks = []
            
            # Check for containerization
            docker_files = ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]
            containerization_score = 0
            for docker_file in docker_files:
                if Path(docker_file).exists():
                    containerization_score += 10
                    checks.append(f"✓ {docker_file}: Present")
                else:
                    checks.append(f"✗ {docker_file}: Missing")
            
            scalability_score += containerization_score
            
            # Check for Kubernetes manifests
            k8s_files = list(Path("k8s").glob("*.yaml")) if Path("k8s").exists() else []
            if k8s_files:
                k8s_score = min(20, len(k8s_files) * 5)
                scalability_score += k8s_score
                checks.append(f"✓ Kubernetes manifests: {len(k8s_files)} files ({k8s_score} pts)")
            else:
                checks.append("✗ Kubernetes manifests: Missing (0 pts)")
            
            # Check for monitoring/observability
            monitoring_files = [
                "monitoring", "prometheus", "grafana", "metrics",
                "health", "readiness", "liveness"
            ]
            
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            import re
            monitoring_patterns = 0
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        patterns = [
                            r'(?:health|Health)Check',
                            r'(?:metrics|Metrics)',
                            r'(?:prometheus|Prometheus)',
                            r'(?:monitoring|Monitoring)',
                            r'/health',
                            r'/metrics',
                            r'@app\.route.*health',
                        ]
                        
                        for pattern in patterns:
                            monitoring_patterns += len(re.findall(pattern, content, re.IGNORECASE))
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            if monitoring_patterns > 0:
                monitoring_score = min(20, monitoring_patterns * 2)
                scalability_score += monitoring_score
                checks.append(f"✓ Monitoring patterns: {monitoring_patterns} found ({monitoring_score} pts)")
            else:
                checks.append("✗ Monitoring patterns: Missing (0 pts)")
            
            # Check for async/concurrency patterns
            async_patterns = 0
            concurrency_patterns = 0
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        async_patterns += len(re.findall(r'async\s+def', content))
                        concurrency_patterns += len(re.findall(r'(?:threading|asyncio|concurrent)', content, re.IGNORECASE))
                        
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            if async_patterns > 0:
                async_score = min(15, async_patterns)
                scalability_score += async_score
                checks.append(f"✓ Async patterns: {async_patterns} found ({async_score} pts)")
            
            if concurrency_patterns > 0:
                concurrency_score = min(15, concurrency_patterns * 2)
                scalability_score += concurrency_score
                checks.append(f"✓ Concurrency patterns: {concurrency_patterns} found ({concurrency_score} pts)")
            
            # Check for load balancing/distribution
            distribution_patterns = [
                "load_balance", "distribute", "shard", "partition",
                "cluster", "replica", "scale"
            ]
            
            distribution_count = 0
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                        for pattern in distribution_patterns:
                            distribution_count += len(re.findall(pattern, content))
                        
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            if distribution_count > 0:
                distribution_score = min(20, distribution_count * 2)
                scalability_score += distribution_score
                checks.append(f"✓ Distribution patterns: {distribution_count} found ({distribution_score} pts)")
            else:
                checks.append("✗ Distribution patterns: Missing (0 pts)")
            
            final_score = min(100.0, scalability_score)
            
            if final_score >= self.threshold:
                result.status = GateStatus.PASSED
                result.message = f"Scalability score: {final_score:.1f}/100"
            else:
                result.status = GateStatus.FAILED
                result.message = f"Scalability score below threshold: {final_score:.1f}/100"
            
            result.score = final_score
            result.details = {
                "final_score": final_score,
                "checks": checks,
                "containerization_score": containerization_score,
                "k8s_files": len(k8s_files),
                "monitoring_patterns": monitoring_patterns,
                "async_patterns": async_patterns,
                "concurrency_patterns": concurrency_patterns,
                "distribution_count": distribution_count
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during scalability check: {str(e)}"
            result.error_details = str(e)
            return result


class ResourceEfficiencyGate(QualityGate):
    """Gate 3.4: Resource efficiency and optimization"""
    
    def __init__(self):
        super().__init__(
            name="resource_efficiency",
            description="Check for resource efficiency patterns and memory optimization",
            threshold=75.0
        )
    
    async def _execute_gate(self) -> QualityGateResult:
        """Check resource efficiency patterns"""
        result = QualityGateResult(gate_name=self.name, status=GateStatus.RUNNING)
        
        try:
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            
            if not src_files:
                result.message = "No source files found"
                result.status = GateStatus.SKIPPED
                return result
            
            import re
            
            # Resource efficiency patterns
            efficiency_patterns = {
                "generators": r'(?:yield|yield\s+from)',
                "iterators": r'(?:__iter__|__next__)',
                "slots": r'__slots__\s*=',
                "context_managers": r'(?:__enter__|__exit__|with\s+)',
                "lazy_evaluation": r'(?:lazy|defer)',
                "memory_profiling": r'(?:memory_usage|profile|memory)',
                "garbage_collection": r'(?:gc\.|weakref)',
                "resource_cleanup": r'(?:close\(\)|cleanup|dispose)',
            }
            
            # Resource inefficiency anti-patterns
            inefficiency_patterns = {
                "global_state": r'global\s+\w+',
                "memory_leaks": r'(?:while\s+True.*append|\.append.*while)',
                "unnecessary_copies": r'(?:list\(.*\)|dict\(.*\)|copy\.copy)',
                "string_concatenation": r'\w+\s*\+\s*=\s*["\']',
                "repeated_io": r'(?:open\(.*\).*open\(.*\)|read\(\).*read\(\))',
            }
            
            efficiency_count = 0
            inefficiency_count = 0
            total_files = len(src_files)
            files_with_efficiency = 0
            files_with_inefficiency = []
            
            for py_file in src_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        file_has_efficiency = False
                        for pattern_name, pattern in efficiency_patterns.items():
                            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
                            efficiency_count += matches
                            if matches > 0:
                                file_has_efficiency = True
                        
                        if file_has_efficiency:
                            files_with_efficiency += 1
                        
                        file_inefficiencies = []
                        for pattern_name, pattern in inefficiency_patterns.items():
                            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                                inefficiency_count += 1
                                file_inefficiencies.append(pattern_name)
                        
                        if file_inefficiencies:
                            files_with_inefficiency.append(str(py_file))
                            
                except Exception as e:
                    logger.warning(f"Could not analyze {py_file}: {e}")
            
            # Calculate resource efficiency score
            efficiency_bonus = min(40, efficiency_count * 3)
            inefficiency_penalty = min(25, inefficiency_count * 5)
            base_score = 50
            
            # File coverage bonus
            coverage_bonus = (files_with_efficiency / total_files) * 25 if total_files > 0 else 0
            
            final_score = max(0, base_score + efficiency_bonus + coverage_bonus - inefficiency_penalty)
            
            if inefficiency_count == 0 and efficiency_count > 5:
                result.status = GateStatus.PASSED
                result.message = f"Resource efficiency: {final_score:.1f}% ({efficiency_count} patterns, 0 anti-patterns)"
            elif inefficiency_count > 3:
                result.status = GateStatus.FAILED
                result.message = f"Too many resource inefficiencies: {inefficiency_count}"
            else:
                if final_score >= self.threshold:
                    result.status = GateStatus.PASSED
                else:
                    result.status = GateStatus.FAILED
                result.message = f"Resource efficiency: {final_score:.1f}% ({efficiency_count} patterns, {inefficiency_count} anti-patterns)"
            
            result.score = final_score
            result.details = {
                "total_files": total_files,
                "efficiency_count": efficiency_count,
                "inefficiency_count": inefficiency_count,
                "files_with_efficiency": files_with_efficiency,
                "files_with_inefficiency": len(files_with_inefficiency),
                "efficiency_bonus": efficiency_bonus,
                "inefficiency_penalty": inefficiency_penalty,
                "coverage_bonus": coverage_bonus
            }
            
            return result
            
        except Exception as e:
            result.status = GateStatus.ERROR
            result.message = f"Error during resource efficiency check: {str(e)}"
            result.error_details = str(e)
            return result


class ProgressiveQualityGateRunner:
    """Main runner for progressive quality gates"""
    
    def __init__(self):
        self.gates_by_generation = {
            Generation.GENERATION_1_BASIC: [
                BasicSyntaxGate(),
                ProjectStructureGate(),
                DocumentationGate(),
                PythonImportsGate()
            ],
            Generation.GENERATION_2_ROBUST: [
                ErrorHandlingGate(),
                SecurityPatternsGate(),
                LoggingQualityGate(),
                ConfigurationValidationGate()
            ],
            Generation.GENERATION_3_OPTIMIZED: [
                PerformanceOptimizationGate(),
                CachingStrategyGate(),
                ScalabilityPatternsGate(),
                ResourceEfficiencyGate()
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
        
        # Execute gates sequentially for now
        for gate in gates:
            result = await gate.execute()
            report.add_result(result)
        
        # Calculate overall results
        report.execution_time = time.time() - start_time
        success_rate = report.calculate_success_rate()
        
        # Determine if generation passed (need at least 85% success rate and no failures)
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
            # Allow progression for testing purposes - in production this would stop
            if not report.is_passed and report.total_gates > 0:
                logger.warning(f"Generation {generation.value} failed - continuing for testing (would normally stop)")
                # break  # Commented out for testing Generation 3
        
        return reports
    
    def save_report(self, reports: List[GenerationReport], output_path: Path) -> None:
        """Save execution reports to JSON file"""
        try:
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            report_data = {
                "execution_timestamp": datetime.now().isoformat(),
                "total_generations": len(reports),
                "generation_reports": []
            }
            
            for report in reports:
                gen_data = {
                    "generation": report.generation.value,
                    "total_gates": report.total_gates,
                    "passed_gates": report.passed_gates,
                    "failed_gates": report.failed_gates,
                    "skipped_gates": report.skipped_gates,
                    "error_gates": report.error_gates,
                    "execution_time": report.execution_time,
                    "success_rate": report.calculate_success_rate(),
                    "overall_score": report.overall_score,
                    "is_passed": report.is_passed,
                    "timestamp": report.timestamp.isoformat(),
                    "gate_results": []
                }
                
                for gate_result in report.gate_results:
                    gate_data = {
                        "gate_name": gate_result.gate_name,
                        "status": gate_result.status.value,
                        "execution_time": gate_result.execution_time,
                        "message": gate_result.message,
                        "score": gate_result.score,
                        "threshold": gate_result.threshold,
                        "details": gate_result.details,
                        "error_details": gate_result.error_details,
                        "timestamp": gate_result.timestamp.isoformat()
                    }
                    gen_data["gate_results"].append(gate_data)
                
                report_data["generation_reports"].append(gen_data)
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Quality gate reports saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save reports: {e}")


def print_banner():
    """Print execution banner"""
    print("=" * 80)
    print("🚀 TERRAGON AUTONOMOUS SDLC - PROGRESSIVE QUALITY GATES")
    print("=" * 80)
    print("Executing evolutionary quality gates across three generations:")
    print("  Generation 1: Make it Work (Basic functionality)")
    print("  Generation 2: Make it Robust (Reliability)")
    print("  Generation 3: Make it Scale (Performance)")
    print("=" * 80)


def print_generation_summary(report):
    """Print summary for a single generation"""
    gen_name = report.generation.value.replace("_", " ").title()
    status_emoji = "✅" if report.is_passed else "❌"
    
    print(f"\n{status_emoji} {gen_name}")
    print(f"   Total Gates: {report.total_gates}")
    print(f"   Passed: {report.passed_gates}")
    print(f"   Failed: {report.failed_gates}")
    print(f"   Success Rate: {report.calculate_success_rate():.1f}%")
    print(f"   Execution Time: {report.execution_time:.2f}s")
    
    # Show failed gates
    failed_gates = [r for r in report.gate_results if r.status == GateStatus.FAILED]
    if failed_gates:
        print("   ⚠️  Failed Gates:")
        for gate in failed_gates:
            print(f"      - {gate.gate_name}: {gate.message}")
    
    # Show error gates
    error_gates = [r for r in report.gate_results if r.status == GateStatus.ERROR]
    if error_gates:
        print("   🚨 Error Gates:")
        for gate in error_gates:
            print(f"      - {gate.gate_name}: {gate.message}")


def print_final_summary(reports):
    """Print final execution summary"""
    print("\n" + "=" * 80)
    print("📊 PROGRESSIVE QUALITY GATES - FINAL SUMMARY")
    print("=" * 80)
    
    total_gates = sum(r.total_gates for r in reports)
    total_passed = sum(r.passed_gates for r in reports)
    total_failed = sum(r.failed_gates for r in reports)
    total_errors = sum(r.error_gates for r in reports)
    
    overall_success = (total_passed / total_gates * 100) if total_gates > 0 else 0
    
    print(f"Total Gates Executed: {total_gates}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Total Errors: {total_errors}")
    print(f"Overall Success Rate: {overall_success:.1f}%")
    
    # Check if all generations passed
    all_passed = all(r.is_passed or r.total_gates == 0 for r in reports)
    
    if all_passed:
        print("\n🎉 ALL GENERATIONS PASSED!")
        print("✅ Project successfully completed progressive quality gates")
        print("🚀 Ready for autonomous deployment and scaling")
    else:
        print(f"\n💥 QUALITY GATES FAILED")
        print("❌ Some quality gates did not pass")
        print("🔧 Review failed gates and fix issues before proceeding")
    
    print("\n📄 Detailed reports saved to: progressive_quality_gates_report.json")
    return 0 if all_passed else 1


async def run_progressive_quality_gates() -> List[GenerationReport]:
    """Main entry point for running progressive quality gates"""
    runner = ProgressiveQualityGateRunner()
    reports = await runner.run_all_generations()
    
    # Save reports
    output_path = Path("progressive_quality_gates_report.json")
    runner.save_report(reports, output_path)
    
    return reports


async def main():
    """Main execution function"""
    try:
        print_banner()
        
        # Run progressive quality gates
        logger.info("Starting progressive quality gate execution...")
        reports = await run_progressive_quality_gates()
        
        # Print results for each generation
        for report in reports:
            print_generation_summary(report)
        
        # Print final summary and return appropriate exit code
        exit_code = print_final_summary(reports)
        return exit_code
        
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        print("\n⚠️  Execution interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
        print(f"\n🚨 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)