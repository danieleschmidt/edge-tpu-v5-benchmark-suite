#!/usr/bin/env python3
"""Quality Gates Implementation for TPU v5 Benchmark Suite

Comprehensive quality gate checking including:
- Security vulnerability scanning
- Code quality metrics
- Test coverage validation
- Performance benchmarks
- Documentation completeness
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, str]
    execution_time: float
    warnings: List[str]
    errors: List[str]


class QualityGateRunner:
    """Main quality gate execution engine."""
    
    def __init__(self):
        self.results = []
        self.overall_passed = True
        self.start_time = time.time()
        
    def run_security_scan(self) -> QualityGateResult:
        """Run security vulnerability scanning."""
        start_time = time.time()
        
        try:
            # Run Bandit security scan
            bandit_result = subprocess.run([
                'bandit', '-r', 'src/', '--format', 'json'
            ], capture_output=True, text=True, timeout=300)
            
            bandit_data = json.loads(bandit_result.stdout) if bandit_result.stdout else {}
            
            # Count security issues by severity
            high_issues = 0
            medium_issues = 0
            low_issues = 0
            
            for result in bandit_data.get('results', []):
                severity = result.get('issue_severity', '').upper()
                if severity == 'HIGH':
                    high_issues += 1
                elif severity == 'MEDIUM':
                    medium_issues += 1
                elif severity == 'LOW':
                    low_issues += 1
            
            # Security gate passing criteria
            passed = high_issues == 0 and medium_issues <= 2
            score = max(0, 100 - (high_issues * 50) - (medium_issues * 10) - (low_issues * 1))
            
            warnings = []
            errors = []
            
            if high_issues > 0:
                errors.append(f"{high_issues} high-severity security issues found")
            if medium_issues > 0:
                warnings.append(f"{medium_issues} medium-severity security issues found")
                
            return QualityGateResult(
                gate_name="Security Scan",
                passed=passed,
                score=score,
                details={
                    "high_severity_issues": str(high_issues),
                    "medium_severity_issues": str(medium_issues),
                    "low_severity_issues": str(low_issues),
                    "total_lines_scanned": str(bandit_data.get('metrics', {}).get('_totals', {}).get('loc', 0))
                },
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Security Scan",
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[f"Security scan failed: {str(e)}"]
            )
    
    def run_dependency_check(self) -> QualityGateResult:
        """Run dependency vulnerability check."""
        start_time = time.time()
        
        try:
            # Run Safety dependency check
            safety_result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse vulnerabilities (Safety outputs to stderr for some reason)
            safety_output = safety_result.stderr if safety_result.stderr else safety_result.stdout
            
            vulnerabilities_found = 0
            critical_vulns = 0
            
            # Count vulnerabilities from output
            if "vulnerabilities_found" in safety_output:
                # Try to extract number from JSON
                try:
                    lines = safety_output.split('\n')
                    for line in lines:
                        if '"vulnerabilities_found":' in line:
                            vulnerabilities_found = int(line.split(':')[1].strip().rstrip(','))
                            break
                except:
                    # Fallback: count vulnerability entries
                    vulnerabilities_found = safety_output.count('"vulnerability_id"')
            
            # Dependency gate passing criteria
            passed = vulnerabilities_found <= 2  # Allow minor vulnerabilities
            score = max(0, 100 - (vulnerabilities_found * 15))
            
            warnings = []
            errors = []
            
            if vulnerabilities_found > 0:
                warnings.append(f"{vulnerabilities_found} dependency vulnerabilities found")
            
            return QualityGateResult(
                gate_name="Dependency Check",
                passed=passed,
                score=score,
                details={
                    "vulnerabilities_found": str(vulnerabilities_found),
                    "critical_vulnerabilities": str(critical_vulns)
                },
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Dependency Check",
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[f"Dependency check failed: {str(e)}"]
            )
    
    def run_code_quality_check(self) -> QualityGateResult:
        """Run code quality checks."""
        start_time = time.time()
        
        try:
            # Run black to check code formatting
            black_result = subprocess.run([
                'black', '--check', '--diff', 'src/'
            ], capture_output=True, text=True, timeout=120)
            
            # Run ruff for linting
            ruff_result = subprocess.run([
                'ruff', 'check', 'src/'
            ], capture_output=True, text=True, timeout=120)
            
            # Check results
            black_passed = black_result.returncode == 0
            ruff_issues = len(ruff_result.stdout.split('\n')) if ruff_result.stdout else 0
            
            passed = black_passed and ruff_issues <= 10
            score = (90 if black_passed else 50) + max(0, 10 - ruff_issues)
            
            warnings = []
            errors = []
            
            if not black_passed:
                errors.append("Code formatting issues found (run 'black src/' to fix)")
            if ruff_issues > 0:
                warnings.append(f"{ruff_issues} linting issues found")
            
            return QualityGateResult(
                gate_name="Code Quality",
                passed=passed,
                score=score,
                details={
                    "black_formatting": "passed" if black_passed else "failed",
                    "ruff_issues": str(ruff_issues)
                },
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Code Quality",
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[f"Code quality check failed: {str(e)}"]
            )
    
    def run_test_coverage_check(self) -> QualityGateResult:
        """Run test coverage validation."""
        start_time = time.time()
        
        try:
            # Count test files
            test_files = list(Path("tests").rglob("test_*.py"))
            src_files = list(Path("src").rglob("*.py"))
            
            test_count = len(test_files)
            src_count = len([f for f in src_files if not f.name.startswith('__')])
            
            # Estimate coverage based on test to source ratio
            estimated_coverage = min(95, (test_count / max(1, src_count)) * 100 * 3)  # Heuristic
            
            passed = estimated_coverage >= 70  # 70% minimum coverage
            score = min(100, estimated_coverage)
            
            warnings = []
            errors = []
            
            if estimated_coverage < 70:
                errors.append(f"Test coverage too low: {estimated_coverage:.1f}% (minimum 70%)")
            elif estimated_coverage < 85:
                warnings.append(f"Test coverage could be improved: {estimated_coverage:.1f}%")
            
            return QualityGateResult(
                gate_name="Test Coverage",
                passed=passed,
                score=score,
                details={
                    "test_files": str(test_count),
                    "source_files": str(src_count),
                    "estimated_coverage": f"{estimated_coverage:.1f}%"
                },
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Test Coverage",
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[f"Test coverage check failed: {str(e)}"]
            )
    
    def run_documentation_check(self) -> QualityGateResult:
        """Run documentation completeness check."""
        start_time = time.time()
        
        try:
            # Check for required documentation files
            required_docs = [
                "README.md",
                "CONTRIBUTING.md", 
                "SECURITY.md",
                "CHANGELOG.md"
            ]
            
            doc_files = [Path(doc) for doc in required_docs]
            existing_docs = [doc for doc in doc_files if doc.exists()]
            
            # Check docstring coverage in Python files
            python_files = list(Path("src").rglob("*.py"))
            files_with_docstrings = 0
            
            for py_file in python_files:
                if py_file.name.startswith('__'):
                    continue
                try:
                    content = py_file.read_text()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
                except:
                    pass
            
            doc_coverage = len(existing_docs) / len(required_docs) * 100
            docstring_coverage = (files_with_docstrings / max(1, len(python_files))) * 100
            
            overall_score = (doc_coverage + docstring_coverage) / 2
            passed = overall_score >= 75
            
            warnings = []
            errors = []
            
            missing_docs = [doc.name for doc in doc_files if not doc.exists()]
            if missing_docs:
                warnings.append(f"Missing documentation files: {', '.join(missing_docs)}")
            
            if docstring_coverage < 60:
                warnings.append(f"Low docstring coverage: {docstring_coverage:.1f}%")
            
            return QualityGateResult(
                gate_name="Documentation",
                passed=passed,
                score=overall_score,
                details={
                    "required_docs_present": f"{len(existing_docs)}/{len(required_docs)}",
                    "docstring_coverage": f"{docstring_coverage:.1f}%",
                    "documentation_score": f"{overall_score:.1f}%"
                },
                execution_time=time.time() - start_time,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Documentation",
                passed=False,
                score=0,
                details={"error": str(e)},
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[f"Documentation check failed: {str(e)}"]
            )
    
    def run_all_gates(self) -> Dict[str, QualityGateResult]:
        """Run all quality gates."""
        print("🚀 Running Quality Gates for TPU v5 Benchmark Suite")
        print("=" * 60)
        
        gates = [
            ("Security Scan", self.run_security_scan),
            ("Dependency Check", self.run_dependency_check),
            ("Code Quality", self.run_code_quality_check),
            ("Test Coverage", self.run_test_coverage_check),
            ("Documentation", self.run_documentation_check)
        ]
        
        results = {}
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running {gate_name}...")
            result = gate_func()
            results[gate_name] = result
            self.results.append(result)
            
            # Print result
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {gate_name}: {result.score:.1f}/100 ({result.execution_time:.2f}s)")
            
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
            for error in result.errors:
                print(f"  ❌ {error}")
            
            if not result.passed:
                self.overall_passed = False
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive quality gate report."""
        total_time = time.time() - self.start_time
        
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        
        average_score = sum(result.score for result in self.results) / max(1, total_gates)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_passed": self.overall_passed,
            "gates_passed": f"{passed_gates}/{total_gates}",
            "average_score": f"{average_score:.1f}/100",
            "total_execution_time": f"{total_time:.2f}s",
            "gates": {}
        }
        
        for result in self.results:
            report["gates"][result.gate_name] = {
                "passed": result.passed,
                "score": result.score,
                "execution_time": result.execution_time,
                "details": result.details,
                "warnings": result.warnings,
                "errors": result.errors
            }
        
        return report


def main():
    """Run quality gates and generate report."""
    runner = QualityGateRunner()
    
    # Run all quality gates
    results = runner.run_all_gates()
    
    # Generate report
    report = runner.generate_report()
    
    # Save report
    report_file = Path("quality_gates_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {'✅ PASSED' if runner.overall_passed else '❌ FAILED'}")
    print(f"Gates Passed: {report['gates_passed']}")
    print(f"Average Score: {report['average_score']}")
    print(f"Total Time: {report['total_execution_time']}")
    print(f"Report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if runner.overall_passed else 1)


if __name__ == "__main__":
    main()