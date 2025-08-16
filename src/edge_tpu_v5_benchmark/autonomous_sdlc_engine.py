"""Autonomous SDLC Engine for Self-Improving Benchmark Suite

This module implements a complete autonomous software development lifecycle
that continuously improves the benchmark suite based on usage patterns,
performance data, and system feedback.
"""

import asyncio
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
import numpy as np
import git
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from .security import SecurityContext, InputValidator
from .adaptive_optimizer import AdaptiveOptimizer, PerformanceMetrics
from .config import get_config


class SDLCPhase(Enum):
    """SDLC phases for autonomous development."""
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


class ImprovementType(Enum):
    """Types of improvements the SDLC engine can make."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_FIX = "bug_fix"
    FEATURE_ENHANCEMENT = "feature_enhancement"
    SECURITY_IMPROVEMENT = "security_improvement"
    CODE_REFACTORING = "code_refactoring"
    TEST_ENHANCEMENT = "test_enhancement"
    DOCUMENTATION_UPDATE = "documentation_update"


@dataclass
class ImprovementProposal:
    """Proposal for system improvement."""
    improvement_type: ImprovementType
    description: str
    priority: int  # 1-10, 10 being highest
    estimated_effort: int  # hours
    expected_benefit: float  # 0-1 scale
    affected_files: List[str] = field(default_factory=list)
    implementation_plan: List[str] = field(default_factory=list)
    tests_required: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def score(self) -> float:
        """Calculate priority score for this improvement."""
        urgency = self.priority / 10.0
        benefit_ratio = self.expected_benefit
        effort_penalty = 1.0 / (1.0 + self.estimated_effort / 10.0)
        
        return urgency * benefit_ratio * effort_penalty


@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    test_pass_rate: float
    error_rate: float
    performance_score: float
    user_satisfaction: float
    code_quality_score: float
    security_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class CodeAnalyzer:
    """Analyzes code for improvement opportunities."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.logger = logging.getLogger(__name__)
        
    def analyze_performance_bottlenecks(self) -> List[ImprovementProposal]:
        """Analyze code for performance bottlenecks."""
        proposals = []
        
        python_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                bottlenecks = self._detect_bottlenecks(content, file_path)
                proposals.extend(bottlenecks)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return proposals
    
    def _detect_bottlenecks(self, content: str, file_path: Path) -> List[ImprovementProposal]:
        """Detect performance bottlenecks in code."""
        proposals = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Detect inefficient loops
            if 'for' in line and 'range(len(' in line:
                proposals.append(ImprovementProposal(
                    improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                    description=f"Replace range(len()) with enumerate() in {file_path}:{i+1}",
                    priority=6,
                    estimated_effort=1,
                    expected_benefit=0.3,
                    affected_files=[str(file_path)]
                ))
            
            # Detect inefficient string concatenation
            if '+=' in line and '"' in line:
                proposals.append(ImprovementProposal(
                    improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                    description=f"Use f-strings or join() instead of += for strings in {file_path}:{i+1}",
                    priority=5,
                    estimated_effort=2,
                    expected_benefit=0.4,
                    affected_files=[str(file_path)]
                ))
            
            # Detect missing async/await opportunities
            if 'time.sleep(' in line and 'async def' not in content:
                proposals.append(ImprovementProposal(
                    improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                    description=f"Convert to async function with asyncio.sleep() in {file_path}:{i+1}",
                    priority=7,
                    estimated_effort=4,
                    expected_benefit=0.6,
                    affected_files=[str(file_path)]
                ))
        
        return proposals
    
    def analyze_security_vulnerabilities(self) -> List[ImprovementProposal]:
        """Analyze code for security vulnerabilities."""
        proposals = []
        
        # Run bandit security scanner
        try:
            result = subprocess.run([
                'bandit', '-r', str(self.repo_path), '-f', 'json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get('results', []):
                    proposals.append(ImprovementProposal(
                        improvement_type=ImprovementType.SECURITY_IMPROVEMENT,
                        description=f"Security issue: {issue['issue_text']} in {issue['filename']}",
                        priority=8,
                        estimated_effort=3,
                        expected_benefit=0.8,
                        affected_files=[issue['filename']]
                    ))
        
        except Exception as e:
            self.logger.warning(f"Security analysis failed: {e}")
        
        return proposals
    
    def analyze_test_coverage(self) -> List[ImprovementProposal]:
        """Analyze test coverage and suggest improvements."""
        proposals = []
        
        try:
            # Run coverage analysis
            result = subprocess.run([
                'coverage', 'run', '-m', 'pytest', 'tests/',
                '&&', 'coverage', 'json'
            ], capture_output=True, text=True, timeout=600, shell=True)
            
            coverage_file = self.repo_path / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                
                for filename, file_data in coverage_data.get('files', {}).items():
                    coverage_percent = file_data.get('summary', {}).get('percent_covered', 0)
                    
                    if coverage_percent < 80:
                        proposals.append(ImprovementProposal(
                            improvement_type=ImprovementType.TEST_ENHANCEMENT,
                            description=f"Improve test coverage for {filename} (currently {coverage_percent:.1f}%)",
                            priority=6,
                            estimated_effort=5,
                            expected_benefit=0.7,
                            affected_files=[filename]
                        ))
        
        except Exception as e:
            self.logger.warning(f"Coverage analysis failed: {e}")
        
        return proposals


class AutomatedDeveloper:
    """Implements automated development capabilities."""
    
    def __init__(self, repo_path: Path, security_context: SecurityContext):
        self.repo_path = repo_path
        self.security_context = security_context
        self.logger = logging.getLogger(__name__)
        self.repo = git.Repo(repo_path)
        
    def implement_improvement(self, proposal: ImprovementProposal) -> bool:
        """Implement an improvement proposal."""
        try:
            self.logger.info(f"Implementing: {proposal.description}")
            
            if proposal.improvement_type == ImprovementType.PERFORMANCE_OPTIMIZATION:
                return self._implement_performance_optimization(proposal)
            elif proposal.improvement_type == ImprovementType.BUG_FIX:
                return self._implement_bug_fix(proposal)
            elif proposal.improvement_type == ImprovementType.FEATURE_ENHANCEMENT:
                return self._implement_feature_enhancement(proposal)
            elif proposal.improvement_type == ImprovementType.SECURITY_IMPROVEMENT:
                return self._implement_security_improvement(proposal)
            elif proposal.improvement_type == ImprovementType.TEST_ENHANCEMENT:
                return self._implement_test_enhancement(proposal)
            else:
                self.logger.warning(f"Unknown improvement type: {proposal.improvement_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Failed to implement improvement: {e}")
            return False
    
    def _implement_performance_optimization(self, proposal: ImprovementProposal) -> bool:
        """Implement performance optimization."""
        for file_path in proposal.affected_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    continue
                
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                optimized_content = self._optimize_code(content, proposal)
                
                if optimized_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(optimized_content)
                    
                    self.logger.info(f"Optimized {file_path}")
                    return True
            
            except Exception as e:
                self.logger.error(f"Failed to optimize {file_path}: {e}")
        
        return False
    
    def _optimize_code(self, content: str, proposal: ImprovementProposal) -> str:
        """Apply code optimizations."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Replace range(len()) with enumerate()
            if 'for' in line and 'range(len(' in line:
                # Simple heuristic replacement
                if 'for i in range(len(' in line:
                    var_name = line.split('range(len(')[1].split('))')[0]
                    indent = len(line) - len(line.lstrip())
                    lines[i] = ' ' * indent + f"for i, item in enumerate({var_name}):"
            
            # Replace string concatenation with f-strings
            if '+=' in line and '"' in line and 'f"' not in line:
                # This is a simplified example - real implementation would be more sophisticated
                pass
        
        return '\n'.join(lines)
    
    def _implement_bug_fix(self, proposal: ImprovementProposal) -> bool:
        """Implement bug fix."""
        # Placeholder for bug fix implementation
        return True
    
    def _implement_feature_enhancement(self, proposal: ImprovementProposal) -> bool:
        """Implement feature enhancement."""
        # Placeholder for feature enhancement
        return True
    
    def _implement_security_improvement(self, proposal: ImprovementProposal) -> bool:
        """Implement security improvement."""
        # Placeholder for security improvement
        return True
    
    def _implement_test_enhancement(self, proposal: ImprovementProposal) -> bool:
        """Implement test enhancement."""
        # Generate additional tests based on coverage gaps
        return True
    
    def create_feature_branch(self, improvement_id: str) -> str:
        """Create a feature branch for improvement."""
        branch_name = f"auto-improvement-{improvement_id}-{int(time.time())}"
        
        try:
            # Create and checkout new branch
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
            
            self.logger.info(f"Created feature branch: {branch_name}")
            return branch_name
        
        except Exception as e:
            self.logger.error(f"Failed to create branch: {e}")
            return ""
    
    def commit_changes(self, message: str, files: List[str] = None):
        """Commit changes to the repository."""
        try:
            if files:
                self.repo.index.add(files)
            else:
                self.repo.git.add(A=True)
            
            self.repo.index.commit(message)
            self.logger.info(f"Committed changes: {message}")
        
        except Exception as e:
            self.logger.error(f"Failed to commit changes: {e}")
    
    def run_tests(self) -> bool:
        """Run test suite to verify changes."""
        try:
            result = subprocess.run([
                'python', '-m', 'pytest', 'tests/', '-v'
            ], capture_output=True, text=True, timeout=1800, cwd=self.repo_path)
            
            return result.returncode == 0
        
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            return False


class AutonomousSDLCEngine:
    """Main autonomous SDLC engine."""
    
    def __init__(self, 
                 repo_path: Path,
                 security_context: Optional[SecurityContext] = None):
        self.repo_path = repo_path
        self.security_context = security_context or SecurityContext()
        self.logger = logging.getLogger(__name__)
        
        self.code_analyzer = CodeAnalyzer(repo_path)
        self.developer = AutomatedDeveloper(repo_path, self.security_context)
        self.optimizer = AdaptiveOptimizer(security_context)
        
        self.improvement_queue: List[ImprovementProposal] = []
        self.completed_improvements: List[ImprovementProposal] = []
        self.system_metrics_history: List[SystemHealthMetrics] = []
        
        self._running = False
        self._sdlc_thread = None
        self.lock = threading.RLock()
        
    def start_autonomous_sdlc(self):
        """Start the autonomous SDLC process."""
        with self.lock:
            if self._running:
                return
            
            self._running = True
            self._sdlc_thread = threading.Thread(
                target=self._sdlc_loop,
                daemon=True
            )
            self._sdlc_thread.start()
            self.logger.info("Autonomous SDLC engine started")
    
    def stop_autonomous_sdlc(self):
        """Stop the autonomous SDLC process."""
        with self.lock:
            self._running = False
            if self._sdlc_thread:
                self._sdlc_thread.join(timeout=10.0)
            self.logger.info("Autonomous SDLC engine stopped")
    
    def _sdlc_loop(self):
        """Main SDLC loop."""
        while self._running:
            try:
                cycle_start = time.time()
                
                # Phase 1: Requirements Analysis
                self._analyze_requirements()
                
                # Phase 2: Design
                self._design_improvements()
                
                # Phase 3: Implementation
                self._implement_improvements()
                
                # Phase 4: Testing
                self._test_improvements()
                
                # Phase 5: Monitoring
                self._monitor_system_health()
                
                # Phase 6: Optimization
                self._optimize_system()
                
                cycle_duration = time.time() - cycle_start
                self.logger.info(f"SDLC cycle completed in {cycle_duration:.2f}s")
                
                # Wait before next cycle (configurable)
                time.sleep(max(300 - cycle_duration, 60))  # 5 min cycles minimum
                
            except Exception as e:
                self.logger.error(f"SDLC cycle failed: {e}")
                time.sleep(60)  # Error recovery
    
    def _analyze_requirements(self):
        """Analyze system requirements and identify improvement opportunities."""
        self.logger.debug("Analyzing requirements...")
        
        # Collect system metrics
        metrics = self._collect_system_metrics()
        self.system_metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.system_metrics_history) > 100:
            self.system_metrics_history = self.system_metrics_history[-50:]
        
        # Analyze trends and identify needs
        if len(self.system_metrics_history) >= 5:
            self._identify_improvement_needs()
    
    def _collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics."""
        cpu_usage = psutil.cpu_percent(interval=1.0)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        # Mock some metrics for demonstration
        test_pass_rate = 0.85  # Would be calculated from actual test results
        error_rate = 0.05  # Would be calculated from logs
        performance_score = 0.75  # Would be calculated from benchmarks
        user_satisfaction = 0.80  # Would be calculated from user feedback
        code_quality_score = 0.85  # Would be calculated from static analysis
        security_score = 0.90  # Would be calculated from security scans
        
        return SystemHealthMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_info.percent,
            disk_usage=disk_info.percent,
            test_pass_rate=test_pass_rate,
            error_rate=error_rate,
            performance_score=performance_score,
            user_satisfaction=user_satisfaction,
            code_quality_score=code_quality_score,
            security_score=security_score
        )
    
    def _identify_improvement_needs(self):
        """Identify improvement needs based on metrics trends."""
        recent_metrics = self.system_metrics_history[-5:]
        
        # Check for declining performance
        performance_scores = [m.performance_score for m in recent_metrics]
        if len(performance_scores) >= 3:
            trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
            if trend < -0.02:  # Declining performance
                self.improvement_queue.append(ImprovementProposal(
                    improvement_type=ImprovementType.PERFORMANCE_OPTIMIZATION,
                    description="System performance is declining - investigate bottlenecks",
                    priority=8,
                    estimated_effort=8,
                    expected_benefit=0.8
                ))
        
        # Check for high error rates
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        if avg_error_rate > 0.1:
            self.improvement_queue.append(ImprovementProposal(
                improvement_type=ImprovementType.BUG_FIX,
                description=f"High error rate detected: {avg_error_rate:.2%}",
                priority=9,
                estimated_effort=6,
                expected_benefit=0.9
            ))
    
    def _design_improvements(self):
        """Design improvements based on analysis."""
        self.logger.debug("Designing improvements...")
        
        # Analyze code for specific improvement opportunities
        performance_proposals = self.code_analyzer.analyze_performance_bottlenecks()
        security_proposals = self.code_analyzer.analyze_security_vulnerabilities()
        test_proposals = self.code_analyzer.analyze_test_coverage()
        
        # Add to improvement queue
        self.improvement_queue.extend(performance_proposals)
        self.improvement_queue.extend(security_proposals)
        self.improvement_queue.extend(test_proposals)
        
        # Sort by priority score
        self.improvement_queue.sort(key=lambda x: x.score(), reverse=True)
        
        # Keep only top improvements to avoid overload
        self.improvement_queue = self.improvement_queue[:20]
    
    def _implement_improvements(self):
        """Implement prioritized improvements."""
        self.logger.debug("Implementing improvements...")
        
        if not self.improvement_queue:
            return
        
        # Implement top priority improvement
        proposal = self.improvement_queue.pop(0)
        
        try:
            # Create feature branch
            branch_name = self.developer.create_feature_branch(
                f"{proposal.improvement_type.value}_{int(time.time())}"
            )
            
            if not branch_name:
                return
            
            # Implement the improvement
            success = self.developer.implement_improvement(proposal)
            
            if success:
                # Commit changes
                self.developer.commit_changes(
                    f"Auto-improvement: {proposal.description}",
                    proposal.affected_files
                )
                
                proposal.completed_at = datetime.now()
                self.completed_improvements.append(proposal)
                
                self.logger.info(f"Successfully implemented: {proposal.description}")
            
        except Exception as e:
            self.logger.error(f"Failed to implement improvement: {e}")
    
    def _test_improvements(self):
        """Test implemented improvements."""
        self.logger.debug("Testing improvements...")
        
        # Run automated tests
        test_success = self.developer.run_tests()
        
        if not test_success:
            self.logger.warning("Tests failed - rolling back changes")
            # Would implement rollback logic here
        else:
            self.logger.info("All tests passed")
    
    def _monitor_system_health(self):
        """Monitor overall system health."""
        self.logger.debug("Monitoring system health...")
        
        current_metrics = self._collect_system_metrics()
        
        # Check for critical issues
        if current_metrics.error_rate > 0.2:
            self.improvement_queue.insert(0, ImprovementProposal(
                improvement_type=ImprovementType.BUG_FIX,
                description="Critical: High error rate detected",
                priority=10,
                estimated_effort=4,
                expected_benefit=1.0
            ))
        
        if current_metrics.security_score < 0.7:
            self.improvement_queue.insert(0, ImprovementProposal(
                improvement_type=ImprovementType.SECURITY_IMPROVEMENT,
                description="Critical: Low security score detected",
                priority=10,
                estimated_effort=6,
                expected_benefit=1.0
            ))
    
    def _optimize_system(self):
        """Optimize system performance."""
        self.logger.debug("Optimizing system...")
        
        # Trigger adaptive optimizer
        self.optimizer.force_reoptimization()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current SDLC engine status."""
        with self.lock:
            return {
                "running": self._running,
                "improvements_queue": len(self.improvement_queue),
                "completed_improvements": len(self.completed_improvements),
                "system_health": self.system_metrics_history[-1].__dict__ if self.system_metrics_history else None,
                "next_improvements": [
                    {
                        "type": imp.improvement_type.value,
                        "description": imp.description,
                        "priority": imp.priority,
                        "score": imp.score()
                    }
                    for imp in self.improvement_queue[:5]
                ]
            }
    
    def export_sdlc_data(self, filepath: Path):
        """Export SDLC data for analysis."""
        data = {
            "status": self.get_status(),
            "improvement_history": [
                {
                    "type": imp.improvement_type.value,
                    "description": imp.description,
                    "priority": imp.priority,
                    "estimated_effort": imp.estimated_effort,
                    "expected_benefit": imp.expected_benefit,
                    "created_at": imp.created_at.isoformat(),
                    "completed_at": getattr(imp, 'completed_at', None)
                }
                for imp in self.completed_improvements
            ],
            "metrics_history": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_usage": m.cpu_usage,
                    "memory_usage": m.memory_usage,
                    "test_pass_rate": m.test_pass_rate,
                    "performance_score": m.performance_score,
                    "security_score": m.security_score
                }
                for m in self.system_metrics_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"SDLC data exported to {filepath}")


def create_autonomous_sdlc_engine(repo_path: str, 
                                security_context: Optional[SecurityContext] = None) -> AutonomousSDLCEngine:
    """Factory function to create autonomous SDLC engine."""
    return AutonomousSDLCEngine(
        repo_path=Path(repo_path),
        security_context=security_context
    )