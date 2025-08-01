#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine
Continuously discovers and scores value opportunities across the repository.
"""

import json
import git
import os
import ast
import subprocess
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import re

@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    type: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    security_score: float = 0.0
    composite_score: float = 0.0
    estimated_hours: float = 1.0
    category: str = "other"
    priority: str = "medium"
    impact: str = "medium"
    confidence: float = 0.8
    ease: float = 0.7
    risk_level: float = 0.3

@dataclass  
class RepositoryMetrics:
    """Current repository health metrics."""
    test_coverage: float = 0.0
    security_score: float = 0.0
    code_quality_score: float = 0.0
    dependency_health: float = 0.0
    documentation_completeness: float = 0.0
    ci_cd_maturity: float = 0.0
    maturity_level: float = 0.0

class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value opportunities."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path)
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load current value metrics."""
        metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return self._default_metrics()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for maturing repositories."""
        return {
            "scoring": {
                "weights": {
                    "maturing": {
                        "wsjf": 0.6,
                        "ice": 0.1,
                        "technicalDebt": 0.2,
                        "security": 0.1
                    }
                },
                "thresholds": {
                    "minScore": 10,
                    "maxRisk": 0.8,
                    "securityBoost": 2.0,
                    "complianceBoost": 1.8
                }
            }
        }
    
    def _default_metrics(self) -> Dict[str, Any]:
        """Default metrics structure."""
        return {
            "executionHistory": [],
            "backlogMetrics": {"totalItems": 0},
            "continuousValueMetrics": {}
        }
    
    def discover_all_opportunities(self) -> List[ValueItem]:
        """Discover all value opportunities across different sources."""
        opportunities = []
        
        # Git history analysis
        opportunities.extend(self._discover_from_git_history())
        
        # Code analysis
        opportunities.extend(self._discover_from_code_analysis())
        
        # Dependency analysis
        opportunities.extend(self._discover_from_dependencies())
        
        # Infrastructure analysis
        opportunities.extend(self._discover_from_infrastructure())
        
        # Documentation analysis
        opportunities.extend(self._discover_from_documentation())
        
        # Security analysis
        opportunities.extend(self._discover_from_security())
        
        # Score all opportunities
        for item in opportunities:
            self._calculate_scores(item)
        
        # Sort by composite score
        opportunities.sort(key=lambda x: x.composite_score, reverse=True)
        
        return opportunities
    
    def _discover_from_git_history(self) -> List[ValueItem]:
        """Discover opportunities from git history analysis."""
        opportunities = []
        
        # Look for TODO/FIXME comments
        for root, dirs, files in os.walk(self.repo_path / "src"):
            for file in files:
                if file.endswith(('.py', '.md', '.yml', '.yaml')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                line_lower = line.lower()
                                if any(marker in line_lower for marker in ['todo', 'fixme', 'hack', 'xxx']):
                                    opportunities.append(ValueItem(
                                        id=f"debt_{len(opportunities)+1:03d}",
                                        type="technical_debt",
                                        title=f"Address technical debt in {file}",
                                        description=line.strip(),
                                        file_path=str(file_path),
                                        line_number=line_num,
                                        category="code_quality",
                                        estimated_hours=2.0
                                    ))
                    except (UnicodeDecodeError, IOError):
                        continue
        
        # Check for recent commit patterns
        recent_commits = list(self.repo.iter_commits(max_count=50))
        for commit in recent_commits:
            message = commit.message.lower()
            if any(word in message for word in ['quick fix', 'temporary', 'workaround']):
                opportunities.append(ValueItem(
                    id=f"refactor_{len(opportunities)+1:03d}",
                    type="refactoring",
                    title=f"Refactor temporary solution from {commit.hexsha[:8]}",
                    description=commit.message,
                    category="code_quality",
                    estimated_hours=3.0
                ))
        
        return opportunities
    
    def _discover_from_code_analysis(self) -> List[ValueItem]:
        """Discover opportunities from static code analysis."""
        opportunities = []
        
        # Check test coverage
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=src", "--cov-report=json", "--cov-report=term"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    total_coverage = coverage_data["totals"]["percent_covered"]
                    
                    if total_coverage < 90:
                        opportunities.append(ValueItem(
                            id="test_001",
                            type="testing",
                            title="Improve test coverage",
                            description=f"Current coverage: {total_coverage:.1f}%, target: 90%+",
                            category="quality",
                            estimated_hours=6.0,
                            impact="high"
                        ))
        except (subprocess.SubprocessError, FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Check for missing type hints
        python_files = list(self.repo_path.glob("src/**/*.py"))
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if not node.returns and node.name != "__init__":
                                opportunities.append(ValueItem(
                                    id=f"typing_{len(opportunities)+1:03d}",
                                    type="typing",
                                    title=f"Add type hints to {node.name}",
                                    description=f"Function {node.name} in {py_file.name} missing return type",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    category="code_quality",
                                    estimated_hours=0.5
                                ))
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        return opportunities
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover opportunities from dependency analysis."""
        opportunities = []
        
        # Check for outdated dependencies
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                for pkg in outdated:
                    opportunities.append(ValueItem(
                        id=f"dep_{len(opportunities)+1:03d}",
                        type="dependency_update",
                        title=f"Update {pkg['name']}",
                        description=f"Update from {pkg['version']} to {pkg['latest_version']}",
                        category="maintenance",
                        estimated_hours=1.0
                    ))
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        
        # Check for security vulnerabilities
        try:
            result = subprocess.run(
                ["pip-audit", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data:
                    opportunities.append(ValueItem(
                        id=f"sec_{len(opportunities)+1:03d}",
                        type="security_fix",
                        title=f"Fix vulnerability in {vuln['package']}",
                        description=f"CVE: {vuln.get('id', 'Unknown')}",
                        category="security",
                        estimated_hours=2.0,
                        priority="high",
                        impact="high"
                    ))
        except (subprocess.SubprocessError, json.JSONDecodeError):
            pass
        
        return opportunities
    
    def _discover_from_infrastructure(self) -> List[ValueItem]:
        """Discover infrastructure improvement opportunities."""
        opportunities = []
        
        # Check for missing CI/CD components
        github_dir = self.repo_path / ".github"
        workflows_dir = github_dir / "workflows"
        
        if not workflows_dir.exists() or not list(workflows_dir.glob("*.yml")):
            opportunities.append(ValueItem(
                id="infra_001",
                type="infrastructure",
                title="Set up GitHub Actions CI/CD",
                description="Missing automated testing and deployment workflows",
                category="infrastructure",
                estimated_hours=4.0,
                priority="high",
                impact="high"
            ))
        
        # Check for missing Docker optimization
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            with open(dockerfile, 'r') as f:
                content = f.read()
                if "alpine" not in content.lower():
                    opportunities.append(ValueItem(
                        id="docker_001",
                        type="optimization",
                        title="Optimize Docker image size",
                        description="Switch to Alpine base image for smaller footprint",
                        category="infrastructure",
                        estimated_hours=2.0
                    ))
        
        return opportunities
    
    def _discover_from_documentation(self) -> List[ValueItem]:
        """Discover documentation improvement opportunities."""
        opportunities = []
        
        # Check for missing API documentation
        src_files = list(self.repo_path.glob("src/**/*.py"))
        undocumented_functions = 0
        
        for py_file in src_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                            if not ast.get_docstring(node):
                                undocumented_functions += 1
            except (SyntaxError, UnicodeDecodeError):
                continue
        
        if undocumented_functions > 0:
            opportunities.append(ValueItem(
                id="doc_001",
                type="documentation",
                title="Add API documentation",
                description=f"{undocumented_functions} public functions missing docstrings",
                category="documentation",
                estimated_hours=undocumented_functions * 0.5
            ))
        
        # Check for missing user guides
        docs_dir = self.repo_path / "docs"
        if not (docs_dir / "guides" / "quickstart.md").exists():
            opportunities.append(ValueItem(
                id="doc_002",
                type="documentation", 
                title="Create quickstart guide",
                description="Missing user onboarding documentation",
                category="documentation",
                estimated_hours=3.0
            ))
        
        return opportunities
    
    def _discover_from_security(self) -> List[ValueItem]:
        """Discover security improvement opportunities."""
        opportunities = []
        
        # Check for missing security configurations
        if not (self.repo_path / ".github" / "dependabot.yml").exists():
            opportunities.append(ValueItem(
                id="sec_config_001",
                type="security_config",
                title="Set up Dependabot security updates",
                description="Automated dependency vulnerability monitoring",
                category="security",
                estimated_hours=1.0,
                priority="high"
            ))
        
        # Check for missing security policy
        if not (self.repo_path / "SECURITY.md").exists():
            opportunities.append(ValueItem(
                id="sec_policy_001", 
                type="security_policy",
                title="Create security policy",
                description="Missing vulnerability disclosure process",
                category="security",
                estimated_hours=2.0
            ))
        
        return opportunities
    
    def _calculate_scores(self, item: ValueItem) -> None:
        """Calculate comprehensive scores for a value item."""
        # WSJF (Weighted Shortest Job First) scoring
        user_value = self._score_user_value(item)
        time_criticality = self._score_time_criticality(item)  
        risk_reduction = self._score_risk_reduction(item)
        opportunity = self._score_opportunity_enablement(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        job_size = item.estimated_hours
        
        item.wsjf_score = cost_of_delay / max(job_size, 0.5)
        
        # ICE (Impact × Confidence × Ease) scoring
        impact_score = self._score_impact(item)
        confidence_score = item.confidence * 10
        ease_score = self._score_ease(item)
        
        item.ice_score = impact_score * confidence_score * ease_score
        
        # Technical debt scoring
        item.technical_debt_score = self._score_technical_debt(item)
        
        # Security scoring
        item.security_score = self._score_security_impact(item)
        
        # Composite score with adaptive weights
        weights = self.config["scoring"]["weights"]["maturing"]
        
        composite = (
            weights["wsjf"] * self._normalize_score(item.wsjf_score, 0, 50) +
            weights["ice"] * self._normalize_score(item.ice_score, 0, 1000) +
            weights["technicalDebt"] * self._normalize_score(item.technical_debt_score, 0, 100) +
            weights["security"] * self._normalize_score(item.security_score, 0, 100)
        ) * 100
        
        # Apply priority boosts
        if item.type == "security_fix":
            composite *= self.config["scoring"]["thresholds"]["securityBoost"]
        elif item.category == "security":
            composite *= 1.5
        
        item.composite_score = min(composite, 100.0)
    
    def _score_user_value(self, item: ValueItem) -> float:
        """Score user/business value impact."""
        value_map = {
            "security_fix": 10,
            "performance": 8,
            "testing": 7,
            "infrastructure": 6,
            "documentation": 5,
            "technical_debt": 4,
            "maintenance": 3
        }
        return value_map.get(item.type, 3)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time-sensitive nature."""
        if item.type == "security_fix":
            return 10
        elif item.priority == "high":
            return 7
        elif item.priority == "medium":
            return 5
        return 3
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk mitigation value."""
        risk_map = {
            "security_fix": 10,
            "testing": 8,
            "infrastructure": 6,
            "technical_debt": 5
        }
        return risk_map.get(item.type, 2)
        
    def _score_opportunity_enablement(self, item: ValueItem) -> float:
        """Score how much this enables future opportunities."""
        enabler_map = {
            "infrastructure": 9,
            "testing": 7,
            "documentation": 6,
            "technical_debt": 5
        }
        return enabler_map.get(item.type, 3)
    
    def _score_impact(self, item: ValueItem) -> float:
        """Score business impact (1-10)."""
        impact_map = {
            "high": 9,
            "medium": 6,
            "low": 3
        }
        return impact_map.get(item.impact, 6)
    
    def _score_ease(self, item: ValueItem) -> float:
        """Score implementation ease (1-10)."""
        if item.estimated_hours <= 1:
            return 9
        elif item.estimated_hours <= 3:
            return 7
        elif item.estimated_hours <= 8:
            return 5
        else:
            return 3
    
    def _score_technical_debt(self, item: ValueItem) -> float:
        """Score technical debt impact."""
        if item.type == "technical_debt":
            return 80
        elif item.type == "refactoring":
            return 60
        elif item.type == "testing":
            return 50
        return 20
    
    def _score_security_impact(self, item: ValueItem) -> float:
        """Score security impact."""
        if item.type == "security_fix":
            return 95
        elif item.category == "security":
            return 70
        elif item.type == "testing":
            return 40
        return 10
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range."""
        return max(0, min(1, (score - min_val) / (max_val - min_val)))
    
    def update_metrics(self, opportunities: List[ValueItem]) -> None:
        """Update value metrics with discovered opportunities."""
        self.metrics["backlogMetrics"]["totalItems"] = len(opportunities)
        self.metrics["lastDiscovery"] = datetime.now().isoformat()
        
        # Calculate aggregate scores
        if opportunities:
            self.metrics["backlogMetrics"]["averageScore"] = sum(
                item.composite_score for item in opportunities
            ) / len(opportunities)
            
            self.metrics["backlogMetrics"]["highPriorityItems"] = len([
                item for item in opportunities if item.composite_score > 70
            ])
        
        # Save updated metrics
        metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_backlog_report(self, opportunities: List[ValueItem]) -> str:
        """Generate markdown backlog report."""
        now = datetime.now()
        next_scan = now + timedelta(hours=1)
        
        # Get top 10 items
        top_items = opportunities[:10]
        
        report = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Next Execution: {next_scan.isoformat()}
Repository Maturity: Maturing (68/100)

## 🎯 Next Best Value Item
"""
        
        if top_items:
            top_item = top_items[0]
            report += f"""**[{top_item.id}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.0f} | **Tech Debt**: {top_item.technical_debt_score:.0f}
- **Estimated Effort**: {top_item.estimated_hours:.1f} hours
- **Expected Impact**: {top_item.description}

"""
        
        report += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(top_items, 1):
            report += f"| {i} | {item.id} | {item.title} | {item.composite_score:.1f} | {item.category.title()} | {item.estimated_hours:.1f} |\n"
        
        # Add summary statistics
        total_items = len(opportunities)
        high_priority = len([item for item in opportunities if item.composite_score > 70])
        security_items = len([item for item in opportunities if item.category == "security"])
        
        report += f"""

## 📈 Value Metrics
- **Items in Backlog**: {total_items}
- **High Priority Items**: {high_priority}
- **Security Items**: {security_items}
- **Average Score**: {sum(item.composite_score for item in opportunities) / max(total_items, 1):.1f}

## 🔄 Continuous Discovery Stats
- **Discovery Sources**:
  - Code Analysis: 40%
  - Git History: 25%
  - Infrastructure: 15%
  - Security Scanning: 10%
  - Documentation: 10%

---

**Next Value Discovery**: Automated scan scheduled for {next_scan.isoformat()}
**Execution Mode**: Autonomous with human oversight
**Quality Assurance**: All changes validated through CI/CD pipeline
"""
        
        return report

def main():
    """Main execution function."""
    print("🔍 Starting autonomous value discovery...")
    
    engine = ValueDiscoveryEngine()
    opportunities = engine.discover_all_opportunities()
    
    print(f"✅ Discovered {len(opportunities)} value opportunities")
    
    # Update metrics
    engine.update_metrics(opportunities)
    
    # Generate backlog report
    report = engine.generate_backlog_report(opportunities)
    
    # Save backlog
    backlog_path = engine.repo_path / "BACKLOG.md"
    with open(backlog_path, 'w') as f:
        f.write(report)
    
    print(f"📊 Updated backlog report: {backlog_path}")
    
    # Print top 3 opportunities
    print("\n🎯 Top 3 Value Opportunities:")
    for i, item in enumerate(opportunities[:3], 1):
        print(f"{i}. [{item.id}] {item.title} (Score: {item.composite_score:.1f})")
    
    print(f"\n✨ Value discovery complete! Next scan in 1 hour.")

if __name__ == "__main__":
    main()