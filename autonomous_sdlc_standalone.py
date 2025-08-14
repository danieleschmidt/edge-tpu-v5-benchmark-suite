#!/usr/bin/env python3
"""
Autonomous SDLC Standalone Runner

This standalone version runs the complete Terragon Autonomous SDLC system
without requiring heavy dependencies.
"""

import asyncio
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import the standalone progressive quality gates
sys.path.insert(0, str(Path(".").absolute()))
from progressive_gates_standalone import (
    run_progressive_quality_gates,
    Generation,
    GenerationReport,
    GateStatus
)


@dataclass
class SDLCMetrics:
    """SDLC execution metrics and statistics"""
    total_execution_time: float = 0.0
    gates_executed: int = 0
    gates_passed: int = 0
    gates_failed: int = 0
    security_issues: int = 0
    performance_score: float = 0.0
    quality_score: float = 0.0
    generation_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all gates"""
        if self.gates_executed == 0:
            return 0.0
        return (self.gates_passed / self.gates_executed) * 100


class AutonomousSDLC:
    """Autonomous Software Development Lifecycle Manager"""
    
    def __init__(self):
        self.metrics = SDLCMetrics()
        self.execution_history: List[SDLCMetrics] = []
        
    async def execute_autonomous_sdlc(self) -> Tuple[SDLCMetrics, List[GenerationReport]]:
        """Execute the complete autonomous SDLC process"""
        print("🚀 Starting Autonomous SDLC Execution")
        start_time = datetime.now()
        
        try:
            # Execute progressive quality gates
            reports = await run_progressive_quality_gates()
            
            # Calculate metrics from reports
            self.metrics = self._calculate_metrics(reports, start_time)
            
            # Store execution history
            self.execution_history.append(self.metrics)
            
            return self.metrics, reports
            
        except Exception as e:
            print(f"Autonomous SDLC execution failed: {e}")
            raise
    
    def _calculate_metrics(self, reports: List[GenerationReport], start_time: datetime) -> SDLCMetrics:
        """Calculate SDLC metrics from generation reports"""
        metrics = SDLCMetrics()
        metrics.total_execution_time = (datetime.now() - start_time).total_seconds()
        
        # Aggregate gate statistics
        for report in reports:
            metrics.gates_executed += report.total_gates
            metrics.gates_passed += report.passed_gates
            metrics.gates_failed += report.failed_gates
            
            # Store generation-specific results
            metrics.generation_results[report.generation.value] = {
                "total_gates": report.total_gates,
                "passed_gates": report.passed_gates,
                "failed_gates": report.failed_gates,
                "success_rate": report.calculate_success_rate(),
                "execution_time": report.execution_time,
                "is_passed": report.is_passed
            }
            
            # Extract specific metrics
            for gate_result in report.gate_results:
                if gate_result.gate_name == "security_patterns":
                    security_details = gate_result.details or {}
                    metrics.security_issues += security_details.get("vulnerability_count", 0)
                elif gate_result.gate_name == "performance_optimization":
                    metrics.performance_score = gate_result.score or 0.0
        
        # Calculate overall quality score
        metrics.quality_score = metrics.calculate_overall_success_rate()
        
        return metrics
    
    def get_deployment_readiness(self) -> Dict[str, Any]:
        """Assess deployment readiness based on quality gates"""
        if not self.metrics.generation_results:
            return {
                "ready": False,
                "reason": "No quality gates executed",
                "recommendations": ["Execute quality gates first"]
            }
        
        # Check Generation 1 (mandatory for deployment)
        gen1_result = self.metrics.generation_results.get("gen1_basic", {})
        if not gen1_result.get("is_passed", False):
            return {
                "ready": False,
                "reason": "Generation 1 (Basic) quality gates failed",
                "recommendations": [
                    "Fix basic syntax and structure issues",
                    "Ensure all imports work correctly",
                    "Complete documentation requirements"
                ]
            }
        
        # Check for critical security issues
        if self.metrics.security_issues > 0:
            return {
                "ready": False,
                "reason": f"Security vulnerabilities found: {self.metrics.security_issues}",
                "recommendations": [
                    "Fix all security vulnerabilities",
                    "Review security patterns implementation", 
                    "Update vulnerable dependencies"
                ]
            }
        
        # Assess overall quality
        overall_success = self.metrics.calculate_overall_success_rate()
        if overall_success < 85.0:
            return {
                "ready": False,
                "reason": f"Overall quality score too low: {overall_success:.1f}%",
                "recommendations": [
                    "Improve failing quality gates",
                    "Address performance optimization issues",
                    "Enhance error handling and logging"
                ]
            }
        
        # Check Generation 2 (robustness)
        gen2_result = self.metrics.generation_results.get("gen2_robust", {})
        if gen2_result and not gen2_result.get("is_passed", False):
            return {
                "ready": "conditional",
                "reason": "Generation 2 (Robustness) has issues",
                "recommendations": [
                    "Address error handling gaps",
                    "Fix security vulnerabilities", 
                    "Improve logging and monitoring"
                ],
                "deployment_risk": "medium"
            }
        
        # Check Generation 3 (scalability)
        gen3_result = self.metrics.generation_results.get("gen3_optimized", {})
        if gen3_result and not gen3_result.get("is_passed", False):
            return {
                "ready": "conditional",
                "reason": "Generation 3 (Scalability) has issues",
                "recommendations": [
                    "Optimize performance bottlenecks",
                    "Implement caching strategies",
                    "Improve resource efficiency"
                ],
                "deployment_risk": "low"
            }
        
        # All checks passed
        return {
            "ready": True,
            "reason": "All quality gates passed",
            "quality_score": overall_success,
            "deployment_confidence": "high"
        }
    
    def export_metrics(self, output_path: Path) -> None:
        """Export SDLC metrics to JSON file"""
        try:
            export_data = {
                "sdlc_metrics": {
                    "total_execution_time": self.metrics.total_execution_time,
                    "gates_executed": self.metrics.gates_executed,
                    "gates_passed": self.metrics.gates_passed,
                    "gates_failed": self.metrics.gates_failed,
                    "security_issues": self.metrics.security_issues,
                    "performance_score": self.metrics.performance_score,
                    "quality_score": self.metrics.quality_score,
                    "overall_success_rate": self.metrics.calculate_overall_success_rate(),
                    "timestamp": self.metrics.timestamp.isoformat()
                },
                "generation_results": self.metrics.generation_results,
                "deployment_readiness": self.get_deployment_readiness(),
                "execution_history_count": len(self.execution_history)
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"📊 SDLC metrics exported to {output_path}")
            
        except Exception as e:
            print(f"Failed to export metrics: {e}")
    
    def generate_ci_cd_config(self) -> Dict[str, Any]:
        """Generate CI/CD configuration based on quality gate results"""
        config = {
            "version": "2.1",
            "jobs": {
                "quality_gates": {
                    "docker": [{"image": "python:3.9"}],
                    "steps": [
                        "checkout",
                        {
                            "run": {
                                "name": "Install Dependencies",
                                "command": "pip install -r requirements.txt"
                            }
                        },
                        {
                            "run": {
                                "name": "Run Progressive Quality Gates",
                                "command": "python3 progressive_gates_standalone.py"
                            }
                        }
                    ]
                }
            },
            "workflows": {
                "autonomous_sdlc": {
                    "jobs": ["quality_gates"]
                }
            }
        }
        
        # Add conditional deployment based on quality gates
        deployment_readiness = self.get_deployment_readiness()
        if deployment_readiness.get("ready"):
            config["jobs"]["deploy"] = {
                "docker": [{"image": "python:3.9"}],
                "steps": [
                    "checkout",
                    {
                        "run": {
                            "name": "Deploy Application",
                            "command": "./deploy.sh"
                        }
                    }
                ]
            }
            config["workflows"]["autonomous_sdlc"]["jobs"].append("deploy")
        
        return config


def create_sdlc_summary_report(metrics: SDLCMetrics, reports: List[GenerationReport]) -> str:
    """Create a comprehensive SDLC summary report"""
    report_lines = [
        "=" * 80,
        "🚀 AUTONOMOUS SDLC EXECUTION REPORT",
        "=" * 80,
        f"Execution Time: {metrics.total_execution_time:.2f} seconds",
        f"Total Quality Gates: {metrics.gates_executed}",
        f"Gates Passed: {metrics.gates_passed}",
        f"Gates Failed: {metrics.gates_failed}",
        f"Overall Success Rate: {metrics.calculate_overall_success_rate():.1f}%",
        f"Security Issues Found: {metrics.security_issues}",
        f"Performance Score: {metrics.performance_score:.1f}%",
        "",
        "📊 Generation Breakdown:",
        "-" * 40
    ]
    
    generation_names = {
        "gen1_basic": "Generation 1: Make it Work",
        "gen2_robust": "Generation 2: Make it Robust", 
        "gen3_optimized": "Generation 3: Make it Scale"
    }
    
    for gen_key, gen_data in metrics.generation_results.items():
        gen_name = generation_names.get(gen_key, gen_key)
        status = "✅ PASSED" if gen_data["is_passed"] else "❌ FAILED"
        report_lines.extend([
            f"{gen_name}:",
            f"  Status: {status}",
            f"  Gates: {gen_data['passed_gates']}/{gen_data['total_gates']} passed",
            f"  Success Rate: {gen_data['success_rate']:.1f}%",
            f"  Execution Time: {gen_data['execution_time']:.2f}s",
            ""
        ])
    
    # Add deployment readiness
    sdlc = AutonomousSDLC()
    sdlc.metrics = metrics
    deployment = sdlc.get_deployment_readiness()
    
    report_lines.extend([
        "🚀 Deployment Readiness Assessment:",
        "-" * 35,
        f"Status: {deployment.get('ready', 'Unknown')}",
        f"Reason: {deployment.get('reason', 'N/A')}",
    ])
    
    if deployment.get("recommendations"):
        report_lines.append("Recommendations:")
        for rec in deployment["recommendations"]:
            report_lines.append(f"  • {rec}")
    
    if deployment.get("deployment_risk"):
        report_lines.append(f"Deployment Risk: {deployment['deployment_risk']}")
    
    if deployment.get("deployment_confidence"):
        report_lines.append(f"Deployment Confidence: {deployment['deployment_confidence']}")
    
    report_lines.extend([
        "",
        "🎯 Key Quality Indicators:",
        "-" * 25,
        f"• Code Quality Score: {metrics.quality_score:.1f}%",
        f"• Security Vulnerabilities: {metrics.security_issues}",
        f"• Performance Optimization: {metrics.performance_score:.1f}%",
        f"• Total Gates Executed: {metrics.gates_executed}",
        "",
        "=" * 80,
        "Generated by Terragon Autonomous SDLC System v1.0.0",
        f"Timestamp: {metrics.timestamp.isoformat()}",
        "🚀 Evolutionary Software Development with Progressive Quality Gates",
        "=" * 80
    ])
    
    return "\n".join(report_lines)


def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Terragon Autonomous SDLC System with Progressive Quality Gates"
    )
    
    parser.add_argument(
        "--export-metrics",
        type=str,
        help="Export SDLC metrics to JSON file"
    )
    
    parser.add_argument(
        "--export-report",
        type=str,
        help="Export summary report to text file"
    )
    
    parser.add_argument(
        "--check-deployment",
        action="store_true",
        help="Check deployment readiness and exit"
    )
    
    parser.add_argument(
        "--generate-ci-config",
        type=str,
        help="Generate CI/CD configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser


async def main():
    """Main execution function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 TERRAGON AUTONOMOUS SDLC SYSTEM")
    print("=" * 80)
    print("Evolutionary Software Development with Progressive Quality Gates")
    print("Version: 1.0.0 | Build: Autonomous | Generation: 1-3")
    print("Powered by Terragon Labs Advanced AI Systems")
    print("=" * 80)
    
    try:
        # Execute autonomous SDLC
        print("\n🔄 Executing Autonomous SDLC Process...")
        print("   Phase 1: Generation 1 - Make it Work (Basic functionality)")
        print("   Phase 2: Generation 2 - Make it Robust (Reliability & security)")
        print("   Phase 3: Generation 3 - Make it Scale (Performance & optimization)")
        print("=" * 80)
        
        # Create SDLC instance
        sdlc = AutonomousSDLC()
        metrics, reports = await sdlc.execute_autonomous_sdlc()
        
        # Generate and display summary report
        summary_report = create_sdlc_summary_report(metrics, reports)
        print(summary_report)
        
        # Export metrics if requested
        if args.export_metrics:
            metrics_path = Path(args.export_metrics)
            sdlc.export_metrics(metrics_path)
        
        # Export report if requested
        if args.export_report:
            report_path = Path(args.export_report)
            with open(report_path, 'w') as f:
                f.write(summary_report)
            print(f"📄 Full report exported to: {report_path}")
        
        # Generate CI/CD config if requested
        if args.generate_ci_config:
            ci_config = sdlc.generate_ci_cd_config()
            config_path = Path(args.generate_ci_config)
            with open(config_path, 'w') as f:
                json.dump(ci_config, f, indent=2)
            print(f"⚙️  CI/CD config generated: {config_path}")
        
        # Check deployment readiness
        deployment = sdlc.get_deployment_readiness()
        
        if args.check_deployment:
            print(f"\n🚀 Deployment Status: {deployment.get('ready', 'Unknown')}")
            print(f"Reason: {deployment.get('reason', 'N/A')}")
            if deployment.get("recommendations"):
                print("Recommendations:")
                for rec in deployment["recommendations"]:
                    print(f"  • {rec}")
            
            # Exit with appropriate code
            if deployment.get("ready") is True:
                return 0
            elif deployment.get("ready") == "conditional":
                return 2  # Warning
            else:
                return 1  # Failure
        
        # Determine overall exit code based on results
        overall_success = metrics.calculate_overall_success_rate()
        
        print(f"\n📈 Final Assessment:")
        print(f"   Overall Success Rate: {overall_success:.1f}%")
        print(f"   Deployment Ready: {deployment.get('ready', 'Unknown')}")
        
        if deployment.get("ready") is True and overall_success >= 90.0:
            print("\n🎉 AUTONOMOUS SDLC COMPLETED SUCCESSFULLY!")
            print("✅ All quality gates passed - Ready for autonomous deployment")
            print("🚀 System meets all quality, security, and performance standards")
            return 0
        elif deployment.get("ready") == "conditional" or overall_success >= 75.0:
            print("\n⚠️  AUTONOMOUS SDLC COMPLETED WITH WARNINGS")
            print("🔧 Some issues identified - Review recommendations before deployment")
            print("📋 Conditional deployment possible with monitored rollout")
            return 2
        else:
            print("\n💥 AUTONOMOUS SDLC EXECUTION FAILED")
            print("❌ Critical issues found - Must address before deployment")
            print("🔴 System does not meet minimum quality standards")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n🚨 Autonomous SDLC execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)