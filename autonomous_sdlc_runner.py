#!/usr/bin/env python3
"""
Autonomous SDLC Runner

This is the main entry point for executing the Terragon Autonomous SDLC system
with progressive quality gates and comprehensive reporting.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path("src").absolute()))

from edge_tpu_v5_benchmark.autonomous_sdlc import (
    run_autonomous_sdlc,
    create_sdlc_summary_report,
    AutonomousSDLC
)


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
    print("=" * 80)
    
    try:
        # Execute autonomous SDLC
        print("\n🔄 Executing Autonomous SDLC...")
        metrics, reports = await run_autonomous_sdlc()
        
        # Create SDLC instance for additional operations
        sdlc = AutonomousSDLC()
        sdlc.metrics = metrics
        
        # Generate summary report
        summary_report = create_sdlc_summary_report(metrics, reports)
        print(summary_report)
        
        # Export metrics if requested
        if args.export_metrics:
            metrics_path = Path(args.export_metrics)
            sdlc.export_metrics(metrics_path)
            print(f"\n📊 Metrics exported to: {metrics_path}")
        
        # Export report if requested
        if args.export_report:
            report_path = Path(args.export_report)
            with open(report_path, 'w') as f:
                f.write(summary_report)
            print(f"📄 Report exported to: {report_path}")
        
        # Generate CI/CD config if requested
        if args.generate_ci_config:
            import json
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
        
        # Determine overall exit code
        overall_success = metrics.calculate_overall_success_rate()
        
        if deployment.get("ready") is True and overall_success >= 90.0:
            print("\n🎉 AUTONOMOUS SDLC COMPLETED SUCCESSFULLY!")
            print("✅ All quality gates passed - Ready for deployment")
            return 0
        elif deployment.get("ready") == "conditional" or overall_success >= 75.0:
            print("\n⚠️  AUTONOMOUS SDLC COMPLETED WITH WARNINGS")
            print("🔧 Some issues found - Review recommendations")
            return 2
        else:
            print("\n💥 AUTONOMOUS SDLC FAILED")
            print("❌ Critical issues found - Address before deployment")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n🚨 Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)