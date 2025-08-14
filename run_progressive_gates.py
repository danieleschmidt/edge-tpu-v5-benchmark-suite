#!/usr/bin/env python3
"""
Progressive Quality Gates Runner for Terragon Autonomous SDLC

This script executes progressive quality gates across three generations:
- Generation 1: Make it Work (Basic functionality)
- Generation 2: Make it Robust (Reliability and error handling)
- Generation 3: Make it Scale (Performance and optimization)
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path("src").absolute()))

from edge_tpu_v5_benchmark.progressive_quality_gates import (
    run_progressive_quality_gates,
    Generation,
    GateStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    print("\n📄 Detailed reports saved to: quality_gate_reports.json")
    return 0 if all_passed else 1


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