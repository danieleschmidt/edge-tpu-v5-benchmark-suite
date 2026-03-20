"""
PowerEstimator: estimate power draw and efficiency for inference on Edge TPU v5.

Spec reference:
  - Power budget: ~2 W
  - Efficiency: 50 TOPS/W (INT8)
"""
from __future__ import annotations
from .model_benchmark import ModelBenchmark


class PowerEstimator:
    """Estimate power consumption and efficiency for a model on Edge TPU v5.

    Args:
        power_budget_w: Maximum power budget in watts (default 2.0 W).
        efficiency_tops_per_w: Compute efficiency in TOPS/W (default 50.0 TOPS/W).
    """

    def __init__(
        self,
        power_budget_w: float = 2.0,
        efficiency_tops_per_w: float = 50.0,
    ) -> None:
        self.power_budget_w = power_budget_w
        self.efficiency_tops_per_w = efficiency_tops_per_w

    def estimate(
        self,
        model_benchmark: ModelBenchmark,
        batch_size: int = 1,
    ) -> dict:
        """Estimate power and efficiency metrics.

        Returns:
            dict with keys:
              - power_w (float): estimated power draw in watts
              - tops_utilized (float): effective TOPS being used
              - tops_per_watt (float): achieved efficiency
              - within_budget (bool): True if power_w <= power_budget_w
        """
        ops = model_benchmark.op_count() * batch_size  # MACs
        # Assume utilisation of 1 second worth of compute
        tops_utilized = ops / 1e12  # TOPS for this inference

        if self.efficiency_tops_per_w > 0:
            power_w = tops_utilized / self.efficiency_tops_per_w
        else:
            power_w = float("inf")

        tops_per_watt = (
            tops_utilized / power_w if power_w > 0 else self.efficiency_tops_per_w
        )

        return {
            "power_w": power_w,
            "tops_utilized": tops_utilized,
            "tops_per_watt": tops_per_watt,
            "within_budget": power_w <= self.power_budget_w,
        }

    def __repr__(self) -> str:
        return (
            f"PowerEstimator(power_budget_w={self.power_budget_w}, "
            f"efficiency_tops_per_w={self.efficiency_tops_per_w})"
        )
