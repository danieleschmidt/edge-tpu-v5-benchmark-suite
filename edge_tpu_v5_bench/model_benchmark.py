"""
ModelBenchmark: compute parameter counts, operation counts, and memory estimates
for neural network models specified via layer specs.
"""
from __future__ import annotations
from typing import List, Dict, Any


class ModelBenchmark:
    """Benchmarks a neural network model described by a list of layer specs.

    Each layer spec is a dict with key 'type' and type-specific fields:
      - conv: 'in_channels', 'out_channels', 'kernel_size' (int or [h,w])
      - linear: 'in_features', 'out_features'
      - depthwise_conv: 'channels', 'kernel_size' (in_channels == out_channels)
    """

    def __init__(self, model_name: str, layer_specs: List[Dict[str, Any]]) -> None:
        self.model_name = model_name
        self.layer_specs = layer_specs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kernel_area(kernel_size) -> int:
        if isinstance(kernel_size, (list, tuple)):
            result = 1
            for k in kernel_size:
                result *= int(k)
            return result
        return int(kernel_size) ** 2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def param_count(self) -> int:
        """Return the total number of trainable parameters."""
        total = 0
        for spec in self.layer_specs:
            ltype = spec.get("type", "").lower()
            if ltype in ("conv", "conv2d"):
                kw = self._kernel_area(spec.get("kernel_size", 3))
                total += spec["in_channels"] * spec["out_channels"] * kw
                total += spec["out_channels"]  # bias
            elif ltype in ("depthwise_conv", "depthwise"):
                channels = spec.get("channels", spec.get("in_channels", 1))
                kw = self._kernel_area(spec.get("kernel_size", 3))
                total += channels * kw
                total += channels  # bias
            elif ltype in ("linear", "fc", "dense"):
                total += spec["in_features"] * spec["out_features"]
                total += spec["out_features"]  # bias
            elif ltype in ("batchnorm", "bn", "batchnorm2d"):
                channels = spec.get("num_features", spec.get("channels", 1))
                total += 2 * channels  # gamma + beta
            # Unknown layers contribute 0 params (safe default)
        return total

    def op_count(self) -> int:
        """Return the total number of multiply-accumulate operations (MACs).

        For a spatial conv layer we assume a default spatial output of 1×1
        (i.e. we count flops per output element only), unless 'output_size'
        is provided in the spec as [H, W].
        """
        total = 0
        for spec in self.layer_specs:
            ltype = spec.get("type", "").lower()
            out_h, out_w = spec.get("output_size", [1, 1])
            if ltype in ("conv", "conv2d"):
                kw = self._kernel_area(spec.get("kernel_size", 3))
                # MACs per output pixel: in_channels * kernel_h * kernel_w
                macs_per_pixel = spec["in_channels"] * kw
                total += macs_per_pixel * spec["out_channels"] * out_h * out_w
            elif ltype in ("depthwise_conv", "depthwise"):
                channels = spec.get("channels", spec.get("in_channels", 1))
                kw = self._kernel_area(spec.get("kernel_size", 3))
                total += channels * kw * out_h * out_w
            elif ltype in ("linear", "fc", "dense"):
                total += spec["in_features"] * spec["out_features"]
            elif ltype in ("batchnorm", "bn", "batchnorm2d"):
                channels = spec.get("num_features", spec.get("channels", 1))
                total += 2 * channels  # normalize + scale
            # Unknown layers: 0 ops
        return total

    def memory_mb(self, dtype_bytes: int = 1) -> float:
        """Estimate the memory required to store all parameters in MB.

        Args:
            dtype_bytes: bytes per element (1=INT8, 2=FP16, 4=FP32).
        """
        return self.param_count() * dtype_bytes / (1024 ** 2)

    def __repr__(self) -> str:
        return (
            f"ModelBenchmark(name={self.model_name!r}, "
            f"params={self.param_count()}, ops={self.op_count()})"
        )
