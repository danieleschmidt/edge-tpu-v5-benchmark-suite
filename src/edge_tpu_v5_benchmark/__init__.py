"""Edge TPU v5 Benchmark Suite

First comprehensive open-source benchmark harness for Google's TPU v5 edge cards.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .benchmark import TPUv5Benchmark
from .models import ModelLoader
from .power import PowerProfiler

__all__ = [
    "TPUv5Benchmark",
    "ModelLoader", 
    "PowerProfiler",
    "__version__"
]