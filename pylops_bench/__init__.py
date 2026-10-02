"""Shared utilities for the PyLops benchmarks."""

from pylops_bench.metrics import compute_metrics

try:
    from pylops_bench.version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["compute_metrics"]
