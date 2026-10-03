from pathlib import Path

import pytest

from pylops_bench import loading

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def load_class(relpath: str, clsname: str):
    """Load ``clsname`` from a benchmark file given relative to ``benchmarks/``."""
    return loading.load_class(BENCHMARKS_DIR / relpath, clsname)


def instantiate(relpath: str, clsname: str, **overrides):
    return loading.instantiate(BENCHMARKS_DIR / relpath, clsname, **overrides)


@pytest.fixture
def bench_loader():
    return instantiate
