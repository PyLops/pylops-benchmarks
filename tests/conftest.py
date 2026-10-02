import importlib.util
from pathlib import Path

import pytest

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def load_class(relpath: str, clsname: str):
    """Load ``clsname`` from a benchmark file given relative to ``benchmarks/``."""
    path = BENCHMARKS_DIR / relpath
    modname = "_bench_" + "_".join(path.with_suffix("").parts[-3:])
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, clsname)


def default_parameters(cls) -> dict:
    """First value of every parameter of a benchopt class."""
    params = {}
    for key, values in getattr(cls, "parameters", {}).items():
        names = [k.strip() for k in key.split(",")]
        value = values[0]
        if len(names) == 1:
            params[names[0]] = value
        else:
            params.update(dict(zip(names, value)))
    return params


def instantiate(relpath: str, clsname: str, **overrides):
    cls = load_class(relpath, clsname)
    params = {**default_parameters(cls), **overrides}
    return cls.get_instance(**params)


@pytest.fixture
def bench_loader():
    return instantiate
