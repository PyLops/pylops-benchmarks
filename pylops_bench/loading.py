"""Load BenchOpt classes from benchmark files outside of ``benchopt run``.

Used by the tests and by the website, which needs to rebuild the datasets
listed in the result tables to plot them.
"""

import importlib.util
import re
from pathlib import Path

NAME_RE = re.compile(r"""^\s*name\s*=\s*["'](.+?)["']""", re.MULTILINE)


def load_class(path: str | Path, clsname: str):
    """Load class ``clsname`` from the Python file ``path``."""
    path = Path(path)
    modname = "_bench_" + "_".join(path.with_suffix("").parts[-3:])
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, clsname)


def default_parameters(cls) -> dict:
    """First value of every parameter of a BenchOpt class."""
    params = {}
    for key, values in getattr(cls, "parameters", {}).items():
        names = [k.strip() for k in key.split(",")]
        value = values[0]
        if len(names) == 1:
            params[names[0]] = value
        else:
            params.update(dict(zip(names, value)))
    return params


def instantiate(path: str | Path, clsname: str, **overrides):
    """Instantiate ``clsname`` from ``path`` with its default parameters,
    updated with ``overrides``."""
    cls = load_class(path, clsname)
    params = {**default_parameters(cls), **overrides}
    return cls.get_instance(**params)


def find_by_name(folder: str | Path, name: str) -> Path | None:
    """Return the file in ``folder`` whose class ``name`` attribute is
    ``name`` (e.g. the dataset file of a dataset listed in a result table)."""
    for path in sorted(Path(folder).glob("*.py")):
        match = NAME_RE.search(path.read_text())
        if match and match.group(1) == name:
            return path
    return None
