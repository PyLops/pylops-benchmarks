"""Find the benchmarks to run on CI and write them to ``$GITHUB_OUTPUT``.

A benchmark is a folder ``benchmarks/<name>/`` containing an ``objective.py``.
Folders starting with ``_`` (e.g. ``_template``) are skipped.

Selection:

- ``--benchmark NAME``: only that benchmark (``all`` selects every benchmark);
- ``--all``: every benchmark;
- ``--base REF``: benchmarks with files changed between ``REF`` and ``HEAD``.
  Changes to shared code (``pylops_bench/``, ``pyproject.toml``, ``uv.lock``)
  select every benchmark. If ``REF`` is missing or invalid, every benchmark
  is selected.

Outputs ``dirs`` (JSON list of benchmark names) and ``found`` (``true`` or
``false``).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = ("pylops_bench/", "pyproject.toml", "uv.lock")


def all_benchmarks() -> list[str]:
    return sorted(
        p.parent.name
        for p in (ROOT / "benchmarks").glob("*/objective.py")
        if not p.parent.name.startswith("_")
    )


def changed_files(base: str) -> list[str] | None:
    if not base or set(base) == {"0"}:
        return None
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return out.splitlines()


def select(benchmarks: list[str], files: list[str] | None) -> list[str]:
    if files is None or any(f.startswith(SHARED) for f in files):
        return benchmarks
    return [
        b for b in benchmarks if any(f.startswith(f"benchmarks/{b}/") for f in files)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--benchmark", default="")
    parser.add_argument("--base", default="")
    args = parser.parse_args()

    benchmarks = all_benchmarks()
    if args.all or args.benchmark == "all":
        selected = benchmarks
    elif args.benchmark:
        if args.benchmark not in benchmarks:
            parser.error(f"unknown benchmark {args.benchmark!r}, have {benchmarks}")
        selected = [args.benchmark]
    else:
        selected = select(benchmarks, changed_files(args.base))

    print(f"Selected benchmarks: {selected}")
    lines = [f"dirs={json.dumps(selected)}", f"found={str(bool(selected)).lower()}"]
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
