"""Upsert a raw BenchOpt result file into ``results/<benchmark>.csv``.

Usage::

    python scripts/collect_results.py <benchmark> <raw.parquet> [--dry-run]

``<benchmark>`` is the name of the folder in ``benchmarks/``. Run this
script in the same environment used to run the benchmark, as the PyLops and
BenchOpt versions are read from it.

The raw file can be a ``.csv`` or a ``.parquet`` file, but only the latter
holds the solver estimates, which are saved in
``<estimates-dir>/<benchmark>/`` and plotted on the benchmark page.
"""

import argparse
from pathlib import Path

import pandas as pd

from pylops_bench.results import update_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("benchmark", help="benchmark folder name in benchmarks/")
    parser.add_argument(
        "raw_file", type=Path, help="raw BenchOpt output (.parquet or .csv)"
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--estimates-dir", type=Path, default=ROOT / "estimates")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the table, do not write it"
    )
    args = parser.parse_args()

    if not (ROOT / "benchmarks" / args.benchmark / "objective.py").exists():
        parser.error(f"unknown benchmark {args.benchmark!r}")

    table = update_results(
        args.raw_file,
        args.benchmark,
        args.results_dir,
        estimates_dir=args.estimates_dir,
        dry_run=args.dry_run,
    )
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table)
    if not args.dry_run:
        print(f"\nUpdated {args.results_dir / (args.benchmark + '.csv')}")


if __name__ == "__main__":
    main()
