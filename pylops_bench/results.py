"""Turn raw BenchOpt outputs into the tidy, per-benchmark result tables.

Each benchmark has a single table ``results/<benchmark>.csv`` with one row
per (objective, dataset, dataset parameters, solver, solver parameters)
configuration. Re-running a configuration replaces its row (upsert) and
updates its ``run_date``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["MAE", "PSNR"]
ID_COLUMNS = ["objective", "dataset", "solver"]


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _strip_params(name: str) -> str:
    """``"FISTA[eps=0.1]"`` -> ``"FISTA"``."""
    return str(name).split("[", 1)[0]


def _norm(value) -> str:
    """Normalise a value so that keys compare equal after a CSV round trip."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return str(value)


def param_columns(df: pd.DataFrame) -> list[str]:
    """Columns holding objective, dataset and solver parameters."""
    prefixes = ("objective_param_", "dataset_param_", "solver_param_")
    return sorted(c for c in df.columns if c.startswith(prefixes))


def key_columns(df: pd.DataFrame) -> list[str]:
    return ID_COLUMNS + param_columns(df)


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a raw BenchOpt result table into one row per configuration.

    Metrics are averaged over repetitions and the time is the median over
    repetitions. Versions of PyLops and BenchOpt are taken from the current
    environment, so this function must run in the environment used for the
    benchmark.
    """
    missing = [f"objective_{m}" for m in METRICS if f"objective_{m}" not in raw]
    if missing:
        msg = f"raw results are missing the metric columns {missing}"
        raise KeyError(msg)

    df = pd.DataFrame(
        {
            "objective": raw["objective_name"].map(_strip_params),
            "dataset": raw["dataset_name"].map(_strip_params),
            "solver": raw["solver_name"].map(_strip_params),
        }
    )
    renames = {"p_obj_": "objective_param_", "p_dataset_": "dataset_param_"}
    renames["p_solver_"] = "solver_param_"
    for col in raw.columns:
        for old, new in renames.items():
            if col.startswith(old):
                df[new + col[len(old) :]] = raw[col]
    for m in METRICS:
        df[m] = raw[f"objective_{m}"]
    df["time_s"] = raw["time"]
    df["run_date"] = pd.to_datetime(raw["run_date"])

    keys = key_columns(df)
    # group on normalised keys so that NaN parameters (not used by a given
    # solver) do not drop rows
    group = df[keys].apply(lambda col: col.map(_norm))
    df["_key"] = group.apply(tuple, axis=1)
    grouped = df.groupby("_key", sort=False)
    out = grouped.agg(
        {
            **{k: "first" for k in keys},
            **{m: "mean" for m in METRICS},
            "time_s": "median",
            "run_date": "max",
        }
    ).reset_index(drop=True)
    out["n_repetitions"] = grouped.size().to_numpy()
    out["run_date"] = out["run_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["pylops_version"] = _package_version("pylops")
    out["benchopt_version"] = _package_version("benchopt")
    return _order(out)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    keys = key_columns(df)
    rest = [c for c in df.columns if c not in keys]
    first = ["MAE", "PSNR", "time_s", "n_repetitions"]
    rest = [c for c in first if c in rest] + [c for c in rest if c not in first]
    df = df[keys + rest]
    sort_key = df[keys].apply(lambda col: col.map(_norm))
    order = sort_key.apply(tuple, axis=1).sort_values().index
    return df.loc[order].reset_index(drop=True)


def upsert(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Merge ``new`` rows into ``existing``, replacing rows with equal keys."""
    if existing is None or existing.empty:
        return _order(new.copy())
    merged = pd.concat([existing, new], ignore_index=True)
    keys = key_columns(merged)
    norm_keys = merged[keys].apply(lambda col: col.map(_norm))
    keep = ~norm_keys.duplicated(keep="last")
    return _order(merged[keep])


def results_path(results_dir: Path | str, benchmark: str) -> Path:
    return Path(results_dir) / f"{benchmark}.csv"


def update_results(
    raw_csv: Path | str,
    benchmark: str,
    results_dir: Path | str = "results",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Tidy ``raw_csv`` and upsert it into ``results/<benchmark>.csv``."""
    new = tidy(pd.read_csv(raw_csv))
    path = results_path(results_dir, benchmark)
    existing = pd.read_csv(path) if path.exists() else None
    merged = upsert(existing, new)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
    return merged
