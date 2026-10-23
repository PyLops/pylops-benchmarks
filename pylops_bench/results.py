"""Turn raw BenchOpt outputs into the tidy, per-benchmark result tables.

Each benchmark has a single table ``results/<benchmark>.csv`` with one row
per (objective, dataset, dataset parameters, solver, solver parameters)
configuration. Re-running a configuration replaces its row (upsert) and
updates its ``run_date``.

If the raw results come from a parquet file with the solver estimates (saved
by ``Objective.save_final_results``), each estimate is written to
``estimates/<benchmark>/<key>.npz`` and its file name is stored in the
``estimate`` column of the table.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from benchopt.results import read_results

METRICS = ["MAE", "PSNR"]
ID_COLUMNS = ["objective", "dataset", "solver"]
ESTIMATE = "estimate"


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


def estimate_key(row: pd.Series, keys: list[str]) -> str:
    """Short hash of the configuration of ``row``.

    Empty values are skipped, so that the key does not change when other
    solvers add parameter columns to the table.
    """
    items = sorted((k, _norm(row[k])) for k in keys if _norm(row[k]) != "")
    return hashlib.md5(repr(items).encode()).hexdigest()[:12]


def _last_valid(values: pd.Series):
    """Last value that is not missing (estimates of the last repetition)."""
    for value in reversed(values.tolist()):
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return None


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
    estimates = "final_results" in raw
    if estimates:
        df["_estimate"] = raw["final_results"]

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
    if estimates:
        # ``_estimate`` holds arrays: it is dropped by ``update_results``
        # once they are saved, and never written to the result table
        out["_estimate"] = [_last_valid(s) for _, s in grouped["_estimate"]]
    out["run_date"] = out["run_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["pylops_version"] = _package_version("pylops")
    out["benchopt_version"] = _package_version("benchopt")
    return _order(out)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    keys = key_columns(df)
    rest = [c for c in df.columns if c not in keys]
    first = ["MAE", "PSNR", "time_s", "n_repetitions", ESTIMATE]
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


def _compact(x) -> np.ndarray:
    """Single precision copy of an estimate, to halve its size on disk."""
    x = np.asarray(x)
    return x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)


def save_estimates(
    new: pd.DataFrame, merged: pd.DataFrame, estimates_dir: Path | str
) -> None:
    """Write the estimates of ``new`` in ``estimates_dir`` and remove the
    files no longer referenced by the ``merged`` result table."""
    estimates_dir = Path(estimates_dir)
    estimates_dir.mkdir(parents=True, exist_ok=True)
    if ESTIMATE in new:
        for _, row in new.dropna(subset=[ESTIMATE]).iterrows():
            x = _compact(row["_estimate"])
            np.savez_compressed(estimates_dir / row[ESTIMATE], x=x)
    used = set(merged[ESTIMATE].dropna()) if ESTIMATE in merged else set()
    for path in estimates_dir.glob("*.npz"):
        if path.name not in used:
            path.unlink()


def update_results(
    raw_file: Path | str,
    benchmark: str,
    results_dir: Path | str = "results",
    estimates_dir: Path | str | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Tidy ``raw_file`` (``.csv`` or ``.parquet``) and upsert it into
    ``results/<benchmark>.csv``.

    If ``estimates_dir`` is given, the solver estimates held by the raw file
    are saved in ``estimates_dir/<benchmark>/``, and the files there that
    are no longer referenced by the table are removed.
    """
    new = tidy(read_results(raw_file))
    if estimates_dir is not None and "_estimate" in new:
        keys = key_columns(new)
        has_estimate = new["_estimate"].notna()
        new.loc[has_estimate, ESTIMATE] = [
            f"{estimate_key(row, keys)}.npz" for _, row in new[has_estimate].iterrows()
        ]
        new = _order(new)
    path = results_path(results_dir, benchmark)
    existing = pd.read_csv(path) if path.exists() else None
    merged = upsert(existing, new.drop(columns="_estimate", errors="ignore"))
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
        if estimates_dir is not None:
            save_estimates(new, merged, Path(estimates_dir) / benchmark)
    return merged
