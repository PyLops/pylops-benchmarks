"""Sphinx extension generating one page per benchmark from ``results/*.csv``.

At ``builder-inited`` it writes, in ``auto_benchmarks/``:

- ``index.md``: a summary table with the best solver of every benchmark;
- ``<benchmark>.md``: the benchmark README followed, for every dataset
  configuration, by the dataset docstring and a leaderboard sorted by PSNR;
- ``figures/*.png``: the true model and data of every dataset configuration
  in the result table, drawn by the optional ``plot`` method of the dataset.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import shutil
from pathlib import Path

import matplotlib
import pandas as pd
from sphinx.util import logging

from pylops_bench.loading import NAME_RE, default_parameters, find_by_name, load_class

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTDIR = "auto_benchmarks"
FIGDIR = "figures"

logger = logging.getLogger(__name__)


def _fmt(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _params(row: pd.Series, prefix: str) -> str:
    items = [
        f"{col[len(prefix) :]}={_fmt(row[col])}"
        for col in row.index
        if col.startswith(prefix) and not pd.isna(row[col])
    ]
    return ", ".join(items)


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def _solver_files(bench_dir: Path) -> dict[str, str]:
    """Map solver names to their file names, read from the class ``name``."""
    files = {}
    for path in sorted((bench_dir / "solvers").glob("*.py")):
        match = NAME_RE.search(path.read_text())
        if match:
            files[match.group(1)] = path.name
    return files


def _dataset_params(row: pd.Series, defaults: dict) -> dict:
    """Dataset parameters of a result row, cast to the type of the default
    value (pandas turns integer columns with missing values into floats)."""
    params = {}
    for col in row.index:
        if not col.startswith("dataset_param_") or pd.isna(row[col]):
            continue
        key = col[len("dataset_param_") :]
        value = row[col]
        value = value.item() if hasattr(value, "item") else value
        default = defaults.get(key)
        if isinstance(default, (bool, int, float, str)):
            value = type(default)(value)
        params[key] = value
    return params


def _dataset_class(bench_dir: Path, name: str):
    """Dataset class called ``name``, or ``None`` if it is not found."""
    path = find_by_name(bench_dir / "datasets", name)
    return None if path is None else load_class(path, "Dataset")


def _figure(cls, bench_dir: Path, row: pd.Series, figdir: Path) -> str | None:
    """Plot the true model and data of the dataset configuration in ``row``
    and return the figure path relative to the benchmark page, or ``None``
    if the dataset has no ``plot`` method."""
    if not hasattr(cls, "plot"):
        return None
    name = row["dataset"]
    params = _dataset_params(row, default_parameters(cls))
    slug = re.sub(r"\W+", "_", name.lower()).strip("_")
    key = hashlib.md5(repr(sorted(params.items())).encode()).hexdigest()[:8]
    fname = f"{bench_dir.name}_{slug}_{key}.png"
    try:
        dataset = cls.get_instance(**params)
        fig = dataset.plot(**dataset.get_data())
        fig.savefig(figdir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # a broken plot must not block the results
        logger.info(f"generate_benchmarks: cannot plot {name!r} ({params}): {exc}")
        return None
    return f"{FIGDIR}/{fname}"


def _readme(bench_dir: Path) -> tuple[str, str]:
    """Return title and body of the benchmark README."""
    readme = bench_dir / "README.md"
    title = bench_dir.name.replace("_", " ").capitalize()
    if not readme.exists():
        return title, ""
    lines = readme.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        title, lines = lines[0][2:].strip(), lines[1:]
    return title, "\n".join(lines).strip()


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |"]
    out.append("|" + "|".join(" --- " for _ in header) + "|")
    out += ["| " + " | ".join(_escape(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _leaderboard(df: pd.DataFrame, solver_url) -> str:
    df = df.sort_values("PSNR", ascending=False)
    rows = []
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        url = solver_url(row["solver"])
        solver = f"[{row['solver']}]({url})" if url else row["solver"]
        rows.append(
            [
                str(rank),
                solver,
                f"{row['MAE']:.4g}",
                f"{row['PSNR']:.2f}",
                f"{row['time_s']:.3f}",
            ]
        )
    header = ["Rank", "Solver", "MAE", "PSNR [dB]", "Time [s]"]
    return _table(header, rows)


def _benchmark_page(
    app, bench_dir: Path, df: pd.DataFrame | None, figdir: Path
) -> str:
    cfg = app.config
    name = bench_dir.name
    title, body = _readme(bench_dir)
    files = _solver_files(bench_dir)
    blob = f"{cfg.github_url}/blob/{cfg.github_branch}"
    tree = f"{cfg.github_url}/tree/{cfg.github_branch}/benchmarks/{name}"

    def solver_url(solver):
        if solver not in files:
            return None
        return f"{blob}/benchmarks/{name}/solvers/{files[solver]}"

    lines = [f"# {title}", "", body, "", f"Source: [`benchmarks/{name}`]({tree})."]
    lines += ["", "## Results", ""]
    if df is None or df.empty:
        lines.append("This benchmark has not been run yet.")
        return "\n".join(lines) + "\n"

    csv_url = f"{blob}/results/{name}.csv"
    lines += [f"Raw results: [`results/{name}.csv`]({csv_url}).", ""]
    dataset_cols = ["objective", "dataset"] + [
        c for c in df.columns if c.startswith("dataset_param_")
    ]
    groups = df.groupby(dataset_cols, dropna=False, sort=True)
    for _, group in groups:
        first = group.iloc[0]
        params = _params(first, "dataset_param_")
        lines += [f"### {first['dataset']}", ""]
        cls = _dataset_class(bench_dir, first["dataset"])
        if cls is not None and cls.__doc__:
            lines += [inspect.cleandoc(cls.__doc__), ""]
        if params:
            lines += [f"Dataset parameters: `{params}`", ""]
        figure = None if cls is None else _figure(cls, bench_dir, first, figdir)
        if figure:
            lines += [f"![True model and data of {first['dataset']}]({figure})", ""]
        lines += [_leaderboard(group, solver_url), ""]
    return "\n".join(lines) + "\n"


def _summary(entries: list[tuple[str, str, pd.DataFrame | None]]) -> str:
    rows = []
    for name, title, df in entries:
        link = f"[{title}]({name}.md)"
        if df is None or df.empty:
            rows.append([link, "-", "-", "-", "not run yet"])
            continue
        best = df.loc[df["PSNR"].idxmax()]
        rows.append(
            [
                link,
                str(df["solver"].nunique()),
                str(len(df)),
                best["solver"],
                f"{best['PSNR']:.2f}",
            ]
        )
    header = [
        "Benchmark",
        "Solvers",
        "Configurations",
        "Best solver (PSNR)",
        "Best PSNR [dB]",
    ]
    return _table(header, rows)


def generate(app) -> None:
    cfg = app.config
    outdir = Path(app.srcdir) / OUTDIR
    shutil.rmtree(outdir, ignore_errors=True)
    figdir = outdir / FIGDIR
    figdir.mkdir(parents=True)

    entries = []
    bench_dirs = sorted(
        p.parent
        for p in Path(cfg.benchmarks_dir).glob("*/objective.py")
        if not p.parent.name.startswith("_")
    )
    for bench_dir in bench_dirs:
        csv = Path(cfg.results_dir) / f"{bench_dir.name}.csv"
        df = pd.read_csv(csv) if csv.exists() else None
        page = _benchmark_page(app, bench_dir, df, figdir)
        (outdir / f"{bench_dir.name}.md").write_text(page)
        entries.append((bench_dir.name, _readme(bench_dir)[0], df))

    index = ["# Benchmarks", ""]
    if entries:
        index += [_summary(entries), ""]
    else:
        index += ["No benchmarks found.", ""]
    index += ["```{toctree}", ":hidden:", ""]
    index += [name for name, _, _ in entries]
    index += ["```", ""]
    (outdir / "index.md").write_text("\n".join(index))


def setup(app):
    app.add_config_value("benchmarks_dir", "", "env")
    app.add_config_value("results_dir", "", "env")
    app.add_config_value("github_url", "", "env")
    app.add_config_value("github_branch", "main", "env")
    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
