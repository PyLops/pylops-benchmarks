# Session summary: scaffolding pylops-benchmarks

**Date:** 2026-10-02
**Outcome:** PR [#1](https://github.com/PyLops/pylops-benchmarks/pull/1) from branch `skeleton`

## Goal

The goal was a benchmark repository for PyLops, modelled on
[deepinv/benchmarks](https://github.com/deepinv/benchmarks) and run with
[BenchOpt](https://benchopt.github.io).

- **Problem:** each benchmark is a PyLops problem made of a true model, an operator and the modelled data, plus optional noise.
- **Solvers:** several combinations of solver and regularizer/preconditioner are run on each problem.
- **Metrics:** every result is scored against the true model with PyLops MAE and PSNR.
- **First use case:** `SeismicInterpolation.ipynb`, with 3 inversion types.
- **CI:** the benchmarks run on CI, results are stored as CSV files in the repo, and a script builds a simple website from them.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| Solver sampling | `run_once` with a fixed `niter`; only final results are recorded, not convergence curves |
| Results storage | One CSV per benchmark (`results/<name>.csv`). Rerunning a configuration updates its row and its `run_date` timestamp. No history or regression tracking |
| Website | A minimal standalone Sphinx site in this repo (not part of the PyLops docs), using the `shibuya` theme like PyLops |
| Benchmark CI | Runs on push to `main` and on manual dispatch, plus a smoke test on PRs |
| Docs CI | A separate workflow, run manually and on a weekly cron |
| PSNR | `psnr(x_true, x_est, xmax=None, xmin=None)`, so it uses the full dynamic range of the signed seismic data |
| Randomness | Seeds are always fixed, both for the operator (subsampling mask) and for the noise |
| Solvers | Operators are built explicitly; an exact match with `SeismicInterpolation` is not required |
| Tooling | uv as project manager; hatchling + hatch-vcs build, as in PyLops; no ruff |
| Notebooks | `benchmarks/*/notebooks/` is gitignored, so each benchmark can have local test notebooks |

## What was built

```
pylops_bench/          shared package: metrics.compute_metrics (MAE, PSNR), results (tidy + upsert)
benchmarks/
  _template/           copy-me skeleton (blocky 1D signal, Convolve1D, CGLS)
  seismic_interpolation/
    objective.py       "Reconstruction": evaluate_result -> MAE, PSNR
    datasets/linear_events.py   3 linear events, seeded Restriction (40% traces), seeded noise
    solvers/lsqr_second_derivative.py   regularized_inversion + SecondDerivative (scipy LSQR)
    solvers/fista_fk.py                 FISTA, Real @ Restriction(complex) @ FFT2D.H
    solvers/fista_radon_linear.py       FISTA, Restriction @ Radon2D (linear, numba)
results/               one CSV per benchmark (written by CI)
scripts/collect_results.py   raw BenchOpt CSV -> upsert into results/<name>.csv
docs/                  Sphinx + MyST; generate_benchmarks.py writes one leaderboard page per benchmark
tests/                 metrics, results upsert, dataset reproducibility, objective
.github/scripts/find_benchmarks.py   selects changed benchmarks for the CI matrix
.github/workflows/     pr_smoke.yml, run_benchmarks.yml, docs.yml
```

The contract between the pieces:
- Datasets return `dict(x_true, Op, y, meta)`, where `meta` holds `dims`, axes, sampling and `iava`.
- Solvers receive `Op`, `y` and `meta`. They build their operators in `set_objective`, which BenchOpt does not time, and solve in `run`, which is timed.
- Solvers return `dict(x=...)`.

**CI:**
- **`pr_smoke.yml`** runs `pytest` and a Sphinx build with warnings treated as errors. For each changed benchmark it also runs `benchopt test --skip-env`, a single-repetition run, and a dry run of `collect_results.py`.
- **`run_benchmarks.yml`** runs the selected benchmarks in a parallel matrix, using `uv sync --locked`.
  - It can optionally test against a PyLops git ref; `UV_NO_SYNC=1` stops `uv run` from undoing that override.
  - A single final job commits `results/*.csv`, and runs are serialized so these commits never race.
- **`docs.yml`** builds the site and deploys it to GitHub Pages.

## Findings along the way

- **`engine="pylops"` runs CGLS.** `regularized_inversion(engine="pylops")` uses CGLS, not LSQR. The LSQR solver therefore keeps the default `engine="scipy"`; a solver using the PyLops engine should be named CGLS.
- **`FFT2D` needs a complex restriction.** A float64 `Restriction` cannot sit after `FFT2D.H`, because its adjoint fails with `UFuncOutputCastingError`. The f-k solver therefore rebuilds a complex `Restriction` from `meta["iava"]`, as `SeismicInterpolation` does.
- **FISTA step size uses randomness.** It comes from an ARPACK eigenvalue estimate, so both FISTA solvers pass a seeded `eigsdict=dict(v0=...)` to keep runs reproducible.
- **BenchOpt `test` under uv.** Its install tests try to create conda environments and need `pip`, which uv environments do not include. Use `benchopt test <dir> --skip-env`.
- **BenchOpt result cache.** BenchOpt caches results in `benchmarks/<name>/__cache__`; use `--no-cache` for fresh timings. CI always passes it.

## Verification done

- `uv run pytest`: 11 passed.
- `uv run benchopt test --skip-env` passes on both benchmarks.
- `benchopt run` followed by `collect_results.py` twice: rows are replaced, not duplicated.
- `sphinx-build -W` succeeds with and without results.
- `actionlint` passes on all workflows.
- `uv build` produces a wheel that contains only `pylops_bench`.

## Follow-ups

- In GitHub settings, set the Pages source to "GitHub Actions".
- Give workflows read/write permission, and allow the bot through branch protection on `main`.
- After merging, trigger `run_benchmarks.yml` manually to produce the first `results/seismic_interpolation.csv`, then run `docs.yml`.
- Check the site URL used in the README once Pages is enabled.
- Add more inversion types and problems by copying `benchmarks/_template`.
