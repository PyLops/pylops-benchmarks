# Contributing

## Repository layout

```
pylops_bench/          shared code (metrics, result tables) used by all benchmarks
benchmarks/<name>/     one BenchOpt benchmark per problem
  objective.py         metrics (always pylops_bench.metrics.compute_metrics)
  datasets/*.py        true model, operator and data
  solvers/*.py         one solver + regularizer/preconditioner per file
  config.yml           BenchOpt per-benchmark settings
  run_config.yml       options used by CI (`benchopt run --config`)
  README.md            description shown on the website
benchmarks/_template/  copy-me skeleton
results/<name>.csv     one result table per benchmark (written by CI)
scripts/               result collection
docs/                  Sphinx website built from results/
```

## Set up

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups      # creates .venv with the package, tests (dev) and docs (doc) deps
```

Prefix commands with `uv run` (or activate `.venv`). Add dependencies with
`uv add <pkg>` (or `uv add --group dev|doc <pkg>`) so that `uv.lock` stays in
sync; CI installs with `uv sync --locked`.

## Add a new problem

1. Copy `benchmarks/_template` to `benchmarks/<name>`.
2. Write the dataset(s) in `datasets/`. `get_data` must return
   `dict(x_true=..., Op=..., y=..., meta=...)`, where `meta` contains at least
   `dims` (the shape of the model) and anything solvers need to build their
   own operators (axes, sampling, ...).
3. **All randomness must be seeded**: expose a `seed` parameter and draw the
   operator randomness (e.g. subsampling masks) and the noise from
   `np.random.default_rng` generators spawned from it
   (`np.random.SeedSequence(seed).spawn(...)`), so that changing the noise
   level never changes the operator.
4. Keep `objective.py` as in the template: metrics are always PyLops MAE and
   PSNR via `pylops_bench.metrics.compute_metrics`. Set `test_config` to the
   name of your dataset.
5. Describe the problem in `README.md` (its first heading is the page title).

## Add a solver

Add one file in `benchmarks/<name>/solvers/`. Each solver is one combination
of solver and regularizer/preconditioner:

- `sampling_strategy = "run_once"`, with the number of iterations and the
  other hyperparameters exposed as `parameters` (lists of values are swept);
- build all operators in `set_objective` (not timed) and only solve in `run`
  (timed);
- `get_result` returns `dict(x=...)` with `x` of shape `meta["dims"]`;
- if the solver uses randomness (e.g. a starting vector for power
  iterations), seed it.

Check composite operators with a dot-test (`pylops.utils.dottest`) while
developing a new solver.

## Run locally

```bash
uv run benchopt test benchmarks/<name> --skip-env # BenchOpt consistency tests
uv run benchopt run benchmarks/<name> --no-plot   # run all solvers/datasets
uv run benchopt run benchmarks/<name> --no-plot -s "FISTA-FFT2D[eps=[0.01,0.1]]"
uv run pytest                                     # repository tests
```

`--skip-env` skips BenchOpt's tests that create conda environments to
install requirements (dependencies are managed by uv here). BenchOpt caches
results in `benchmarks/<name>/__cache__`: pass `--no-cache`
to force a fresh run (e.g. to get new timings).

To update the result table from a run (CI does this automatically):

```bash
uv run benchopt run benchmarks/<name> --no-plot --no-cache --output /tmp/raw.csv
uv run python scripts/collect_results.py <name> /tmp/raw.csv
```

Rows are keyed on objective, dataset, solver and all their parameters: a
rerun replaces the corresponding row (and its `run_date`), a new
configuration adds a row.

Build the website:

```bash
uv run sphinx-build -W docs docs/_build/html
```

## Continuous integration

All workflows install the locked environment with `uv sync --locked`.

- **PR smoke test** (`pr_smoke.yml`): `pytest`, then `benchopt test`
  and a short `benchopt run` for every changed benchmark, and a docs build.
  Nothing is committed.
- **Run benchmarks** (`run_benchmarks.yml`): on push to `main` (changed
  benchmarks only) or manually (one or all benchmarks, optionally against a
  PyLops git ref). Results are upserted into `results/<name>.csv` and
  committed by `github-actions[bot]`.
- **Docs** (`docs.yml`): manually or weekly, builds the website from
  `results/` and deploys it to GitHub Pages.
