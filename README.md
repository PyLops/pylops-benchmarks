# pylops-benchmarks

PyLops Benchmarks: a zoo of problems + inversion setups = comparison metrics.

Each benchmark is a [PyLops](https://github.com/PyLops/pylops) inverse
problem, defined by a true model, an operator and the associated modelled
data (plus possibly some noise). Different combinations of solvers and
regularizers/preconditioners are run with
[BenchOpt](https://benchopt.github.io), and each estimate is compared with
the true model using PyLops' MAE and PSNR.

| Benchmark | Description |
| --------- | ----------- |
| [`image_deblurring`](benchmarks/image_deblurring) | Image deblurring |
| [`poststack`](benchmarks/poststack) | Post-stack inversion |
| [`prestack`](benchmarks/prestack) | Pre-stack inversion |
| [`seismic_deblending`](benchmarks/seismic_deblending) | Deblending of a seismic gather from continously blended data |
| [`seismic_interpolation`](benchmarks/seismic_interpolation) | Reconstruction of a seismic gather from randomly subsampled traces |

Results are stored in [`results/`](results) (one CSV per benchmark) and
published at https://pylops.github.io/pylops-benchmarks.

## Quick start

The project is managed with [uv](https://docs.astral.sh/uv/). Run the following:

```bash
uv sync --all-groups  # only once
uv run benchopt run benchmarks/{BENCH} --no-plot
```

where `${BENCH}` is the name of the benchmark to run.

## Build the website locally

The website is a Sphinx site in `docs/`, generated from the CSV files in
`results/` (benchmarks without a CSV are shown as "not run yet"):

```bash
uv sync --group doc
uv run sphinx-build -W docs docs/_build/html
```

To preview the pages with results from a local run instead of the committed
ones, collect them into a separate folder and point the build to it:

```bash
uv run benchopt run benchmarks/seismic_interpolation --no-plot --no-cache --output /tmp/raw.csv
uv run python scripts/collect_results.py seismic_interpolation /tmp/raw.csv --results-dir /tmp/results
uv run sphinx-build -W -D results_dir=/tmp/results docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) to add problems and solvers, and for
details on the CI.
