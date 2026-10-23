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
| [`compressive_sensing`](benchmarks/compressive_sensing) | Recovery of a sparse vector from few random measurements |
| [`image_deblurring`](benchmarks/image_deblurring) | Image deblurring |
| [`image_denoising`](benchmarks/image_denoising) | Image denoising |
| [`mri`](benchmarks/mri) | Reconstruction of an image from undersampled k-space (MRI) data |
| [`seismic_deblending`](benchmarks/seismic_deblending) | Deblending of a seismic gather from continously blended data |
| [`seismic_interpolation`](benchmarks/seismic_interpolation) | Reconstruction of a seismic gather from randomly subsampled traces |
| [`seismic_poststack`](benchmarks/seismic_poststack) | Seismic post-stack inversion |
| [`seismic_prestack`](benchmarks/seismic_prestack) | Seismic pre-stack inversion |

Results are stored in [`results/`](results) (one CSV per benchmark), the
estimates of the solvers in the
[`estimates`](https://github.com/PyLops/pylops-benchmarks/tree/estimates)
branch, and both are published at https://pylops.github.io/pylops-benchmarks.

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
uv run benchopt run benchmarks/seismic_interpolation --no-plot --no-cache --output /tmp/raw.parquet
uv run python scripts/collect_results.py seismic_interpolation /tmp/raw.parquet --results-dir /tmp/results --estimates-dir /tmp/estimates
uv run sphinx-build -W -D results_dir=/tmp/results -D estimates_dir=/tmp/estimates docs docs/_build/html
```

The leaderboards link a figure of the estimate of every solver, plotted from
the estimates stored on the `estimates` branch (see
[CONTRIBUTING.md](CONTRIBUTING.md)).

See [CONTRIBUTING.md](CONTRIBUTING.md) to add problems and solvers, and for
details on the CI.
