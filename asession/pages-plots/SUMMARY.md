# Session summary: dataset figures and descriptions on benchmark pages

**Date:** 2026-10-03
**Outcome:** PR [#4](https://github.com/PyLops/pylops-benchmarks/pull/4) from branch `pages-plots`

## Goal

Each benchmark page has a Results section with one subsection per dataset
configuration (e.g. "Linear events" in seismic interpolation). The goal was to
show a figure of the true model and the data in each subsection, with the
plotting code living in the benchmark's `datasets/*.py`.

The user also asked whether the shape of the gather (140 traces x 200 time
samples) belongs in the benchmark README, given that more datasets may be
added later.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| Where figures come from | They are generated at Sphinx build time by `docs/generate_benchmarks.py`. Neither benchopt plots nor committed PNGs are used |
| Plotting API | Each dataset can have an optional `plot(self, x_true, Op, y, meta)` method that returns a matplotlib `Figure`, with matplotlib imported inside the method |
| Which configurations are plotted | Every dataset configuration in `results/<bench>.csv`, rebuilt with the exact parameters stored there |
| Plot failures | Logged at info level and skipped, so the weekly docs build is not blocked (`-W` turns warnings into errors) |
| README content | Generic, describing only the problem. Dataset specifics (sizes, sampling, content) go in the dataset class docstring, which the page shows at the top of that dataset's subsection |
| Docstring format | Rendered as Markdown on the page, so it is written in Markdown, not reStructuredText |

### Why not benchopt or committed PNGs

benchopt 1.10 custom plots (`BasePlot` in a `plots/` folder) do not fit:
- They run on the results of a `benchopt run`, not on the dataset.
- They render only in benchopt's own HTML report. CI runs with `--no-plot`, and the site is built by our own Sphinx extension.
- `plotting/image_utils.py` shows images as grayscale clipped to [0, 1], with no colormap or axes. That does not work for signed seismic amplitudes.

Committed PNGs would go stale whenever a dataset's code or parameters change, and they add binary files to git.

## What was built

```
pylops_bench/loading.py    load_class, default_parameters, instantiate (moved from tests/conftest.py),
                           find_by_name (dataset file from its `name`), NAME_RE
docs/generate_benchmarks.py
  _dataset_class           dataset class from the name in the CSV
  _dataset_params          dataset_param_* columns cast to the type of each default value
  _figure                  get_data() + plot() -> auto_benchmarks/figures/<bench>_<slug>_<hash>.png
  _benchmark_page          per subsection: docstring, parameters, figure, leaderboard
benchmarks/seismic_interpolation/datasets/linear_events.py
                           detailed docstring; plot(): true model | data with missing traces as zeros
benchmarks/_template/datasets/blocky_signal.py   simple line-plot example of plot()
tests/conftest.py          thin wrappers around pylops_bench.loading
tests/test_seismic_interpolation.py   test that plot() returns a Figure
pyproject.toml, uv.lock    matplotlib added to the `doc` group
CONTRIBUTING.md, benchmarks/_template/README.md   README/docstring split, optional plot()
```

The figures go in `docs/auto_benchmarks/figures/`. That folder is ignored by git
and recreated at every build.

## Findings along the way

- **Integer parameters read back as floats.** pandas turns an integer column with missing values into floats, so `seed=42` comes back as `42.0`, and `np.random.SeedSequence` rejects floats. `_dataset_params` casts each value to the type of the dataset's default value.
- **matplotlib was only an indirect dependency.** It was already installed as a dependency of benchopt, but it is now listed explicitly in the `doc` group.
- **Dark mode.** The PNGs have a white background, so they appear as a white box in the theme's dark mode.

## Verification done

- `uv run pytest`: 12 passed.
- `uv run sphinx-build -W docs docs/_build/html` succeeds. The docstring and figure render under "Linear events", and the figure was checked visually.
- `uv run benchopt test benchmarks/{seismic_interpolation,_template} --skip-env -k dataset` passes.

## Follow-ups

- Consider a dark-mode-friendly figure style (transparent background, or light and dark variants).
- When new datasets are added, check whether solver settings tuned for this geometry (e.g. `pxmax=0.002` in the Radon solvers) still suit them.
- `benchmarks/seismic_deblending/` is untracked in the working tree and was not part of this PR.
