# Session summary: solver estimate figures on benchmark pages

**Date:** 2026-10-23
**Outcome:** PR [#17](https://github.com/PyLops/pylops-benchmarks/pull/17) from branch `estimate-figures` (based on `main`, separate from the PnP/RED work on `pnp-red`)

## Goal

The benchmark pages showed the true model and data of each dataset, and a
leaderboard of metrics. The user wanted to know whether a figure could be made
for every inversion, and then either shown on the page or linked from the
table rows.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| Display | A **Figure** column in each leaderboard, with a `view` link to the PNG of that row. There is no gallery |
| Storage | An orphan **`estimates`** branch, force-pushed as a single parentless commit at every run, so it adds no git history |
| Format | Raw estimates (`.npz`), plotted at Sphinx build time like the dataset figures, so a change of plot style needs no re-run |
| CI git steps | The user did not want them tried locally against a bare remote, and will test them in CI |

### Why this design

- BenchOpt keeps only the metrics, and running the solvers again at docs time is not possible (runs take up to 6 h).
- BenchOpt 1.10 has an `Objective.save_final_results(**result)` hook. What it returns is stored inline in the **parquet** output (as safetensors/pickle bytes), and `benchopt.results.read_results` unpacks it back into an ndarray. A CSV output would turn the arrays into strings, so CI now writes `.parquet`.
- If the estimates were committed to `main`, every run would add MBs to the git history (Ascent is 512x512 per solver). Workflow artifacts expire after 90 days.

## What was built

```
benchmarks/*/objective.py      save_final_results(self, x): return x
pylops_bench/results.py        update_results reads .csv/.parquet (benchopt read_results);
                               tidy keeps the last repetition's estimate (_estimate column);
                               estimate_key: md5 of the non-empty normalised key values;
                               save_estimates: <key>.npz in float32/complex64, prunes
                               files no longer in the table; new `estimate` CSV column
scripts/collect_results.py     --estimates-dir (default estimates/)
benchmarks/*/datasets/*.py     optional plot_estimate(x_true, x, meta): true | estimate | error
                               (1D: overlay plus error; prestack: 3x3 grid, one row per parameter)
docs/generate_benchmarks.py    _load_dataset (once per dataset configuration), _figure,
                               _estimate_figures (title "<solver> (<params>): PSNR xx dB"),
                               Figure column in _leaderboard; config value estimates_dir
docs/conf.py                   estimates_dir = <repo>/estimates
.github/workflows/run_benchmarks.yml
  run job                      --output .parquet; fetch this benchmark from the estimates
                               branch (git archive); collect; upload estimates-<bench>
  commit job                   rebuild the tree from the branch plus the artifacts, then
                               GIT_INDEX_FILE + write-tree + commit-tree + force-push
                               to refs/heads/estimates; then the CSV commit as before
.github/workflows/docs.yml     extract the estimates branch into estimates/ before the build
.github/workflows/pr_smoke.yml --output .parquet (dry-run collect)
.gitignore                     /estimates/
README.md, CONTRIBUTING.md     plot_estimate, the estimates branch, local preview commands
tests/                         estimate saving/keying/overwrite/prune/dry-run/CSV tests;
                               a plot_estimate test per benchmark
```

## Findings along the way

- **With `run_once`, every repetition row has the estimate** in `final_results`. `tidy` keeps the last one.
- **pandas `groupby.agg` rejects functions that return arrays**, so the estimates are aggregated with a list comprehension over the groups.
- **Keys must be stable.** Empty values are skipped in the hash, so that parameter columns added by other solvers, and integers read back as floats from the CSV, do not change a row's key.
- **MyST turns links to existing local files into download links** (`_downloads/<hash>/...png`), so the figures need no extra Sphinx config.
- **Sizes:** the estimates were 40 KB for compressive sensing (9 rows) and 496 KB for image deblurring (12 rows).

## Verification done

- `uv run pytest`: 56 passed.
- `compressive_sensing` and `image_deblurring` were run locally with parquet output, collected into scratch folders and built with `sphinx-build -W -D results_dir=... -D estimates_dir=...`. The Figure column and the download links render. The deblurring, compressive sensing, poststack and template figures were checked visually.
- The CI git plumbing was **not** tested locally.

## Follow-ups

- Watch the first `run_benchmarks.yml` run after the merge (it re-runs every benchmark, because every `objective.py` changed). It creates the `estimates` branch. Check the publish step, and check that `download-artifact` with no matching artifacts does not block the CSV commit (`continue-on-error`).
- Trigger `docs.yml` by hand afterwards to see the links, instead of waiting for the Monday build.
- If all of a benchmark's estimates are pruned, it uploads no artifact, so its old folder stays on the branch. This is a rare edge case.
- The estimate figures of denoising, MRI, deblending, interpolation and prestack have tests but have not been checked visually.
- `benchmarks/_template/README.md` does not mention `plot_estimate` (CONTRIBUTING.md and the template dataset cover it).
