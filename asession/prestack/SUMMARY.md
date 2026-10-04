# Session summary: pre-stack (AVO) inversion benchmark

**Date:** 2026-10-04
**Outcome:** PR [#13](https://github.com/PyLops/pylops-benchmarks/pull/13) from branch `prestack`

## Goal

Turn `benchmarks/prestack/` into a real benchmark along the lines of
`benchmarks/poststack/`, using the PyLops pre-stack tutorial
(`notebooks/prestack.py`) as the starting point for the dataset and the
problem. At the start the folder held a copy of `_template` plus that
notebook.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| Datasets | One 2D Marmousi dataset only. The tutorial's 1D example is not included |
| Model size | First the same grid as poststack (550 x 3 x 267). At that size the TV solvers cost about 3 s per iteration, so the user chose the tutorial's grid instead: first 300 depth samples, one every six traces (300 x 3 x 134) |
| Notebook | Deleted once the benchmark replaced it |
| Dataset figure | One figure: the 3 true models side by side, the data at θ=0 and the full angle gather at the middle x |
| Regularization weights | A separate weight for VP, VS and rho in every regularizer and damping term. The user expected that a single weight for all parameters would not work well |
| Warm start | All TV solvers start from the trace-by-trace solution, since it helped PrimalDual |
| Time budget | First about 30 s per solver, as in poststack. Later raised to about 60 s |

## What was built

```
benchmarks/prestack/
  objective.py          as poststack, docstring for elastic parameters
  config.yml, run_config.yml   as poststack (timeout 30m)
  README.md             problem, operator, data; solver table
  datasets/marmousi.py  Dataset "Marmousi" (noise_std=0.05, seed=10); plot()
  datasets/poststack_model.npz   copy of poststack's model, cropped in code
  solvers/lstsq_trace_by_trace.py   damped normal equations per trace, epsI per parameter
  solvers/lsqr_laplacian.py         Laplacian per parameter, LSQR from the trace-by-trace solution
  solvers/splitbregman_blocky.py    L1 on D_z and L2 on D_xx, one term per parameter
  solvers/primal_dual_tv.py         TV per parameter, PyProximal PrimalDual, tau exposed
  solvers/linearized_admm_tv.py     TV per parameter, PyProximal LinearizedADMM, tau exposed
  solvers/admml2_tv.py              TV per parameter, PyProximal ADMML2
tests/test_prestack.py  shapes, reproducibility, noise, objective, plot
```

### Problem

- **Model:** `ln(vp, vs, rho)` with `vp = 1000 * model`, `vs = vp / 2`,
  `rho = vp / 3 + 300`, stacked along axis 1. The shape is `(nz, 3, nx)`,
  the native ordering of `PrestackLinearModelling`, so no transposes are
  needed.
- **Operator:** `PrestackLinearModelling` with Aki-Richards, `vsvp=0.5`, 21
  angles in 0–40°, and a 15 Hz Ricker wavelet of 41 samples.
- **Background model:** moving average of 30 samples in z and 25 in x.
- **`meta`:** `dims`, `wav`, `theta`, `vsvp`, `mback`, `LOp` (largest
  eigenvalue of `GᵀG` for the dense 1D operator, about 95), `x` and `z`.

### Per-parameter weights

Every regularizer acts within a single parameter, so it never mixes the
second axis:

- **L2 terms** (Laplacian, second derivative): one term per parameter,
  `Restriction(dims, [ip], axis=1) @ Reg`, each with its own `epsR`.
- **Split Bregman L1:** the same split, which is needed because Split
  Bregman's `epsRL1` is both the threshold and the penalty weight, so the
  weights cannot simply be folded into the operator.
- **TV:** `K = VStack([D_z, D_x]) @ Diagonal(eps / eps.max(), axis=1)`
  with `L21(ndim=2, sigma=eps.max())`. This keeps ‖K‖² ≤ 8 and reduces to
  the poststack form when all weights are equal.
- **`ADMM-TV` left out:** `pyproximal.TV(dims)` differentiates along every
  axis, including the parameter axis.

The weights are exposed as scalar parameters (`*_vp`, `*_vs`, `*_rho`), not
tuples, so CSV rows and `default_parameters` stay flat.

## Tuning

1. **Equivalence check.** With equal weights, LSTSQ-TraceByTrace,
   LSQR-Laplacian and SplitBregman-Blocky match `PrestackInversion` to
   within 6e-13.
2. **Iteration counts.** Set from solo timings to fill the budget, then the
   weights were tuned at those counts by coordinate search on a log scale:
   first a common factor for all three parameters, then each parameter.
   The script lived in the job's temporary folder and is not committed.
3. **Per-parameter weights help a lot.** The best weights always grow from
   VP to VS to rho, roughly 1 : 1–2 : 2–4. The gain from tuning them is up
   to +7 dB at 30 s.
4. **PrimalDual.** The user expected it to do better, given its poststack
   results. Two changes moved it from 29.4 to 31.8 dB at 30 s:
   - `tau=0.3` instead of 1 (see `benchmarks/prestack/primal_dual_tau.md`,
     not committed);
   - a warm start from the trace-by-trace solution (+1.6 dB). This
     requires the per-parameter `epsI`: a single `epsI=0.1` loses about
     1.2 dB.
5. **Warm start for all TV solvers.** Applied to LinearizedADMM and ADMML2
   too, and their weights were retuned.
6. **60 s budget.** Iteration counts doubled and the weights retuned around
   their previous values. The Split Bregman search was stopped early at the
   user's request, keeping its best so far.

### Final results (single run, `benchopt run -r 1`)

| Solver | Time | MAE | PSNR |
|---|---|---|---|
| ADMML2-TV (55 iter) | 32.3 s | 0.0228 | 34.05 |
| PrimalDual-TV (60 iter, tau 0.3) | 58.9 s | 0.0263 | 32.48 |
| LinearizedADMM-TV (65 iter, tau 2.8) | 57.2 s | 0.0264 | 32.48 |
| SplitBregman-Blocky (105 x 3) | 63.5 s | 0.0278 | 32.11 |
| LSQR-Laplacian (converges before 2200) | 26.5 s | 0.0293 | 31.65 |
| LSTSQ-TraceByTrace | 0.09 s | 0.0384 | 29.47 |

The background model alone gives 26.50 dB.

## Findings along the way

- **Early stopping hid the weak tutorial weights.** With equal weights,
  ADMML2, LSQR and Split Bregman lose PSNR as they iterate further. The
  tutorial's weights only work because those solvers stop early.
- **ADMML2 converges within 30 s.** At 105 iterations it is slightly worse
  than at 55, so it stays at 55.
- **PrimalDual and LinearizedADMM end up equivalent.** Both reach the same
  weights and 32.48 dB. LinearizedADMM passes `mu = 0.99·tau/8 ≈ 0.35` to
  the L2 proximal step, almost the same as PrimalDual's `tau=0.3`.
- **Timings are sensitive to machine load.** One run gave 74 s for ADMML2
  (32 s in a clean run).
- **Bug in the user's notebook.** In
  `pylops_notebooks/pyproximal/ProxSolvers-Prestack.ipynb`, the ADMML2
  result was reshaped with `.reshape(npar, nz, nx).swapaxes(0, 1)`, which
  is only right for the dense trace-by-trace solve. The iterative solvers
  return `(nz, npar, nx)`, so `.reshape(dims)` is correct. Also,
  `PrimalDual` returns `x` directly, not a tuple.

## Verification done

- `uv run pytest`: 25 passed, including 5 in `tests/test_prestack.py`.
- `uv run benchopt test benchmarks/prestack --skip-env`: 28 passed, 7 skipped.
- `uv run sphinx-build -W docs docs/_build/html` succeeds.
- `uv run benchopt run benchmarks/prestack -r 1 --no-cache`: all solvers run,
  and `collect_results.py --dry-run` reads the output.

## Follow-ups

- `results/prestack.csv` will be written by CI after merge.
- PrimalDual and LinearizedADMM are still improving at 60 s. A fully
  linearized primal-dual (data term inside K, one Op/Opᵀ per iteration) was
  started but stopped by the user.
- The Split Bregman search did not reach its fine stage.
- The stray `post_stack/` folder (a copy of `pylops_bench`) is untracked and
  was not part of this PR.
