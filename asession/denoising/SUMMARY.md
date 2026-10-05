# Session summary: image denoising benchmark

**Date:** 2026-10-05
**Outcome:** branch `denoising` (from `main`), not committed yet

## Goal

Turn `benchmarks/denoising/` into a real benchmark, using the PyProximal
denoising tutorial (`notebooks/denoising.py`) as the starting point for the
dataset and the solvers. At the start the folder held a copy of `_template`
plus that notebook.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| Noise | Random (uniform) noise only. The tutorial's salt-and-pepper case and the L1 data term are not included |
| Iterations | Run to convergence (PSNR within 0.01 dB of a longer run), not a time budget |
| Solvers | The four tutorial setups, plus PyLops-only LSQR and Split Bregman baselines |
| Notebook | Deleted once the benchmark replaced it |

## What was built

```
benchmarks/denoising/
  objective.py          as image_deblurring, name "Denoising"
  config.yml, run_config.yml   as the template (timeout 30m)
  README.md             problem, operator, data; solver table
  datasets/ascent.npy   scipy.datasets.ascent() as uint8 (pooch is not a dependency)
  datasets/ascent.py    Dataset "Ascent" (noise_level=0.2, seed=0); plot()
  solvers/lsqr_gradient.py             Tikhonov on the gradient, LSQR
  solvers/linearized_admm_tikhonov.py  Tikhonov on the gradient, LinearizedADMM
  solvers/linearized_admm_tv.py        anisotropic TV, LinearizedADMM
  solvers/splitbregman_tv.py           anisotropic TV, Split Bregman
  solvers/proximal_gradient_tv.py      isotropic TV, ProximalGradient (pyproximal.TV prox)
  solvers/primal_dual_tv.py            isotropic TV, PrimalDual (L21 on the gradient)
tests/test_denoising.py  reproducibility, noise, shapes, objective, plot
README.md                denoising row added to the benchmark table
```

- **Model:** ascent / 255, values in [0, 1], 512 x 512.
- **Operator:** `Identity`.
- **Data:** `x + noise_level * max|x| * U(-1, 1)`, from a seeded generator. The noisy data has a PSNR of 18.75 dB.

## Tuning

1. Coordinate log-search on the weight at the tutorial's 100 iterations, then iterations doubled until the PSNR settled, then the weight retuned.
2. Smallest converged iteration count found from a scan, together with `tau` (the step mostly changes speed, not the solution).
3. Fine scan of the weight, for each of the three underlying problems, with its fastest solver. Solvers that solve the same problem then use the same weight.

| Solver | Final parameters | PSNR | Time |
|---|---|---|---|
| ProximalGradient-TV | eps 0.084, tau 1, niter 1, niter_tv 100, rtol_tv 1e-5 | 27.54 | 0.29 s |
| PrimalDual-TV | eps 0.084, tau 0.2, niter 50 | 27.54 | 0.25 s |
| LinearizedADMM-TV | eps 0.07, tau 4, niter 50 | 27.47 | 0.37 s |
| SplitBregman-TV | eps 0.1, mu 0.1429 (eps²/mu = 0.07), 20 x 1 | 27.47 | 0.73 s |
| LSQR-Gradient | epsR 0.95, niter 10 | 25.38 | 0.07 s |
| LinearizedADMM-Tikhonov | eps 0.9 (= epsR²), tau 4, niter 30 | 25.38 | 0.16 s |

All solvers are within 0.002 dB of a run with 4x the iterations.

## Findings along the way

- **ProximalGradient step.** The tutorial uses `tau = 1/8`, from ‖∇‖², but the data-term gradient here is 1-Lipschitz. With `tau=1`, one step is the exact TV prox, and repeating it changes nothing. The TV prox's default `rtol=1e-4` stops early (27.48 dB); `rtol=1e-5` reaches the PrimalDual solution in about 0.3 s, against 6 s for the tutorial's 100 x 10.
- **Split Bregman derivatives.** `FirstDerivative` defaults to `kind="centered"`, which gives a different, worse TV (24.0 dB). With `kind="forward"` it matches LinearizedADMM-TV.
- **Split Bregman weights.** In PyLops `epsRL1` is both the shrinkage threshold and the Bregman penalty weight, so the effective TV weight at convergence is `eps²/mu`. A sweep along `eps²/mu = const` confirmed it. Coordinate search on `eps` and `mu` separately moves poorly along this ridge.
- **Step sizes.** LinearizedADMM converges much faster with `tau=4` than with the tutorial's 1, and PrimalDual with `tau=0.2` (`mu=1/(8 tau)`) faster than with `1/√8`.
- **LSQR.** It converges in about 10 iterations; early iterations overshoot slightly, with higher PSNR at 3–5 iterations.

## Second round: more configurations

The user asked for HQS with TV and wavelet sparsity, and picked the extra suggestions to add: TV + wavelets, Huber-TV, a nonconvex gradient penalty, and patched DCT/FFT shrinkage. Seven solvers were added and tuned the same way: weight search, iterations to convergence, then a level and window-size scan.

| Solver | Final parameters | PSNR | Time |
|---|---|---|---|
| LinearizedADMM-NonconvexTV (ETP) | eps 0.035, gamma 2, tau 4, niter 50 | 27.58 | 2.5 s |
| PrimalDual-TV-DWT2D | db4 level 4, eps_tv 0.0714, eps_w 0.0168, tau 0.2, niter 50 | 27.57 | 1.1 s |
| LinearizedADMM-NonconvexTV (SCAD) | eps 0.07, a 12.4 | 27.56 | 0.7 s |
| PrimalDual-HuberTV | eps 0.084, alpha 0.0025, tau 0.2, niter 50 | 27.55 | 0.35 s |
| HQS-TV | eps 0.084, tau 30, niter 3 | 27.54 | 1.3 s |
| LinearizedADMM-NonconvexTV (log) | eps 0.07, gamma 0.057 | 27.48 | 2.3 s |
| FISTA-Patch2D (FFT) | nwin 32, eps 0.2, niter 100 | 26.18 | 6.6 s |
| PrimalDual-DWT2D (haar+db4) | level 5, eps 0.077, tau 0.5, niter 50 | 25.70 | 1.1 s |
| FISTA-Patch2D (DCT) | nwin 32, eps 0.183, niter 100 | 25.60 | 3.1 s |
| FISTA-DWT2D soft / hard | db4, level 5 / 3, eps 0.34 / 0.13, 1 iteration | 25.07 / 24.73 | 0.02 s |

### Findings

- **HQS for denoising.** The fixed point z is exactly the TV solution with weight `eps (1 + tau)`. The solver therefore returns z and divides the weight by `1 + tau`, so `eps` means the same as in the other TV solvers and `tau` only sets the speed (30 → 3 iterations).
- **PyProximal Huber.** `pyproximal.Huber.prox` selects the quadratic or linear branch with `|x| > alpha` instead of `|x| > alpha + tau`, which is wrong in between. Checked numerically: for x=1.5, alpha=1, tau=1 it returns 0.5, but the minimizer is 0.75. `HuberCircular` uses a single norm for the whole vector. The Huber-TV solver therefore defines its own per-pixel isotropic Huber prox, `HuberL21`, in the solver file. Worth an upstream issue.
- **Huber-TV.** Its `alpha` went to 0, so it ends up as plain TV: no PSNR gain on this image.
- **Wavelets.** A single orthogonal wavelet in analysis form gives exactly soft shrinkage, so `PrimalDual-DWT2D` only runs the stacked haar+db4 prior. All wavelet priors stay about 2 dB below TV, and adding db4 to TV gains only +0.03 dB.
- **Patch2D speed.** If the transform acts on all windows at once (`DCT`/`FFT2D` with `axes=(2, 3)` over a 4D array), `Patch2D` applies it in one call. ARPACK `eigs` took more than 60 s for the complex patched FFT, so the step comes from 30 seeded power iterations with a 5 % safety margin (`pylops.optimization.eigs.power_iteration` is unseeded). The FFT solver went from 96 s to under 7 s.
- **Nonconvex penalties.** They give at most +0.1 dB over anisotropic TV (ETP). SCAD's `a` and log's `gamma` drift toward the L1 limit.

## Verification done

- `uv run pytest`: 30 passed.
- `uv run benchopt test benchmarks/denoising --skip-env`: 56 passed, 14 skipped.
- `uv run benchopt run benchmarks/denoising -r 1 --no-cache`: all 13 solvers (17 configurations) run. `collect_results.py` writes the table to a temporary results directory.
- `uv run sphinx-build -W docs ...` succeeds, both with those results (leaderboard and dataset figure rendered) and without them.

## Follow-ups

- `results/denoising.csv` will be written by CI after merge.
- Possible extensions: a salt-and-pepper dataset with an L1 data term (needs per-dataset weights, e.g. a swept list), or Gaussian noise.
