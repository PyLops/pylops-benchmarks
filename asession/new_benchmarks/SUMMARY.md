# Session summary: compressive sensing and MRI benchmarks

**Date:** 2026-10-05 / 2026-10-06
**Outcome:** PR [#15](https://github.com/PyLops/pylops-benchmarks/pull/15) from branch `new-benchmarks`

## Goal

Suggest new benchmarks based on the PyLops and PyProximal tutorials, then
implement the ones the user picked.

## Decisions taken with the user

| Topic | Decision |
| ----- | -------- |
| New benchmarks | `compressive_sensing` and `mri` (from a ranked list of tutorial-based suggestions) |
| Skipped for now | LSM, MDD, wavelet estimation, deghosting/wavefield decomposition, matrix completion, and new datasets for denoising and poststack |
| MRI | One mask only; include PnP and RED (to be added to more benchmarks later) |
| Denoiser for PnP/RED | BM3D, added as a dependency, in a shared helper in `pylops_bench/` |
| PyProximal | Upgraded to 0.13.0, which ships RED |
| Extensions | `image_deblurring`: relaxed Mumford-Shah. `seismic_interpolation`: PnP, later removed because it was poor |
| MRI TV | Add an `iso`/`aniso` option to PrimalDual-TV and LinearizedADMM-TV |
| Renaming | `denoising` → `image_denoising`, `poststack` → `seismic_poststack`, `prestack` → `seismic_prestack` |

## What was built

```
pyproject.toml / uv.lock     pyproximal >= 0.13.0, bm3d
pylops_bench/denoisers.py    bm3d_denoise(x, sigma, scale): BM3D (hard-thresholding stage) on the real part
benchmarks/compressive_sensing/
  datasets/sparse_spikes.py  n=256, m=64, k=16, noise 0.01; Gaussian matrix with unit-norm columns
  solvers/                   LSQR, ISTA, FISTA, AA-ISTA, TwIST, IRLS, ADMM-BP, PrimalDual-BPDN, OMP, IHT
benchmarks/mri/
  datasets/shepp_logan.py    195x170 phantom; 30 random radial lines (~18 % of k-space); complex noise 0.01
  datasets/shepp_logan_phantom.npy   copied from pylops testdata
  solvers/                   ZeroFilled, FISTA-DWT2D, SplitBregman-TV, PrimalDual-TV (iso/aniso),
                             LinearizedADMM-TV (iso/aniso), PnP-ADMM-BM3D, RED-BM3D
benchmarks/image_deblurring/solvers/linearized_admm_rms.py   relaxed Mumford-Shah on the gradient
tests/test_compressive_sensing.py, test_mri.py, test_denoisers.py
README.md                    benchmark table (new rows, new names, alphabetical)
```

The renamed folders, `results/*.csv` and test files were moved with `git mv`,
so history and the existing result rows carry over. The objective names
(e.g. "Denoising") were kept, as they are part of the results upsert key.

## Results (local runs)

**Compressive sensing** (k/m = 0.25, just inside the L1 phase transition):

| Solver | PSNR |
|---|---|
| TwIST, AA-ISTA | 51.2 |
| FISTA / ISTA / PrimalDual-BPDN | 51.0 / 50.8 / 50.8 |
| ADMM-BP / IRLS | 50.1 / 49.0 |
| IHT / LSQR / OMP | 23.3 / 21.2 / 20.7 |

**MRI:**

| Solver | PSNR | Time |
|---|---|---|
| LinearizedADMM-TV aniso (eps 2e-3) | 42.9 | 4 s |
| PrimalDual-TV aniso (eps 2e-3, 4000 it) | 42.7 | 19 s |
| SplitBregman-TV (eps 0.03, 50 x 5) | 41.9 | 6 s |
| LinearizedADMM-TV iso / PrimalDual-TV iso | 36.3 / 36.2 | 3 / 9 s |
| PnP-ADMM-BM3D (sigma 0.05, 50 it) | 26.6 | 24 s |
| RED-BM3D (fixed point, sigma 1, sigmad 0.1) | 22.8 | 13 s |
| FISTA-DWT2D (haar) / ZeroFilled | 21.5 / 17.7 | |

**Image deblurring:** LinearizedADMM-rMS (sigma 0.1, kappa 10, tau 50, 300 it)
reaches 21.7 dB, close to the leader PrimalDual-TV (21.3–22.2 dB).

## Findings along the way

- **MRI2D masks.** The `*-uni` masks of `pylops.medical.MRI2D` use an unseeded
  `default_rng()`, so the dataset builds its own seeded radial mask and passes it
  as an array. With a mask array the operator is `Diagonal(mask) @ FFT2D`, so
  `Op^H Op` is a projection, `‖Op‖ = 1`, and CGLS/LSQR converge to the
  zero-filled image (CGLS was dropped for that reason).
- **Compressive sensing difficulty.** With m=96, k=12 every solver recovered
  the model exactly (PSNR > 120 dB noiseless), so the problem was made harder.
  At k=16, m=64 the greedy L0 methods fail: OMP's first 6 picks are right, then
  a wrong one corrupts the residual and only 8/16 of the support is found, even
  with more atoms. IHT stalls in a wrong support.
- **Regularization weight conventions.** PyLops `ista`/`fista` minimize
  `‖y - Ax‖² + eps‖x‖₁` (no 1/2), PyProximal `L1(sigma)` with `L2` the 1/2 form.
  So PyLops `eps=0.03` ~ PyProximal `0.015`, and AA-ISTA/TwIST with 0.01 reach
  the same minimizer as FISTA.
- **SplitBregman-TV in MRI.** It first gave 31 dB because `FirstDerivative`
  defaults to `kind="centered"`. With `"backward"` it reaches 41.9 dB. Its lead
  over PrimalDual/LinearizedADMM came from the regularizer, not the algorithm:
  anisotropic TV beats isotropic TV by ~6 dB on this phantom. PrimalDual with
  anisotropic TV at weight `0.03² = 9e-4` reproduces the SplitBregman solution
  (same PSNR, residual and TV).
- **PyLops `splitbregman` docs.** `epsRL1s` is both the split-penalty weight and
  the shrinkage threshold, so the effective L1 weight is `epsRL1s**2` (with the
  `mu/2` data term), not `epsRL1s` as documented (also seen in the denoising
  session). `epsRL2s` matches the docs. With `tol=0` the outer loop never runs and
  `x0` is returned. Filed as
  [PyLops/pylops#808](https://github.com/PyLops/pylops/issues/808), with a
  reproducer and a follow-up comment.
- **PnP/RED with BM3D.** They are below TV on the Shepp-Logan phantom. RED with
  ADMM costs ~100 s, the fixed-point variant ~15 s at similar quality. On the
  seismic interpolation gather, PnP plateaued at ~30 dB, below even the LSQR
  baseline (38 dB) and far from the f-k/Radon solvers (50–60 dB), so it was
  removed. BM3D works in single precision: results differ at the ~1e-6 level
  between runs.
- **Relaxed Mumford-Shah.** It needs a large `tau` (50) in LinearizedADMM on the
  0–255 image. Weights from the 1D tutorial don't transfer.
- **Haar+db4 prior.** It was already available in `image_deblurring`
  (`PrimalDual-DWT2D` with `wavelets="haar+db4"`), so nothing was added.
- **Stateful proximal operators.** Warm-started `L2` proxes, the `RED` term and
  the decaying-sigma denoiser keep state between calls. They are built in
  `run()` so that repeated runs give the same result.
- **Formatting.** `ruff format benchmarks/` reformatted some untouched files;
  the tracked ones were reverted.

## Verification done

- `uv run pytest`: 45 passed.
- `uv run benchopt test --skip-env` on every benchmark (including all existing
  ones after the PyProximal 0.13 upgrade and after the renaming): all pass.
- `uv run benchopt run` of `compressive_sensing` and `mri` (with
  `run_config.yml`) and of the new solvers in `image_deblurring` and
  `seismic_interpolation`; results collected with `collect_results.py` into a
  temporary folder.
- `uv run sphinx-build -W` with those results, and after the renaming
  (renamed pages show the existing leaderboards).
- `.github/scripts/find_benchmarks.py --all` lists the eight benchmarks with
  the new names.

## Follow-ups

- CI reruns every benchmark (shared code changed) and writes the results,
  including the first `results/compressive_sensing.csv` and `results/mri.csv`.
- Published URLs of the renamed benchmark pages change
  (e.g. `auto_benchmarks/denoising.html` → `auto_benchmarks/image_denoising.html`).
- Possible extensions: compressive sensing with decaying amplitudes or a sweep
  over `k`; PnP/RED in more benchmarks; matrix completion and the seismic
  benchmarks (LSM, MDD) from the suggestion list.
