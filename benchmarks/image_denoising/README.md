# Image denoising

Recovery of a clean image from a version corrupted by additive random noise.

- **True model**: the clean image.
- **Operator**: `pylops.Identity`.
- **Data**: the noisy image, i.e. the clean image plus uniform random noise
  (`noise_level`, drawn with a fixed `seed`).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSQR-Gradient` | Regularized least squares with a smoothness (Tikhonov) regularizer on the gradient, solved with LSQR |
| `LinearizedADMM-Tikhonov` | Same smoothness regularizer on the gradient, solved with the linearized ADMM (PyProximal) |
| `LinearizedADMM-TV` | Anisotropic total variation regularization, solved with the linearized ADMM (PyProximal) |
| `SplitBregman-TV` | Anisotropic total variation regularization, solved with Split Bregman |
| `LinearizedADMM-NonconvexTV` | Nonconvex sparsity penalty (log, SCAD or ETP) on each derivative, solved with the linearized ADMM (PyProximal) |
| `ProximalGradient-TV` | Isotropic total variation regularization, solved with the proximal gradient method (PyProximal) |
| `HQS-TV` | Isotropic total variation regularization, solved with half quadratic splitting (PyProximal) |
| `PrimalDual-TV` | Isotropic total variation regularization, solved with the primal-dual algorithm (PyProximal) |
| `FISTA-DWT2D` | L1 (soft thresholding) or L0 (hard thresholding) sparsity of the wavelet coefficients (synthesis, i.e. wavelet shrinkage), solved with FISTA |
| `PrimalDual-DWT2D` | L1 sparsity of the image in one or more wavelet domains (analysis, e.g. Haar, db4 or both stacked), solved with the primal-dual algorithm (PyProximal) |
| `PrimalDual-TV-DWT2D` | Isotropic total variation plus L1 sparsity in the wavelet domain, solved with the primal-dual algorithm (PyProximal) |
| `FISTA-Patch2D` | L1 sparsity in overlapping local 2D DCT or FFT windows (`pylops.signalprocessing.Patch2D`), solved with FISTA |
| `PnP-ADMM-BM3D` | Plug-and-Play prior with the BM3D denoiser in place of the proximal operator of the regularizer, solved with ADMM (PyProximal) |
| `RED-BM3D` | Regularization by Denoising with the BM3D denoiser, solved with fixed-point iterations (PyProximal) |

All gradients are first-order forward differences and all wavelet
transforms are `pylops.signalprocessing.DWT2D`. Solvers that share a
formulation solve the same problem, so they converge to the same image:
their weights are tuned so that this image has the highest PSNR, and their
iterations so that they have converged.

`PnP-ADMM-BM3D` and `RED-BM3D` use the BM3D denoiser (hard-thresholding
stage, `pylops_bench.denoisers.bm3d_denoise`). Since the operator is the
identity, a single call of BM3D with the noise standard deviation of the data
(about 0.115) is already a strong baseline (29.6 dB). `PnP-ADMM-BM3D`
reaches the same PSNR: its best denoiser strength is close to the noise
level, and the iterations do not improve on a single call. `RED-BM3D` behaves
differently: its fixed point balances the data with a weak denoiser applied
repeatedly, and it is best with a large RED weight and a denoiser strength
well below the noise level (`sigma=5`, `sigmad=0.03`), reaching about 0.5 dB
less than a single BM3D call. Both solvers outperform the total variation
and wavelet solvers.
