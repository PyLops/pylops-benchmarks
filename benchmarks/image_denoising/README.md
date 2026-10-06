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

All gradients are first-order forward differences and all wavelet
transforms are `pylops.signalprocessing.DWT2D`. Solvers that share a
formulation solve the same problem, so they converge to the same image:
their weights are tuned so that this image has the highest PSNR, and their
iterations so that they have converged.
