# Image deblurring

Recovery of a sharp image from its blurred version, with a known point-spread
function.

- **True model**: the sharp image.
- **Operator**: `pylops.signalprocessing.Convolve2D`, a 2D convolution with
  a Gaussian point-spread function.
- **Data**: the blurred image, plus optional white Gaussian noise
  (`noise_std`, drawn with a fixed `seed`).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSQR-Laplacian` | Regularized least squares with a Laplacian (smoothness) regularizer, solved with LSQR |
| `FISTA-DWT2D` | L1 (soft thresholding) or L0 (hard thresholding) sparsity-promoting inversion in the wavelet domain (2D DWT as preconditioner), solved with FISTA |
| `SplitBregman-TV` | Anisotropic total variation regularization, solved with Split Bregman |
| `SplitBregman-TV-DWT2D` | Anisotropic total variation plus L1 sparsity in the wavelet domain, solved with Split Bregman |
| `PrimalDual-TV` | Isotropic total variation regularization, with or without bound constraints on the pixel values, solved with the primal-dual algorithm (PyProximal) |
| `PrimalDual-DWT2D` | L1 sparsity of the image in one or more wavelet domains (analysis form, e.g. Haar, db4 or both stacked), solved with the primal-dual algorithm (PyProximal) |
| `LinearizedADMM-rMS` | Relaxed Mumford-Shah (truncated quadratic) penalty on the image gradient, nonconvex, solved with the linearized ADMM (PyProximal) |

All wavelet transforms are `pylops.signalprocessing.DWT2D` with 3 levels.
