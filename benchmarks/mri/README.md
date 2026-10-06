# MRI

Reconstruction of an image from undersampled k-space data, as in
compressed-sensing Magnetic Resonance Imaging.

- **True model**: the (real) image.
- **Operator**: `pylops.medical.MRI2D`, a 2D Fourier transform followed by
  a binary k-space mask (radial lines through the center of k-space, drawn
  with a fixed `seed`).
- **Data**: the sampled (complex) Fourier coefficients, plus optional
  complex white Gaussian noise (`noise_std`, drawn with a fixed `seed`).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `ZeroFilled` | Adjoint of the MRI operator (missing coefficients set to zero), equal to the minimum-norm least-squares solution |
| `FISTA-DWT2D` | L1 sparsity-promoting inversion in the wavelet domain (2D DWT as preconditioner), solved with FISTA |
| `SplitBregman-TV` | Anisotropic total variation regularization, solved with Split Bregman |
| `PrimalDual-TV` | Isotropic or anisotropic total variation regularization, solved with the primal-dual algorithm (PyProximal) |
| `LinearizedADMM-TV` | Isotropic or anisotropic total variation regularization, solved with the linearized ADMM (PyProximal) |
| `PnP-ADMM-BM3D` | Plug-and-Play prior with the BM3D denoiser in place of the proximal operator of the regularizer, solved with ADMM (PyProximal) |
| `RED-BM3D` | Regularization by Denoising with the BM3D denoiser, solved with fixed-point iterations (PyProximal) |

All solvers work with complex unknowns and return the real part of the
estimate.
