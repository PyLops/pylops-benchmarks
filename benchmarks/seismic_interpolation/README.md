# Seismic interpolation

Reconstruction of a 2D seismic gather (140 traces x 200 time samples, three
linear events) from a random subset of its traces.

- **True model**: the fully sampled gather.
- **Operator**: `pylops.Restriction` along the spatial axis, keeping
  `perc_subsampling` of the traces (selected with a fixed `seed`).
- **Data**: the restricted gather, plus optional white Gaussian noise
  (`noise_std`, also drawn with a fixed seed).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSQR-SecondDerivative` | Regularized least squares with a second derivative along the spatial axis, solved with LSQR |
| `FISTA-FFT2D` | L1 sparsity-promoting inversion in the f-k domain (2D FFT as preconditioner), solved with FISTA with a decaying sparsity weight |
| `FISTA-Radon2D` | L1 sparsity-promoting inversion in the linear Radon (tau-p) domain, solved with FISTA with a decaying sparsity weight |
| `OMP-Radon2D` | Greedy L0 inversion in the linear Radon domain, solved with Orthogonal Matching Pursuit |
| `AIHT-Radon2D` | L0-regularized inversion in the linear Radon domain, solved with Accelerated Iterative Hard Thresholding (PyProximal) with a decaying sparsity weight |
| `POCS-L0-Radon2D` | L0-regularized inversion in the linear Radon domain with exact data consistency, alternating hard thresholding with a decaying sparsity weight and projection onto the data-consistent affine set (PyProximal HQS) |

All Radon solvers use the same dictionary, `pylops.signalprocessing.FourierRadon2D`
with a linear kernel.
