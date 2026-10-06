# Seismic post-stack inversion

Estimation of the acoustic impedance of the subsurface from band-limited
post-stack seismic data.

- **True model**: the logarithm of the acoustic impedance, $\ln \text{AI}$.
- **Operator**: `pylops.avo.poststack.PoststackLinearModelling`, a first
  derivative along depth followed by a convolution with the wavelet,
  $d = \frac{1}{2} w * \frac{\mathrm{d} \ln \text{AI}}{\mathrm{d} z}$,
  applied to every trace.
- **Data**: the post-stack seismic data, plus white Gaussian noise
  (`noise_std`, drawn with a fixed `seed`).

The wavelet and a smooth background model are given to the solvers, which
use the latter as starting guess.

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSTSQ-TraceByTrace` | Damped least squares solved independently for each trace through the normal equations, with a direct solver |
| `LSQR-Laplacian` | Regularized least squares with a 2D Laplacian (smoothness) regularizer, solved with LSQR starting from the trace-by-trace solution |
| `SplitBregman-Blocky` | Blockiness-promoting inversion with L1 on the first derivative along depth and L2 on the second derivative along space, solved with Split Bregman |
| `LinearizedADMM-TV` | Isotropic total variation regularization, solved with linearized ADMM (PyProximal) |
| `ADMML2-TV` | Isotropic total variation regularization, solved with ADMM specialized for an L2 misfit (PyProximal) |
| `ADMM-TV` | Total variation regularization, solved with ADMM with an iterative TV proximal operator (PyProximal) |
| `PrimalDual-TV` | Isotropic total variation regularization, solved with the primal-dual algorithm (PyProximal) |

All total variation solvers use `pylops.Gradient` with forward differences.
