# Compressive sensing

Recovery of a sparse vector from fewer random measurements than unknowns.

- **True model**: a vector with a few nonzero samples (spikes) at random
  locations.
- **Operator**: `pylops.MatrixMult`, a random Gaussian sensing matrix with
  unit-norm columns (drawn with a fixed `seed`).
- **Data**: the measurements, plus optional white Gaussian noise
  (`noise_std`, drawn with a fixed `seed`).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSQR` | Damped least squares (no sparsity), solved with LSQR |
| `ISTA` | L1-regularized least squares (LASSO), solved with ISTA |
| `FISTA` | L1-regularized least squares (LASSO), solved with FISTA |
| `AA-ISTA` | L1-regularized least squares (LASSO), solved with Anderson-accelerated ISTA (PyProximal) |
| `TwIST` | L1-regularized least squares (LASSO), solved with the two-step iterative shrinkage/thresholding algorithm (PyProximal) |
| `IRLS` | Basis pursuit (L1 norm with exact data fit), solved with iteratively reweighted least squares |
| `ADMM-BP` | Basis pursuit, with the data fit as a projection onto an affine set, solved with ADMM (PyProximal) |
| `PrimalDual-BPDN` | Basis pursuit denoise (L1 norm with the data misfit bounded by the noise level), solved with the primal-dual algorithm (PyProximal) |
| `OMP` | Greedy L0 recovery with as many atoms as nonzero samples, solved with Orthogonal Matching Pursuit |
| `IHT` | L0-constrained least squares (as many nonzero samples as the true model), solved with accelerated Iterative Hard Thresholding (PyProximal) |

The number of nonzero samples of the true model is given to the L0 solvers
(`OMP`, `IHT`) and the noise level to `PrimalDual-BPDN`.
