# Pre-stack inversion

Estimation of the elastic parameters of the subsurface (P-wave velocity,
S-wave velocity and density) from band-limited pre-stack seismic data
(angle gathers), using their amplitude variation with angle (AVO).

- **True model**: the logarithm of the elastic parameters,
  $\mathbf{m} = [\ln V_P, \ln V_S, \ln \rho]$, stacked along the second axis.
- **Operator**: `pylops.avo.prestack.PrestackLinearModelling`, a first
  derivative along depth, followed by the Aki-Richards linearization of the
  AVO response and a convolution with the wavelet,
  $d(t, \theta) = w(t) * \sum_{i=1}^{3} G_i(t, \theta) \frac{\mathrm{d} \ln m_i(t)}{\mathrm{d} t}$,
  applied to every trace.
- **Data**: the pre-stack seismic data, plus white Gaussian noise
  (`noise_std`, drawn with a fixed `seed`).

The wavelet, the angles and a smooth background model are given to the
solvers, which use the latter as starting guess. The three parameters have
different sensitivities to the data, so every solver weights its
regularization (or damping) separately for $V_P$, $V_S$ and $\rho$
(parameters with suffixes `_vp`, `_vs` and `_rho`).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `LSTSQ-TraceByTrace` | Damped least squares solved independently for each trace through the normal equations, with a direct solver |
| `LSQR-Laplacian` | Regularized least squares with a 2D Laplacian (smoothness) regularizer on each parameter, solved with LSQR starting from the trace-by-trace solution |
| `SplitBregman-Blocky` | Blockiness-promoting inversion with L1 on the first derivative along depth and L2 on the second derivative along space of each parameter, solved with Split Bregman |
| `LinearizedADMM-TV` | Isotropic total variation regularization of each parameter, solved with linearized ADMM (PyProximal), starting from the trace-by-trace solution |
| `ADMML2-TV` | Isotropic total variation regularization of each parameter, solved with ADMM specialized for an L2 misfit (PyProximal), starting from the trace-by-trace solution |
| `PrimalDual-TV` | Isotropic total variation regularization of each parameter, solved with the primal-dual algorithm (PyProximal), starting from the trace-by-trace solution |

All total variation solvers use forward differences along depth and space
(never across parameters).
