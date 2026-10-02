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
| `FISTA-FFT2D` | Sparsity-promoting inversion in the f-k domain (2D FFT as preconditioner) |
| `FISTA-Radon2D` | Sparsity-promoting inversion in the linear Radon (tau-p) domain |
