# Seismic deblending

Deblending of a continuously blended common-receiver gather.

- **True model**: the unblended gather.
- **Operator**: `pylops.waveeqprocessing.BlendingContinuous`, with sources
  fired every `overlap * nt` samples plus a random jitter in the ignition
  times (drawn with a fixed `seed`).
- **Data**: the blended (continuous) recording, plus optional white Gaussian
  noise (`noise_std`, also drawn with a fixed seed).

All solvers promote sparsity in the same dictionary, an overlapping patched
f-k transform (`pylops.signalprocessing.Patch2D` of `FFT2D`), either in
*synthesis* form (inversion for the f-k coefficients) or in *analysis* form
(inversion for the data, with the regularizer applied to its f-k transform).
Thresholds decay with the iterations (continuation).

Solvers compared:

| Solver | Formulation |
| ------ | ----------- |
| `Acc. Proximal Gradient` | L1 or L0 regularized least squares, synthesis or analysis, solved with the accelerated proximal gradient algorithm (PyProximal) |
| `POCS` | L1 or L0 regularization with exact data consistency, synthesis or analysis, alternating thresholding and projection onto the data-consistent affine set (PyProximal HQS) |
| `ADMM` | L1 or L0 regularization with exact data consistency, synthesis or analysis, solved with ADMM (PyProximal) |
| `PrimalDual` | L1 or L0 regularization with exact data consistency, synthesis or analysis, solved with the primal-dual algorithm (PyProximal) |
