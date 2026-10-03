import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Real, Restriction
from pylops.optimization.sparsity import omp
from pylops.signalprocessing import FFT2D


class Solver(BaseSolver):
    r"""Greedy L0 inversion in the f-k domain, solved with OMP.

    .. math::
        \min_p \|p\|_0 \quad \text{s.t.} \quad
        \|y - \Re\{Op F^H p\}\|_2 \le \sigma,
        \quad x = \Re\{F^H p\}

    where :math:`F` is the 2D Fourier transform. Solved with Orthogonal
    Matching Pursuit (:func:`pylops.optimization.sparsity.omp`), which adds
    one f-k coefficient per outer iteration and refits the selected ones with
    ``niter_inner`` LSQR iterations. As in
    :func:`pylops.waveeqprocessing.SeismicInterpolation`, the restriction
    operator is rebuilt with complex dtype (from ``meta["iava"]``) so that it
    can be applied to the complex output of :math:`F^H`.
    """

    name = "OMP-FFT2D"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "nfft": [256],
        "sigma": [1e-3],
        "niter_outer": [400],
        "niter_inner": [20],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        iava = meta["iava"]
        cdtype = np.result_type(Op.dtype, np.complex64)
        Rop = Restriction(dims=self.dims, iava=iava, axis=0, dtype=cdtype)
        Realop = Real((len(iava), self.dims[1]), dtype=cdtype)
        self.Pop = FFT2D(
            dims=self.dims,
            nffts=(self.nfft, self.nfft),
            sampling=(meta["dx"], meta["dt"]),
            dtype=cdtype,
        ).H
        self.SIop = Realop @ Rop @ self.Pop
        self.y = y.astype(self.SIop.dtype)

    def run(self, _):
        self.p = omp(
            self.SIop,
            self.y,
            niter_outer=self.niter_outer,
            niter_inner=self.niter_inner,
            sigma=self.sigma,
        )[0]

    def get_result(self):
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
