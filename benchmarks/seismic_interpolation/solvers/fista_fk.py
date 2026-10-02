import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Real, Restriction
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FFT2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the f-k domain, solved with FISTA.

    .. math::
        \min_p \|y - \Re\{Op F^H p\}\|_2^2 + \epsilon \|p\|_1,
        \quad x = \Re\{F^H p\}

    where :math:`F` is the 2D Fourier transform. As in
    :func:`pylops.waveeqprocessing.SeismicInterpolation`, the restriction
    operator is rebuilt with complex dtype (from ``meta["iava"]``) so that it
    can be applied to the complex output of :math:`F^H`.
    """

    name = "FISTA-FFT2D"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "nfft": [1024],
        "eps": [1e-2],
        "niter": [100],
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
        # fixed starting vector for the power iterations used to pick the
        # FISTA step size, so that runs are fully reproducible
        self.eigsdict = dict(
            v0=np.random.default_rng(0)
            .standard_normal(self.SIop.shape[1])
            .astype(self.SIop.dtype)
        )

    def run(self, _):
        self.p = fista(
            self.SIop,
            self.y,
            niter=self.niter,
            eps=self.eps,
            eigsdict=self.eigsdict,
        )[0]

    def get_result(self):
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
