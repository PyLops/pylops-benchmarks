import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Real, Restriction
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FFT2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the f-k domain, solved with FISTA.

    .. math::
        \min_p \|y - \Re\{Op F^H p\}\|_2^2 + \epsilon_k \|p\|_1,
        \quad x = \Re\{F^H p\}

    where :math:`F` is the 2D Fourier transform. The sparsity weight decays
    with the iterations (continuation),
    :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)` with
    :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`. As in
    :func:`pylops.waveeqprocessing.SeismicInterpolation`, the restriction
    operator is rebuilt with complex dtype (from ``meta["iava"]``) so that it
    can be applied to the complex output of :math:`F^H`.
    """

    name = "FISTA-FFT2D"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "nfft": [1024],
        "eps": [3e-3],
        "eps0_factor": [5.0],
        "gamma": [0.97],
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
        # threshold multiplier: eps0_factor * gamma**k, never below 1
        self.decay = np.maximum(
            1.0, self.eps0_factor * self.gamma ** np.arange(self.niter)
        )

    def run(self, _):
        self.p = fista(
            self.SIop,
            self.y,
            niter=self.niter,
            eps=self.eps,
            eigsdict=self.eigsdict,
            decay=self.decay,
        )[0]

    def get_result(self):
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
