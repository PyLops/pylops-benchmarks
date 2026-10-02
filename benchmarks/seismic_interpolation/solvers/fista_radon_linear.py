import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FourierRadon2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the linear Radon domain, solved with FISTA.

    .. math::
        \min_p \|y - Op R p\|_2^2 + \epsilon_k \|p\|_1, \quad x = R p

    where :math:`R` is the linear Radon transform (implemented in the
    frequency domain with :class:`pylops.signalprocessing.FourierRadon2D`).
    The sparsity weight decays with the iterations (continuation),
    :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)` with
    :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`.
    """

    name = "FISTA-Radon2D"

    requirements = ["pip::pylops", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "npx": [200],
        "pxmax": [2e-3],
        "nfft": [256],
        "eps": [2e-1],
        "eps0_factor": [5.0],
        "gamma": [0.97],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.y = y
        px = np.linspace(-self.pxmax, self.pxmax, self.npx)
        self.Pop = FourierRadon2D(
            meta["taxis"],
            meta["xaxis"],
            px,
            self.nfft,
            kind="linear",
            engine="numba",
            dtype=Op.dtype,
        )
        self.SIop = Op @ self.Pop
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
        return dict(x=(self.Pop @ self.p).reshape(self.dims))
