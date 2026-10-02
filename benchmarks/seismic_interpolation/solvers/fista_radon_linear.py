import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import Radon2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the linear Radon domain, solved with FISTA.

    .. math::
        \min_p \|y - Op R p\|_2^2 + \epsilon \|p\|_1, \quad x = R p

    where :math:`R` is the linear (slant-stack) Radon transform.
    """

    name = "FISTA-Radon2D"

    requirements = ["pip::pylops", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "npx": [200],
        "pxmax": [2e-3],
        "eps": [8e-1],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.y = y
        px = np.linspace(-self.pxmax, self.pxmax, self.npx)
        self.Pop = Radon2D(
            meta["taxis"],
            meta["xaxis"],
            px,
            centeredh=True,
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

    def run(self, _):
        self.p = fista(
            self.SIop,
            self.y,
            niter=self.niter,
            eps=self.eps,
            eigsdict=self.eigsdict,
        )[0]

    def get_result(self):
        return dict(x=(self.Pop @ self.p).reshape(self.dims))
