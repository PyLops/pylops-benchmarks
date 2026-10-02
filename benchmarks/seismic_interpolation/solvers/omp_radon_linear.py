import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import omp
from pylops.signalprocessing import FourierRadon2D


class Solver(BaseSolver):
    r"""Greedy L0 inversion in the linear Radon domain, solved with OMP.

    .. math::
        \min_p \|p\|_0 \quad \text{s.t.} \quad \|y - Op R p\|_2 \le \sigma,
        \quad x = R p

    where :math:`R` is the linear Radon transform (implemented in the
    frequency domain with :class:`pylops.signalprocessing.FourierRadon2D`).
    Solved with Orthogonal Matching Pursuit
    (:func:`pylops.optimization.sparsity.omp`), which adds one Radon
    coefficient per outer iteration and refits the selected ones with
    ``niter_inner`` LSQR iterations.
    """

    name = "OMP-Radon2D"

    requirements = ["pip::pylops", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "npx": [200],
        "pxmax": [2e-3],
        "nfft": [256],
        "sigma": [1e-2],
        "niter_outer": [80],
        "niter_inner": [5],
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

    def run(self, _):
        self.p = omp(
            self.SIop,
            self.y,
            niter_outer=self.niter_outer,
            niter_inner=self.niter_inner,
            sigma=self.sigma,
        )[0]

    def get_result(self):
        return dict(x=(self.Pop @ self.p).reshape(self.dims))
