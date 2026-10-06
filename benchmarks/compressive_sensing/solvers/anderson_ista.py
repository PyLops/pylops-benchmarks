import numpy as np
from benchopt import BaseSolver
from pyproximal import L1, L2
from pyproximal.optimization.primal import AndersonProximalGradient


class Solver(BaseSolver):
    r"""L1-regularized least squares (LASSO), solved with Anderson-accelerated
    ISTA.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \|x\|_1

    Solved with :func:`pyproximal.optimization.primal.AndersonProximalGradient`,
    which extrapolates from the last ``nhistory`` iterates, with step size
    :math:`0.95 / \|Op\|_2^2`.
    """

    name = "AA-ISTA"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [1e-2],
        "nhistory": [5],
        "niter": [3000],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.proxf = L2(Op=Op, b=y)
        self.proxg = L1()
        self.tau = 0.95 / np.linalg.norm(Op.A, 2) ** 2
        self.x0 = np.zeros(Op.shape[1])

    def run(self, _):
        self.x = AndersonProximalGradient(
            self.proxf,
            self.proxg,
            x0=self.x0,
            epsg=self.eps,
            tau=self.tau,
            niter=self.niter,
            nhistory=self.nhistory,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
