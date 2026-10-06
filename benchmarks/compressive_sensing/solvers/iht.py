import numpy as np
from benchopt import BaseSolver
from pyproximal import L2, L0Ball
from pyproximal.optimization.primal import ProximalGradient


class Solver(BaseSolver):
    r"""L0-constrained least squares, solved with (accelerated) Iterative
    Hard Thresholding.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 \quad \text{s.t.} \quad
        \|x\|_0 \le k

    Solved with :func:`pyproximal.optimization.primal.ProximalGradient` and
    the projection onto the L0 ball (:class:`pyproximal.L0Ball`), with FISTA
    acceleration and step size :math:`0.95 / \|Op\|_2^2`. The radius of the
    ball is the number of nonzero samples of the true model (``meta["k"]``).
    """

    name = "IHT"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "niter": [1000],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.proxf = L2(Op=Op, b=y)
        self.proxg = L0Ball(meta["k"])
        self.tau = 0.95 / np.linalg.norm(Op.A, 2) ** 2
        self.x0 = np.zeros(Op.shape[1])

    def run(self, _):
        self.x = ProximalGradient(
            self.proxf,
            self.proxg,
            tau=self.tau,
            x0=self.x0,
            niter=self.niter,
            acceleration="fista",
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
