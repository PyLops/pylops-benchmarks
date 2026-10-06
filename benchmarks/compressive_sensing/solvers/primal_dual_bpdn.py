import numpy as np
from benchopt import BaseSolver
from pyproximal import L1, EuclideanBall
from pyproximal.optimization.primaldual import PrimalDual


class Solver(BaseSolver):
    r"""Basis pursuit denoise, solved with the primal-dual algorithm.

    .. math::
        \min_x \|x\|_1 \quad \text{s.t.} \quad \|Op x - y\|_2 \le r

    The radius of the data misfit ball is set from the noise level,
    :math:`r = \rho \sqrt{m} \sigma` (``meta["noise_std"]``, with ``rho``
    a scaling factor), and the problem is solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual`, with :math:`K = Op`,
    the projection onto the ball (:class:`pyproximal.proximal.EuclideanBall`)
    and :math:`\tau = \mu = 0.99 / \|Op\|_2`.
    """

    name = "PrimalDual-BPDN"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "rho": [1.0],
        "niter": [2000],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.dims = meta["dims"]
        radius = self.rho * np.sqrt(Op.shape[0]) * meta["noise_std"]
        self.proxf = L1()
        self.proxg = EuclideanBall(y, radius)
        self.tau = self.mu = 0.99 / np.linalg.norm(Op.A, 2)
        self.x0 = np.zeros(Op.shape[1])

    def run(self, _):
        self.x = PrimalDual(
            self.proxf,
            self.proxg,
            self.Op,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            niter=self.niter,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
