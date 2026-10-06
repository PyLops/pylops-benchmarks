import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L2, L21
from pyproximal.optimization.primaldual import PrimalDual


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with the
    primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon \|\nabla x\|_{2,1}

    where :math:`\|\nabla x\|_{2,1}` is the sum over pixels of the norm of
    the gradient (isotropic TV). Solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual`, with
    :math:`K = \nabla`, :math:`\mu = 1 / (\tau \|\nabla\|_2^2)` and
    :math:`\|\nabla\|_2^2 \le 8`.
    """

    name = "PrimalDual-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.084],
        "tau": [0.2],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)
        self.proxf = L2(b=y)
        self.proxg = L21(ndim=2, sigma=self.eps)
        self.mu = 1.0 / (self.tau * 8.0)
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.x = PrimalDual(
            self.proxf,
            self.proxg,
            self.Gop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            theta=1.0,
            niter=self.niter,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
