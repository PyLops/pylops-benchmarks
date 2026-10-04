import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient, VStack
from pyproximal import L2, L21, Box
from pyproximal import VStack as ProxVStack
from pyproximal.optimization.primaldual import PrimalDual


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, optionally with bound
    constraints, solved with the primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \|\nabla x\|_{2,1}
        \;\text{s.t.}\; x_{min} \le x \le x_{max}

    where :math:`\|\nabla x\|_{2,1}` is the sum over pixels of the norm of
    the gradient (isotropic TV) and the bounds are the dynamic range of the
    image, 0-255 (``box=True``) or none (``box=False``). Both the data term
    and the TV term act on :math:`K = [Op; \nabla]`, so that each iteration
    of :func:`pyproximal.optimization.primaldual.PrimalDual` only requires
    closed-form proximal operators, with
    :math:`\tau = \mu = 0.95 / \|K\|_2` and
    :math:`\|K\|_2^2 \le \|Op\|_2^2 + \|\nabla\|_2^2 \le 1 + 8`.
    """

    name = "PrimalDual-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "box": [True, False],
        "eps": [0.5],
        "niter": [3000],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        Gop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.Kop = VStack([Op, Gop])
        self.proxg = ProxVStack(
            [L2(b=y), L21(ndim=2, sigma=self.eps)], nn=[Op.shape[0], Gop.shape[0]]
        )
        self.proxf = Box(0.0, 255.0) if self.box else Box(-np.inf, np.inf)
        self.tau = self.mu = 0.95 / np.sqrt(1.0 + 8.0)
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.x = PrimalDual(
            self.proxf,
            self.proxg,
            self.Kop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            niter=self.niter,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
