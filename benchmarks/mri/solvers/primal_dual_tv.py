import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient, VStack
from pyproximal import L1, L2, L21, Box
from pyproximal import VStack as ProxVStack
from pyproximal.optimization.primaldual import PrimalDual


# proximal operator of the TV term on the stacked derivatives of the gradient
TV_PROX = {
    "iso": lambda eps: L21(ndim=2, sigma=eps),
    "aniso": lambda eps: L1(sigma=eps),
}


class Solver(BaseSolver):
    r"""Isotropic or anisotropic total variation regularization, solved with
    the primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \, \mathrm{TV}(x)

    where :math:`\mathrm{TV}(x)` is either the isotropic total variation
    :math:`\|\nabla x\|_{2,1}` (sum over pixels of the norm of the gradient,
    ``tv="iso"``) or the anisotropic one :math:`\|\nabla x\|_1` (sum of the
    absolute values of both derivatives, ``tv="aniso"``), with :math:`\nabla`
    the 2D gradient (forward differences).
    Both the data term and the TV term act on
    :math:`K = [Op; \nabla]`, so that each iteration of
    :func:`pyproximal.optimization.primaldual.PrimalDual` only requires
    closed-form proximal operators, with :math:`\tau = \mu = 0.95 / \|K\|_2`
    and :math:`\|K\|_2^2 \le \|Op\|_2^2 + \|\nabla\|_2^2 \le 1 + 8`. The real
    part of the solution is kept.
    """

    name = "PrimalDual-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        # weight tuned per kind of TV
        "tv, eps": [("iso", 3e-3), ("aniso", 2e-3)],
        "niter": [4000],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        Gop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.Kop = VStack([Op, Gop])
        self.proxg = ProxVStack(
            [L2(b=y), TV_PROX[self.tv](self.eps)], nn=[Op.shape[0], Gop.shape[0]]
        )
        self.proxf = Box(-np.inf, np.inf)
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
        return dict(x=np.real(self.x).reshape(self.dims))
