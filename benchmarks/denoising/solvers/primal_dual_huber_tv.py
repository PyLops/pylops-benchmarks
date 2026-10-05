import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L2, ProxOperator
from pyproximal.optimization.primaldual import PrimalDual


class HuberL21(ProxOperator):
    r"""Isotropic Huber norm of a stacked gradient,
    :math:`\sigma \sum_i H_\alpha(\|v_i\|_2)`, where :math:`v_i` collects the
    ``ndim`` derivatives of pixel :math:`i` and :math:`H_\alpha(t)` is
    :math:`t^2 / (2 \alpha)` for :math:`t \le \alpha` and
    :math:`t - \alpha / 2` otherwise. Its proximal operator shrinks each
    :math:`v_i` by :math:`1 - \tau\sigma / \max(\|v_i\|_2, \tau\sigma + \alpha)`.

    (:class:`pyproximal.HuberCircular` takes the norm of the whole vector,
    not per pixel.)
    """

    def __init__(self, ndim, sigma=1.0, alpha=1.0):
        super().__init__(None, False)
        self.ndim = ndim
        self.sigma = sigma
        self.alpha = alpha

    def __call__(self, x):
        t = np.linalg.norm(x.reshape(self.ndim, -1), axis=0)
        h = np.where(t <= self.alpha, t**2 / (2 * self.alpha), t - self.alpha / 2)
        return float(self.sigma * np.sum(h))

    def prox(self, x, tau):
        v = x.reshape(self.ndim, -1)
        lam = tau * self.sigma
        t = np.linalg.norm(v, axis=0)
        return (v * (1.0 - lam / np.maximum(t, lam + self.alpha))).ravel()


class Solver(BaseSolver):
    r"""Isotropic Huber total variation regularization, solved with the
    primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2
        + \epsilon \sum_i H_\alpha(\|(\nabla x)_i\|_2)

    where :math:`(\nabla x)_i` is the gradient (forward differences) at
    pixel :math:`i` and :math:`H_\alpha` the Huber function: quadratic for
    gradients smaller than :math:`\alpha` and linear (as in TV) above, which
    reduces the staircasing of TV in smooth areas. Solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual`, with
    :math:`K = \nabla`, :math:`\mu = 1 / (\tau \|\nabla\|_2^2)` and
    :math:`\|\nabla\|_2^2 \le 8`. The proximal operator of the Huber term is
    defined in this file, per pixel.
    """

    name = "PrimalDual-HuberTV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.084],
        "alpha": [0.0025],
        "tau": [0.2],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)
        self.proxf = L2(b=y)
        self.proxg = HuberL21(ndim=2, sigma=self.eps, alpha=self.alpha)
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
