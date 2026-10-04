from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L2, L21
from pyproximal.optimization.primal import LinearizedADMM


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with linearized ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \|\nabla x\|_{2,1}

    where :math:`\|\nabla x\|_{2,1}` is the sum over samples of the norm of
    the gradient (isotropic TV). Solved with
    :func:`pyproximal.optimization.primal.LinearizedADMM` with
    :math:`f = \frac{1}{2} \|y - Op x\|_2^2` (proximal operator computed with
    20 warm-started LSQR iterations), :math:`g = \epsilon \|\cdot\|_{2,1}`
    and :math:`A = \nabla`, starting from the background model, with
    :math:`\tau = 1` and :math:`\mu = 0.99 \tau / \|\nabla\|_2^2`,
    :math:`\|\nabla\|_2^2 \le 8`.
    """

    name = "LinearizedADMM-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.05],
        "niter": [1600],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.x0 = meta["mback"].ravel()
        self.proxf = L2(Op=Op, b=y, x0=self.x0, niter=20, warm=True)
        self.proxg = L21(ndim=2, sigma=self.eps)
        self.Dop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.tau = 1.0
        self.mu = 0.99 * self.tau / 8.0

    def run(self, _):
        self.x = LinearizedADMM(
            self.proxf,
            self.proxg,
            self.Dop,
            tau=self.tau,
            mu=self.mu,
            x0=self.x0,
            niter=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
