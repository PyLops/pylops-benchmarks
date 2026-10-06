import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L1, L2
from pyproximal.optimization.primal import LinearizedADMM


class Solver(BaseSolver):
    r"""Anisotropic total variation regularization, solved with the
    linearized ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon \|\nabla x\|_1

    where :math:`\nabla` is the 2D gradient (forward differences), so that
    :math:`\|\nabla x\|_1` is the sum of the absolute values of the
    derivatives along the two axes (anisotropic TV). Solved with
    :func:`pyproximal.optimization.primal.LinearizedADMM`, with
    :math:`\mu = \tau / \|\nabla\|_2^2` and :math:`\|\nabla\|_2^2 \le 8`.
    """

    name = "LinearizedADMM-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.07],
        "tau": [4.0],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)
        self.proxf = L2(b=y)
        self.proxg = L1(sigma=self.eps)
        self.mu = self.tau / 8.0
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.x = LinearizedADMM(
            self.proxf,
            self.proxg,
            self.Gop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            niter=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
