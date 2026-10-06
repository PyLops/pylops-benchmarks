import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import ETP, L2, SCAD, Log
from pyproximal.optimization.primal import LinearizedADMM

PENALTIES = {"log": Log, "scad": SCAD, "etp": ETP}


class Solver(BaseSolver):
    r"""Nonconvex sparsity on the gradient (anisotropic), solved with the
    linearized ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon \sum_i \phi((\nabla x)_i)

    where :math:`\nabla` is the 2D gradient (forward differences) and
    :math:`\phi` a nonconvex sparsity penalty applied to each derivative:
    the logarithmic penalty (:class:`pyproximal.Log`), SCAD
    (:class:`pyproximal.SCAD`) or the exponential-type penalty
    (:class:`pyproximal.ETP`), with shape parameter ``gamma`` (``a`` for
    SCAD). Large gradients are penalized less than with TV, which keeps
    edges sharper. Solved with
    :func:`pyproximal.optimization.primal.LinearizedADMM`, with
    :math:`\mu = \tau / \|\nabla\|_2^2` and :math:`\|\nabla\|_2^2 \le 8`;
    convergence is not guaranteed for nonconvex penalties.
    """

    name = "LinearizedADMM-NonconvexTV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        # weight and shape parameter tuned per penalty
        "penalty, eps, gamma": [
            ("log", 0.07, 0.057),
            ("scad", 0.07, 12.4),
            ("etp", 0.035, 2.0),
        ],
        "tau": [4.0],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)
        self.proxf = L2(b=y)
        self.proxg = PENALTIES[self.penalty](self.eps, self.gamma)
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
