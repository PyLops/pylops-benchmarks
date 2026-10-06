import numpy as np
from benchopt import BaseSolver
from pyproximal import L1, AffineSet
from pyproximal.optimization.primal import ADMM


class Solver(BaseSolver):
    r"""Basis pursuit, solved with ADMM.

    .. math::
        \min_x \|x\|_1 \quad \text{s.t.} \quad Op x = y

    The data are fitted exactly (noise included) through the indicator
    function of the affine set :math:`\{x: Op x = y\}`
    (:class:`pyproximal.AffineSet`, whose projection is computed with
    ``niter_affine`` inner iterations), and solved with
    :func:`pyproximal.optimization.primal.ADMM`.
    """

    name = "ADMM-BP"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "tau": [0.1],
        "niter_affine": [20],
        "niter": [300],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.proxf = AffineSet(Op, y, niter=self.niter_affine)
        self.proxg = L1()
        self.x0 = np.zeros(Op.shape[1])

    def run(self, _):
        self.x = ADMM(
            self.proxf, self.proxg, x0=self.x0, tau=self.tau, niter=self.niter
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
