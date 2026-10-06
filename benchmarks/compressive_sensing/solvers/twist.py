import numpy as np
from benchopt import BaseSolver
from pyproximal import L1
from pyproximal.optimization.primal import TwIST


class Solver(BaseSolver):
    r"""L1-regularized least squares (LASSO), solved with TwIST.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \|x\|_1

    Solved with the Two-step Iterative Shrinkage/Thresholding algorithm
    (:func:`pyproximal.optimization.primal.TwIST`), whose parameters are set
    from the largest eigenvalue of :math:`Op^T Op` (computed from the
    explicit matrix) and an assumed smallest eigenvalue ``eigmin``.
    """

    name = "TwIST"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [1e-2],
        "eigmin": [1e-2],
        "niter": [1000],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.proxg = L1(sigma=self.eps)
        self.eigs = (np.linalg.norm(Op.A, 2) ** 2, self.eigmin)
        self.x0 = np.zeros(Op.shape[1])

    def run(self, _):
        self.x = TwIST(
            self.proxg,
            self.Op,
            self.y,
            x0=self.x0,
            eigs=self.eigs,
            niter=self.niter,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
