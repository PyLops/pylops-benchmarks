import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import ista


class Solver(BaseSolver):
    r"""L1-regularized least squares (LASSO), solved with ISTA.

    .. math::
        \min_x \|y - Op x\|_2^2 + \epsilon \|x\|_1

    Solved with the Iterative Shrinkage-Thresholding Algorithm
    (:func:`pylops.optimization.sparsity.ista`), with step size
    :math:`1 / \|Op\|_2^2` computed from the explicit matrix.
    """

    name = "ISTA"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [3e-2],
        "niter": [5000],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.alpha = 1.0 / np.linalg.norm(Op.A, 2) ** 2

    def run(self, _):
        self.x = ista(
            self.Op, self.y, niter=self.niter, eps=self.eps, alpha=self.alpha
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
