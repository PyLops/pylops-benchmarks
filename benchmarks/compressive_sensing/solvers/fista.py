import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista


class Solver(BaseSolver):
    r"""L1-regularized least squares (LASSO), solved with FISTA.

    .. math::
        \min_x \|y - Op x\|_2^2 + \epsilon \|x\|_1

    Solved with the Fast Iterative Shrinkage-Thresholding Algorithm (with Nesterov acceleration)
    (:func:`pylops.optimization.sparsity.fista`), with step size
    :math:`1 / \|Op\|_2^2` computed from the explicit matrix.
    """

    name = "FISTA"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [3e-2],
        "niter": [500],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.alpha = 1.0 / np.linalg.norm(Op.A, 2) ** 2

    def run(self, _):
        self.x = fista(
            self.Op, self.y, niter=self.niter, eps=self.eps, alpha=self.alpha
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
