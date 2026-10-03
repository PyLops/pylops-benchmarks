from benchopt import BaseSolver
from pylops.basicoperators import Laplacian
from pylops.optimization.leastsquares import regularized_inversion


class Solver(BaseSolver):
    r"""Regularized least squares with a smoothness prior, solved with LSQR.

    .. math::
        \min_x \|y - Op x\|_2^2 + \epsilon_R^2 \|\nabla^2 x\|_2^2

    where :math:`\nabla^2` is the 2D Laplacian. Solved with
    :func:`pylops.optimization.leastsquares.regularized_inversion`
    (default ``engine="scipy"``, i.e., :func:`scipy.sparse.linalg.lsqr`).
    """

    name = "LSQR-Laplacian"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsR": [0.1],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Regop = Laplacian(self.dims, dtype=Op.dtype)

    def run(self, _):
        self.x = regularized_inversion(
            self.Op, self.y, [self.Regop], epsRs=[self.epsR], iter_lim=self.niter
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
