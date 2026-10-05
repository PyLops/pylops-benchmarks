from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pylops.optimization.leastsquares import regularized_inversion


class Solver(BaseSolver):
    r"""Regularized least squares with a smoothness prior on the gradient
    (Tikhonov), solved with LSQR.

    .. math::
        \min_x \|y - x\|_2^2 + \epsilon_R^2 \|\nabla x\|_2^2

    where :math:`\nabla` is the 2D gradient (forward differences). Solved
    with :func:`pylops.optimization.leastsquares.regularized_inversion`
    (default ``engine="scipy"``, i.e., :func:`scipy.sparse.linalg.lsqr`).
    """

    name = "LSQR-Gradient"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsR": [0.95],
        "niter": [10],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Regop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)

    def run(self, _):
        self.x = regularized_inversion(
            self.Op, self.y, [self.Regop], epsRs=[self.epsR], iter_lim=self.niter
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
