import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import SecondDerivative
from pylops.optimization.leastsquares import regularized_inversion


class Solver(BaseSolver):
    r"""Regularized least squares with spatial smoothing, solved with LSQR.

    .. math::
        \min_x \|y - Op x\|_2^2 + \epsilon_R^2 \|D_2 x\|_2^2
        + \text{damp}^2 \|x\|_2^2

    where :math:`D_2` is the second derivative along the spatial axis.
    Solved with :func:`pylops.optimization.leastsquares.regularized_inversion`
    (default ``engine="scipy"``, i.e., :func:`scipy.sparse.linalg.lsqr`).
    """

    name = "LSQR-SecondDerivative"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsR": [np.sqrt(0.1)],
        "damp": [np.sqrt(1e-4)],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Regop = SecondDerivative(self.dims, axis=0, dtype=Op.dtype)

    def run(self, _):
        self.x = regularized_inversion(
            self.Op,
            self.y,
            [self.Regop],
            epsRs=[self.epsR],
            damp=self.damp,
            iter_lim=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
