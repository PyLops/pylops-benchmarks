import numpy as np
from benchopt import BaseSolver
from pylops.avo.poststack import PoststackLinearModelling
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Damped least squares solved trace by trace with a direct solver.

    .. math::
        \min_{x_i} \|y_i - G x_i\|_2^2 + \epsilon_I \|x_i - x_{0,i}\|_2^2

    for every trace :math:`i`, where :math:`G` is the dense 1D post-stack
    modelling matrix and :math:`x_0` is the background model. The regularized
    normal equations
    :math:`(G^T G + \epsilon_I I) \Delta x_i = G^T (y_i - G x_{0,i})`
    are solved for all traces at once with :func:`scipy.linalg.lstsq`, and
    :math:`x_i = x_{0,i} + \Delta x_i`.
    """

    name = "LSTSQ-TraceByTrace"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI": [4e-2],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.mback = meta["mback"]
        G = PoststackLinearModelling(meta["wav"], nt0=self.dims[0]).todense()
        self.GT = G.T
        self.PP = G.T @ G + self.epsI * np.eye(self.dims[0])

    def run(self, _):
        yr = (self.y - self.Op @ self.mback.ravel()).reshape(self.dims)
        self.x = self.mback + lstsq(self.PP, self.GT @ yr)[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
