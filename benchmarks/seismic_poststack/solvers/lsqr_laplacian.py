import numpy as np
from benchopt import BaseSolver
from pylops.avo.poststack import PoststackLinearModelling
from pylops.basicoperators import Laplacian
from pylops.optimization.leastsquares import regularized_inversion
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Regularized least squares with a spatial Laplacian, solved with LSQR
    starting from a trace-by-trace inversion.

    .. math::
        \min_{x} \|y - Op x\|_2^2 + \epsilon_R^2 \|\nabla^2 (x - x_0)\|_2^2
        + \text{damp}^2 \|x - x_0\|_2^2

    where :math:`\nabla^2` is the 2D Laplacian and the starting guess
    :math:`x_0` is the damped trace-by-trace least-squares solution
    (see ``LSTSQ-TraceByTrace``, with damping :math:`\epsilon_I`) computed
    around the background model. Since
    :func:`pylops.optimization.leastsquares.regularized_inversion` solves for
    the update :math:`x - x_0`, both the Laplacian and the damping act on the
    update. Solved with the default ``engine="scipy"``, i.e.,
    :func:`scipy.sparse.linalg.lsqr`.
    """

    name = "LSQR-Laplacian"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI": [4e-2],
        "epsR": [3.0],
        "damp": [1e-2],
        "niter": [5000],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.mback = meta["mback"]
        G = PoststackLinearModelling(meta["wav"], nt0=self.dims[0]).todense()
        self.GT = G.T
        self.PP = G.T @ G + self.epsI * np.eye(self.dims[0])
        self.Regop = Laplacian(self.dims, dtype=Op.dtype)

    def run(self, _):
        # starting guess: damped trace-by-trace inversion
        yr = (self.y - self.Op @ self.mback.ravel()).reshape(self.dims)
        x0 = self.mback + lstsq(self.PP, self.GT @ yr)[0]

        self.x = regularized_inversion(
            self.Op,
            self.y,
            [self.Regop],
            x0=x0.ravel(),
            epsRs=[self.epsR],
            damp=self.damp,
            iter_lim=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
