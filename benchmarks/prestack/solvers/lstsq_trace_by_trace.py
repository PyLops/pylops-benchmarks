import numpy as np
from benchopt import BaseSolver
from pylops.avo.prestack import PrestackLinearModelling
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Damped least squares solved trace by trace with a direct solver.

    .. math::
        \min_{x_i} \|y_i - G x_i\|_2^2 + \sum_{p} \epsilon_{I,p}
        \|x_{i,p} - x_{0,i,p}\|_2^2

    for every trace :math:`i`, where :math:`G` is the dense 1D pre-stack
    modelling matrix, :math:`x_0` is the background model and
    :math:`p \in \{V_P, V_S, \rho\}` (each parameter has its own damping).
    The regularized normal equations
    :math:`(G^T G + E_I) \Delta x_i = G^T (y_i - G x_{0,i})`, with
    :math:`E_I` diagonal, are solved for all traces at once with
    :func:`scipy.linalg.lstsq`, and :math:`x_i = x_{0,i} + \Delta x_i`.
    """

    name = "LSTSQ-TraceByTrace"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI_vp": [5.5e-2],
        "epsI_vs": [1.1e-1],
        "epsI_rho": [2.2e-1],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.ntheta = len(meta["theta"])
        self.mback = meta["mback"]
        nz = self.dims[0]
        G = PrestackLinearModelling(
            meta["wav"],
            meta["theta"],
            vsvp=meta["vsvp"],
            nt0=nz,
            linearization="akirich",
            explicit=True,
        ).A
        epsI = np.repeat([self.epsI_vp, self.epsI_vs, self.epsI_rho], nz)
        self.GT = G.T
        self.PP = G.T @ G + np.diag(epsI)

    def run(self, _):
        nz, npar, nx = self.dims
        # the dense operator works on (parameter, depth) and (angle, depth)
        yr = self.y - self.Op @ self.mback.ravel()
        yr = yr.reshape(nz, self.ntheta, nx).swapaxes(0, 1).reshape(-1, nx)
        dx = lstsq(self.PP, self.GT @ yr)[0]
        self.x = self.mback + dx.reshape(npar, nz, nx).swapaxes(0, 1)

    def get_result(self):
        return dict(x=self.x)
