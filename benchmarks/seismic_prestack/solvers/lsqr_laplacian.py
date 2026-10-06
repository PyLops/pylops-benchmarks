import numpy as np
from benchopt import BaseSolver
from pylops.avo.prestack import PrestackLinearModelling
from pylops.basicoperators import Laplacian, Restriction
from pylops.optimization.leastsquares import regularized_inversion
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Regularized least squares with a spatial Laplacian on each parameter,
    solved with LSQR starting from a trace-by-trace inversion.

    .. math::
        \min_{x} \|y - Op x\|_2^2
        + \sum_{p} \epsilon_{R,p}^2 \|\nabla^2 (x_p - x_{0,p})\|_2^2
        + \text{damp}^2 \|x - x_0\|_2^2

    where :math:`\nabla^2` is the 2D Laplacian (depth and space) of the
    parameter :math:`p \in \{V_P, V_S, \rho\}`, each with its own weight, and
    the starting guess :math:`x_0` is the damped trace-by-trace
    least-squares solution (see ``LSTSQ-TraceByTrace``, with damping
    :math:`\epsilon_I` for all parameters) computed around the background
    model. Since :func:`pylops.optimization.leastsquares.regularized_inversion`
    solves for the update :math:`x - x_0`, both the Laplacian and the damping
    act on the update. Solved with the default ``engine="scipy"``, i.e.,
    :func:`scipy.sparse.linalg.lsqr`.
    """

    name = "LSQR-Laplacian"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI": [8e-2],
        "epsR_vp": [0.875],
        "epsR_vs": [1.25],
        "epsR_rho": [2.5],
        "damp": [4e-2],
        "niter": [2200],
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
        self.GT = G.T
        self.PP = G.T @ G + self.epsI * np.eye(G.shape[1])
        # one Laplacian (depth and space) per parameter
        Lop = Laplacian(self.dims, axes=(0, 2), dtype=Op.dtype)
        self.Regs = [
            Restriction(self.dims, [ip], axis=1, dtype=Op.dtype) @ Lop
            for ip in range(self.dims[1])
        ]
        self.epsRs = [self.epsR_vp, self.epsR_vs, self.epsR_rho]

    def run(self, _):
        nz, npar, nx = self.dims
        # starting guess: damped trace-by-trace inversion
        yr = self.y - self.Op @ self.mback.ravel()
        yr = yr.reshape(nz, self.ntheta, nx).swapaxes(0, 1).reshape(-1, nx)
        dx = lstsq(self.PP, self.GT @ yr)[0]
        x0 = self.mback + dx.reshape(npar, nz, nx).swapaxes(0, 1)

        self.x = regularized_inversion(
            self.Op,
            self.y,
            self.Regs,
            x0=x0.ravel(),
            epsRs=self.epsRs,
            damp=self.damp,
            iter_lim=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
