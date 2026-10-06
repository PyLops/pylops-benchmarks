import numpy as np
from benchopt import BaseSolver
from pylops.avo.prestack import PrestackLinearModelling
from pylops.basicoperators import Diagonal, FirstDerivative, VStack
from pyproximal import L21
from pyproximal.optimization.primal import ADMML2
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Isotropic total variation regularization of each parameter, solved
    with ADMM specialized for an L2 misfit, starting from a trace-by-trace
    inversion.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2
        + \sum_{p} \epsilon_p \|\nabla x_p\|_{2,1}

    where :math:`\|\nabla x_p\|_{2,1}` is the sum over samples of the norm of
    the gradient along depth and space (isotropic TV) of the parameter
    :math:`p \in \{V_P, V_S, \rho\}`, each with its own weight. Written as
    :math:`\epsilon \|K x\|_{2,1}` with :math:`\epsilon = \max_p \epsilon_p`
    and :math:`K = \nabla W`, where :math:`W` scales each parameter by
    :math:`\epsilon_p / \epsilon`, and solved with
    :func:`pyproximal.optimization.primal.ADMML2`, where the
    :math:`x`-update is a least-squares problem solved with 20 LSQR
    iterations, with :math:`\tau = 0.99 / \|K\|_2^2`,
    :math:`\|K\|_2^2 \le 8`.
    The starting guess is the damped trace-by-trace least-squares solution
    (see ``LSTSQ-TraceByTrace``, with the same per-parameter damping)
    computed around the background model.
    """

    name = "ADMML2-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI_vp": [5.5e-2],
        "epsI_vs": [1.1e-1],
        "epsI_rho": [2.2e-1],
        "eps_vp": [0.0075],
        "eps_vs": [0.0105],
        "eps_rho": [0.021],
        "niter": [55],
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
        epsI = np.repeat([self.epsI_vp, self.epsI_vs, self.epsI_rho], nz)
        self.PP = G.T @ G + np.diag(epsI)
        eps = np.array([self.eps_vp, self.eps_vs, self.eps_rho])
        self.proxg = L21(ndim=2, sigma=eps.max())
        # gradient along depth and space, scaled per parameter
        Wop = Diagonal(eps / eps.max(), dims=self.dims, axis=1, dtype=Op.dtype)
        self.Dop = (
            VStack(
                [
                    FirstDerivative(
                        self.dims, axis=ax, edge=True, kind="forward", dtype=Op.dtype
                    )
                    for ax in (0, 2)
                ]
            )
            @ Wop
        )
        self.tau = 0.99 / 8.0

    def run(self, _):
        nz, npar, nx = self.dims
        # starting guess: damped trace-by-trace inversion
        yr = self.y - self.Op @ self.mback.ravel()
        yr = yr.reshape(nz, self.ntheta, nx).swapaxes(0, 1).reshape(-1, nx)
        dx = lstsq(self.PP, self.GT @ yr)[0]
        x0 = (self.mback + dx.reshape(npar, nz, nx).swapaxes(0, 1)).ravel()

        self.x = ADMML2(
            self.proxg,
            self.Op,
            self.y,
            self.Dop,
            x0=x0,
            tau=self.tau,
            niter=self.niter,
            iter_lim=20,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
