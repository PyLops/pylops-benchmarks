import numpy as np
from benchopt import BaseSolver
from pylops.avo.prestack import PrestackLinearModelling
from pylops.basicoperators import Diagonal, FirstDerivative, VStack
from pyproximal import L2, L21
from pyproximal.optimization.primaldual import PrimalDual
from scipy.linalg import lstsq


class Solver(BaseSolver):
    r"""Isotropic total variation regularization of each parameter, solved
    with the primal-dual algorithm starting from a trace-by-trace inversion.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2
        + \sum_{p} \epsilon_p \|\nabla x_p\|_{2,1}

    where :math:`\|\nabla x_p\|_{2,1}` is the sum over samples of the norm of
    the gradient along depth and space (isotropic TV) of the parameter
    :math:`p \in \{V_P, V_S, \rho\}`, each with its own weight. Written as
    :math:`\epsilon \|K x\|_{2,1}` with :math:`\epsilon = \max_p \epsilon_p`
    and :math:`K = \nabla W`, where :math:`W` scales each parameter by
    :math:`\epsilon_p / \epsilon`, and solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual` with
    :math:`f = \frac{1}{2} \|y - Op x\|_2^2` (proximal operator computed with
    20 warm-started LSQR iterations) and :math:`g = \epsilon \|\cdot\|_{2,1}`,
    with :math:`\mu = 0.95 / (\tau \|K\|_2^2)`, :math:`\|K\|_2^2 \le 8`. The
    starting guess is the damped trace-by-trace least-squares solution (see
    ``LSTSQ-TraceByTrace``, with the same per-parameter damping) computed
    around the background model.
    """

    name = "PrimalDual-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI_vp": [5.5e-2],
        "epsI_vs": [1.1e-1],
        "epsI_rho": [2.2e-1],
        "eps_vp": [0.024],
        "eps_vs": [0.035],
        "eps_rho": [0.07],
        "tau": [0.3],
        "niter": [60],
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
        self.Kop = (
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
        self.mu = 0.95 / (self.tau * 8.0)

    def run(self, _):
        nz, npar, nx = self.dims
        # starting guess: damped trace-by-trace inversion
        yr = self.y - self.Op @ self.mback.ravel()
        yr = yr.reshape(nz, self.ntheta, nx).swapaxes(0, 1).reshape(-1, nx)
        dx = lstsq(self.PP, self.GT @ yr)[0]
        x0 = (self.mback + dx.reshape(npar, nz, nx).swapaxes(0, 1)).ravel()

        proxf = L2(Op=self.Op, b=self.y, x0=x0, niter=20, warm=True)
        self.x = PrimalDual(
            proxf,
            self.proxg,
            self.Kop,
            tau=self.tau,
            mu=self.mu,
            theta=1.0,
            x0=x0,
            niter=self.niter,
            gfirst=True,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
