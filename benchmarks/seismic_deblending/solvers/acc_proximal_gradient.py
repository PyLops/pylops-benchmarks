import numpy as np
from benchopt import BaseSolver
from pyproximal import L0, L1, L2, Orthogonal
from pyproximal.optimization.primal import ProximalGradient

from pylops_bench.deblending import patched_fk


class Solver(BaseSolver):
    r"""Sparsity-promoting deblending in the patched f-k domain, solved with the accelerated proximal gradient.

    .. math::
        \text{synthesis:}\; \min_p \|y - B S p\|_2^2 + \epsilon_k \|p\|_q,
        \quad x = S p \\
        \text{analysis:}\; \min_x \|y - B x\|_2^2 + \epsilon_k \|S^H x\|_q

    where :math:`B` is the continuous blending operator, :math:`S` the
    synthesis operator of the overlapping patched f-k transform and
    :math:`q \in \{0, 1\}`. The threshold decays with
    the iterations, :math:`\epsilon_k = \epsilon_0 \gamma^k`. The step size
    is :math:`0.99 / L` with :math:`L` the fold of the blended data.
    """

    name = "Acc. Proximal Gradient"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "formulation": ["synthesis", "analysis"],
        "regularizer": ["l1", "l0"],
        "gamma": [0.9],
        "niter": [50],
    }

    # initial thresholds eps0 of the continuation
    EPS0 = {"l1": 20.0, "l0": 200.0}

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Sop, _ = patched_fk(self.dims)
        self.synthesis = self.formulation == "synthesis"
        self.Aop = Op @ self.Sop if self.synthesis else Op
        self.tau = 0.99 / meta["LBop"]
        self.l2 = L2(Op=self.Aop, b=y)
        prox = L1() if self.regularizer == "l1" else L0()
        self.prox = prox if self.synthesis else Orthogonal(prox, self.Sop.H)
        self.eps = self.EPS0[self.regularizer] * self.gamma ** np.arange(self.niter)
        self.x0 = np.zeros(
            self.Aop.shape[1], dtype=np.complex128 if self.synthesis else np.float64
        )

    def run(self, _):
        self.sol = ProximalGradient(
            self.l2,
            self.prox,
            tau=self.tau,
            epsg=self.eps,
            niter=self.niter,
            x0=self.x0,
            acceleration="fista",
        )

    def get_result(self):
        x = self.Sop @ self.sol if self.synthesis else self.sol
        return dict(x=np.real(x).reshape(self.dims))
