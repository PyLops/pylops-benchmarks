import numpy as np
from benchopt import BaseSolver
from pylops import Identity
from pyproximal import L0, L1, AffineSet
from pyproximal.optimization.primaldual import PrimalDual

from pylops_bench.deblending import patched_fk


class Solver(BaseSolver):
    r"""Data-consistent sparse deblending in the patched f-k domain, solved with primal-dual.

    .. math::
        \text{synthesis:}\; \min_p \sigma \|p\|_q \;\text{s.t.}\; B S p = y,
        \quad x = S p \\
        \text{analysis:}\; \min_x \sigma \|S^H x\|_q \;\text{s.t.}\; B x = y

    where :math:`B` is the continuous blending operator, :math:`S` the
    synthesis operator of the overlapping patched f-k transform and
    :math:`q \in \{0, 1\}`. Solved with
    :func:`pyproximal.optimization.primal.PrimalDual`, with
    :class:`pyproximal.AffineSet` (5 CG iterations) for the data constraint
    and a linear operator of norm 1 in the regularization term (the identity
    in the synthesis case, :math:`S^H` in the analysis case).
    """

    name = "PrimalDual"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "formulation": ["synthesis", "analysis"],
        "regularizer": ["l1", "l0"],
        "niter_affine": [5],
        "niter": [50],
    }

    SIGMA = {"l1": 1.0, "l0": 10.0}

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Sop, _ = patched_fk(self.dims)
        self.synthesis = self.formulation == "synthesis"
        self.Aop = Op @ self.Sop if self.synthesis else Op
        self.Kop = (
            Identity(self.Aop.shape[1], dtype=np.complex128)
            if self.synthesis
            else self.Sop.H
        )
        self.proxf = AffineSet(self.Aop, y, niter=self.niter_affine)
        self.proxg = (L1 if self.regularizer == "l1" else L0)(
            sigma=self.SIGMA[self.regularizer]
        )
        # ||K|| = 1 for both the identity and the patched f-k transform
        self.tau = 0.99
        self.mu = 0.99 / self.tau
        self.x0 = np.zeros(
            self.Aop.shape[1], dtype=np.complex128 if self.synthesis else np.float64
        )

    def run(self, _):
        self.sol = PrimalDual(
            self.proxf,
            self.proxg,
            self.Kop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            gfirst=False,
            niter=self.niter,
        )

    def get_result(self):
        x = self.Sop @ self.sol if self.synthesis else self.sol
        return dict(x=np.real(x).reshape(self.dims))
