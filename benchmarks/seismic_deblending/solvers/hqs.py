import numpy as np
from benchopt import BaseSolver
from pyproximal import L0, L1, AffineSet, Orthogonal
from pyproximal.optimization.primal import HQS

from pylops_bench.deblending import max_fk_amplitude, patched_fk


class Solver(BaseSolver):
    r"""Data-consistent sparse deblending in the patched f-k domain, solved with POCS.

    .. math::
        \text{synthesis:}\; \min_p \epsilon_k \|p\|_q \;\text{s.t.}\; B S p = y,
        \quad x = S p \\
        \text{analysis:}\; \min_x \epsilon_k \|S^H x\|_q \;\text{s.t.}\; B x = y

    where :math:`B` is the continuous blending operator, :math:`S` the
    synthesis operator of the overlapping patched f-k transform and
    :math:`q \in \{0, 1\}`. Solved by alternating thresholding (proximal
    operator of :class:`pyproximal.L1` or :class:`pyproximal.L0`) and
    projection onto the affine set of data-consistent models
    (:class:`pyproximal.AffineSet`, 5 CG iterations), i.e.,
    :func:`pyproximal.optimization.primal.HQS`. The threshold decays with the
    iterations, :math:`\epsilon_k = \epsilon_0 \gamma^k`.
    """

    name = "POCS"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "formulation": ["synthesis", "analysis"],
        "regularizer": ["l1", "l0"],
        "gamma": [0.9],
        "niter_affine": [5],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Sop, Sop1 = patched_fk(self.dims)
        self.synthesis = self.formulation == "synthesis"
        self.Aop = Op @ self.Sop if self.synthesis else Op
        self.tau = 0.99 / meta["LBop"]
        self.proxf = AffineSet(self.Aop, y, niter=self.niter_affine)
        # initial threshold; L0 is scaled with the largest f-k amplitude
        # of the pseudo-deblended data in the analysis case
        if self.regularizer == "l1":
            self.eps0 = 20.0
        elif self.synthesis:
            self.eps0 = 100.0
        else:
            self.eps0 = 0.6 * max_fk_amplitude(Op, Sop1, y)
        self.x0 = np.zeros(
            self.Aop.shape[1], dtype=np.complex128 if self.synthesis else np.float64
        )

    def _sigma(self, k):
        return self.eps0 * self.gamma**k

    def run(self, _):
        # the proximal operators evaluate the sigma schedule by call count
        prox = (L1 if self.regularizer == "l1" else L0)(sigma=self._sigma)
        if not self.synthesis:
            prox = Orthogonal(prox, self.Sop.H)
        self.sol = HQS(
            self.proxf,
            prox,
            x0=self.x0,
            tau=self.tau,
            gfirst=False,
            niter=self.niter,
        )[0]

    def get_result(self):
        x = self.Sop @ self.sol if self.synthesis else self.sol
        return dict(x=np.real(x).reshape(self.dims))
