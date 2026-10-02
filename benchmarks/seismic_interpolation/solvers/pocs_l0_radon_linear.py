import numpy as np
from benchopt import BaseSolver
from pylops.signalprocessing import FourierRadon2D
from pyproximal import L0, AffineSet
from pyproximal.optimization.primal import HQS


class Solver(BaseSolver):
    r"""Data-consistent L0 inversion in the linear Radon domain, solved with POCS.

    .. math::
        \min_p \epsilon_k \|p\|_0 \quad \text{s.t.} \quad Op R p = y,
        \quad x = R p

    where :math:`R` is the linear Radon transform (implemented in the
    frequency domain with :class:`pylops.signalprocessing.FourierRadon2D`).
    Solved by alternating hard thresholding (proximal operator of
    :class:`pyproximal.L0`) and projection onto the affine set of
    data-consistent models (:class:`pyproximal.AffineSet`, ``niter_affine``
    CG iterations), i.e., :func:`pyproximal.optimization.primal.HQS`. The
    sparsity weight decays with the iterations (continuation),
    :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)` with
    :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`. The
    data-consistent iterate is returned.
    """

    name = "POCS-L0-Radon2D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "npx": [200],
        "pxmax": [2e-3],
        "nfft": [256],
        "eps": [1e-2],
        "eps0_factor": [5.0],
        "gamma": [0.97],
        "niter_affine": [8],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.y = y
        px = np.linspace(-self.pxmax, self.pxmax, self.npx)
        self.Pop = FourierRadon2D(
            meta["taxis"],
            meta["xaxis"],
            px,
            self.nfft,
            kind="linear",
            engine="numba",
            dtype=Op.dtype,
        )
        self.SIop = Op @ self.Pop
        self.proxf = AffineSet(Op=self.SIop, b=self.y, niter=self.niter_affine)

    def _sigma(self, k):
        return self.eps * max(1.0, self.eps0_factor * self.gamma**k)

    def run(self, _):
        # L0 counts its calls to evaluate the sigma schedule, so it is
        # created here to restart the schedule at every repetition
        self.p = HQS(
            self.proxf,
            L0(sigma=self._sigma),
            x0=np.zeros(self.SIop.shape[1], dtype=self.SIop.dtype),
            tau=1.0,
            niter=self.niter,
            gfirst=True,
        )[0]

    def get_result(self):
        return dict(x=(self.Pop @ self.p).reshape(self.dims))
