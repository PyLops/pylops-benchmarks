import numpy as np
from benchopt import BaseSolver
from pylops.signalprocessing import FourierRadon2D
from pyproximal import L0, L2
from pyproximal.optimization.primal import ProximalGradient


class Solver(BaseSolver):
    r"""L0-regularized inversion in the linear Radon domain, solved with AIHT.

    .. math::
        \min_p \frac{1}{2}\|y - Op R p\|_2^2 + \epsilon_k \|p\|_0,
        \quad x = R p

    where :math:`R` is the linear Radon transform (implemented in the
    frequency domain with :class:`pylops.signalprocessing.FourierRadon2D`).
    Solved with Accelerated Iterative Hard Thresholding, i.e.,
    :func:`pyproximal.optimization.primal.ProximalGradient` with
    ``vandenberghe`` acceleration, :class:`pyproximal.L2` and
    :class:`pyproximal.L0`. The sparsity weight decays with the iterations
    (continuation), :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)`
    with :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`.
    """

    name = "AIHT-Radon2D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    sampling_strategy = "run_once"

    parameters = {
        "npx": [200],
        "pxmax": [2e-3],
        "nfft": [256],
        "eps": [2.0],
        "eps0_factor": [5.0],
        "gamma": [0.97],
        "niter": [300],
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
        self.proxf = L2(Op=self.SIop, b=self.y)
        # step size from the largest eigenvalue of SIop^H SIop, computed with
        # a fixed starting vector so that runs are fully reproducible
        v0 = (
            np.random.default_rng(0)
            .standard_normal(self.SIop.shape[1])
            .astype(self.SIop.dtype)
        )
        maxeig = np.abs(
            (self.SIop.H @ self.SIop).eigs(neigs=1, symmetric=True, v0=v0)
        )
        self.tau = 0.9 / maxeig[0]

    def _sigma(self, k):
        return self.eps * max(1.0, self.eps0_factor * self.gamma**k)

    def run(self, _):
        # L0 counts its calls to evaluate the sigma schedule, so it is
        # created here to restart the schedule at every repetition
        self.p = ProximalGradient(
            self.proxf,
            L0(sigma=self._sigma),
            x0=np.zeros(self.SIop.shape[1], dtype=self.SIop.dtype),
            tau=self.tau,
            niter=self.niter,
            acceleration="vandenberghe",
        )

    def get_result(self):
        return dict(x=(self.Pop @ self.p).reshape(self.dims))
