import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Real, Restriction
from pylops.signalprocessing import FFT2D
from pyproximal import L0, L2
from pyproximal.optimization.primal import ProximalGradient


class Solver(BaseSolver):
    r"""L0-regularized inversion in the f-k domain, solved with AIHT.

    .. math::
        \min_p \frac{1}{2}\|y - \Re\{Op F^H p\}\|_2^2 + \epsilon_k \|p\|_0,
        \quad x = \Re\{F^H p\}

    where :math:`F` is the 2D Fourier transform. Solved with Accelerated
    Iterative Hard Thresholding, i.e.,
    :func:`pyproximal.optimization.primal.ProximalGradient` with
    ``vandenberghe`` acceleration, :class:`pyproximal.L2` and
    :class:`pyproximal.L0`. The sparsity weight decays with the iterations
    (continuation), :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)`
    with :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`. As in
    :func:`pylops.waveeqprocessing.SeismicInterpolation`, the restriction
    operator is rebuilt with complex dtype (from ``meta["iava"]``) so that it
    can be applied to the complex output of :math:`F^H`.
    """

    name = "AIHT-FFT2D"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "nfft": [1024],
        "eps": [3e-3],
        "eps0_factor": [100.0],
        "gamma": [0.97],
        "niter": [200],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        iava = meta["iava"]
        cdtype = np.result_type(Op.dtype, np.complex64)
        Rop = Restriction(dims=self.dims, iava=iava, axis=0, dtype=cdtype)
        Realop = Real((len(iava), self.dims[1]), dtype=cdtype)
        self.Pop = FFT2D(
            dims=self.dims,
            nffts=(self.nfft, self.nfft),
            sampling=(meta["dx"], meta["dt"]),
            dtype=cdtype,
        ).H
        self.SIop = Realop @ Rop @ self.Pop
        self.y = y.astype(self.SIop.dtype)
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
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
