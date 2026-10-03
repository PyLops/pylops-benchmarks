import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Real, Restriction
from pylops.signalprocessing import FFT2D
from pyproximal import L0, AffineSet
from pyproximal.optimization.primal import HQS


class Solver(BaseSolver):
    r"""Data-consistent L0 inversion in the f-k domain, solved with POCS.

    .. math::
        \min_p \epsilon_k \|p\|_0 \quad \text{s.t.} \quad
        \Re\{Op F^H p\} = y,
        \quad x = \Re\{F^H p\}

    where :math:`F` is the 2D Fourier transform. Solved by alternating hard
    thresholding (proximal operator of :class:`pyproximal.L0`) and projection
    onto the affine set of data-consistent models
    (:class:`pyproximal.AffineSet`, ``niter_affine`` CG iterations), i.e.,
    :func:`pyproximal.optimization.primal.HQS`. Since :math:`F` is
    orthonormal and zero-padded, :math:`F^H F = I`, so the projection is
    exact after a single CG iteration. The sparsity weight decays with the
    iterations (continuation),
    :math:`\epsilon_k = \max(\epsilon, \epsilon_0 \gamma^k)` with
    :math:`\epsilon_0 = \text{eps0\_factor} \cdot \epsilon`. The
    data-consistent iterate is returned. As in
    :func:`pylops.waveeqprocessing.SeismicInterpolation`, the restriction
    operator is rebuilt with complex dtype (from ``meta["iava"]``) so that it
    can be applied to the complex output of :math:`F^H`.
    """

    name = "POCS-L0-FFT2D"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "nfft": [1024],
        "eps": [3e-3],
        "eps0_factor": [100.0],
        "gamma": [0.97],
        "niter_affine": [1],
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
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
