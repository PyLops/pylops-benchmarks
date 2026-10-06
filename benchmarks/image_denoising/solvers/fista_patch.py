import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import DCT, FFT2D, Patch2D, patch2d_design


def max_eig(Aop, niter=30, seed=0):
    """Largest eigenvalue of ``Aop^H Aop`` by power iterations from a fixed
    starting vector (ARPACK is slow for the complex patched FFT)."""
    v = np.random.default_rng(seed).standard_normal(Aop.shape[1]).astype(Aop.dtype)
    for _ in range(niter):
        w = Aop.H @ (Aop @ v)
        lam = np.linalg.norm(w)
        v = w / lam
    return lam


class Solver(BaseSolver):
    r"""Sparsity in overlapping local 2D DCT or FFT windows (synthesis),
    solved with FISTA.

    .. math::
        \min_p \|y - P p\|_2^2 + \epsilon \|p\|_1, \quad x = P p

    where :math:`P` is :class:`pylops.signalprocessing.Patch2D`: it applies
    the inverse 2D DCT (:class:`pylops.signalprocessing.DCT`) or FFT
    (:class:`pylops.signalprocessing.FFT2D`) to the coefficients of each
    window (all windows at once) and blends overlapping windows with a
    taper. Windows of ``nwin`` x ``nwin`` pixels overlap by half. The FISTA step is set by
    power iterations with a fixed starting vector, with a 5% safety
    margin.
    """

    name = "FISTA-Patch2D"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        # weight tuned per transform
        "transform, eps": [("dct", 0.183), ("fft", 0.2)],
        "nwin": [32],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        nwin = (self.nwin, self.nwin)
        nover = (self.nwin // 2, self.nwin // 2)
        nwins = patch2d_design(self.dims, nwin, nover, nwin, verb=False)[0]
        # transform of all windows at once, along the last two axes
        if self.transform == "dct":
            Top = DCT((*nwins, *nwin), axes=(2, 3), dtype=Op.dtype)
        else:
            # real FFT along the last axis: only the non-negative frequencies
            Top = FFT2D(
                (*nwins, *nwin), axes=(2, 3), real=True, norm="ortho", dtype="complex128"
            )
        nop = tuple(Top.dimsd[2:])
        dims = patch2d_design(self.dims, nwin, nover, nop, verb=False)[1]
        self.Pop = Patch2D(Top.H, dims, self.dims, nwin, nover, nop, tapertype="hanning")
        self.Aop = Op @ self.Pop
        self.y = y
        self.alpha = 1.0 / (1.05 * max_eig(self.Aop))

    def run(self, _):
        self.p = fista(
            self.Aop, self.y, niter=self.niter, eps=self.eps, alpha=self.alpha
        )[0]

    def get_result(self):
        return dict(x=np.real(self.Pop @ self.p).reshape(self.dims))
