import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import DWT2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the wavelet domain, solved with FISTA.

    .. math::
        \min_p \|y - Op W^H p\|_2^2 + \epsilon \|p\|_1, \quad x = \Re\{W^H p\}

    where :math:`W` is the 2D Discrete Wavelet Transform
    (:class:`pylops.signalprocessing.DWT2D`). As :math:`W` is orthogonal,
    the Fourier transform orthonormal and the mask a projection,
    :math:`\|Op W^H\|_2 = 1` and the step size is set to 1.
    """

    name = "FISTA-DWT2D"

    requirements = ["pip::pylops", "pip::PyWavelets"]
    sampling_strategy = "run_once"

    parameters = {
        "wavelet": ["haar"],
        "level": [3],
        "eps": [1e-2],
        "niter": [500],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Wop = DWT2D(
            self.dims, wavelet=self.wavelet, level=self.level, dtype=Op.dtype
        )
        self.Aop = Op @ self.Wop.H
        self.y = y

    def run(self, _):
        self.p = fista(self.Aop, self.y, niter=self.niter, eps=self.eps, alpha=1.0)[0]

    def get_result(self):
        return dict(x=np.real(self.Wop.H @ self.p).reshape(self.dims))
