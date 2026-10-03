import numpy as np
from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import DWT2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the wavelet domain, solved with FISTA.

    .. math::
        \min_p \|y - Op W^H p\|_2^2 + \epsilon \|p\|_q, \quad x = W^H p

    where :math:`W` is the 2D Discrete Wavelet Transform
    (:class:`pylops.signalprocessing.DWT2D`) and :math:`q = 1` with soft
    thresholding (``threshkind="soft"``) or :math:`q = 0` with hard
    thresholding (``threshkind="hard"``). The sparsity weight is set per
    thresholding kind, as the two penalties have different scales.
    """

    name = "FISTA-DWT2D"

    requirements = ["pip::pylops", "pip::PyWavelets"]
    sampling_strategy = "run_once"

    parameters = {
        "wavelet": ["haar"],
        "level": [3],
        "threshkind, eps": [("soft", 1e-1), ("hard", 1.0)],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Wop = DWT2D(self.dims, wavelet=self.wavelet, level=self.level)
        self.Aop = Op @ self.Wop.H
        self.y = y
        # fixed starting vector for the power iterations used to pick the
        # FISTA step size, so that runs are fully reproducible
        self.eigsdict = dict(
            v0=np.random.default_rng(0).standard_normal(self.Aop.shape[1])
        )

    def run(self, _):
        self.p = fista(
            self.Aop,
            self.y,
            niter=self.niter,
            eps=self.eps,
            threshkind=self.threshkind,
            eigsdict=self.eigsdict,
        )[0]

    def get_result(self):
        return dict(x=(self.Wop.H @ self.p).reshape(self.dims))
