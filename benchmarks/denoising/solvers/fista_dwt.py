from benchopt import BaseSolver
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import DWT2D


class Solver(BaseSolver):
    r"""Sparsity-promoting inversion in the wavelet domain (synthesis),
    solved with FISTA.

    .. math::
        \min_p \|y - W^H p\|_2^2 + \epsilon \|p\|_q, \quad x = W^H p

    where :math:`W` is the 2D Discrete Wavelet Transform
    (:class:`pylops.signalprocessing.DWT2D`) and :math:`q = 1` with soft
    thresholding (``threshkind="soft"``) or :math:`q = 0` with hard
    thresholding (``threshkind="hard"``). The sparsity weight is set per
    thresholding kind, as the two penalties have different scales. Since
    :math:`W` is orthogonal, the step is set to 1 (no power iterations) and
    FISTA converges in one iteration to classic wavelet shrinkage.
    """

    name = "FISTA-DWT2D"

    requirements = ["pip::pylops", "pip::PyWavelets"]
    sampling_strategy = "run_once"

    parameters = {
        "wavelet": ["db4"],
        # decomposition level and weight tuned per thresholding kind
        "threshkind, level, eps": [("soft", 5, 0.34), ("hard", 3, 0.13)],
        "niter": [1],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Wop = DWT2D(self.dims, wavelet=self.wavelet, level=self.level)
        self.Aop = Op @ self.Wop.H
        self.y = y

    def run(self, _):
        self.p = fista(
            self.Aop,
            self.y,
            niter=self.niter,
            eps=self.eps,
            alpha=1.0,
            threshkind=self.threshkind,
        )[0]

    def get_result(self):
        return dict(x=(self.Wop.H @ self.p).reshape(self.dims))
