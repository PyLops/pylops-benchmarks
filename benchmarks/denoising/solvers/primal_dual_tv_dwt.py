import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient, VStack
from pylops.signalprocessing import DWT2D
from pyproximal import L1, L2, L21
from pyproximal import VStack as ProxVStack
from pyproximal.optimization.primaldual import PrimalDual


class Solver(BaseSolver):
    r"""Isotropic total variation plus sparsity in the wavelet domain
    (analysis), solved with the primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon_{TV} \|\nabla x\|_{2,1}
        + \epsilon_W \|W x\|_1

    where :math:`\nabla` is the 2D gradient (forward differences) and
    :math:`W` the 2D Discrete Wavelet Transform
    (:class:`pylops.signalprocessing.DWT2D`). Solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual` with
    :math:`K = [\nabla; W]`, :math:`\mu = 1 / (\tau \|K\|_2^2)` and
    :math:`\|K\|_2^2 \le 8 + 1`.
    """

    name = "PrimalDual-TV-DWT2D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::PyWavelets"]
    sampling_strategy = "run_once"

    parameters = {
        "wavelet": ["db4"],
        "level": [4],
        "eps_tv": [0.0714],
        "eps_w": [0.0168],
        "tau": [0.2],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        Gop = Gradient(self.dims, edge=False, kind="forward", dtype=Op.dtype)
        Wop = DWT2D(self.dims, wavelet=self.wavelet, level=self.level)
        self.Kop = VStack([Gop, Wop])
        self.proxf = L2(b=y)
        self.proxg = ProxVStack(
            [L21(ndim=2, sigma=self.eps_tv), L1(sigma=self.eps_w)],
            nn=[Gop.shape[0], Wop.shape[0]],
        )
        self.mu = 1.0 / (self.tau * 9.0)
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.x = PrimalDual(
            self.proxf,
            self.proxg,
            self.Kop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            niter=self.niter,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
