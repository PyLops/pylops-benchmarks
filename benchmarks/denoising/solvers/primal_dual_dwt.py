import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import VStack
from pylops.signalprocessing import DWT2D
from pyproximal import L1, L2
from pyproximal import VStack as ProxVStack
from pyproximal.optimization.primaldual import PrimalDual


class Solver(BaseSolver):
    r"""Sparsity in one or more wavelet domains (analysis), solved with the
    primal-dual algorithm.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon \sum_i \|W_i x\|_1

    where the :math:`W_i` are 2D Discrete Wavelet Transforms
    (:class:`pylops.signalprocessing.DWT2D`) with the wavelets listed in
    ``wavelets`` (e.g., ``"haar+db4"`` combines a Haar and a Daubechies-4
    prior). The transforms are stacked into :math:`K = [W_1; W_2; \ldots]`
    and the problem is solved with
    :func:`pyproximal.optimization.primaldual.PrimalDual`, with
    :math:`\mu = 1 / (\tau \|K\|_2^2)` and :math:`\|K\|_2^2` equal to the
    number of (orthogonal) wavelet transforms.
    """

    name = "PrimalDual-DWT2D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::PyWavelets"]
    sampling_strategy = "run_once"

    parameters = {
        # a single orthogonal wavelet gives the same solution as soft
        # thresholding (FISTA-DWT2D): only the redundant, stacked prior is run
        "wavelets": ["haar+db4"],
        "level": [5],
        "eps": [0.077],
        "tau": [0.5],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        Wops = [
            DWT2D(self.dims, wavelet=wavelet, level=self.level)
            for wavelet in self.wavelets.split("+")
        ]
        self.proxf = L2(b=y)
        if len(Wops) == 1:
            self.Kop = Wops[0]
            self.proxg = L1(sigma=self.eps)
        else:
            self.Kop = VStack(Wops)
            self.proxg = ProxVStack(
                [L1(sigma=self.eps) for _ in Wops], nn=[W.shape[0] for W in Wops]
            )
        self.mu = 1.0 / (self.tau * len(Wops))
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
