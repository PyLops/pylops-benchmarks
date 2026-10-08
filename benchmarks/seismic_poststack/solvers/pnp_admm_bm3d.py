from benchopt import BaseSolver
from pyproximal import L2
from pyproximal.optimization.pnp import PlugAndPlay
from pyproximal.optimization.primal import ADMM

from pylops_bench.denoisers import bm3d_denoise


class Solver(BaseSolver):
    r"""Plug-and-Play prior with the BM3D denoiser, solved with ADMM.

    The proximal operator of the regularizer in
    :func:`pyproximal.optimization.primal.ADMM` is replaced by the BM3D
    denoiser (:func:`pylops_bench.denoisers.bm3d_denoise`), with noise
    standard deviation :math:`\sigma \tau`
    (:func:`pyproximal.optimization.pnp.PlugAndPlay`), starting from the
    background model. The proximal operator of the data term
    :math:`\frac{1}{2} \|y - Op x\|_2^2` is evaluated with 20 warm-started
    LSQR iterations.
    """

    name = "PnP-ADMM-BM3D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::bm3d"]
    sampling_strategy = "run_once"

    parameters = {
        "sigma": [0.2],
        "tau": [1.0],
        "niter": [30],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Op = Op
        self.y = y
        self.x0 = meta["mback"].ravel()

    def run(self, _):
        # warm-started proximal operators keep state between calls: build
        # them at every run so that repeated runs give the same result
        proxf = L2(Op=self.Op, b=self.y, x0=self.x0, niter=20, warm=True)
        self.x = PlugAndPlay(
            proxf,
            lambda x, tau: bm3d_denoise(x, self.sigma * tau),
            self.dims,
            x0=self.x0,
            solver=ADMM,
            tau=self.tau,
            niter=self.niter,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
