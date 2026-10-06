import numpy as np
from benchopt import BaseSolver
from pyproximal import L2
from pyproximal import RED as REDTerm
from pyproximal.optimization.red import RED

from pylops_bench.denoisers import bm3d_denoise


class Solver(BaseSolver):
    r"""Regularization by Denoising (RED) with the BM3D denoiser, solved
    with fixed-point iterations.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2
        + \sigma x^T (x - f_{\sigma_d}(x))

    where :math:`f_{\sigma_d}` is the BM3D denoiser
    (:func:`pylops_bench.denoisers.bm3d_denoise`) with noise standard
    deviation :math:`\sigma_d`. Solved with
    :func:`pyproximal.optimization.red.RED` and ``solver="fixedpoint"``: each
    iteration denoises the current estimate and solves the normal equations
    :math:`(Op^H Op + \sigma I) x = Op^H y + \sigma f_{\sigma_d}(x)` with a
    few warm-started inner iterations. The real part of the solution is
    kept.
    """

    name = "RED-BM3D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::bm3d"]
    sampling_strategy = "run_once"

    parameters = {
        "sigma": [1.0],
        "sigmad": [0.1],
        "niter": [30],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Op = Op
        self.y = y
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        # the warm-started proximal operator and the RED term keep state
        # between calls: build them at every run so that repeated runs give
        # the same result
        proxf = L2(Op=self.Op, b=self.y, niter=20, warm=True)
        red = REDTerm(
            bm3d_denoise, self.dims, sigma=self.sigma, sigmad=self.sigmad, call=False
        )
        self.x = RED(
            proxf,
            red,
            x0=self.x0,
            solver="fixedpoint",
            niter=self.niter,
            niter_inner=1,
        )

    def get_result(self):
        return dict(x=np.real(self.x).reshape(self.dims))
