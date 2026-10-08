from functools import partial

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
    deviation :math:`\sigma_d` relative to the 0-255 range of the image.
    Solved with :func:`pyproximal.optimization.red.RED` and
    ``solver="fixedpoint"``: each iteration denoises the current estimate and
    solves the normal equations
    :math:`(Op^H Op + \sigma I) x = Op^H y + \sigma f_{\sigma_d}(x)` with
    ``niter_inner`` LSQR iterations started from the current estimate.
    """

    name = "RED-BM3D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::bm3d"]
    sampling_strategy = "run_once"

    parameters = {
        "sigma": [0.02],
        "sigmad": [0.08],
        "niter": [60],
        "niter_inner": [50],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Op = Op
        self.y = y
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        # the RED term keeps state between calls: build it at every run so
        # that repeated runs give the same result (the fixed-point solver
        # only uses the operator and data of the L2 term)
        proxf = L2(Op=self.Op, b=self.y)
        red = REDTerm(
            partial(bm3d_denoise, scale=255.0),
            self.dims,
            sigma=self.sigma,
            sigmad=self.sigmad,
            call=False,
        )
        self.x = RED(
            proxf,
            red,
            x0=self.x0,
            solver="fixedpoint",
            niter=self.niter,
            niter_inner=self.niter_inner,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
