from benchopt import BaseSolver
from pyproximal import L2
from pyproximal import RED as REDTerm
from pyproximal.optimization.red import RED

from pylops_bench.denoisers import bm3d_denoise


class Solver(BaseSolver):
    r"""Regularization by Denoising (RED) with the BM3D denoiser, solved
    with fixed-point iterations.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2
        + \sigma x^T (x - f_{\sigma_d}(x))

    where :math:`f_{\sigma_d}` is the BM3D denoiser
    (:func:`pylops_bench.denoisers.bm3d_denoise`) with noise standard
    deviation :math:`\sigma_d`.
    Solved with :func:`pyproximal.optimization.red.RED` and
    ``solver="fixedpoint"``: each iteration denoises the current estimate and
    solves :math:`(1 + \sigma) x = y + \sigma f_{\sigma_d}(x)` (one LSQR
    iteration, exact for the identity operator), starting from the noisy
    image.
    """

    name = "RED-BM3D"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::bm3d"]
    sampling_strategy = "run_once"

    parameters = {
        "sigma": [5.0],
        "sigmad": [0.03],
        "niter": [30],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Op = Op
        self.y = y
        self.x0 = y.copy()

    def run(self, _):
        # the RED term keeps state between calls: build it at every run so
        # that repeated runs give the same result (the fixed-point solver
        # only uses the operator and data of the L2 term)
        proxf = L2(Op=self.Op, b=self.y)
        red = REDTerm(
            bm3d_denoise,
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
            niter_inner=1,
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
