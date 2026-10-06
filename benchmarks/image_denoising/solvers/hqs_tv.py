import numpy as np
from benchopt import BaseSolver
from pyproximal import L2, TV
from pyproximal.optimization.primal import HQS


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with Half Quadratic
    Splitting.

    .. math::
        \min_{x, z} \frac{1}{2} \|y - x\|_2^2 + \epsilon \, TV(z)
        + \frac{1}{2 \tau} \|x - z\|_2^2

    solved with :func:`pyproximal.optimization.primal.HQS`, alternating the
    closed-form proximal operator of the data term and the proximal operator
    of the TV term (:class:`pyproximal.TV`, computed iteratively with at most
    ``niter_tv`` iterations and relative tolerance ``rtol_tv``). For
    denoising, the fixed point :math:`z` is the TV-denoised image with weight
    :math:`\epsilon (1 + \tau)`, so :math:`z` is returned and the weight is
    rescaled internally: ``eps`` is the TV weight of the solution, the same as
    in the other TV solvers, and ``tau`` only changes the convergence speed.
    """

    name = "HQS-TV"

    requirements = ["pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.084],
        "tau": [30.0],
        "niter": [3],
        "niter_tv": [100],
        "rtol_tv": [1e-5],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.proxf = L2(b=y)
        self.proxg = TV(
            dims=self.dims,
            sigma=self.eps / (1.0 + self.tau),
            niter=self.niter_tv,
            rtol=self.rtol_tv,
        )
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.z = HQS(
            self.proxf,
            self.proxg,
            x0=self.x0,
            tau=self.tau,
            niter=self.niter,
            gfirst=False,
        )[1]

    def get_result(self):
        return dict(x=self.z.reshape(self.dims))
