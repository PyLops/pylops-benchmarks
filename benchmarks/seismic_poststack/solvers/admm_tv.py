from benchopt import BaseSolver
from pyproximal import L2, TV
from pyproximal.optimization.primal import ADMM


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \, \text{TV}(x)

    Solved with :func:`pyproximal.optimization.primal.ADMM` with
    :math:`f = \frac{1}{2} \|y - Op x\|_2^2` (proximal operator computed with
    20 warm-started LSQR iterations) and :math:`g = \epsilon \, \text{TV}`,
    whose proximal operator is itself computed iteratively
    (:class:`pyproximal.TV`, relative tolerance :math:`10^{-2}`), starting
    from the background model, with :math:`\tau = 0.99 / \|Op\|_2^2`.
    """

    name = "ADMM-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.06],
        "niter": [350],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.x0 = meta["mback"].ravel()
        self.proxf = L2(Op=Op, b=y, x0=self.x0, niter=20, warm=True)
        self.proxg = TV(dims=self.dims, sigma=self.eps, rtol=1e-2)
        self.tau = 0.99 / meta["LOp"]

    def run(self, _):
        self.x = ADMM(
            self.proxf, self.proxg, x0=self.x0, tau=self.tau, niter=self.niter
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
