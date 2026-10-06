import numpy as np
from benchopt import BaseSolver
from pyproximal import L2, TV
from pyproximal.optimization.primal import ProximalGradient


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with the proximal
    gradient method.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2 + \epsilon \, TV(x)

    where :math:`TV(x)` is the sum over pixels of the norm of the gradient
    (isotropic TV). Solved with
    :func:`pyproximal.optimization.primal.ProximalGradient` with step
    ``tau``: the gradient step is taken on the data term and the proximal
    operator of the TV term (:class:`pyproximal.TV`) is itself computed
    iteratively, with at most ``niter_tv`` iterations and relative tolerance
    ``rtol_tv``. Since the operator is the identity, the gradient of the
    data term is 1-Lipschitz and ``tau=1`` makes a single proximal gradient
    step solve the problem, as accurately as the TV proximal operator.
    """

    name = "ProximalGradient-TV"

    requirements = ["pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.084],
        "tau": [1.0],
        "niter": [1],
        "niter_tv": [100],
        "rtol_tv": [1e-5],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.proxf = L2(b=y)
        self.proxg = TV(
            dims=self.dims, sigma=self.eps, niter=self.niter_tv, rtol=self.rtol_tv
        )
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        self.x = ProximalGradient(
            self.proxf, self.proxg, x0=self.x0, tau=self.tau, niter=self.niter
        )

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
