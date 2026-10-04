from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L21
from pyproximal.optimization.primal import ADMML2


class Solver(BaseSolver):
    r"""Isotropic total variation regularization, solved with ADMM
    specialized for an L2 misfit.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \|\nabla x\|_{2,1}

    where :math:`\|\nabla x\|_{2,1}` is the sum over samples of the norm of
    the gradient (isotropic TV). Solved with
    :func:`pyproximal.optimization.primal.ADMML2`, where the
    :math:`x`-update is a least-squares problem solved with 20 LSQR
    iterations, starting from the background model, with
    :math:`\tau = 0.99 / \|\nabla\|_2^2`, :math:`\|\nabla\|_2^2 \le 8`.
    """

    name = "ADMML2-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.03],
        "niter": [250],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.x0 = meta["mback"].ravel()
        self.Dop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.proxg = L21(ndim=2, sigma=self.eps)
        self.tau = 0.99 / 8.0

    def run(self, _):
        self.x = ADMML2(
            self.proxg,
            self.Op,
            self.y,
            self.Dop,
            x0=self.x0,
            tau=self.tau,
            niter=self.niter,
            iter_lim=20,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
