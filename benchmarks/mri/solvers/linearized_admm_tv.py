import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L1, L2, L21
from pyproximal.optimization.primal import LinearizedADMM


# proximal operator of the TV term on the stacked derivatives of the gradient
TV_PROX = {
    "iso": lambda eps: L21(ndim=2, sigma=eps),
    "aniso": lambda eps: L1(sigma=eps),
}


class Solver(BaseSolver):
    r"""Isotropic or anisotropic total variation regularization, solved with
    the linearized ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2 + \epsilon \, \mathrm{TV}(x)

    where :math:`\mathrm{TV}(x)` is either the isotropic total variation
    :math:`\|\nabla x\|_{2,1}` (sum over pixels of the norm of the gradient,
    ``tv="iso"``) or the anisotropic one :math:`\|\nabla x\|_1` (sum of the
    absolute values of both derivatives, ``tv="aniso"``), with :math:`\nabla`
    the 2D gradient (forward differences).
    Solved with :func:`pyproximal.optimization.primal.LinearizedADMM`, with
    :math:`\mu = 0.99 \tau / \|\nabla\|_2^2` and
    :math:`\|\nabla\|_2^2 \le 8`. The proximal operator of the data term is
    evaluated with a few warm-started inner iterations. The real part of
    the solution is kept.
    """

    name = "LinearizedADMM-TV"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        # weight tuned per kind of TV
        "tv, eps": [("iso", 3e-3), ("aniso", 2e-3)],
        "tau": [20.0],
        "niter": [500],
    }

    def set_objective(self, Op, y, meta):
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.Op = Op
        self.y = y
        self.proxg = TV_PROX[self.tv](self.eps)
        self.mu = 0.99 * self.tau / 8.0
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        # warm-started proximal operators keep state between calls: build
        # them at every run so that repeated runs give the same result
        proxf = L2(Op=self.Op, b=self.y, niter=20, warm=True)
        self.x = LinearizedADMM(
            proxf,
            self.proxg,
            self.Gop,
            x0=self.x0,
            tau=self.tau,
            mu=self.mu,
            niter=self.niter,
        )[0]

    def get_result(self):
        return dict(x=np.real(self.x).reshape(self.dims))
