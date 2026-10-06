import numpy as np
from benchopt import BaseSolver
from pylops.basicoperators import Gradient
from pyproximal import L2, RelaxedMumfordShah
from pyproximal.optimization.primal import LinearizedADMM


class Solver(BaseSolver):
    r"""Relaxed Mumford-Shah regularization on the gradient, solved with the
    linearized ADMM.

    .. math::
        \min_x \frac{1}{2} \|y - Op x\|_2^2
        + \sum_i \min\left(\sigma (\nabla x)_i^2, \kappa\right)

    where :math:`\nabla` is the 2D gradient (forward differences). Small
    gradients are penalized quadratically (smoothing) and large ones by the
    constant :math:`\kappa` (edges are not penalized further), see
    :class:`pyproximal.RelaxedMumfordShah`. Solved with
    :func:`pyproximal.optimization.primal.LinearizedADMM`, with
    :math:`\mu = 0.99 \tau / \|\nabla\|_2^2` and
    :math:`\|\nabla\|_2^2 \le 8`; the proximal operator of the data term is
    evaluated with 10 warm-started inner iterations. Convergence is not
    guaranteed, as the penalty is nonconvex.
    """

    name = "LinearizedADMM-rMS"

    requirements = ["pip::pylops", "pip::pyproximal"]
    sampling_strategy = "run_once"

    parameters = {
        "sigma": [0.1],
        "kappa": [10.0],
        "tau": [50.0],
        "niter": [300],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Gop = Gradient(self.dims, edge=True, kind="forward", dtype=Op.dtype)
        self.proxg = RelaxedMumfordShah(sigma=self.sigma, kappa=self.kappa)
        self.mu = 0.99 * self.tau / 8.0
        self.x0 = np.zeros(Op.shape[1], dtype=Op.dtype)

    def run(self, _):
        # the warm-started proximal operator keeps state between calls: build
        # it at every run so that repeated runs give the same result
        proxf = L2(Op=self.Op, b=self.y, niter=10, warm=True)
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
        return dict(x=self.x.reshape(self.dims))
