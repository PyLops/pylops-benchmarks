from benchopt import BaseSolver
from pylops.basicoperators import FirstDerivative
from pylops.optimization.sparsity import splitbregman


class Solver(BaseSolver):
    r"""Anisotropic total variation regularization, solved with Split Bregman.

    .. math::
        \min_x \frac{1}{2} \|y - x\|_2^2
        + \frac{\epsilon^2}{\mu} \left( \|D_z x\|_1 + \|D_x x\|_1 \right)

    where :math:`D_z` and :math:`D_x` are the first derivatives (forward
    differences) along the two axes of the image. Solved with
    :func:`pylops.optimization.sparsity.splitbregman`, with LSQR for the
    inner least-squares problems. In PyLops, ``epsRL1s`` is both the
    soft-thresholding value and the weight of the Bregman penalty term, so
    at convergence the TV weight is :math:`\epsilon^2 / \mu` (``mu`` is the
    weight of the data term): different pairs with the same ratio give the
    same solution, but converge at different rates.
    """

    name = "SplitBregman-TV"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.1],
        "mu": [0.1429],
        "niter_outer": [20],
        "niter_inner": [1],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Regs = [
            FirstDerivative(
                self.dims, axis=axis, kind="forward", edge=False, dtype=Op.dtype
            )
            for axis in (0, 1)
        ]

    def run(self, _):
        self.x = splitbregman(
            self.Op,
            self.y,
            self.Regs,
            niter_outer=self.niter_outer,
            niter_inner=self.niter_inner,
            mu=self.mu,
            epsRL1s=[self.eps] * len(self.Regs),
            tol=1e-4,
            tau=1.0,
            iter_lim=5,
            damp=1e-4,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
