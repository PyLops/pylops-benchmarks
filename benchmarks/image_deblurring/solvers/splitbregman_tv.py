from benchopt import BaseSolver
from pylops.basicoperators import FirstDerivative
from pylops.optimization.sparsity import splitbregman


class Solver(BaseSolver):
    r"""Anisotropic total variation regularization, solved with Split Bregman.

    .. math::
        \min_x \frac{\mu}{2} \|y - Op x\|_2^2
        + \epsilon \left( \|D_z x\|_1 + \|D_x x\|_1 \right)

    where :math:`D_z` and :math:`D_x` are the first derivatives along the
    two axes of the image. Solved with
    :func:`pylops.optimization.sparsity.splitbregman`, with LSQR for the
    inner least-squares problems.
    """

    name = "SplitBregman-TV"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [2.0],
        "mu": [1.5],
        "niter_outer": [10],
        "niter_inner": [5],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.Regs = [
            FirstDerivative(self.dims, axis=axis, edge=False, dtype=Op.dtype)
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
