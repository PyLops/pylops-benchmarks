from benchopt import BaseSolver
from pylops.optimization.basic import lsqr


class Solver(BaseSolver):
    r"""Damped least squares solved with LSQR (no sparsity).

    .. math::
        \min_x \|y - Op x\|_2^2 + \text{damp}^2 \|x\|_2^2

    For an underdetermined system and a small damping, this is close to the
    minimum-norm solution: a reference for the sparsity-promoting solvers.
    """

    name = "LSQR"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "damp": [1e-2],
        "niter": [100],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]

    def run(self, _):
        self.x = lsqr(self.Op, self.y, damp=self.damp, niter=self.niter)[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
