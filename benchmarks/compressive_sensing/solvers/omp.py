from benchopt import BaseSolver
from pylops.optimization.sparsity import omp


class Solver(BaseSolver):
    r"""Greedy L0 recovery, solved with Orthogonal Matching Pursuit.

    .. math::
        \min_x \|y - Op x\|_2 \quad \text{s.t.} \quad \|x\|_0 \le k

    Solved with :func:`pylops.optimization.sparsity.omp`, which adds one
    column of :math:`Op` per outer iteration and refits the selected ones
    with ``niter_inner`` LSQR iterations. The number of outer iterations is
    the number of nonzero samples of the true model (``meta["k"]``).
    """

    name = "OMP"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "niter_inner": [50],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.k = meta["k"]

    def run(self, _):
        self.x = omp(
            self.Op,
            self.y,
            niter_outer=self.k,
            niter_inner=self.niter_inner,
            sigma=1e-10,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
