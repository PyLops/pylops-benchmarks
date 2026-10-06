from benchopt import BaseSolver
from pylops.optimization.sparsity import irls


class Solver(BaseSolver):
    r"""Basis pursuit, solved with Iteratively Reweighted Least Squares.

    .. math::
        \min_x \|x\|_1 \quad \text{s.t.} \quad Op x = y

    Solved with :func:`pylops.optimization.sparsity.irls` with
    ``kind="model"``: each outer iteration solves a weighted minimum-norm
    problem, with weights equal to the absolute values of the previous
    estimate, regularized by :math:`\epsilon_I`.
    """

    name = "IRLS"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "epsI": [1e-2],
        "nouter": [50],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]

    def run(self, _):
        self.x = irls(
            self.Op, self.y, nouter=self.nouter, epsI=self.epsI, kind="model"
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
