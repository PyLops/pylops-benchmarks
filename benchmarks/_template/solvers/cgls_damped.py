from benchopt import BaseSolver
from pylops.optimization.basic import cgls


class Solver(BaseSolver):
    r"""Damped least squares solved with CGLS.

    .. math::
        \min_x \|y - Op x\|_2^2 + \text{damp}^2 \|x\|_2^2
    """

    name = "CGLS"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "damp": [1e-2],
        "niter": [50],
    }

    def set_objective(self, Op, y, meta):
        # build operators here: this part is not timed
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]

    def run(self, _):
        # only this part is timed
        self.x = cgls(self.Op, self.y, damp=self.damp, niter=self.niter)[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
