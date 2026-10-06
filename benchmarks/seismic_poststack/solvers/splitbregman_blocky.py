from benchopt import BaseSolver
from pylops.basicoperators import FirstDerivative, SecondDerivative
from pylops.optimization.sparsity import splitbregman


class Solver(BaseSolver):
    r"""Blockiness-promoting inversion, solved with Split Bregman.

    .. math::
        \min_x \frac{\mu}{2} \|y - Op x\|_2^2 + \epsilon \|D_z x\|_1
        + \frac{\epsilon_R}{2} \|D_x^2 x\|_2^2

    where :math:`D_z` is the forward first derivative along depth (promoting
    blocky layers) and :math:`D_x^2` is the second derivative along space
    (promoting lateral continuity). Solved with
    :func:`pylops.optimization.sparsity.splitbregman` starting from the
    background model, with LSQR for the inner least-squares problems.
    """

    name = "SplitBregman-Blocky"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps": [0.05],
        "epsR": [0.4],
        "mu": [0.1],
        "niter_outer": [95],
        "niter_inner": [10],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.mback = meta["mback"]
        self.RegL1 = FirstDerivative(self.dims, axis=0, kind="forward", dtype=Op.dtype)
        self.RegL2 = SecondDerivative(self.dims, axis=1, dtype=Op.dtype)

    def run(self, _):
        self.x = splitbregman(
            self.Op,
            self.y,
            [self.RegL1],
            RegsL2=[self.RegL2],
            epsRL1s=[self.eps],
            epsRL2s=[self.epsR],
            mu=self.mu,
            niter_outer=self.niter_outer,
            niter_inner=self.niter_inner,
            x0=self.mback.ravel(),
            iter_lim=5,
            damp=1e-3,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
