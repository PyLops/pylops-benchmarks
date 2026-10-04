from benchopt import BaseSolver
from pylops.basicoperators import FirstDerivative, Restriction, SecondDerivative
from pylops.optimization.sparsity import splitbregman


class Solver(BaseSolver):
    r"""Blockiness-promoting inversion, solved with Split Bregman.

    .. math::
        \min_x \frac{\mu}{2} \|y - Op x\|_2^2
        + \sum_{p} \epsilon_p \|D_z x_p\|_1
        + \sum_{p} \frac{\epsilon_{R,p}}{2} \|D_x^2 x_p\|_2^2

    where :math:`D_z` is the centered first derivative along depth (promoting
    blocky layers) and :math:`D_x^2` is the second derivative along space
    (promoting lateral continuity) of the parameter
    :math:`p \in \{V_P, V_S, \rho\}`, each with its own weights. Solved with
    :func:`pylops.optimization.sparsity.splitbregman` starting from the
    background model, with LSQR for the inner least-squares problems.
    """

    name = "SplitBregman-Blocky"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {
        "eps_vp": [0.035],
        "eps_vs": [0.025],
        "eps_rho": [0.05],
        "epsR_vp": [0.0175],
        "epsR_vs": [0.05],
        "epsR_rho": [0.1],
        "mu": [0.07],
        "niter_outer": [105],
        "niter_inner": [3],
    }

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]
        self.mback = meta["mback"]
        Dz = FirstDerivative(self.dims, axis=0, dtype=Op.dtype)
        Dxx = SecondDerivative(self.dims, axis=2, dtype=Op.dtype)
        # one regularization term per parameter
        Rs = [
            Restriction(self.dims, [ip], axis=1, dtype=Op.dtype)
            for ip in range(self.dims[1])
        ]
        self.RegsL1 = [R @ Dz for R in Rs]
        self.RegsL2 = [R @ Dxx for R in Rs]
        self.epsRL1s = [self.eps_vp, self.eps_vs, self.eps_rho]
        self.epsRL2s = [self.epsR_vp, self.epsR_vs, self.epsR_rho]

    def run(self, _):
        self.x = splitbregman(
            self.Op,
            self.y,
            self.RegsL1,
            RegsL2=self.RegsL2,
            epsRL1s=self.epsRL1s,
            epsRL2s=self.epsRL2s,
            mu=self.mu,
            niter_outer=self.niter_outer,
            niter_inner=self.niter_inner,
            x0=self.mback.ravel(),
            iter_lim=5,
            damp=1e-3,
        )[0]

    def get_result(self):
        return dict(x=self.x.reshape(self.dims))
