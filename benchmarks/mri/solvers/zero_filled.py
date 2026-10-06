import numpy as np
from benchopt import BaseSolver


class Solver(BaseSolver):
    r"""Zero-filled reconstruction (adjoint of the MRI operator).

    .. math::
        x = \Re\{Op^H y\}

    Since the 2D Fourier transform is orthonormal and the mask a projection,
    this is also the minimum-norm least-squares solution (e.g., what CGLS or
    LSQR converge to).
    """

    name = "ZeroFilled"

    requirements = ["pip::pylops"]
    sampling_strategy = "run_once"

    parameters = {}

    def set_objective(self, Op, y, meta):
        self.Op = Op
        self.y = y
        self.dims = meta["dims"]

    def run(self, _):
        self.x = self.Op.H @ self.y

    def get_result(self):
        return dict(x=np.real(self.x).reshape(self.dims))
