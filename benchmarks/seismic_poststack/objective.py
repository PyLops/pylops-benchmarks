import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Post-stack inversion of seismic data for acoustic impedance.

    The true model ``x_true`` is the logarithm of the acoustic impedance, the
    operator ``Op`` is the post-stack linear modelling operator (derivative
    followed by convolution with the wavelet) and ``y`` is the (noisy)
    post-stack seismic data. Solvers receive ``Op``, ``y`` and some problem
    metadata (``meta``, including the wavelet, a background model and the
    squared norm of ``Op``) and
    must return an estimate of ``x_true`` with shape ``meta["dims"]``.
    """

    name = "Reconstruction"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops", "pip::pyproximal"]
    min_benchopt_version = "1.10"

    # every solver runs once with a fixed number of iterations
    sampling_strategy = "run_once"

    # name of the dataset used by `benchopt test`
    test_config = {"dataset": {"name": "Marmousi"}}

    def set_data(self, x_true, Op, y, meta):
        self.x_true = x_true
        self.Op = Op
        self.y = y
        self.meta = meta

    def evaluate_result(self, x):
        return compute_metrics(self.x_true, x)

    def get_one_result(self):
        return dict(x=np.zeros(self.meta["dims"]))

    def get_objective(self):
        return dict(Op=self.Op, y=self.y, meta=self.meta)
