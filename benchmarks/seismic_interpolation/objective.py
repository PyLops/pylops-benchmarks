import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Reconstruction of a fully sampled seismic gather from a subset of traces.

    The true model ``x_true`` is the fully sampled gather, the operator ``Op``
    is a restriction operator that selects the available traces and ``y`` are
    the recorded (and possibly noisy) traces. Solvers receive ``Op``, ``y``
    and some problem metadata (``meta``) and must return an estimate of
    ``x_true`` with shape ``meta["dims"]``.
    """

    name = "Reconstruction"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops", "pip::numba"]
    min_benchopt_version = "1.10"

    # every solver runs once with a fixed number of iterations
    sampling_strategy = "run_once"

    # `benchopt test` looks for a dataset called `Simulated` by default
    test_config = {"dataset": {"name": "Linear events"}}

    def set_data(self, x_true, Op, y, meta):
        self.x_true = x_true
        self.Op = Op
        self.y = y
        self.meta = meta

    def evaluate_result(self, x):
        return compute_metrics(self.x_true, x)

    def save_final_results(self, x):
        # stored in the parquet output, then plotted on the benchmark page
        return x

    def get_one_result(self):
        return dict(x=np.zeros(self.meta["dims"]))

    def get_objective(self):
        return dict(Op=self.Op, y=self.y, meta=self.meta)
