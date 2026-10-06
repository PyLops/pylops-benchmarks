import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Recovery of a sparse vector from a small number of random measurements.

    The datasets return ``dict(x_true=..., Op=..., y=..., meta=...)``, where
    ``x_true`` is the sparse vector, ``Op`` the random sensing matrix and
    ``y`` the measurements. The metrics are the PyLops MAE and PSNR between
    the true and the recovered vector.
    """

    name = "Compressive sensing"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops", "pip::pyproximal"]
    min_benchopt_version = "1.10"

    sampling_strategy = "run_once"

    # name of the dataset used by `benchopt test`
    test_config = {"dataset": {"name": "Sparse spikes"}}

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
