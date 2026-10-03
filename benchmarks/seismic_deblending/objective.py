import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Deblending of continuously blended seismic data.

    The datasets return ``dict(x_true=..., Op=..., y=..., meta=...)``, where
    ``x_true`` is the unblended gather, ``Op`` the blending operator and ``y``
    the blended data. The metrics are the PyLops MAE and PSNR between the
    true and the deblended gather.
    """

    name = "Deblending"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::numba"]
    min_benchopt_version = "1.10"

    sampling_strategy = "run_once"

    # name of the dataset used by `benchopt test`
    test_config = {"dataset": {"name": "Mobil AVO"}}

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
