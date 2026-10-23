import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Template objective: copy this folder to start a new benchmark.

    Every benchmark follows the same contract:

    - datasets return ``dict(x_true=..., Op=..., y=..., meta=...)``, where
      ``meta`` holds whatever solvers need to build their own operators
      (axes, sampling, ...) and always includes ``dims``;
    - solvers receive ``Op``, ``y`` and ``meta`` and return ``dict(x=...)``;
    - the metrics are always PyLops MAE and PSNR (``compute_metrics``).
    """

    name = "Reconstruction"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops"]
    min_benchopt_version = "1.10"

    sampling_strategy = "run_once"

    # name of the dataset used by `benchopt test`
    test_config = {"dataset": {"name": "Blocky signal"}}

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
