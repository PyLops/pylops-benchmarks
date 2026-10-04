import numpy as np
from benchopt import BaseObjective

from pylops_bench.metrics import compute_metrics


class Objective(BaseObjective):
    """Deblurring of an image blurred by a known point-spread function.

    The datasets return ``dict(x_true=..., Op=..., y=..., meta=...)``, where
    ``x_true`` is the sharp image, ``Op`` the 2D convolution operator and
    ``y`` the blurred data. The metrics are the PyLops MAE and PSNR between
    the true and the deblurred image.
    """

    name = "Deblurring"
    url = "https://github.com/PyLops/pylops-benchmarks"

    requirements = ["pip::pylops", "pip::pyproximal", "pip::PyWavelets"]
    min_benchopt_version = "1.10"

    sampling_strategy = "run_once"

    # name of the dataset used by `benchopt test`
    test_config = {"dataset": {"name": "Python logo"}}

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
