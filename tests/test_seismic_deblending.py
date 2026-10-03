"""Checks for the seismic deblending benchmark.

The dataset must be reproducible (fixed seeds, with the noise drawn from a
stream independent of the ignition times) and the objective must return
the benchmark metrics.
"""

import numpy as np
import pytest

BENCH = "seismic_deblending"
DATASET = f"{BENCH}/datasets/mobil_avo.py"


@pytest.fixture(scope="module")
def data():
    from conftest import instantiate

    return instantiate(DATASET, "Dataset").get_data()


def test_dataset_is_reproducible(bench_loader):
    d1 = bench_loader(DATASET, "Dataset").get_data()
    d2 = bench_loader(DATASET, "Dataset").get_data()
    np.testing.assert_array_equal(
        d1["meta"]["ignition_times"], d2["meta"]["ignition_times"]
    )
    np.testing.assert_array_equal(d1["y"], d2["y"])


def test_dataset_noise_does_not_change_ignition_times(bench_loader):
    d0 = bench_loader(DATASET, "Dataset", noise_std=0.0).get_data()
    d1 = bench_loader(DATASET, "Dataset", noise_std=0.1).get_data()
    np.testing.assert_array_equal(
        d0["meta"]["ignition_times"], d1["meta"]["ignition_times"]
    )
    assert not np.allclose(d0["y"], d1["y"])
    d2 = bench_loader(DATASET, "Dataset", noise_std=0.1).get_data()
    np.testing.assert_array_equal(d1["y"], d2["y"])


@pytest.mark.filterwarnings("ignore:divide by zero")
def test_objective(bench_loader, data):
    objective = bench_loader(f"{BENCH}/objective.py", "Objective")
    objective.set_data(**data)
    metrics = objective.evaluate_result(data["x_true"])
    assert metrics["MAE"] == 0.0
    assert np.isinf(metrics["PSNR"])
