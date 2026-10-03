"""Checks for the seismic interpolation benchmark.

The dataset must be reproducible (fixed seeds, with the noise drawn from a
stream independent of the subsampling mask) and the objective must return
the benchmark metrics.
"""

import numpy as np
import pytest

BENCH = "seismic_interpolation"
DATASET = f"{BENCH}/datasets/linear_events.py"


@pytest.fixture(scope="module")
def data():
    from conftest import instantiate

    return instantiate(DATASET, "Dataset").get_data()


def test_dataset_is_reproducible(bench_loader):
    d1 = bench_loader(DATASET, "Dataset").get_data()
    d2 = bench_loader(DATASET, "Dataset").get_data()
    np.testing.assert_array_equal(d1["meta"]["iava"], d2["meta"]["iava"])
    np.testing.assert_array_equal(d1["y"], d2["y"])


def test_dataset_noise_does_not_change_mask(bench_loader):
    d0 = bench_loader(DATASET, "Dataset", noise_std=0.0).get_data()
    d1 = bench_loader(DATASET, "Dataset", noise_std=0.1).get_data()
    np.testing.assert_array_equal(d0["meta"]["iava"], d1["meta"]["iava"])
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


def test_dataset_plot(bench_loader, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    fig = bench_loader(DATASET, "Dataset").plot(**data)
    assert isinstance(fig, Figure)
    plt.close(fig)
