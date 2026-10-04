"""Checks for the pre-stack inversion benchmark.

The dataset must be reproducible (fixed seed for the noise, the only random
component) and the objective must return the benchmark metrics.
"""

import numpy as np
import pytest

BENCH = "prestack"
DATASET = f"{BENCH}/datasets/marmousi.py"

pytestmark = pytest.mark.filterwarnings("ignore:A new implementation of convmtx")


@pytest.fixture(scope="module")
def data():
    from conftest import instantiate

    return instantiate(DATASET, "Dataset").get_data()


def test_dataset_shapes(data):
    nz, npar, nx = data["meta"]["dims"]
    assert data["x_true"].shape == (nz, npar, nx)
    assert npar == 3
    assert data["meta"]["mback"].shape == data["x_true"].shape
    assert data["y"].size == nz * len(data["meta"]["theta"]) * nx


def test_dataset_is_reproducible(bench_loader, data):
    d = bench_loader(DATASET, "Dataset").get_data()
    np.testing.assert_array_equal(d["y"], data["y"])


def test_dataset_noise(bench_loader, data):
    d0 = bench_loader(DATASET, "Dataset", noise_std=0.0).get_data()
    np.testing.assert_allclose(d0["y"], data["Op"] @ data["x_true"].ravel())
    assert not np.allclose(d0["y"], data["y"])


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
