"""Checks for the image denoising benchmark.

The dataset must be reproducible (fixed seed for the noise) and the
objective must return the benchmark metrics.
"""

import numpy as np
import pytest

BENCH = "image_denoising"
DATASET = f"{BENCH}/datasets/ascent.py"


@pytest.fixture(scope="module")
def data():
    from conftest import instantiate

    return instantiate(DATASET, "Dataset").get_data()


def test_dataset_is_reproducible(bench_loader):
    d1 = bench_loader(DATASET, "Dataset", noise_level=0.2).get_data()
    d2 = bench_loader(DATASET, "Dataset", noise_level=0.2).get_data()
    np.testing.assert_array_equal(d1["y"], d2["y"])


def test_dataset_noise(bench_loader):
    d0 = bench_loader(DATASET, "Dataset", noise_level=0.0).get_data()
    d1 = bench_loader(DATASET, "Dataset", noise_level=0.2).get_data()
    np.testing.assert_array_equal(d0["x_true"], d1["x_true"])
    np.testing.assert_allclose(d0["y"], d0["Op"] @ d0["x_true"].ravel())
    assert not np.allclose(d0["y"], d1["y"])


def test_dataset_shapes(data):
    assert data["x_true"].shape == data["meta"]["dims"]
    assert data["Op"].shape == (data["y"].size, data["x_true"].size)


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
