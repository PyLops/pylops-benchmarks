"""Checks for the compressive sensing benchmark.

The dataset must be reproducible (fixed seeds, with independent streams for
the sensing matrix, the model and the noise) and the objective must return
the benchmark metrics.
"""

import numpy as np
import pytest

BENCH = "compressive_sensing"
DATASET = f"{BENCH}/datasets/sparse_spikes.py"


@pytest.fixture(scope="module")
def data():
    from conftest import instantiate

    return instantiate(DATASET, "Dataset").get_data()


def test_dataset_is_reproducible(bench_loader):
    d1 = bench_loader(DATASET, "Dataset").get_data()
    d2 = bench_loader(DATASET, "Dataset").get_data()
    np.testing.assert_array_equal(d1["Op"].A, d2["Op"].A)
    np.testing.assert_array_equal(d1["x_true"], d2["x_true"])
    np.testing.assert_array_equal(d1["y"], d2["y"])


def test_dataset_noise_does_not_change_operator_or_model(bench_loader):
    d0 = bench_loader(DATASET, "Dataset", noise_std=0.0).get_data()
    d1 = bench_loader(DATASET, "Dataset", noise_std=0.01).get_data()
    np.testing.assert_array_equal(d0["Op"].A, d1["Op"].A)
    np.testing.assert_array_equal(d0["x_true"], d1["x_true"])
    np.testing.assert_allclose(d0["y"], d0["Op"] @ d0["x_true"])
    assert not np.allclose(d0["y"], d1["y"])


def test_dataset_sparsity(data):
    x = data["x_true"]
    assert np.count_nonzero(x) == data["meta"]["k"]
    assert np.all(np.abs(x[x != 0]) >= 1.0)
    np.testing.assert_allclose(np.linalg.norm(data["Op"].A, axis=0), 1.0)


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


def test_dataset_plot_estimate(bench_loader, data):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # estimates are stored in single precision
    x = data["x_true"].astype("float32")
    fig = bench_loader(DATASET, "Dataset").plot_estimate(
        x_true=data["x_true"], x=x, meta=data["meta"]
    )
    assert isinstance(fig, Figure)
    plt.close(fig)
