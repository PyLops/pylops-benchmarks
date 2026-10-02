import numpy as np
import pytest
from pylops.utils.metrics import mae, psnr

from pylops_bench.metrics import compute_metrics


def test_compute_metrics_matches_pylops():
    rng = np.random.default_rng(0)
    x_true = rng.standard_normal((20, 30))
    x_est = x_true + 0.1 * rng.standard_normal((20, 30))
    metrics = compute_metrics(x_true, x_est)
    assert set(metrics) == {"MAE", "PSNR"}
    assert metrics["MAE"] == pytest.approx(mae(x_true.ravel(), x_est.ravel()))
    assert metrics["PSNR"] == pytest.approx(
        psnr(x_true.ravel(), x_est.ravel(), xmax=None, xmin=None)
    )


def test_compute_metrics_psnr_uses_full_range():
    x_true = np.array([-2.0, 0.0, 1.0])
    x_est = x_true + 0.1
    # range is max - min = 3, mse = 0.01
    assert compute_metrics(x_true, x_est)["PSNR"] == pytest.approx(
        10 * np.log10(9 / 0.01)
    )


def test_compute_metrics_size_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(4), np.zeros(5))
