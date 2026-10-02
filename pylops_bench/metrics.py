"""Metrics used by every benchmark.

All benchmarks compare the true model with the estimated one using the
PyLops implementations of the Mean Absolute Error (MAE) and the Peak
Signal-to-Noise Ratio (PSNR).
"""

import numpy as np
from pylops.utils.metrics import mae, psnr


def compute_metrics(x_true: np.ndarray, x_est: np.ndarray) -> dict[str, float]:
    """Compute the benchmark metrics between true and estimated models.

    Parameters
    ----------
    x_true : :obj:`numpy.ndarray`
        True model
    x_est : :obj:`numpy.ndarray`
        Estimated model (must have the same number of elements as
        ``x_true``)

    Returns
    -------
    metrics : :obj:`dict`
        Dictionary with keys ``MAE`` and ``PSNR``. The PSNR uses the
        dynamic range of the true model (``xmax=None``, ``xmin=None``), so
        that signed models (e.g., seismic data) are handled correctly.

    """
    x_true = np.asarray(x_true).ravel()
    x_est = np.asarray(x_est).ravel()
    if x_true.size != x_est.size:
        msg = f"x_true and x_est have different sizes ({x_true.size} vs {x_est.size})"
        raise ValueError(msg)
    if np.iscomplexobj(x_est) and not np.iscomplexobj(x_true):
        x_est = np.real(x_est)
    return dict(
        MAE=float(mae(x_true, x_est)),
        PSNR=float(psnr(x_true, x_est, xmax=None, xmin=None)),
    )
