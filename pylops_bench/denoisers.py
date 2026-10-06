"""Denoisers shared by the Plug-and-Play and RED solvers of the benchmarks."""

import numpy as np


def bm3d_denoise(x: np.ndarray, sigma: float, scale: float = 1.0) -> np.ndarray:
    """Denoise an image with BM3D (hard-thresholding stage only).

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Image to denoise (2-dimensional). Only its real part is denoised.
    sigma : :obj:`float`
        Standard deviation of the noise, relative to ``scale``.
    scale : :obj:`float`, optional
        Scale of the image values: the image is divided by ``scale`` before
        denoising and multiplied back after, so that ``sigma`` does not
        depend on the amplitude of the data (use ``1.0`` for images in 0-1).

    Returns
    -------
    xden : :obj:`numpy.ndarray`
        Denoised image (real, same shape as ``x``)
    """
    import bm3d

    xden = bm3d.bm3d(
        np.real(x) / scale,
        sigma_psd=sigma,
        stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING,
    )
    return scale * xden
