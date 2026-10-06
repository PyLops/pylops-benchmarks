import numpy as np

from pylops_bench.denoisers import bm3d_denoise


def _noisy_image(scale):
    rng = np.random.default_rng(0)
    x = np.zeros((64, 64))
    x[16:48, 16:48] = 1.0
    xn = x + 0.1 * rng.standard_normal(x.shape)
    return scale * x, scale * xn


def test_bm3d_denoise_reduces_error():
    x, xn = _noisy_image(1.0)
    xden = bm3d_denoise(xn, 0.1)
    assert xden.shape == x.shape
    assert np.isrealobj(xden)
    assert np.linalg.norm(xden - x) < 0.5 * np.linalg.norm(xn - x)


def test_bm3d_denoise_scale_invariance():
    x, xn = _noisy_image(1.0)
    _, xn100 = _noisy_image(100.0)
    np.testing.assert_allclose(
        bm3d_denoise(xn100, 0.1, scale=100.0),
        100.0 * bm3d_denoise(xn, 0.1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_bm3d_denoise_complex_input():
    x, xn = _noisy_image(1.0)
    xden = bm3d_denoise(xn + 1j * xn, 0.1)
    # BM3D works in single precision
    np.testing.assert_allclose(xden, bm3d_denoise(xn, 0.1), atol=1e-5)
