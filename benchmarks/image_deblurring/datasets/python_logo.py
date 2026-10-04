from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.signalprocessing import Convolve2D


def gaussian_psf(nh=(15, 25), alpha=(0.1, 0.03)):
    """Separable Gaussian point-spread function, each axis normalized to
    unit integral."""
    hz, hx = (
        np.exp(-a * np.linspace(-(n // 2), n // 2, n) ** 2) for n, a in zip(nh, alpha)
    )
    hz /= np.trapezoid(hz)
    hx /= np.trapezoid(hx)
    return hz[:, np.newaxis] * hx[np.newaxis, :]


class Dataset(BaseDataset):
    """Python logo (126 x 126 pixels, values in 0-255), blurred by a
    separable Gaussian point-spread function of 15 x 25 samples, plus white
    Gaussian noise.
    """

    name = "Python logo"

    requirements = ["pip::pylops"]

    parameters = {
        "noise_std": [5.0],
        "seed": [0],
    }

    def get_data(self):
        x = np.load(Path(__file__).parent / "python.npy").astype("float64")
        dims = x.shape

        # the operator has no randomness: only the noise is drawn, from a
        # fixed stream
        (rng_noise,) = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(1)
        )

        h = gaussian_psf()
        Op = Convolve2D(
            dims, h=h, offset=(h.shape[0] // 2, h.shape[1] // 2), dtype="float64"
        )

        y = Op @ x.ravel()
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(y.shape)
        return dict(x_true=x, Op=Op, y=y, meta=dict(dims=dims, h=h))

    def plot(self, x_true, Op, y, meta):
        """True image, blurred data and point-spread function."""
        import matplotlib.pyplot as plt

        opts = dict(cmap="viridis", vmin=0, vmax=250)
        fig, axs = plt.subplots(
            1, 3, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=(1, 1, 0.6))
        )
        axs[0].imshow(x_true, **opts)
        axs[0].set_title("True model")
        axs[1].imshow(y.reshape(meta["dims"]), **opts)
        axs[1].set_title("Data")
        axs[2].imshow(meta["h"], cmap="viridis")
        axs[2].set_title("Point-spread function")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        return fig
