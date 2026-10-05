from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.basicoperators import Identity


class Dataset(BaseDataset):
    """Ascent image from `scipy.datasets` (512 x 512 pixels, values scaled
    to 0-1), plus uniform noise between -`noise_level` and `noise_level`
    times the maximum of the image (by default ±20 %).
    """

    name = "Ascent"

    requirements = ["pip::pylops"]

    parameters = {
        "noise_level": [0.2],
        "seed": [0],
    }

    def get_data(self):
        x = np.load(Path(__file__).parent / "ascent.npy").astype("float64") / 255.0
        dims = x.shape

        # the operator has no randomness: only the noise is drawn, from a
        # fixed stream
        (rng_noise,) = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(1)
        )

        Op = Identity(x.size, dtype="float64")

        y = Op @ x.ravel()
        if self.noise_level > 0:
            y = y + self.noise_level * np.abs(x).max() * rng_noise.uniform(
                -1, 1, y.shape
            )
        return dict(x_true=x, Op=Op, y=y, meta=dict(dims=dims))

    def plot(self, x_true, Op, y, meta):
        """True image and noisy data."""
        import matplotlib.pyplot as plt

        opts = dict(cmap="gray", vmin=0, vmax=1)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(x_true, **opts)
        axs[0].set_title("True model")
        axs[1].imshow(y.reshape(meta["dims"]), **opts)
        axs[1].set_title("Data")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        return fig
