from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.medical import MRI2D


def radial_mask(dims, nlines, rng):
    """Binary k-space mask made of ``nlines`` radial lines through the center
    of the (centered) k-space, with angles drawn uniformly in :math:`[0, \\pi)`
    from ``rng``. Each line is rasterized to the nearest grid points."""
    ny, nx = dims
    mask = np.zeros(dims)
    radius = np.arange(-max(dims), max(dims), 0.5)
    for theta in rng.uniform(0, np.pi, nlines):
        iy = np.round(ny // 2 + radius * np.sin(theta)).astype(int)
        ix = np.round(nx // 2 + radius * np.cos(theta)).astype(int)
        inside = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        mask[iy[inside], ix[inside]] = 1.0
    return mask


class Dataset(BaseDataset):
    """Shepp-Logan phantom (195 x 170 pixels, values scaled to 0-1),
    sampled in k-space along `nlines` radial lines with random angles (30
    lines by default, about 18 % of k-space), plus complex white Gaussian
    noise on the sampled coefficients.
    """

    name = "Shepp-Logan"

    requirements = ["pip::pylops"]

    parameters = {
        "nlines": [30],
        "noise_std": [0.01],
        "seed": [0],
    }

    def get_data(self):
        x = np.load(Path(__file__).parent / "shepp_logan_phantom.npy").astype("float64")
        x = x / x.max()
        dims = x.shape

        # independent, fixed random streams for the operator and the noise so
        # that changing the noise level never changes the k-space mask (the
        # random masks of MRI2D are not seeded, so the mask is built here)
        rng_op, rng_noise = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2)
        )
        mask = radial_mask(dims, self.nlines, rng_op)
        Op = MRI2D(dims, mask=mask, dtype="complex128")

        y = Op @ x.ravel()
        if self.noise_std > 0:
            noise = rng_noise.standard_normal((2, y.size))
            y = y + self.noise_std * mask.ravel() * (
                noise[0] + 1j * noise[1]
            ) / np.sqrt(2)
        return dict(x_true=x, Op=Op, y=y, meta=dict(dims=dims, mask=mask))

    def plot(self, x_true, Op, y, meta):
        """True image, k-space mask and zero-filled reconstruction."""
        import matplotlib.pyplot as plt

        xzf = np.real(Op.H @ y).reshape(meta["dims"])
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.8))
        axs[0].imshow(x_true, cmap="gray", vmin=0, vmax=1)
        axs[0].set_title("True model")
        axs[1].imshow(meta["mask"], cmap="gray_r")
        axs[1].set_title("k-space mask")
        axs[2].imshow(xzf, cmap="gray", vmin=0, vmax=1)
        axs[2].set_title("Data (zero-filled adjoint)")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        return fig
