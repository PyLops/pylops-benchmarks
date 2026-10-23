from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.avo.poststack import PoststackLinearModelling
from pylops.utils.wavelets import ricker
from scipy.signal import filtfilt


class Dataset(BaseDataset):
    """2D section of a Marmousi-like acoustic impedance model, plus white
    Gaussian noise.

    The model is the logarithm of the acoustic impedance of the PyLops
    ``poststack_model`` (one every three traces), with 550 depth samples and
    267 traces (4 m and 12 m spacing). Data are modelled with a 20 Hz Ricker
    wavelet of 41 samples (4 ms sampling). The background model, used by the
    solvers as starting guess, is the true model smoothed with a moving
    average of 60 samples in depth and 50 in space.
    """

    name = "Marmousi"

    requirements = ["pip::pylops"]

    parameters = {
        "noise_std": [0.1],
        "seed": [10],
    }

    def get_data(self):
        model = np.load(Path(__file__).parent / "poststack_model.npz")
        x_true = np.log(model["model"])
        x, z = model["x"] / 1000.0, model["z"] / 1000.0
        dims = x_true.shape

        # background model
        nsmoothz, nsmoothx = 60, 50
        mback = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, x_true, axis=0)
        mback = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback, axis=1)

        # the operator has no randomness: only the noise is drawn, from a
        # fixed stream
        (rng_noise,) = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(1)
        )

        wav = ricker(np.arange(21) * 0.004, 20)[0] / 2
        Op = PoststackLinearModelling(wav, nt0=dims[0], spatdims=dims[1])

        y = Op @ x_true.ravel()
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(y.shape)

        # squared spectral norm of Op (the same 1D operator is applied to every
        # trace), used by the proximal solvers to set their step sizes
        G = PoststackLinearModelling(wav, nt0=dims[0]).todense()
        LOp = float(np.linalg.eigvalsh(G.T @ G).max())

        meta = dict(
            dims=dims,
            wav=wav,
            mback=mback,
            LOp=LOp,
            x=x,
            z=z,
        )
        return dict(x_true=x_true, Op=Op, y=y, meta=meta)

    def plot(self, x_true, Op, y, meta):
        """True model, noisy data and background model."""
        import matplotlib.pyplot as plt

        x, z = meta["x"], meta["z"]
        extent = (x[0], x[-1], z[-1], z[0])
        mopts = dict(
            cmap="gist_rainbow", vmin=x_true.min(), vmax=x_true.max(), extent=extent
        )

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        axs[0].imshow(x_true, **mopts)
        axs[0].set_title("True model")
        axs[1].imshow(
            y.reshape(meta["dims"]), cmap="gray", vmin=-0.4, vmax=0.4, extent=extent
        )
        axs[1].set_title("Data")
        axs[2].imshow(meta["mback"], **mopts)
        axs[2].set_title("Background model")
        for ax in axs:
            ax.axis("tight")
            ax.set_xlabel("x [km]")
        axs[0].set_ylabel("z [km]")
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True model, inverted model and error."""
        import matplotlib.pyplot as plt

        xx, z = meta["x"], meta["z"]
        x = np.real(x).reshape(meta["dims"])
        extent = (xx[0], xx[-1], z[-1], z[0])
        vmin, vmax = x_true.min(), x_true.max()
        mopts = dict(cmap="gist_rainbow", vmin=vmin, vmax=vmax, extent=extent)
        eclip = 0.2 * (vmax - vmin)

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        axs[0].imshow(x_true, **mopts)
        axs[0].set_title("True model")
        axs[1].imshow(x, **mopts)
        axs[1].set_title("Estimate")
        axs[2].imshow(
            x - x_true, cmap="RdBu_r", vmin=-eclip, vmax=eclip, extent=extent
        )
        axs[2].set_title("Error")
        for ax in axs:
            ax.axis("tight")
            ax.set_xlabel("x [km]")
        axs[0].set_ylabel("z [km]")
        fig.tight_layout()
        return fig
