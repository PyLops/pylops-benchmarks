import numpy as np
from benchopt import BaseDataset
from pylops.signalprocessing import Convolve1D


class Dataset(BaseDataset):
    """Blocky 1D signal blurred by a Gaussian kernel."""

    name = "Blocky signal"

    requirements = ["pip::pylops"]

    parameters = {
        "n": [200],
        "noise_std": [0.01],
        "seed": [0],
    }

    def get_data(self):
        # fixed and independent random streams for the model and the noise
        rng_model, rng_noise = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2)
        )

        x = np.zeros(self.n)
        edges = np.sort(rng_model.choice(np.arange(1, self.n), 8, replace=False))
        for start, end in zip(edges[::2], edges[1::2]):
            x[start:end] = rng_model.uniform(-1, 1)

        t = np.arange(-15, 16)
        h = np.exp(-(t**2) / 20.0)
        Op = Convolve1D(self.n, h=h / h.sum(), offset=15, dtype="float64")

        y = Op @ x + self.noise_std * rng_noise.standard_normal(self.n)
        return dict(x_true=x, Op=Op, y=y, meta=dict(dims=(self.n,)))

    def plot(self, x_true, Op, y, meta):
        """True model and data, shown on the benchmark page (optional)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(x_true, "k", lw=2, label="True model")
        ax.plot(y, "r", lw=1, label="Data")
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True model, estimate of a solver and error, shown on the benchmark
        page (optional)."""
        import matplotlib.pyplot as plt

        x = np.real(x).reshape(meta["dims"])
        fig, axs = plt.subplots(
            2, 1, sharex=True, figsize=(9, 4.5), height_ratios=(2, 1)
        )
        axs[0].plot(x_true, "k", lw=2, label="True model")
        axs[0].plot(x, "r", lw=1, label="Estimate")
        axs[0].legend()
        axs[1].plot(x - x_true, "k", lw=1)
        axs[1].set_title("Error")
        fig.tight_layout()
        return fig
