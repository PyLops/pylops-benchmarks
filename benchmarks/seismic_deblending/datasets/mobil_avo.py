from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.waveeqprocessing.blending import BlendingContinuous


def compute_fold(timings, dt, nt):
    """Number of overlapping shots as a function of time."""
    itimings = np.round(timings / dt).astype(np.int32)
    fold = np.zeros(itimings.max() + nt)
    for itiming in itimings:
        fold[itiming : itiming + nt] += 1
    return fold


class Dataset(BaseDataset):
    """Mobil AVO common-receiver gather, continuously blended.

    The gather has 60 sources and 960 time samples (4 ms sampling). Sources
    are fired every ``overlap * nt`` samples with a random jitter of +/-1 s in
    the ignition times.
    """

    name = "Mobil AVO"

    requirements = ["pip::pylops"]

    parameters = {
        "overlap": [0.5],
        "noise_std": [0.0],
        "seed": [42],
    }

    def get_data(self):
        dt = 0.004
        x = np.load(Path(__file__).parent / "mobil.npy")[:, :960].astype("float64")
        ns, nt = x.shape

        # independent, fixed random streams for the operator and the noise so
        # that changing the noise level never changes the ignition times
        rng_op, rng_noise = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2)
        )

        # jittered ignition times, shots fired every overlap * nt samples
        jitter = 2.0 * rng_op.random(ns) - 1.0
        jitter[-1] = 0.0
        ignition_times = (
            np.arange(0, self.overlap * nt * ns, self.overlap * nt) * dt + jitter
        )
        ignition_times[0] = 0.0
        Bop = BlendingContinuous(nt, 1, ns, dt, ignition_times, dtype="complex128")

        y = Bop @ x[:, np.newaxis]
        y = y.ravel()
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(y.shape)

        fold = compute_fold(ignition_times, dt, nt)
        meta = dict(
            dims=(ns, nt),
            dt=dt,
            ignition_times=ignition_times,
            LBop=float(fold.max()),
        )
        return dict(x_true=x, Op=Bop, y=y, meta=meta)

    def plot(self, x_true, Op, y, meta):
        """True model and pseudo-deblended data (blended data mapped back)."""
        import matplotlib.pyplot as plt

        ns, nt = meta["dims"]
        ypseudo = np.real(Op.H @ y).reshape(meta["dims"])
        clip = 0.3 * np.abs(x_true).max()
        opts = dict(
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="none",
            extent=(0, ns, nt * meta["dt"], 0),
        )

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
        for ax, d, title in zip(
            axs, (x_true, ypseudo), ("True model", "Pseudo-deblended data")
        ):
            ax.imshow(d.T, **opts)
            ax.set_title(title)
            ax.set_xlabel("Source")
        axs[0].set_ylabel("t [s]")
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True model, deblended data and error, with the same clipping."""
        import matplotlib.pyplot as plt

        ns, nt = meta["dims"]
        x = np.real(x).reshape(meta["dims"])
        clip = 0.3 * np.abs(x_true).max()
        opts = dict(
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="none",
            extent=(0, ns, nt * meta["dt"], 0),
        )

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        for ax, d, title in zip(
            axs, (x_true, x, x - x_true), ("True model", "Estimate", "Error")
        ):
            ax.imshow(d.T, **opts)
            ax.set_title(title)
            ax.set_xlabel("Source")
        axs[0].set_ylabel("t [s]")
        fig.tight_layout()
        return fig
