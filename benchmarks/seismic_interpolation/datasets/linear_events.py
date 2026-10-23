import numpy as np
from benchopt import BaseDataset
from pylops.basicoperators import Restriction
from pylops.utils.seismicevents import linear2d, makeaxis
from pylops.utils.wavelets import ricker


class Dataset(BaseDataset):
    """2D gather with three linear events, randomly subsampled in space.

    The gather has 140 traces (2 m spacing) and 200 time samples (4 ms
    sampling). It contains three linear events with intercept times 0.2, 0.4
    and 0.5 s, angles 0, 70 and -80 degrees (velocity 800 m/s) and amplitudes
    1, -2 and 0.5, all with a 20 Hz Ricker wavelet.
    """

    name = "Linear events"

    requirements = ["pip::pylops"]

    parameters = {
        "perc_subsampling": [0.4],
        "noise_std": [0.0],
        "seed": [42],
    }

    def get_data(self):
        par = {
            "ox": 0, "dx": 2, "nx": 140,
            "oy": 0, "dy": 2, "ny": 101,
            "ot": 0, "dt": 0.004, "nt": 200,
            "f0": 20,
        }  # fmt: skip
        v = 800
        t0_m = [0.2, 0.4, 0.5]
        theta_m = [0, 70, -80]
        amp_m = [1.0, -2, 0.5]

        taxis, _, xaxis, _ = makeaxis(par)
        wav = ricker(taxis[:41], f0=par["f0"])[0]
        _, x = linear2d(xaxis, taxis, v, t0_m, theta_m, amp_m, wav)
        dims = (par["nx"], par["nt"])

        # independent, fixed random streams for the operator and the noise so
        # that changing the noise level never changes the subsampling mask
        rng_op, rng_noise = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(2)
        )

        nsub = int(np.round(par["nx"] * self.perc_subsampling))
        iava = np.sort(rng_op.permutation(par["nx"])[:nsub])
        Rop = Restriction(dims=dims, iava=iava, axis=0, dtype="float64")

        y = Rop @ x.ravel()
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(y.shape)

        meta = dict(
            dims=dims,
            taxis=taxis,
            xaxis=xaxis,
            dx=par["dx"],
            dt=par["dt"],
            iava=iava,
        )
        return dict(x_true=x, Op=Rop, y=y, meta=meta)

    def plot(self, x_true, Op, y, meta):
        """True model and data (missing traces shown as zeros)."""
        import matplotlib.pyplot as plt

        xaxis, taxis = meta["xaxis"], meta["taxis"]
        yfull = (Op.H @ y).reshape(meta["dims"])
        clip = np.abs(x_true).max()
        opts = dict(
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="none",
            extent=(xaxis[0], xaxis[-1], taxis[-1], taxis[0]),
        )

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
        for ax, d, title in zip(axs, (x_true, yfull), ("True model", "Data")):
            ax.imshow(d.T, **opts)
            ax.set_title(title)
            ax.set_xlabel("x [m]")
        axs[0].set_ylabel("t [s]")
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True model, reconstruction and error, with the same clipping."""
        import matplotlib.pyplot as plt

        xaxis, taxis = meta["xaxis"], meta["taxis"]
        x = np.real(x).reshape(meta["dims"])
        clip = np.abs(x_true).max()
        opts = dict(
            cmap="gray",
            vmin=-clip,
            vmax=clip,
            aspect="auto",
            interpolation="none",
            extent=(xaxis[0], xaxis[-1], taxis[-1], taxis[0]),
        )

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
        for ax, d, title in zip(
            axs, (x_true, x, x - x_true), ("True model", "Estimate", "Error")
        ):
            ax.imshow(d.T, **opts)
            ax.set_title(title)
            ax.set_xlabel("x [m]")
        axs[0].set_ylabel("t [s]")
        fig.tight_layout()
        return fig
