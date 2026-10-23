from pathlib import Path

import numpy as np
from benchopt import BaseDataset
from pylops.avo.prestack import PrestackLinearModelling
from pylops.utils.wavelets import ricker
from scipy.signal import filtfilt


class Dataset(BaseDataset):
    """2D section of a Marmousi-like elastic model, plus white Gaussian noise.

    The P-wave velocity is the acoustic impedance of the PyLops
    ``poststack_model`` (first 300 depth samples, one every six traces)
    taken as velocity in m/s,
    the S-wave velocity is half of it and the density is
    $\\rho = V_P / 3 + 300$. The model is the logarithm of the three
    parameters, with 300 depth samples, 3 parameters and 134 traces (4 m and
    24 m spacing). Data are modelled with the Aki-Richards linearization
    ($V_S/V_P = 0.5$), 21 angles between 0° and 40° and a 15 Hz Ricker
    wavelet of 41 samples (4 ms sampling). The background model, used by the
    solvers as starting guess, is the true model smoothed with a moving
    average of 30 samples in depth and 25 in space.
    """

    name = "Marmousi"

    requirements = ["pip::pylops"]

    parameters = {
        "noise_std": [0.05],
        "seed": [10],
    }

    def get_data(self):
        # the stored model has one every three traces of the original one
        model = np.load(Path(__file__).parent / "poststack_model.npz")
        vp = 1000 * model["model"][:300, ::2]
        vs = vp / 2
        rho = vp / 3 + 300
        x_true = np.log(np.stack((vp, vs, rho), axis=1))
        x, z = model["x"][::2] / 1000.0, model["z"][:300] / 1000.0
        dims = x_true.shape
        nz, _, nx = dims

        # background model
        nsmoothz, nsmoothx = 30, 25
        mback = filtfilt(np.ones(nsmoothz) / float(nsmoothz), 1, x_true, axis=0)
        mback = filtfilt(np.ones(nsmoothx) / float(nsmoothx), 1, mback, axis=2)

        # the operator has no randomness: only the noise is drawn, from a
        # fixed stream
        (rng_noise,) = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(1)
        )

        theta = np.linspace(0, 40, 21)
        vsvp = 0.5
        wav = ricker(np.arange(21) * 0.004, 15)[0]
        Op = PrestackLinearModelling(
            wav, theta, vsvp=vsvp, nt0=nz, spatdims=nx, linearization="akirich"
        )

        y = Op @ x_true.ravel()
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(y.shape)

        # squared spectral norm of Op (the same 1D operator is applied to every
        # trace), used by the proximal solvers to set their step sizes
        G = PrestackLinearModelling(
            wav, theta, vsvp=vsvp, nt0=nz, linearization="akirich", explicit=True
        ).A
        LOp = float(np.linalg.eigvalsh(G.T @ G).max())

        meta = dict(
            dims=dims,
            wav=wav,
            theta=theta,
            vsvp=vsvp,
            mback=mback,
            LOp=LOp,
            x=x,
            z=z,
        )
        return dict(x_true=x_true, Op=Op, y=y, meta=meta)

    def plot(self, x_true, Op, y, meta):
        """True models, data at zero angle and angle gather in the middle."""
        import matplotlib.pyplot as plt

        x, z, theta = meta["x"], meta["z"], meta["theta"]
        nz, _, nx = meta["dims"]
        d = y.reshape(nz, len(theta), nx)
        ix = nx // 2
        extent = (x[0], x[-1], z[-1], z[0])
        dopts = dict(cmap="gray", vmin=-0.4, vmax=0.4)

        fig = plt.figure(figsize=(12, 8))
        axs = [plt.subplot2grid((2, 6), (0, 2 * i), colspan=2) for i in range(3)]
        for ip, (ax, param) in enumerate(zip(axs, (r"$V_P$", r"$V_S$", r"$\rho$"))):
            ax.imshow(x_true[:, ip], cmap="gist_rainbow", extent=extent,
                      vmin=x_true.min(), vmax=x_true.max())
            ax.set_title(rf"True $\ln$ {param}")
            ax.set_xlabel("x [km]")
        axs[0].set_ylabel("z [km]")
        axd = plt.subplot2grid((2, 6), (1, 0), colspan=4)
        axd.imshow(d[:, 0], extent=extent, **dopts)
        axd.axvline(x[ix], color="w", ls="--")
        axd.set_title(r"Data ($\theta$=0)")
        axd.set_xlabel("x [km]")
        axd.set_ylabel("z [km]")
        axg = plt.subplot2grid((2, 6), (1, 4), colspan=2, sharey=axd)
        axg.imshow(d[:, :, ix], extent=(theta[0], theta[-1], z[-1], z[0]), **dopts)
        axg.set_title(f"Gather (x={x[ix]:.2f} km)")
        axg.set_xlabel(r"$\theta$ [°]")
        for ax in axs + [axd, axg]:
            ax.axis("tight")
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True models, inverted models and errors, one row per parameter."""
        import matplotlib.pyplot as plt

        xx, z = meta["x"], meta["z"]
        x = np.real(x).reshape(meta["dims"])
        extent = (xx[0], xx[-1], z[-1], z[0])

        fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(12, 9))
        for ip, param in enumerate((r"$V_P$", r"$V_S$", r"$\rho$")):
            vmin, vmax = x_true[:, ip].min(), x_true[:, ip].max()
            mopts = dict(cmap="gist_rainbow", vmin=vmin, vmax=vmax, extent=extent)
            eclip = 0.2 * (vmax - vmin)
            axs[ip, 0].imshow(x_true[:, ip], **mopts)
            axs[ip, 0].set_title(rf"True $\ln$ {param}")
            axs[ip, 1].imshow(x[:, ip], **mopts)
            axs[ip, 1].set_title(rf"Estimated $\ln$ {param}")
            axs[ip, 2].imshow(
                x[:, ip] - x_true[:, ip],
                cmap="RdBu_r",
                vmin=-eclip,
                vmax=eclip,
                extent=extent,
            )
            axs[ip, 2].set_title(rf"Error $\ln$ {param}")
            axs[ip, 0].set_ylabel("z [km]")
        for ax in axs.ravel():
            ax.axis("tight")
        for ax in axs[-1]:
            ax.set_xlabel("x [km]")
        fig.tight_layout()
        return fig
