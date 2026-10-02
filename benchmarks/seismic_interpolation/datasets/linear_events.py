import numpy as np
from benchopt import BaseDataset
from pylops.basicoperators import Restriction
from pylops.utils.seismicevents import linear2d, makeaxis
from pylops.utils.wavelets import ricker


class Dataset(BaseDataset):
    """2D gather with three linear events, randomly subsampled in space."""

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
