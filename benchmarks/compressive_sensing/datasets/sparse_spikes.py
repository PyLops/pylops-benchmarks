import numpy as np
from benchopt import BaseDataset
from pylops.basicoperators import MatrixMult


class Dataset(BaseDataset):
    """Sparse vector of `n` = 256 samples with `k` = 16 nonzero spikes at
    random locations (random signs, amplitudes between 1 and 2), measured by a
    random Gaussian matrix with `m` = 64 rows and unit-norm columns, plus
    white Gaussian noise.
    """

    name = "Sparse spikes"

    requirements = ["pip::pylops"]

    parameters = {
        "n": [256],
        "m": [64],
        "k": [16],
        "noise_std": [0.01],
        "seed": [0],
    }

    def get_data(self):
        # independent, fixed random streams for the operator, the model and
        # the noise so that changing one never changes the others
        rng_op, rng_model, rng_noise = (
            np.random.default_rng(s) for s in np.random.SeedSequence(self.seed).spawn(3)
        )

        A = rng_op.standard_normal((self.m, self.n))
        A /= np.linalg.norm(A, axis=0)
        Op = MatrixMult(A, dtype="float64")

        x = np.zeros(self.n)
        support = rng_model.choice(self.n, self.k, replace=False)
        x[support] = rng_model.choice([-1.0, 1.0], self.k) * rng_model.uniform(
            1.0, 2.0, self.k
        )

        y = Op @ x
        if self.noise_std > 0:
            y = y + self.noise_std * rng_noise.standard_normal(self.m)
        meta = dict(dims=(self.n,), k=self.k, noise_std=self.noise_std)
        return dict(x_true=x, Op=Op, y=y, meta=meta)

    def plot(self, x_true, Op, y, meta):
        """True sparse vector and measurements."""
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(10, 3), width_ratios=(2, 1))
        axs[0].stem(x_true, linefmt="k-", markerfmt="ko", basefmt="k-")
        axs[0].set_title("True model")
        axs[1].plot(y, "r.-")
        axs[1].set_title("Data")
        fig.tight_layout()
        return fig

    def plot_estimate(self, x_true, x, meta):
        """True sparse vector, estimate and error."""
        import matplotlib.pyplot as plt

        x = np.real(x).reshape(meta["dims"])
        fig, axs = plt.subplots(
            2, 1, sharex=True, figsize=(10, 4.5), height_ratios=(2, 1)
        )
        axs[0].stem(
            x_true, linefmt="k-", markerfmt="ko", basefmt="k-", label="True model"
        )
        axs[0].plot(x, "r.", ms=8, label="Estimate")
        axs[0].legend()
        axs[1].plot(x - x_true, "k.-", lw=1)
        axs[1].set_title("Error")
        fig.tight_layout()
        return fig
