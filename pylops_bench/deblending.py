"""Helpers shared by the solvers of the seismic deblending benchmark."""

import numpy as np
from pylops.signalprocessing import FFT2D, Patch2D
from pylops.signalprocessing.patch2d import patch2d_design

NWIN = (20, 80)
NOVER = (10, 40)
NOP = (128, 128)
NOP1 = (128, 65)


def patched_fk(dims):
    """Patched FK transform (synthesis operator mapping FK patches to data).

    Returns the operator with cosine-sqrt tapered overlapping patches and the
    same operator without taper (used to scale thresholds from the
    pseudo-deblended data).
    """
    Fop = FFT2D(NWIN, nffts=NOP, real=True)
    _, mdims = patch2d_design(dims, NWIN, NOVER, NOP1)[:2]
    args = (Fop.H, mdims, dims, NWIN, NOVER, NOP1)
    Sop = Patch2D(*args, tapertype="cosinesqrt")
    Sop1 = Patch2D(*args, tapertype=None)
    return Sop, Sop1


def max_fk_amplitude(Bop, Sop1, y):
    """Maximum FK amplitude of the pseudo-deblended data."""
    return np.abs(Sop1.H @ (Bop.H @ y)).max()
