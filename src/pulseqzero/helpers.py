from warnings import warn
from typing import Optional
import numpy as np
import torch
from .events import Scalar, RfPulse, Event, Array
from pypulseq import Opts


def calc_duration(*args: Event) -> Scalar:
    return max(ev.duration for ev in args if ev is not None)


def calc_rf_bandwidth(rf, cutoff=0.5, return_axis=False, return_spectrum=False):
    bw = 0
    spectrum = np.zeros(1)
    w = np.zeros(1)

    if return_spectrum and not return_axis:
        return bw, spectrum
    if return_axis:
        return bw, spectrum, w
    return bw


def calc_rf_center(rf: RfPulse) -> tuple[Scalar, int]:
    warn(
        "pulseq-zeros calc_rf_center does not compute the center shape index (returns 0). "
        "The returned time point is computed on pulse construction - it is encouraged to use rf.center directly. "
        "This behaviour does not affect forwarded pypulseq sequence operations which use the pypulseq implementation."
    )
    return rf.center, 0


def traj_to_grad(k: Array, raster_time: Optional[float] = None) -> tuple[Array, Array]:
    if raster_time is None:
        raster_time = Opts.default.grad_raster_time

    # Compute finite difference for gradients in Hz/m
    g = (k[..., 1:] - k[..., :-1]) / raster_time
    # Compute the slew rate
    sr0 = (g[..., 1:] - g[..., :-1]) / raster_time

    # Gradient is now sampled between k-space points whilst the slew rate is between gradient points
    sr = torch.zeros((*sr0.shape[:-1], sr0.shape[-1] + 1))
    sr[..., 0] = sr0[..., 0]
    sr[..., 1:-1] = 0.5 * (sr0[..., :-1] + sr0[..., 1:])
    sr[..., -1] = sr0[..., -1]

    return g, sr
