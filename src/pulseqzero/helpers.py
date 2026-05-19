from warnings import warn
import torch
import numpy as np
from .events import Scalar, RfPulse


def calc_duration(*args):
    duration = torch.zeros(())
    for event in args:
        if event is not None:
            duration = torch.maximum(duration, torch.as_tensor(event.duration))
    return duration


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
