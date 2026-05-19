from typing import cast
import torch
import numpy as np
from .events import Array, Scalar, RfPulse


def calc_duration(*args):
    import torch  # needed for differentiability
    duration = torch.zeros(())
    for event in args:
        if event is not None:
            duration = torch.maximum(duration, torch.as_tensor(event.duration))
    return duration


def calc_rf_bandwidth(rf, cutoff=0.5, return_axis=False, return_spectrum=False):
    import numpy as np
    bw = 0
    spectrum = np.zeros(1)
    w = np.zeros(1)

    if return_spectrum and not return_axis:
        return bw, spectrum
    if return_axis:
        return bw, spectrum, w
    return bw


def calc_rf_center(rf: RfPulse) -> tuple[Scalar, int]:
    # if hasattr(rf, 'center'):
    #     return rf.center, np.argmin(abs(rf.t - rf.center)).item()

    # # Detect the excitation peak; if i is a plateau take its center
    # rf_max = np.max(np.abs(rf.signal))
    # i_peak = np.where(np.abs(rf.signal) >= rf_max * 0.99999)[0]
    # time_center = (rf.t[i_peak[0]] + rf.t[i_peak[-1]]) / 2
    # id_center = i_peak[round((len(i_peak) - 1) / 2)]

    # return time_center, id_center
    return rf.shape_dur / 2, 0


# not exposed, used by arbitrary pulses
def _calc_shape_center(signal: Array, time: Array) -> tuple[Scalar, int]:
    """Detect the excitation peak; if i is a plateau take its center"""
    if isinstance(signal, torch.Tensor):
        rf_max = torch.max(torch.abs(signal))
        i_peak = torch.where(torch.abs(signal) >= rf_max * 0.99999)[0]
        time_center = cast(Scalar, (time[i_peak[0]] + time[i_peak[-1]]) / 2)
        id_center = int(i_peak[round((len(i_peak) - 1) / 2)])
    else:
        rf_max = np.max(np.abs(signal))
        i_peak = np.where(np.abs(signal) >= rf_max * 0.99999)[0]
        time_center = (time[i_peak[0]] + time[i_peak[-1]]) / 2
        id_center = i_peak[round((len(i_peak) - 1) / 2)]

    return time_center, id_center
