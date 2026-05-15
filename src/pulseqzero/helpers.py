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


def calc_rf_center(rf):
    return rf.shape_dur / 2, 0
