def calc_SAR(file):
    pass


def make_label(label, type, value):
    pass


def calc_duration(*args):
    # We assume that all events have a duration property
    return max(
        (event.duration for event in args if event is not None),
        default=0.0
    )


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


def get_supported_labels():
    return (
        "SLC", "SEG", "REP", "AVG", "SET", "ECO", "PHS", "LIN", "PAR", "NAV",
        "REV", "SMS", "REF", "IMA", "NOISE", "PMC", "NOROT", "NOPOS", "NOSCL",
        "ONCE", "TRID",
    )


from .opts import Opts
from .delay import make_delay, make_trigger, make_digital_output_pulse
from .adc import make_adc
from .grads import scale_grad, split_gradient, add_gradients, make_trapezoid, make_arbitrary_grad, make_extended_trapezoid
from .pulses import make_arbitrary_rf, make_block_pulse, make_gauss_pulse, make_sinc_pulse
from .sequence import Sequence

# copied from pypulseq, not yet differentiable
from .extended_trap_grad import make_extended_trapezoid_area
