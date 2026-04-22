def calc_SAR(file):
    pass


def make_label(label, type, value):
    pass


def _unsupported(name, workaround):
    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"pulseqzero.{name} is not implemented. {workaround}"
        )
    stub.__name__ = name
    return stub


make_adiabatic_pulse = _unsupported(
    "make_adiabatic_pulse",
    "Adiabatic pulses have no differentiable reimplementation yet. "
    "Workaround: design the pulse with `pypulseq.make_adiabatic_pulse` and "
    "wrap its signal via `pulseqzero.make_arbitrary_rf(signal=..., ...)`.",
)
sigpy_n_seq = _unsupported(
    "sigpy_n_seq",
    "sigpy-based pulse design is not supported. "
    "Workaround: build the pulse with sigpy + pypulseq and wrap the signal "
    "via `pulseqzero.make_arbitrary_rf(...)`.",
)
make_slr = _unsupported(
    "make_slr",
    "SLR pulse design is not supported. "
    "Workaround: design with sigpy / pypulseq then wrap via "
    "`pulseqzero.make_arbitrary_rf(...)`.",
)
make_sms = _unsupported(
    "make_sms",
    "SMS pulse design is not supported. "
    "Workaround: design with sigpy / pypulseq then wrap via "
    "`pulseqzero.make_arbitrary_rf(...)`.",
)
SigpyPulseOpts = _unsupported(
    "SigpyPulseOpts",
    "sigpy pulse options are not supported (see `make_slr` / `make_sms`).",
)
align = _unsupported(
    "align",
    "Timing alignment is not implemented in the adapter yet. "
    "Workaround: call `seq.to_pypulseq()` and use pypulseq's `align` on the "
    "translated sequence.",
)
calc_ramp = _unsupported(
    "calc_ramp",
    "calc_ramp is not implemented. "
    "Workaround: compute ramps manually or call `pypulseq.calc_ramp` via "
    "`seq.to_pypulseq()`.",
)
rotate = _unsupported(
    "rotate",
    "Gradient rotation is not implemented. "
    "Workaround: construct rotated gradients manually, or rotate after "
    "`seq.to_pypulseq()`.",
)
points_to_waveform = _unsupported(
    "points_to_waveform",
    "points_to_waveform is not implemented. "
    "Workaround: call `pypulseq.points_to_waveform` directly.",
)
traj_to_grad = _unsupported(
    "traj_to_grad",
    "traj_to_grad is not implemented. "
    "Workaround: call `pypulseq.traj_to_grad` directly.",
)


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


def get_supported_labels():
    return (
        "SLC", "SEG", "REP", "AVG", "SET", "ECO", "PHS", "LIN", "PAR", "NAV",
        "REV", "SMS", "REF", "IMA", "NOISE", "PMC", "NOROT", "NOPOS", "NOSCL",
        "ONCE", "TRID",
    )


from .opts import Opts
from .delay import make_delay, make_trigger, make_digital_output_pulse
from .adc import make_adc
from .grads import scale_grad, split_gradient, split_gradient_at, add_gradients, make_trapezoid, make_arbitrary_grad, make_extended_trapezoid
from .pulses import make_arbitrary_rf, make_block_pulse, make_gauss_pulse, make_sinc_pulse
from .sequence import Sequence

# copied from pypulseq, not yet differentiable
from .extended_trap_grad import make_extended_trapezoid_area
