"""TODO: write the module docstring.

This is pulseq-zero, a wrapper around pypulseq. When swapping imports, sequence
scripts should continue to run as they did under pypulseq. But the underlying
architecture changed: All calls are converted to an intermediate format that
allows to plot and write sequences as before but also to convert to MR-zero
while passing through torch tensors and their backpropagation graph. This way,
gradient descent optimization with a seq script in the loop becomes possible.
"""

__all__ = [
    "ceil",
    "floor",
    "round",
    "Opts",
    "Sequence",
    "calc_duration",
    "calc_rf_bandwidth",
    "calc_rf_center",
    "calc_SAR",
    "make_adc",
    "make_delay",
    "make_trigger",
    "make_digital_output_pulse",
    "make_label",
    "get_supported_labels",
    "make_trapezoid",
    "make_arbitrary_grad",
    "make_extended_trapezoid",
    "make_extended_trapezoid_area",
    "scale_grad",
    "split_gradient",
    "split_gradient_at",
    "add_gradients",
    "make_sinc_pulse",
    "make_gauss_pulse",
    "make_block_pulse",
    "make_arbitrary_rf",
    "make_adiabatic_pulse",
    "sigpy_n_seq",
    "make_slr",
    "make_sms",
    "SigpyPulseOpts",
    "align",
    "calc_ramp",
    "rotate",
    "points_to_waveform",
    "traj_to_grad",
    "round_half_up",
    "enable_trace",
    "disable_trace",
    "make_soft_delay",
    "eps",
    "calc_adc_segments",
]

# does not need to be differentiable, use directly from pypulseq
from pypulseq import Opts, eps, calc_adc_segments

# not differentiable - some might be replaced with differentiable versions
from pypulseq import round_half_up

# differentiable math helper functions
from .math import ceil, floor, round

from .adapter import (
    Sequence,
    calc_duration,
    calc_rf_bandwidth,
    calc_rf_center,
    calc_SAR,
    make_adc,
    make_delay,
    make_trigger,
    make_digital_output_pulse,
    make_label,
    get_supported_labels,
    make_trapezoid,
    make_arbitrary_grad,
    make_extended_trapezoid,
    make_extended_trapezoid_area,
    scale_grad,
    split_gradient,
    split_gradient_at,
    add_gradients,
    make_sinc_pulse,
    make_gauss_pulse,
    make_block_pulse,
    make_arbitrary_rf,
)

# No pulseq-zero support yet, calling will raise NotImplementedError.
# This list should be eliminated before releasing 1.0
from .not_implemented import (
    make_adiabatic_pulse,
    sigpy_n_seq,
    make_slr,
    make_sms,
    SigpyPulseOpts,
    align,
    calc_ramp,
    rotate,
    points_to_waveform,
    traj_to_grad,
    enable_trace,
    disable_trace,
    make_soft_delay,
)

# Re-export the shim-generation function if Martin's pTx pulseq is installed
try:
    from pypulseq import set_tx_mode as set_tx_mode
except ImportError:
    pass
else:
    __all__.append("set_tx_mode")
