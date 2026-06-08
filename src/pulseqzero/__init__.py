"""Temporary pulseq-zero description.

This is pulseq-zero, a wrapper around pypulseq. When swapping imports, sequence
scripts should continue to run as they did under pypulseq. But the underlying
architecture changed: All calls are converted to an intermediate format that
allows to plot and write sequences as before but also to convert to MR-zero
while passing through torch tensors and their backpropagation graph. This way,
gradient descent optimization with a seq script in the loop becomes possible.
"""
# TODO: write the module docstring.

# We mimic the version of the installed pypulseq in case users rely on it
import importlib.metadata
__version__ = importlib.metadata.version("pypulseq")

__all__ = [
    "Opts",
    "Sequence",
    "SigpyPulseOpts",
    "add_gradients",
    "align",
    "calc_SAR",
    "calc_adc_segments",
    "calc_duration",
    "calc_ramp",
    "calc_rf_bandwidth",
    "calc_rf_center",
    "ceil",
    "disable_trace",
    "enable_trace",
    "eps",
    "floor",
    "get_supported_labels",
    "get_supported_rf_uses",
    "make_adc",
    "make_adiabatic_pulse",
    "make_arbitrary_grad",
    "make_arbitrary_rf",
    "make_block_pulse",
    "make_delay",
    "make_digital_output_pulse",
    "make_extended_trapezoid",
    "make_extended_trapezoid_area",
    "make_gauss_pulse",
    "make_label",
    "make_sinc_pulse",
    "make_slr",
    "make_sms",
    "make_soft_delay",
    "make_trapezoid",
    "make_trigger",
    "points_to_waveform",
    "rotate",
    "round",
    "round_half_up",
    "scale_grad",
    "sigpy_n_seq",
    "split_gradient",
    "split_gradient_at",
    "traj_to_grad",
]

# Re-export the shim-generation function if Martin's pTx pulseq is installed
try:
    from pypulseq import set_tx_mode as set_tx_mode  # ty: ignore[unresolved-import]
except ImportError:
    FREUDENSPRUNG_PTX = False
else:
    FREUDENSPRUNG_PTX = True
    __all__.append("set_tx_mode")

# does not need to be differentiable, use directly from pypulseq
from pypulseq import Opts, eps, calc_adc_segments, get_supported_labels
from pypulseq.supported_labels_rf_use import get_supported_rf_uses

# differentiable math helper functions
from .math import ceil, floor, round, round_half_up

from .wrapper.helpers import (
    calc_duration,
    calc_rf_bandwidth,
    calc_rf_center,
    align,
    traj_to_grad,
)
from .wrapper.calc_ramp import calc_ramp
from .wrapper.make_basic import (
    make_adc,
    make_delay,
    make_trigger,
    make_digital_output_pulse,
    make_label,
    make_soft_delay,
)
from .wrapper.make_pulse import (
    make_block_pulse,
    make_gauss_pulse,
    make_sinc_pulse,
    make_arbitrary_rf,
)
from .wrapper.make_grad import (
    make_trapezoid,
    make_arbitrary_grad,
    make_extended_trapezoid,
    make_extended_trapezoid_area,
)
from .wrapper.grad_funcs import (
    scale_grad,
    points_to_waveform,
    rotate,
    split_gradient,
    split_gradient_at,
    add_gradients,
)
from .wrapper.sequence import Sequence

# No pulseq-zero support (documented), calling will raise NotImplementedError.
from .not_implemented import (
    calc_SAR,
    enable_trace,
    disable_trace,
    SigpyPulseOpts,
    sigpy_n_seq,
    make_adiabatic_pulse,
    make_slr,
    make_sms,
)
