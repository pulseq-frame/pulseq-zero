from .math import ceil, floor, round

from .adapter import (
    Opts,
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

try:
    from pypulseq import set_tx_mode  # noqa: F401
except ImportError:
    pass
