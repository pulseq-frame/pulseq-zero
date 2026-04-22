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
    # Not implemented in the adapter — these raise NotImplementedError
    # with a clear message pointing to the workaround.
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
)

try:
    from pypulseq import set_tx_mode  # noqa: F401
except ImportError:
    pass
