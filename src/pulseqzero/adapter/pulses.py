import numpy as np
import pypulseq

from .. import Opts
from ..events import Pulse, TrapGrad
from ..helpers import calc_duration
from ..adapter.delay import make_delay


def _n(x):
    """Detach a torch tensor (or pass a plain number) to a Python float."""
    if hasattr(x, "detach"):
        return x.detach().cpu().item()
    return float(x)


def _wrap_trap(pp_grad):
    return TrapGrad(
        channel=pp_grad.channel,
        amplitude=float(pp_grad.amplitude),
        rise_time=float(pp_grad.rise_time),
        flat_time=float(pp_grad.flat_time),
        fall_time=float(pp_grad.fall_time),
        delay=float(pp_grad.delay),
    )


def _build_pulse(pp_rf, *, flip_angle, freq_offset, phase_offset, delay,
                 shim_array, use, factory_name, pp_kwargs):
    return Pulse(
        flip_angle=flip_angle,
        shape_dur=float(pp_rf.shape_dur),
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        ringdown_time=float(pp_rf.ringdown_time),
        shim_array=shim_array,
        shape=(np.asarray(pp_rf.t), np.asarray(pp_rf.signal)),
        use=use,
        _pp_factory=factory_name,
        _pp_kwargs=pp_kwargs,
    )


def make_sinc_pulse(
    flip_angle,
    apodization=0,
    delay=0,
    duration=4e-3,
    dwell=0,
    center_pos=0.5,
    freq_offset=0,
    max_grad=None,
    max_slew=None,
    phase_offset=0,
    return_delay=False,
    return_gz=False,
    slice_thickness=0,
    system=None,
    time_bw_product=4,
    shim_array=None,
    use="",
):
    if system is None:
        system = Opts.default

    pp_kwargs = dict(
        apodization=_n(apodization),
        duration=_n(duration),
        dwell=_n(dwell),
        center_pos=_n(center_pos),
        max_grad=_n(max_grad if max_grad is not None else 0),
        max_slew=_n(max_slew if max_slew is not None else 0),
        slice_thickness=_n(slice_thickness),
        time_bw_product=_n(time_bw_product),
    )
    result = pypulseq.make_sinc_pulse(
        flip_angle=_n(flip_angle),
        delay=_n(delay),
        freq_offset=_n(freq_offset),
        phase_offset=_n(phase_offset),
        return_gz=return_gz,
        system=system,
        use=use or "undefined",
        **pp_kwargs,
    )
    if return_gz:
        pp_rf, pp_gz, pp_gzr = result
    else:
        pp_rf = result

    rf = _build_pulse(
        pp_rf,
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=max(_n(delay), system.rf_dead_time),
        shim_array=shim_array,
        use=use,
        factory_name="make_sinc_pulse",
        pp_kwargs=pp_kwargs,
    )

    ret_val = rf
    if return_gz:
        ret_val = (rf, _wrap_trap(pp_gz), _wrap_trap(pp_gzr))

    if return_delay and rf.ringdown_time > 0:
        d = make_delay(calc_duration(rf) + rf.ringdown_time)
        ret_val = (*ret_val, d) if isinstance(ret_val, tuple) else (ret_val, d)

    return ret_val


def make_gauss_pulse(
    flip_angle,
    apodization=0,
    bandwidth=None,
    center_pos=0.5,
    delay=0,
    dwell=0,
    duration=4e-3,
    freq_offset=0,
    max_grad=0,
    max_slew=0,
    phase_offset=0,
    return_gz=False,
    return_delay=False,
    slice_thickness=0,
    system=None,
    time_bw_product=4,
    shim_array=None,
    use="",
):
    if system is None:
        system = Opts.default

    if bandwidth is None:
        bandwidth = _n(time_bw_product) / _n(duration)
    pp_kwargs = dict(
        apodization=_n(apodization),
        bandwidth=_n(bandwidth),
        center_pos=_n(center_pos),
        duration=_n(duration),
        dwell=_n(dwell),
        max_grad=_n(max_grad if max_grad is not None else 0),
        max_slew=_n(max_slew if max_slew is not None else 0),
        slice_thickness=_n(slice_thickness),
        time_bw_product=_n(time_bw_product),
    )
    result = pypulseq.make_gauss_pulse(
        flip_angle=_n(flip_angle),
        delay=_n(delay),
        freq_offset=_n(freq_offset),
        phase_offset=_n(phase_offset),
        return_gz=return_gz,
        system=system,
        use=use or "undefined",
        **pp_kwargs,
    )
    if return_gz:
        pp_rf, pp_gz, pp_gzr = result
    else:
        pp_rf = result

    rf = _build_pulse(
        pp_rf,
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=max(_n(delay), system.rf_dead_time),
        shim_array=shim_array,
        use=use,
        factory_name="make_gauss_pulse",
        pp_kwargs=pp_kwargs,
    )

    ret_val = rf
    if return_gz:
        ret_val = (rf, _wrap_trap(pp_gz), _wrap_trap(pp_gzr))

    if return_delay and rf.ringdown_time > 0:
        d = make_delay(calc_duration(rf) + rf.ringdown_time)
        ret_val = (*ret_val, d) if isinstance(ret_val, tuple) else (ret_val, d)

    return ret_val


def make_block_pulse(
    flip_angle,
    delay=0,
    duration=None,
    bandwidth=None,
    time_bw_product=0.25,
    freq_offset=0,
    phase_offset=0,
    return_delay=False,
    system=None,
    shim_array=None,
    use="",
):
    if system is None:
        system = Opts.default

    pp_kwargs = dict(
        duration=None if duration is None else _n(duration),
        bandwidth=None if bandwidth is None else _n(bandwidth),
        time_bw_product=None if time_bw_product is None else _n(time_bw_product),
    )
    pp_rf = pypulseq.make_block_pulse(
        flip_angle=_n(flip_angle),
        delay=_n(delay),
        freq_offset=_n(freq_offset),
        phase_offset=_n(phase_offset),
        system=system,
        use=use or "undefined",
        **pp_kwargs,
    )

    rf = _build_pulse(
        pp_rf,
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=max(_n(delay), system.rf_dead_time),
        shim_array=shim_array,
        use=use,
        factory_name="make_block_pulse",
        pp_kwargs=pp_kwargs,
    )

    if return_delay:
        return (rf, make_delay(calc_duration(rf) + system.rf_ringdown_time))
    return rf


def make_arbitrary_rf(
    signal,
    flip_angle,
    bandwidth=0,
    delay=0,
    dwell=None,
    freq_offset=0,
    no_signal_scaling=False,
    max_grad=0,
    max_slew=0,
    phase_offset=0,
    return_delay=False,
    return_gz=False,
    slice_thickness=0,
    system=None,
    time_bw_product=0,
    shim_array=None,
    use="",
):
    if system is None:
        system = Opts.default

    signal_np = np.asarray(signal.detach().cpu() if hasattr(signal, "detach") else signal)
    pp_kwargs = dict(
        signal=signal_np,
        bandwidth=_n(bandwidth),
        dwell=_n(dwell if dwell is not None else 0),
        no_signal_scaling=bool(no_signal_scaling),
        max_grad=_n(max_grad if max_grad is not None else 0),
        max_slew=_n(max_slew if max_slew is not None else 0),
        slice_thickness=_n(slice_thickness),
        time_bw_product=_n(time_bw_product),
    )
    result = pypulseq.make_arbitrary_rf(
        flip_angle=_n(flip_angle),
        delay=_n(delay),
        freq_offset=_n(freq_offset),
        phase_offset=_n(phase_offset),
        return_gz=return_gz,
        system=system,
        use=use or "undefined",
        **pp_kwargs,
    )
    if return_gz:
        pp_rf, pp_gz = result
    else:
        pp_rf = result

    rf = _build_pulse(
        pp_rf,
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=max(_n(delay), system.rf_dead_time),
        shim_array=shim_array,
        use=use,
        factory_name="make_arbitrary_rf",
        pp_kwargs=pp_kwargs,
    )

    ret_val = rf
    if return_gz:
        ret_val = (rf, _wrap_trap(pp_gz))

    if return_delay and rf.ringdown_time > 0:
        d = make_delay(calc_duration(rf) + rf.ringdown_time)
        ret_val = (*ret_val, d) if isinstance(ret_val, tuple) else (ret_val, d)

    return ret_val
