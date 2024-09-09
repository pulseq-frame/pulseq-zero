from dataclasses import dataclass
from pulseqzero.adapter import Opts, make_delay, make_trapezoid, calc_duration


def make_arbitrary_rf(
        signal: np.ndarray,
        flip_angle: float,
        bandwidth: float = 0,
        delay: float = 0,
        dwell: float = 0,
        freq_offset: float = 0,
        no_signal_scaling: bool = False,
        max_grad: float = 0,
        max_slew: float = 0,
        phase_offset: float = 0,
        return_delay: bool = False,
        return_gz: bool = False,
        slice_thickness: float = 0,
        system: Opts = None,
        time_bw_product: float = 0,
        use: str = str(),
    ):
    pass


def make_block_pulse(
        flip_angle,
        delay=0,
        duration=None,
        bandwith=None,
        time_bw_product=None,
        freq_offset=0,
        phase_offset=0,
        return_delay=False,
        system=None,
        use=None,
    ):
    pass


def make_gauss_pulse(
        flip_angle: float,
        apodization: float = 0,
        bandwidth: float = 0,
        center_pos: float = 0.5,
        delay: float = 0,
        dwell: float = 0,
        duration: float = 4e-3,
        freq_offset: float = 0,
        max_grad: float = 0,
        max_slew: float = 0,
        phase_offset: float = 0,
        return_gz: bool = False,
        return_delay: bool = False,
        slice_thickness: float = 0,
        system: Opts = None,
        time_bw_product: float = 4,
        use: str = str(),
    ):
    pass


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
        use="",
    ):
    if system is None:
        system = Opts.default

    rf = Pulse(
        flip_angle,
        duration,
        freq_offset,
        phase_offset,
        system.rf_dead_time,
        system.rf_ringdown_time,
        delay
    )

    ret_val = (rf, )
    
    if return_gz:
        if max_grad is None:
            max_grad = system.max_grad
        if max_slew is None:
            max_slew = system.max_slew

        BW = time_bw_product / duration
        grad_area = BW / slice_thickness * duration

        gz = make_trapezoid(...)
        gzr = make_trapezoid(...)

        if rf.delay > gz.rise_time:
            gz.delay = rf.delay - gz.rise_time
        if rf.delay < gz.rise_time + gz.delay:
            rf.delay = gz.rise_time + gz.delay
        
        ret_val = (*ret_val, gz, gzr)
    
    if return_delay and rf.ringdown_time > 0:
        delay = make_delay(calc_duration(rf) + rf.ringdown_time)
        ret_val = (*ret_val, delay)
    
    return ret_val


@dataclass
class Pulse:
    flip_angle: ...
    shape_dur: ...
    freq_offset: ...  # ignored by sim
    phase_offset: ...
    dead_time: ...  # ignored by sim
    ringdown_time: ...  # ignored by sim
    delay: ...