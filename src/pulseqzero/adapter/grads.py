from dataclasses import dataclass
import numpy as np
from pulseqzero.adapter import Opts


def make_trapezoid(
    channel,
    amplitude=None,
    area=None,
    delay=0,
    duration=None,
    fall_time=None,
    flat_area=None,
    flat_time=None,
    max_grad=None,
    max_slew=None,
    rise_time=None,
    system=None,
):
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew

    if flat_time is not None:
        if amplitude is not None:
            pass
        elif area is not None:
            assert rise_time is not None
            if fall_time is None:
                fall_time = rise_time
            amplitude = area / (rise_time / 2 + flat_time + fall_time / 2)
        else:
            assert flat_area is not None
            amplitude = flat_area / flat_time

        if rise_time is None:
            rise_time = abs(amplitude) / max_slew
        if fall_time is None:
            fall_time = rise_time

    elif duration is not None:
        if amplitude is None:
            if rise_time is None:
                _, rise_time, flat_time, fall_time = calc_params_for_area(area, max_slew, max_grad)
                assert duration >= rise_time + flat_time + fall_time

                assert area is not None
                dC = 1 / abs(max_slew)
                amplitude = (duration - (duration**2 - 4 * abs(area) * dC)**0.5) / (2 * dC)
            else:
                if fall_time is None:
                    fall_time = rise_time
                amplitude = area / (duration - rise_time / 2 - fall_time / 2)

        if rise_time is None:
            rise_time = amplitude / max_slew
        if fall_time is None:
            fall_time = rise_time
        flat_time = duration - rise_time - fall_time

    else:
        assert area is not None
        amplitude, rise_time, flat_time, fall_time = calc_params_for_area(area, max_slew, max_grad)

    return TrapGrad(
        channel,
        amplitude,
        rise_time,
        flat_time,
        fall_time,
        delay
    )


def calc_params_for_area(area, max_slew, max_grad):
    rise_time = (abs(area) / max_slew)**0.5
    amplitude = area / rise_time
    t_eff = rise_time

    if abs(amplitude) > max_grad:
        t_eff = abs(area) / max_grad
        amplitude = area / t_eff
        rise_time = abs(amplitude) / max_slew
    
    flat_time = t_eff - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


@dataclass
class TrapGrad:
    channel: ...
    amplitude: ...
    rise_time: ...
    flat_time: ...
    fall_time: ...
    delay: ...

    @property
    def area(self):
        return self.amplitude * (self.rise_time / 2 + self.flat_time + self.fall_time / 2)

    @property
    def flat_area(self):
        return self.amplitude * self.flat_time

    @property
    def duration(self):
        return self.delay + self.rise_time + self.flat_time + self.fall_time


def make_arbitrary_grad(
    channel,
    waveform,
    delay=0,
    max_grad=None,
    max_slew=None,
    system=None,
):
    if system is None:
        system = Opts.default
    
    tt = (np.arange(len(waveform)) + 0.5) * system.grad_raster_time
    
    return FreeGrad(
        channel,
        waveform,
        delay,
        tt,
        shape_dur
    )


@dataclass
class FreeGrad:
    channel: ...
    waveform: ...
    delay: ...
    tt: ...
    shape_dur: ...

    @property
    def duration(self):
        return self.delay + self.shape_dur
        
    @property
    def area(self):
        return 0.5 * (
            (self.tt[1:] - self.tt[:-1]) *
            (self.waveform[1:] + self.waveform[:-1])
        ).sum()


def make_extended_trapezoid(
    channel,
    amplitudes=np.zeros(1),
    convert_to_arbitrary=False,
    max_grad=0,
    max_slew=0,
    skip_check=False,
    system=None,
    times=np.zeros(1),
):
    if system is None:
        system = Opts.default
    if max_grad is None:
        max_grad = system.max_grad
    if max_slew is None:
        max_slew = system.max_slew

    return FreeGrad(
        channel,
        amplitudes,
        times[0],
        times - times[0],
        times[-1]
    )
