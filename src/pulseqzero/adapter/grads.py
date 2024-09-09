from dataclasses import dataclass
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


def make_arbitrary_grad(
    channel,
    waveform,
    delay=0,
    max_grad=0,
    max_slew=0,
    system=None,
):
    return Grad()


@dataclass
class Grad:
    # make_trapezoid
    channel: ...
    amplitude: ...
    rise_time: ...
    flat_time: ...
    fall_time: ...
    area: ...
    flat_area: ...
    delay: ...
    first: ...
    last: ...

    # make_arbitrary_grad
    channel: ...
    waveform: ...
    delay: ...
    tt: ...
    shape_dur: ...
    first: ...
    last: ...
