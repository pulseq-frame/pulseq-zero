from typing import Optional
from ..events import Delay, Scalar, Label
from . import _n
import pypulseq as pp
from pypulseq import Opts, get_supported_labels


def make_delay(d: Scalar) -> Delay:
    return Delay(delay=d, _pp_event=pp.make_delay(_n(d)))


def make_trigger(
    channel: str,
    delay: Scalar = 0.0,
    duration: float = 0.0,
    system: Optional[Opts] = None,
) -> Delay:
    return Delay(
        delay=delay, _pp_event=pp.make_trigger(channel, _n(delay), duration, system)
    )


def make_digital_output_pulse(
    channel: str,
    delay: Scalar = 0.0,
    duration: float = 4e-3,
    system: Optional[Opts] = None,
) -> Delay:
    return Delay(
        delay=delay,
        _pp_event=pp.make_digital_output_pulse(channel, _n(delay), duration, system),
    )


def make_label(label: str, type: str, value: int) -> Label:
    if label not in get_supported_labels():
        raise ValueError("invalid label")

    if type == "INC":
        inc = True
    elif type == "SET":
        inc = False
    else:
        raise ValueError("Invalid type. Must be one of 'SET' or 'INC'.")
    
    return Label(label, inc, int(value))
