from typing import Optional
from ..events import Delay, Scalar, Label, SoftDelay
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


def make_soft_delay(
    hint: str,
    numID: Optional[int] = None,
    offset: float = 0.0,
    factor: float = 1.0,
    default_duration: float = 10e-6,
) -> SoftDelay:
    if any(c.isspace() for c in hint):
        raise ValueError("Parameter 'hint' may not contain white space characters.")
    if default_duration <= 0:
        raise ValueError("Default duration must be greater than 0.")
    if factor == 0:
        raise ValueError("Parameter 'factor' cannot be zero.")
    if numID is not None and (not isinstance(numID, int) or numID < 0):
        raise ValueError("Parameter 'numID' must be a non-negative integer or None.")

    return SoftDelay(hint, numID, offset, factor, default_duration)
