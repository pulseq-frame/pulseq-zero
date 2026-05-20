from ..events import Delay, Scalar

def make_delay(d: Scalar) -> Delay:
    return Delay(d)