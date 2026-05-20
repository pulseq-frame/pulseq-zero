from copy import copy
from typing import Optional, TypeVar
from pypulseq import Opts
from ..events import TrapGrad, ExtTrapGrad, ArbitraryGrad

GradType = TypeVar("GradType", TrapGrad, ExtTrapGrad, ArbitraryGrad)


def scale_grad(grad: GradType, scale: float, system: Optional[Opts] = None) -> GradType:
    grad = copy(grad)

    if isinstance(grad, TrapGrad):
        grad.amplitude = scale * grad.amplitude
    elif isinstance(grad, ExtTrapGrad):
        grad.waveform = scale * grad.waveform
    else:
        grad.waveform = scale * grad.waveform
        grad.first = scale * grad.first
        grad.last = scale * grad.last

    return grad
