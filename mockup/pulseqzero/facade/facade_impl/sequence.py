import torch

from .make_adc import Adc
from .make_block_pulse import BlockPulse
from .make_delay import Delay
from .make_extended_trapezoid import ExtendedTrapezoid
from .make_sinc_pulse import SincPulse
from .make_trapezoid import Trapezoid

class Sequence:
    def __init__(self, system=None) -> None:
        self.blocks = []

    def add_block(self, *args):
        self.blocks.append(args)

    def check_timing(self):
        return True, []
    
    def write(self, file_name):
        pass

    def plot(self):
        pass

    def to_mr0(self):
        seq = []

        for block in self.blocks:
            delay = None
            adc = None
            rf = None
            grad_x = None
            grad_y = None
            grad_z = None
            for ev in block:
                if isinstance(ev, Delay):
                    assert delay is None
                    delay = ev
                if isinstance(ev, Adc):
                    assert adc is None
                    adc = ev
                if isinstance(ev, BlockPulse):
                    assert rf is None
                    rf = ev
                if isinstance(ev, SincPulse):
                    assert rf is None
                    rf = ev
                if isinstance(ev, (Trapezoid, ExtendedTrapezoid)):
                    assert ev.channel in ["x", "y", "z"]
                    if ev.channel == "x":
                        assert grad_x is None
                        grad_x = ev
                    elif ev.channel == "y":
                        assert grad_y is None
                        grad_y = ev
                    elif ev.channel == "z":
                        assert grad_z is None
                        grad_z = ev
            
            if rf:
                assert delay is None and adc is None
                seq += parse_pulse(rf, grad_x, grad_y, grad_z)
            elif adc:
                assert delay is None
                seq += parse_adc(adc, grad_x, grad_y, grad_z)
            else:
                seq += parse_spoiler(delay, grad_x, grad_y, grad_z)

        for block in seq:
            print(block)


# Copy the pypulseq importer logic here:
# we want an intermediate representation so we avoid appending to an mr0
# sequence all the time.


class TmpPulse:
    pass

class TmpSpoiler:
    def __init__(self, duration, gx, gy, gz) -> None:
        self.duration = torch.as_tensor(duration)
        self.gradm = torch.stack([
            torch.as_tensor(gx),
            torch.as_tensor(gy),
            torch.as_tensor(gz)
        ])
    
    def __repr__(self) -> str:
        return f"Spoiler({self.duration}, {self.gradm})"

class TmpAdc:
    def __init__(self, event_time, gradm, phase) -> None:
        self.event_time = event_time
        self.gradm = gradm
        self.phase = phase
    
    def __repr__(self) -> str:
        return f"Adc({self.phase}, {self.event_time}, {self.gradm})"


def parse_pulse(rf, grad_x, grad_y, grad_z) -> tuple[TmpSpoiler, TmpPulse, TmpSpoiler]:
    return []


def parse_spoiler(delay, grad_x, grad_y, grad_z) -> tuple[TmpSpoiler]:
    duration = max([x.duration for x in [delay, grad_x, grad_y, grad_z] if x is not None], default=0.0)
    gx = grad_x.area if grad_x is not None else 0.0
    gy = grad_y.area if grad_y is not None else 0.0
    gz = grad_z.area if grad_z is not None else 0.0
    return [TmpSpoiler(duration, gx, gy, gz)]


def parse_adc(adc: Adc, grad_x, grad_y, grad_z) -> tuple[TmpAdc, TmpSpoiler]:
    duration = max([x.duration for x in [adc, grad_x, grad_y, grad_z] if x is not None])
    time = torch.cat([
        torch.as_tensor(0.0).view((1, )),
        adc.delay + torch.arange(adc.num_samples) * adc.dwell,
        torch.as_tensor(duration).view((1, ))
    ])

    gradm = torch.zeros((adc.num_samples + 2, 3))
    if grad_x:
        gradm[:, 0] = torch.vmap(lambda t: integrate(grad_x, t))(time)
    if grad_y:
        gradm[:, 1] = torch.vmap(lambda t: integrate(grad_y, t))(time)
    if grad_z:
        gradm[:, 2] = torch.vmap(lambda t: integrate(grad_z, t))(time)

    event_time = torch.diff(time)
    gradm = torch.diff(gradm, dim=0)
    return TmpAdc(event_time[:, -1], gradm[:,-1, :], adc.phase)


# TODO: make this differentiable...
# https://www.desmos.com/calculator?lang=de
# Maybe either approximate the trapezoid with polynomials or sigmods,
# or manually define the deriviates.
# Note that the torch.vmap used in parse_adc does not work with if statements,
# but we could swap the vmap if this is too complicated and wait with the
# differentiablity for future sequences (is not needed for the ESMRMB optimizations)
def integrate(grad, t):
    if isinstance(grad, Trapezoid):
        t -= grad.delay
        total = grad.rise_time/2 + grad.flat_time + grad.fall_time/2

        if t <= 0.0:
            integral = 0.0
        elif t < grad.rise_time:
            integral = 0.5*t**2 / grad.rise_time
        elif t < grad.rise_time + grad.flat_time:
            t -= grad.rise_time
            integral = grad.rise_time/2 + t
        elif t < grad.rise_time + grad.flat_time + grad.fall_time:
            t = grad.rise_time + grad.flat_time + grad.fall_time - t
            integral = total - 0.5*t**2 / grad.fall_time
        else:
            integral = total
        return grad.amplitude * integral
    else:
        raise NotImplementedError
