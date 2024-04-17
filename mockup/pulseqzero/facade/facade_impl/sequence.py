import torch
import MRzeroCore as mr0
from copy import copy

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
        self.blocks.append([copy(arg) for arg in args])

    def check_timing(self):
        return True, []
    
    def write(self, file_name):
        pass

    def plot(self):
        pass

    def to_mr0(self) -> mr0.Sequence:
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

        reps = []
        rep = []
        for ev in seq:
            if isinstance(ev, TmpPulse):
                rep = []
                reps.append(rep)
            rep.append(ev)
        
        seq = mr0.Sequence()
        for rep_in in reps:
            event_count = 0
            for ev in rep_in:
                if isinstance(ev, TmpAdc):
                    event_count += len(ev.event_time)
                else:
                    event_count += 1

            rep_out = seq.new_rep(event_count)
            rep_out.pulse.angle = torch.as_tensor(rep_in[0].angle)
            rep_out.pulse.phase = torch.as_tensor(rep_in[0].phase)
            if rep_out.pulse.angle > 100 * torch.pi / 180:
                rep_out.pulse.usage = mr0.PulseUsage.REFOC
            else:
                rep_out.pulse.usage = mr0.PulseUsage.EXCIT

            i = 0
            for ev in rep_in[1:]:
                if isinstance(ev, TmpSpoiler):
                    rep_out.event_time[i] = ev.duration
                    rep_out.gradm[i, :] = ev.gradm
                    i += 1
                else:
                    assert isinstance(ev, TmpAdc)
                    num = len(ev.event_time)
                    rep_out.event_time[i:i+num] = torch.as_tensor(ev.event_time)
                    rep_out.gradm[i:i+num, :] = torch.as_tensor(ev.gradm)
                    rep_out.adc_phase[i:i+num] = torch.pi / 2 - ev.phase
                    rep_out.adc_usage[i:i+num] = 1
                    i += num
        
        seq.normalized_grads = False
        return seq


class TmpPulse:
    def __init__(self, angle, phase) -> None:
        self.angle = angle
        self.phase = phase

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
    t = rf.delay + rf.duration / 2
    duration = max([x.duration for x in [rf, grad_x, grad_y, grad_z] if x is not None])

    gx1 = gx2 = gy1 = gy2 = gz1 = gz2 = 0.0
    if grad_x:
        gx1, gx2 = split_gradm(grad_x, t)
    if grad_y:
        gy1, gy2 = split_gradm(grad_y, t)
    if grad_z:
        gz1, gz2 = split_gradm(grad_z, t)
    
    print(gx1, gx2, gy1, gy2, gz1, gz2)

    return (
        TmpSpoiler(t, gx1, gy1, gz1),
        TmpPulse(rf.angle, rf.phase_offset),
        TmpSpoiler(duration - t, gx2, gy2, gz2)
    )


def parse_spoiler(delay, grad_x, grad_y, grad_z) -> tuple[TmpSpoiler]:
    duration = max([x.duration for x in [delay, grad_x, grad_y, grad_z] if x is not None], default=0.0)
    gx = grad_x.area if grad_x is not None else 0.0
    gy = grad_y.area if grad_y is not None else 0.0
    gz = grad_z.area if grad_z is not None else 0.0
    return (TmpSpoiler(duration, gx, gy, gz), )


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
    return (
        TmpAdc(event_time[:-1], gradm[:-1, :], adc.phase_offset),
        TmpSpoiler(event_time[-1], gradm[-1, 0], gradm[-1, 1], gradm[-1, 2])
    )


def split_gradm(grad, t):
    before = integrate(grad, t)
    total = integrate(grad, 1e9)  # Infinity produces 0*inf = NaNs internally
    return (before, total - before)


def integrate(grad, t):
    if isinstance(grad, Trapezoid):
        # heaviside could be replaced with error function for differentiability
        def h(x):
            return torch.heaviside(torch.as_tensor(x), torch.tensor(0.5))

        # https://www.desmos.com/calculator/0q5co02ecm

        d = grad.delay
        t1 = grad.rise_time
        t2 = grad.flat_time
        t3 = grad.fall_time
        T1 = d + t1
        T12 = d + t1 + t2
        T123 = d + t1 + t2 + t3

        # Trapezoid, could be provided as derivative:
        # f1 = h(t - d) * h(T1 - t) * (t - d) / t1
        # f2 = h(t - T1) * h(T12 - t)
        # f3 = h(t - T12) * h(T123 - t) * (T123 - t) / t3
        # f = grad.amplitude * (f1 + f2 + f3)

        F_inf = t1 / 2 + t2 + t3 / 2
        F1 = h(t - d) * h(T1 - t) * 0.5 * (t - d)**2 / t1
        F2 = h(t - T1) * h(T12 - t) * (t1 / 2 + t - T1)
        F3 = h(t - T12) * h(T123 - t) * (F_inf - 0.5 * (T123 - t)**2 / t3)
        F = grad.amplitude * (F1 + F2 + F3 + h(t - T123) * F_inf)

        return F
    else:
        raise NotImplementedError
