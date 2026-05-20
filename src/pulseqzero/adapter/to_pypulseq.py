import inspect

import numpy as np
import pypulseq

from ..events import Adc, Delay, RfPulse, TrapGrad, ExtTrapGrad, ArbitraryGrad


def _n(x):
    """Detach a torch tensor (or pass a plain number) to a Python float."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.shape == () or (arr.ndim == 1 and arr.shape[0] == 1):
        return arr.item()
    return arr


def _translate_pulse(ev: RfPulse, system):
    """Re-create a pulse stored as a pulseq-zero object with pypulseq."""
    return ev.to_pulseq()

    factory_name = ev._pp_factory
    kwargs = dict(ev._pp_kwargs or {})
    kwargs["flip_angle"] = _n(ev.flip_angle)
    kwargs["phase_offset"] = _n(ev.phase_offset)
    kwargs["freq_offset"] = _n(ev.freq_offset)
    kwargs["delay"] = _n(ev.delay)
    kwargs["system"] = system

    factory = getattr(pypulseq, factory_name)
    sig = inspect.signature(factory)
    if "use" in sig.parameters:
        kwargs["use"] = ev.use or "undefined"
    # Drop any kwargs the installed pypulseq version doesn't accept (e.g.
    # `center_pos` was renamed / removed in some 1.5.x builds).
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    result = factory(**kwargs)
    # return_gz=False by default here (we only want the RF), and the factories
    # that can return a tuple only do so when return_gz=True — which we never
    # set. So `result` is always the RF SimpleNamespace.
    return result


def _translate_trap(ev: TrapGrad, system):
    """Re-create a trap gradient stored as pulseq-zero object with pypulseq."""
    return ev.to_pulseq()

    return pypulseq.make_trapezoid(
        channel=ev.channel,
        amplitude=_n(ev.amplitude),
        rise_time=_n(ev.rise_time),
        flat_time=_n(ev.flat_time),
        fall_time=_n(ev.fall_time),
        delay=_n(ev.delay),
        system=system,
    )


def _translate_free(ev: ExtTrapGrad | ArbitraryGrad, system):
    """Re-create a free gradient stored as pulseq-zero object with pypulseq."""
    return ev.to_pulseq()
    
    tt = np.asarray(_n(ev.tt))
    wf = np.asarray(_n(ev.waveform))
    delay = _n(ev.delay)
    # FreeGrad has two origins:
    #  - make_extended_trapezoid stores tt[0] == 0 (times shifted so tt starts
    #    at 0, and delay holds times[0]).
    #  - make_arbitrary_grad stores tt = (arange + 0.5) * grad_raster_time,
    #    so tt[0] == 0.5 * raster > 0.
    # Route back to the matching PyPulseq factory. This keeps the .seq output
    # byte-identical for scripts that use make_extended_trapezoid explicitly
    # (the TSE demo does this heavily).
    if tt.size > 0 and float(tt[0]) == 0.0:
        return pypulseq.make_extended_trapezoid(
            channel=ev.channel,
            amplitudes=wf,
            times=tt + delay,
            system=system,
        )
    return pypulseq.make_arbitrary_grad(
        channel=ev.channel,
        waveform=wf,
        delay=delay,
        first=_n(ev.first_waveform) if ev.first_waveform is not None else None,
        last=_n(ev.last_waveform) if ev.last_waveform is not None else None,
        system=system,
    )


def _translate_adc(ev: Adc, system):
    """Re-create an adc stored as pulseq-zero object with pypulseq."""
    return pypulseq.make_adc(
        num_samples=int(ev.num_samples),
        dwell=_n(ev.dwell),
        delay=_n(ev.delay),
        freq_offset=_n(ev.freq_offset),
        phase_offset=_n(ev.phase_offset),
        system=system,
    )


def _translate_delay(ev: Delay, system):
    """Re-create a delay stored as pulseq-zero object with pypulseq."""
    return pypulseq.make_delay(_n(ev.delay))


def event_to_pp(ev, system):
    """Re-create an event stored as pulseq-zero object with pypulseq."""
    if isinstance(ev, RfPulse):
        return _translate_pulse(ev, system)
    if isinstance(ev, TrapGrad):
        return _translate_trap(ev, system)
    if isinstance(ev, ExtTrapGrad | ArbitraryGrad):
        return _translate_free(ev, system)
    if isinstance(ev, Adc):
        return _translate_adc(ev, system)
    if isinstance(ev, Delay):
        return _translate_delay(ev, system)
    raise TypeError(
        f"pulseqzero: cannot translate event of type {type(ev).__name__} to PyPulseq"
    )
