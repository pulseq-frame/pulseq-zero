from typing import Union, Tuple, Optional
import pypulseq as pp
from warnings import warn
from copy import copy

from .. import get_supported_rf_uses, Opts, FREUDENSPRUNG_PTX
from .make_grad import make_trapezoid
from ..events import RfPulse, TrapGrad, Scalar, Array
from . import _n


def make_sinc_pulse(
    flip_angle: Scalar,
    apodization: float = 0.0,
    delay: Scalar = 0.0,
    duration: Scalar = 4e-3,
    dwell: float = 0.0,
    center_pos: float = 0.5,
    freq_offset: Scalar = 0.0,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: Scalar = 0.0,
    return_gz: bool = False,
    slice_thickness: float = 0.0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 4.0,
    use: str = "undefined",
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    # Martins pTx extension
    shim_array: Optional[Array] = None,
) -> Union[
    RfPulse,
    Tuple[RfPulse, TrapGrad, TrapGrad],
]:
    if system is None:
        system = Opts.default

    valid_uses = get_supported_rf_uses()
    if use != "" and use not in valid_uses:
        raise ValueError(
            f"Invalid use parameter. Must be one of {valid_uses}. Passed: {use}"
        )

    if duration <= 0:
        raise ValueError("RF pulse duration must be positive.")

    if delay < system.rf_dead_time:
        warn(
            f"Specified RF delay {delay * 1e6:.2f} us is less than the dead time {system.rf_dead_time * 1e6:.0f} us."
            " Delay was increased to the dead time.",
            stacklevel=2,
        )
        delay = system.rf_dead_time

    rf = RfPulse(
        flip_angle=flip_angle,
        freq_offset=freq_offset,
        phase_offset=phase_offset,
        delay=delay,
        shape_dur=duration,
        ringdown_time=system.rf_ringdown_time,
        use=use,
        shim_array=shim_array,
        # wrapped pypulseq call - stored if needed for writing, plotting, ...
        _pp_factory=lambda self: pp.make_sinc_pulse(
            flip_angle=_n(self.flip_angle),
            apodization=_n(apodization),
            delay=_n(self.delay),
            duration=_n(self.shape_dur),
            dwell=_n(dwell),
            center_pos=_n(center_pos),
            freq_offset=_n(self.freq_offset),
            max_grad=_n(max_grad),
            max_slew=_n(max_slew),
            phase_offset=_n(self.phase_offset),
            return_gz=False,
            slice_thickness=_n(slice_thickness),
            system=system,
            time_bw_product=_n(time_bw_product),
            use=self.use,
            freq_ppm=_n(freq_ppm),
            phase_ppm=_n(phase_ppm),
            **({"shim_array": shim_array} if FREUDENSPRUNG_PTX else {}),
        ),
    )

    if not return_gz:
        return rf
    else:
        if slice_thickness == 0:
            raise ValueError("Slice thickness must be provided")

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew

        bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration

        gz = make_trapezoid(
            channel="z", system=system, flat_time=duration, flat_area=area
        )
        gzr = make_trapezoid(
            channel="z",
            system=system,
            area=-area * (1 - center_pos) - 0.5 * (gz.area - area),
        )

        if rf.delay > gz.rise_time:
            gz.delay = rf.delay - gz.rise_time
        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

        return rf, gz, gzr
