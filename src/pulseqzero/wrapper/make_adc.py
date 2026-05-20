from ..events import Scalar, Array, Adc
from typing import Optional
from pypulseq import Opts


def make_adc(
    num_samples: int,
    delay: Scalar = 0.0,
    duration: Scalar = 0.0,
    dwell: Scalar = 0.0,
    freq_offset: Scalar = 0.0,
    phase_offset: Scalar = 0.0,
    system: Optional[Opts] = None,
    freq_ppm: Scalar = 0.0,
    phase_ppm: Scalar = 0.0,
    phase_modulation: Optional[Array] = None,
) -> Adc:
    if phase_modulation is not None and len(phase_modulation) != num_samples:
        raise ValueError(
            "ADC Phase modulation vector must have the same length as the number of samples"
        )

    if (dwell == 0 and duration == 0) or (dwell > 0 and duration > 0):
        raise ValueError("Either dwell or duration must be defined")
    if duration > 0:
        dwell = duration / num_samples

    if system is None:
        system = Opts.default
    delay = max(delay, system.adc_dead_time)

    if phase_modulation is not None:
        raise NotImplementedError(
            "ADC phase modulation is not yet supported by pulseq-zero"
        )

    return Adc(
        num_samples,
        dwell,
        delay,
        freq_offset,
        phase_offset,
        freq_ppm,
        phase_ppm,
        system,
    )
