from dataclasses import dataclass
from ..adapter import Opts


def make_adc(
    num_samples,
    delay=0,
    duration=None,
    dwell=None,
    freq_offset=0,
    phase_offset=0,
    system=None
):
    if (dwell is None) == (duration is None):
        raise ValueError("Either dwell or duration must be defined")

    if dwell is None:
        dwell = duration / num_samples
    if system is None:
        system = Opts.default
    if delay < system.adc_dead_time:
        delay = system.adc_dead_time

    return Adc(num_samples, dwell, delay, freq_offset, phase_offset)


@dataclass
class Adc:
    num_samples: ...
    dwell: ...
    delay: ...
    freq_offset: ...  # ignored by sim
    phase_offset: ...

    @property
    def duration(self):
        return self.delay + self.num_samples * self.dwell
