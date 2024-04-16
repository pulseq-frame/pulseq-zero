def make_adc(num_samples, duration, delay, phase_offset=0):
    return Adc(num_samples, duration, delay, phase_offset)


class Adc:
    def __init__(self, num_samples, duration, delay, phase_offset) -> None:
        self.num_samples = num_samples
        self.dwell = duration / num_samples
        self.delay = delay
        self.phase_offset = phase_offset
    
    @property
    def duration(self):
        return self.delay + self.num_samples * self.dwell
