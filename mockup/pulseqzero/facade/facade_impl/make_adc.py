def make_adc(num_samples, duration, delay):
    return Adc(num_samples, duration, delay)


class Adc:
    def __init__(self, num_samples, duration, delay) -> None:
        self.num_samples = num_samples
        self.dwell = duration / num_samples
        self.delay = delay
    
    @property
    def duration(self):
        return self.delay + self.num_samples * self.dwell
