def make_adc(num_samples, duration, delay):
    return Adc(num_samples, duration, delay)


class Adc:
    def __init__(self, num_samples, duration, delay) -> None:
        self.num_samples = num_samples
        self.duration = duration
        self.delay = delay
