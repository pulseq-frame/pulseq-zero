class ExtendedTrapezoid:
    def __init__(self, channel, amplitudes, times) -> None:
        self.channel = channel
        self.amplitudes = amplitudes
        self.times = times
    
    @property
    def duration(self):
        return self.times[-1] + self.times[1] - self.times[0]


def make_extended_trapezoid(channel, amplitudes, times) -> ExtendedTrapezoid:
    return ExtendedTrapezoid(channel, amplitudes, times)
