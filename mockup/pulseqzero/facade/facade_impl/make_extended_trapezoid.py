import torch

class ExtendedTrapezoid:
    def __init__(self, channel, amplitudes: torch.Tensor, times: torch.Tensor) -> None:
        self.channel = channel
        self.amplitudes = amplitudes
        self.times = times
    
    @property
    def duration(self):
        return self.times[-1] + self.times[1] - self.times[0]
    
    @property
    def area(self):
        # Areas under segments equals areas of rects with average amplitude
        amps = (self.amplitudes[1:] + self.amplitudes[:-1]) / 2
        durs = self.times[1:] - self.times[:-1]

        return (amps * durs).sum()


def make_extended_trapezoid(channel, amplitudes, times) -> ExtendedTrapezoid:
    return ExtendedTrapezoid(channel, amplitudes, times)
