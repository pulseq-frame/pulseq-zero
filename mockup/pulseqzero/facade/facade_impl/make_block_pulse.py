def make_block_pulse(flip_angle, duration, system):
    return BlockPulse(flip_angle, 0.0, 0.0, duration)


class BlockPulse:
    def __init__(self, angle, phase, delay, duration) -> None:
        self.angle = angle
        self.phase = phase
        self.delay = delay
        self.duration = duration
    
    @property
    def t_center(self):
        return self.delay + self.duration / 2
