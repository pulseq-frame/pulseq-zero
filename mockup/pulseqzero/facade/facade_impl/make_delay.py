class Delay:
    def __init__(self, delay) -> None:
        self.delay = delay

    @property
    def duration(self):
        return self.delay


def make_delay(d):
    return Delay(d)
