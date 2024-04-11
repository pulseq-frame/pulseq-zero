class Delay:
    def __init__(self, delay) -> None:
        self.delay = delay


def make_delay(d):
    return Delay(d)
