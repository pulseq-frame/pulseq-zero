from dataclasses import dataclass


def make_delay(d):
    return Delay(d)


def make_trigger(channel, delay=0, duration=0, system=None):
    return Delay(delay)


@dataclass
class Delay:
    delay: ...
