from dataclasses import dataclass


def make_delay(d):
    return Delay(d)


def make_trigger(channel, delay=0, duration=0, system=None):
    return Delay(delay)


def make_digital_output_pulse(channel, delay=0, duration=None, system=None):
    return Delay(delay)


@dataclass
class Delay:
    delay: ...

    @property
    def duration(self):
        return self.delay
