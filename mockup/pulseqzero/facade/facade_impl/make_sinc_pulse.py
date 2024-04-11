from .. import facade_impl as fi

def make_sinc_pulse(flip_angle, system, duration, slice_thickness,
                    apodization, time_bw_product, return_gz):
    BW = time_bw_product / duration
    pulse = SincPulse(flip_angle, 0.0, 0.0, duration)

    if return_gz:
        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = fi.make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = fi.make_trapezoid(channel='z', system=system, area=-0.5 * gz.area)

        return pulse, gz, gzr
    else:
        return pulse


class SincPulse:
    def __init__(self, angle, phase, delay, duration) -> None:
        self.angle = angle
        self.phase = phase
        self.delay = delay
        self.duration = duration
        # Some scripts rely on it but we don't care because we don't simulate it:
        self.freq_offset = 0
