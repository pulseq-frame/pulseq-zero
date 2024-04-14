from .. import facade_impl as fi

def make_sinc_pulse(flip_angle, system, duration, slice_thickness, apodization,
                    time_bw_product, return_gz=False, phase_offset=0.0, use=None,
                    freq_offset=0.0):
    BW = time_bw_product / duration
    pulse = SincPulse(flip_angle, phase_offset, 0.0, duration, freq_offset)

    if return_gz:
        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = fi.make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = fi.make_trapezoid(channel='z', system=system, area=-0.5 * gz.area)

        return pulse, gz, gzr
    else:
        return pulse


class SincPulse:
    def __init__(self, angle, phase_offset, delay, duration, freq_offset) -> None:
        self.angle = angle
        self.phase_offset = phase_offset
        self.delay = delay
        self.duration = duration
        self.freq_offset = freq_offset
    
    @property
    def t_center(self):
        return self.delay + self.duration / 2
