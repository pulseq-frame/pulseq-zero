import pypulseq as pp
from . import facade_impl as fi


Opts = pp.Opts
Sequence = pp.Sequence
make_block_pulse = pp.make_block_pulse
make_sinc_pulse = pp.make_sinc_pulse
make_trapezoid = pp.make_trapezoid
make_extended_trapezoid = pp.make_extended_trapezoid
make_adc = pp.make_adc
make_delay = pp.make_delay
calc_duration = pp.calc_duration
calc_rf_center = pp.calc_rf_center


def use_pypulseq():
    global Opts
    global Sequence
    global make_block_pulse
    global make_sinc_pulse
    global make_trapezoid
    global make_extended_trapezoid
    global make_adc
    global make_delay
    global calc_duration
    global calc_rf_center
    Opts = pp.Opts
    Sequence = pp.Sequence
    make_block_pulse = pp.make_block_pulse
    make_sinc_pulse = pp.make_sinc_pulse
    make_trapezoid = pp.make_trapezoid
    make_extended_trapezoid = pp.make_extended_trapezoid
    make_adc = pp.make_adc
    make_delay = pp.make_delay
    calc_duration = pp.calc_duration
    calc_rf_center = pp.calc_rf_center


def use_pulseqzero():
    global Opts
    global Sequence
    global make_block_pulse
    global make_sinc_pulse
    global make_trapezoid
    global make_extended_trapezoid
    global make_adc
    global make_delay
    global calc_duration
    global calc_rf_center
    Opts = fi.Opts
    Sequence = fi.Sequence
    make_block_pulse = fi.make_block_pulse
    make_sinc_pulse = fi.make_sinc_pulse
    make_trapezoid = fi.make_trapezoid
    # make_extended_trapezoid = fi.make_extended_trapezoid
    make_extended_trapezoid = crash
    make_adc = fi.make_adc
    make_delay = fi.make_delay
    # calc_duration = fi.calc_duration
    calc_duration = crash
    # calc_rf_center = fi.calc_rf_center
    calc_rf_center = crash


# Helper so we can't accidentally miss providing a facade for something
def crash(*args, **kwargs):
    assert False
