import pypulseq as pp
use_real_pypulseq = True

def Opts(*args, **kwargs):
    if use_real_pypulseq:
        return pp.Opts(*args, **kwargs)
    else:
        assert False


def Sequence(*args, **kwargs):
    if use_real_pypulseq:
        return pp.Sequence(*args, **kwargs)
    else:
        assert False


def make_block_pulse(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_block_pulse(*args, **kwargs)
    else:
        assert False


def make_sinc_pulse(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_sinc_pulse(*args, **kwargs)
    else:
        assert False


def make_trapezoid(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_trapezoid(*args, **kwargs)
    else:
        assert False


def make_extended_trapezoid(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_extended_trapezoid(*args, **kwargs)
    else:
        assert False


def make_adc(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_adc(*args, **kwargs)
    else:
        assert False


def make_delay(*args, **kwargs):
    if use_real_pypulseq:
        return pp.make_delay(*args, **kwargs)
    else:
        assert False


def calc_duration(*args, **kwargs):
    if use_real_pypulseq:
        return pp.calc_duration(*args, **kwargs)
    else:
        assert False


def calc_rf_center(*args, **kwargs):
    if use_real_pypulseq:
        return pp.calc_rf_center(*args, **kwargs)
    else:
        assert False
