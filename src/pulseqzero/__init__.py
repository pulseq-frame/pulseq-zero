from contextlib import contextmanager

from .math import ceil, floor, round


class Impl:
    def __init__(self):
        self.use_pypulseq()

        # https://gitlab.cs.fau.de/mrzero/pypulseq_rfshim
        # Make pre-defined shims available if rfshim pulseq is installed
        try:
            import pypulseq.set_tx_mode
            self.set_tx_mode = pypulseq.set_tx_mode
        except ImportError:
            pass

    def use_pypulseq(self):
        import pypulseq as pp
        self.mr0_mode = False
        self.calc_SAR = pp.calc_SAR
        self.Sequence = pp.Sequence
        self.add_gradients = pp.add_gradients
        self.align = pp.align
        self.calc_duration = pp.calc_duration
        self.calc_ramp = pp.calc_ramp
        self.calc_rf_bandwidth = pp.calc_rf_bandwidth
        self.calc_rf_center = pp.calc_rf_center
        self.make_adc = pp.make_adc
        self.make_adiabatic_pulse = pp.make_adiabatic_pulse
        try:  # Fix for pypulseq 1.4.2 which doesn't re-export this
            self.make_arbitrary_grad = pp.make_arbitrary_grad
        except ImportError:
            from pypulseq.make_arbitrary_grad import make_arbitrary_grad
            self.make_arbitrary_grad = make_arbitrary_grad
        self.make_arbitrary_rf = pp.make_arbitrary_rf
        self.make_block_pulse = pp.make_block_pulse
        # These were exported by pypulseq 1.4.2 but not by 1.4.2post1 anymore.
        # Since pulseq-zero has no equivalent anyways, we remove the imports.
        # self.sigpy_n_seq = pp.sigpy_n_seq
        # self.make_slr = pp.make_slr
        # self.make_sms = pp.make_sms
        self.make_delay = pp.make_delay
        self.make_digital_output_pulse = pp.make_digital_output_pulse
        self.make_extended_trapezoid = pp.make_extended_trapezoid
        self.make_extended_trapezoid_area = pp.make_extended_trapezoid_area
        self.make_gauss_pulse = pp.make_gauss_pulse
        self.make_label = pp.make_label
        self.make_sinc_pulse = pp.make_sinc_pulse
        self.make_trapezoid = pp.make_trapezoid
        self.SigpyPulseOpts = pp.SigpyPulseOpts
        self.make_trigger = pp.make_trigger
        self.Opts = pp.Opts
        self.points_to_waveform = pp.points_to_waveform
        self.rotate = pp.rotate
        self.scale_grad = pp.scale_grad
        self.split_gradient = pp.split_gradient
        self.split_gradient_at = pp.split_gradient_at
        self.get_supported_labels = pp.get_supported_labels
        self.traj_to_grad = pp.traj_to_grad

    def use_pulseqzero(self):
        from . import adapter as ad
        self.mr0_mode = True
        self.calc_SAR = ad.calc_SAR
        self.Sequence = ad.Sequence
        self.add_gradients = ad.add_gradients
        # self.align = ad.align
        self.calc_duration = ad.calc_duration
        # self.calc_ramp = ad.calc_ramp
        self.calc_rf_bandwidth = ad.calc_rf_bandwidth
        self.calc_rf_center = ad.calc_rf_center
        self.make_adc = ad.make_adc
        # self.make_adiabatic_pulse = ad.make_adiabatic_pulse
        self.make_arbitrary_grad = ad.make_arbitrary_grad
        self.make_arbitrary_rf = ad.make_arbitrary_rf
        self.make_block_pulse = ad.make_block_pulse
        # self.sigpy_n_seq = ad.sigpy_n_seq
        # self.make_slr = ad.make_slr
        # self.make_sms = ad.make_sms
        self.make_delay = ad.make_delay
        self.make_digital_output_pulse = ad.make_digital_output_pulse
        self.make_extended_trapezoid = ad.make_extended_trapezoid
        self.make_extended_trapezoid_area = ad.make_extended_trapezoid_area
        self.make_gauss_pulse = ad.make_gauss_pulse
        self.make_label = ad.make_label
        self.make_sinc_pulse = ad.make_sinc_pulse
        self.make_trapezoid = ad.make_trapezoid
        # self.SigpyPulseOpts = ad.SigpyPulseOpts
        self.make_trigger = ad.make_trigger
        self.Opts = ad.Opts
        # self.points_to_waveform = ad.points_to_waveform
        # self.rotate = ad.rotate
        self.scale_grad = ad.scale_grad
        self.split_gradient = ad.split_gradient
        # self.split_gradient_at = ad.split_gradient_at
        self.get_supported_labels = ad.get_supported_labels
        # self.traj_to_grad = ad.traj_to_grad


pp_impl = Impl()


@contextmanager
def mr0_mode():
    pp_impl.use_pulseqzero()
    try:
        yield
    finally:
        pp_impl.use_pypulseq()


def is_mr0_mode() -> bool:
    return pp_impl.mr0_mode
