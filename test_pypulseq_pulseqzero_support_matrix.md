# Pulseq-zero adapter completeness matrix

Built and probed with `pypulseq` and `pulseqzero` against the README
section 4 coverage table. `pulseqzero` is invoked as a single drop-in
module (no `pp_impl`, no `mr0_mode()`).

api entry | pypulseq | pulseqzero | expected (pulseqzero)
--- | --- | --- | ---
Opts | OK | OK | ok
Sequence.__init__ | OK | OK | ok
Sequence.__str__ | OK | OK | ok
Sequence.add_block | OK | OK | ok
Sequence.check_timing | OK | OK | ok
Sequence.duration | OK | OK | ok
Sequence.remove_duplicates | missing | OK | ok
Sequence.set_definition/get_definition | OK | OK | ok
Sequence.test_report | ERROR | ERROR <-- MISMATCH | ok
Sequence.to_mr0 | missing | OK | ok
Sequence.to_pypulseq | missing | OK | ok
Sequence.write | OK | OK | ok
SigpyPulseOpts | OK | NotImpl | not_implemented
add_gradients | OK | OK | ok
align | OK | OK | ok
calc_SAR | missing | NotImpl | not_implemented
calc_duration | OK | OK | ok
calc_ramp | OK | OK | ok
calc_rf_bandwidth (stub) | OK | OK | ok
calc_rf_center (stub) | OK | OK | ok
disable_trace | missing | NotImpl | not_implemented
enable_trace | missing | NotImpl | not_implemented
get_supported_labels | OK | OK | ok
make_adc | OK | OK | ok
make_adiabatic_pulse | ERROR | NotImpl | not_implemented
make_arbitrary_grad | OK | OK | ok
make_arbitrary_rf | OK | OK | ok
make_block_pulse | OK | OK | ok
make_delay | OK | OK | ok
make_digital_output_pulse | OK | OK | ok
make_extended_trapezoid | OK | OK | ok
make_extended_trapezoid_area | ERROR | OK | ok
make_gauss_pulse | OK | OK | ok
make_label | OK | OK | ok
make_sinc_pulse | OK | OK | ok
make_slr | ERROR | NotImpl | not_implemented
make_sms | ERROR | NotImpl | not_implemented
make_soft_delay | missing | OK | ok
make_trapezoid | OK | OK | ok
make_trigger | OK | OK | ok
points_to_waveform | OK | OK | ok
rotate | ERROR | OK | ok
scale_grad | OK | OK | ok
sigpy_n_seq | ERROR | NotImpl | not_implemented
split_gradient | missing | OK | ok
split_gradient_at | OK | OK | ok
traj_to_grad | OK | OK | ok
