# TOC: pypulseq 1.5.0.post1

This file contains all functions / classes / methods re-exported by pypulseq at
the module level. Sorted by file structure, annotated with the location of the
corresponding pulseq-rs wrapper. This is a table of contents and a checklist.

---

## pypulseq/
__init__.py
- [/] eps                                    (module-level constant ~1e-9)
- [/] round_half_up
add_gradients.py
- [/] add_gradients
add_ramps.py
- [ ] add_ramps
align.py
- [/] align                                  (stub: NotImplementedError)
block_to_events.py
- [ ] block_to_events
calc_duration.py
- [/] calc_duration
calc_ramp.py
- [/] calc_ramp                              (stub: NotImplementedError)
calc_rf_bandwidth.py
- [/] calc_rf_bandwidth                      (numeric stub, returns 0)
calc_rf_center.py
- [/] calc_rf_center                         (numeric stub)
check_timing.py
- [ ] check_timing
- [ ] format_string
- [ ] indent_string
- [ ] print_error_report
compress_shape.py
- [ ] compress_shape
convert.py
- [ ] convert
decompress_shape.py
- [ ] decompress_shape
event_lib.py
- [ ] class EventLibrary
make_adc.py
- [/] make_adc
- [/] calc_adc_segments                      (direct pypulseq re-export)
make_adiabatic_pulse.py
- [/] make_adiabatic_pulse                   (stub: NotImplementedError)
make_arbitrary_grad.py
- [/] make_arbitrary_grad
make_arbitrary_rf.py
- [/] make_arbitrary_rf
make_block_pulse.py
- [/] make_block_pulse
make_delay.py
- [/] make_delay
make_digital_output_pulse.py
- [/] make_digital_output_pulse
make_extended_trapezoid.py
- [/] make_extended_trapezoid
make_extended_trapezoid_area.py
- [/] make_extended_trapezoid_area           (copied from pypulseq, not differentiable)
make_gauss_pulse.py
- [/] make_gauss_pulse
make_label.py
- [/] make_label                             (silent no-op stub)
make_sigpy_pulse.py
- [/] sigpy_n_seq                            (stub: NotImplementedError)
- [/] make_slr                               (stub: NotImplementedError)
- [/] make_sms                               (stub: NotImplementedError)
make_sinc_pulse.py
- [/] make_sinc_pulse
make_soft_delay.py
- [/] make_soft_delay                        (stub: NotImplementedError)
make_trapezoid.py
- [ ] calculate_shortest_params_for_area
- [ ] calculate_shortest_rise_time
- [/] make_trapezoid
make_trigger.py
- [/] make_trigger
opts.py
- [/] class Opts                             (direct pypulseq re-export)
points_to_waveform.py
- [/] points_to_waveform                     (stub: NotImplementedError)
rotate.py
- [/] rotate                                 (stub: NotImplementedError)
scale_grad.py
- [/] scale_grad
sigpy_pulse_opts.py
- [/] class SigpyPulseOpts                   (stub: NotImplementedError)
split_gradient.py
- [/] split_gradient
split_gradient_at.py
- [/] split_gradient_at
supported_labels_rf_use.py
- [/] get_supported_labels
- [ ] get_supported_rf_uses
traj_to_grad.py
- [/] traj_to_grad                           (stub: NotImplementedError)

## pypulseq/SAR/ 
__init__.py                                  (empty)
SAR_calc.py
- [/] calc_SAR                               (silent no-op stub)

## pypulseq/Sequence/
__init__.py                                  (empty)
block.py                                     (methods on Sequence)
- [ ] set_block
- [ ] get_raw_block_content_IDs
- [ ] get_block
- [ ] register_adc_event
- [ ] register_control_event
- [ ] register_grad_event
- [ ] register_label_event
- [ ] register_soft_delay_event
- [ ] register_rf_event
calc_grad_spectrum.py
- [ ] calculate_gradient_spectrum
calc_pns.py
- [ ] calc_pns
ext_test_report.py
- [ ] ext_test_report
install.py
- [ ] class ScannerDefinition
- [ ] class SiemensDefinition
- [ ] register_scanner
- [ ] detect_scanner
- [ ] silent_call
parula.py
- [ ] main                                   (CLI helper / colormap generator)
read_seq.py
- [ ] read                                   (.seq -> Sequence; round-trip ingestion)
sequence.py
- [/] class Sequence                         (pulseqzero has its own adapter Sequence)
write_seq.py
- [ ] write                                  (Sequence -> .seq; goes through to_pypulseq())
- [ ] write_v141

## pypulseq/utils/
__init__.py                                  (empty)
cumsum.py
- [ ] cumsum
paper_plot.py
- [ ] paper_plot
safe_pns_prediction.py
- [ ] safe_example_hw
- [ ] safe_example_gwf
- [ ] safe_hw_check
- [ ] safe_longest_time_const
- [ ] safe_pns_model
- [ ] safe_tau_lowpass
- [ ] safe_gwf_to_pns
- [ ] safe_plot
- [ ] safe_example
seq_plot.py
- [ ] class SeqPlot
tracing.py
- [ ] trace_enabled
- [/] enable_trace                           (stub: NotImplementedError)
- [/] disable_trace                          (stub: NotImplementedError)
- [ ] trace
- [ ] format_trace

## pypulseq/utils/siemens/
__init__.py                                  (empty)
asc_to_hw.py
- [ ] asc_to_acoustic_resonances
- [ ] asc_to_hw
readasc.py
- [ ] readasc