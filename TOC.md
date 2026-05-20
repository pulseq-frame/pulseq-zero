# TOC: pypulseq 1.5.0.post1

This file contains all functions / classes / methods re-exported by pypulseq at
the module level. Sorted by file structure, annotated with the location of the
corresponding pulseq-rs wrapper. This is a table of contents and a checklist. We
do not aim to mimic the exact sub-module structure.

Methods are marked with one of the following:
- [ ] still missing from pulseq-zero
- [x] completed - has differentiable wrapper or doesn't need it
- [?] WIP: should be differentiable but currently just re-export
- [-] knowingly dropped from pulseq-zero (sigpy features)

---

Additional information, to be integrated into the list below:
- missing functions new in 1.5:
  `make_soft_delay`, `Sequence.apply_soft_delay`
  `Opts.set_as_default()`
- how to implement `__version__`?
- support for new pulseq attributes:
  `freq_ppm`, `phase_ppma`, `center`, `phase_modulation`, `oversampling`, `Opt.adc_samples_limit/divisor` `use`
  new function default values

---

## dir(pypulseq)
- [x] Opts                             **re-exported**
- [ ] Sequence
  - [ ] adc_times
  - [ ] add_block
  - [ ] calculate_gradient_spectrum
  - [ ] calculate_kspace
  - [ ] calculate_kspacePP
  - [ ] calculate_pns
  - [ ] check_timing
  - [ ] duration
  - [ ] evaluate_labels
  - [ ] find_block_by_time
  - [ ] flip_grad_axis
  - [ ] get_block
  - [ ] get_definition
  - [ ] get_extension_type_ID
  - [ ] get_extension_type_string
  - [ ] get_gradients
  - [ ] get_raw_block_content_IDs
  - [ ] install
  - [ ] mod_grad_axis
  - [ ] paper_plot
  - [ ] plot
  - [ ] read
  - [ ] register_adc_event
  - [ ] register_grad_event
  - [ ] register_label_event
  - [ ] register_rf_event
  - [ ] register_soft_delay_event
  - [ ] remove_duplicates
  - [ ] rf_from_lib_data
  - [ ] rf_times
  - [ ] set_block
  - [ ] set_definition
  - [ ] set_extension_string_ID
  - [ ] test_report
  - [ ] version_major
  - [ ] version_minor
  - [ ] version_revision
  - [ ] waveforms
  - [ ] waveforms_and_times
  - [ ] waveforms_export
  - [ ] write
- [-] SigpyPulseOpts                   **no sigpy support**
- [ ] add_gradients
- [ ] align
- [x] calc_SAR                         **error deprecated**
- [x] calc_adc_segments                **re-exported**
- [x] calc_duration                     *helpers.py*
- [ ] calc_ramp
- [ ] calc_rf_bandwidth
- [x] calc_rf_center                    *helpers.py*
- [ ] check_timing
- [-] disable_trace                    **no trace support**
- [-] enable_trace                     **no trace support**
- [x] eps                              **re-exported**
- [x] get_supported_labels             **re-exported**
- [x] get_supported_rf_uses            **re-exported**
- [x] make_adc                          *make_adc.py*
- [ ] make_adiabatic_pulse
- [x] make_arbitrary_grad               *make_grad.py*
- [x] make_arbitrary_rf                 *make_pulse.py*
- [x] make_block_pulse                  *make_pulse.py*
- [x] make_delay                        *make_basic.py*
- [x] make_digital_output_pulse         *make_basic.py* **acts as delay in sim**
- [x] make_extended_trapezoid           *make_grad.py*
- [ ] make_extended_trapezoid_area
- [x] make_gauss_pulse                  *make_pulse.py*
- [x] make_label                        *make_basic.py*
- [x] make_sinc_pulse                   *make_pulse.py*
- [-] make_slr                         **no sigpy support**
- [-] make_sms                         **no sigpy support**
- [x] make_trapezoid                    *make_grad.py*
- [x] make_trigger                      *make_basic.py* **acts as delay in sim**
- [x] points_to_waveform                *grad_funcs.py*
- [ ] rotate
- [x] round_half_up                     *math.py*
- [x] scale_grad                        *grad_funcs.py*
- [-] sigpy_n_seq                      **no sigpy support**
- [ ] split_gradient
- [ ] split_gradient_at
- [ ] traj_to_grad
- utils module
  - [ ] cumsum
  - [ ] safe_pns_prediction
  - [-] tracing                        **no trace support**
- pulseq-zero math
  - [x] ceil
  - [x] floor
  - [x] round
  - [x] interp
