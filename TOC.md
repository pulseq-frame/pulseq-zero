# TOC: pypulseq 1.5.0.post1

This file contains all functions / classes / methods re-exported by pypulseq at
the module level. Sorted by file structure, annotated with the location of the
corresponding pulseq-rs wrapper. This is a table of contents and a checklist. We
do not aim to mimic the exact sub-module structure.

Methods are marked with one of the following:
- [ ] still missing from pulseq-zero
- [x] completed - has differentiable wrapper or doesn't need it
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
- [ ] Sequence                          *sequence.py*
  - [ ] adc_times
  - [x] add_block                      **native**
  - [ ] calculate_gradient_spectrum
  - [ ] calculate_kspace
  - [ ] calculate_kspacePP
  - [x] calculate_pns                  **re-export**
  - [x] check_timing                   **re-export**
  - [x] duration                       **native**
  - [ ] evaluate_labels
  - [ ] find_block_by_time
  - [ ] flip_grad_axis
  - [ ] get_block
  - [x] get_definition                 **native**
  - [ ] get_extension_type_ID
  - [ ] get_extension_type_string
  - [ ] get_gradients
  - [ ] get_raw_block_content_IDs
  - [ ] install
  - [ ] mod_grad_axis
  - [x] paper_plot                     **re-export**
  - [x] plot                           **re-export**
  - [-] read                           **no .seq read support**
  - [ ] register_adc_event
  - [ ] register_grad_event
  - [ ] register_label_event
  - [ ] register_rf_event
  - [ ] register_soft_delay_event
  - [x] remove_duplicates              **native**
  - [ ] rf_from_lib_data
  - [ ] rf_times
  - [ ] set_block
  - [x] set_definition                 **native**
  - [ ] set_extension_string_ID
  - [x] test_report                    **re-export**
  - [ ] version_major
  - [ ] version_minor
  - [ ] version_revision
  - [ ] waveforms
  - [ ] waveforms_and_times
  - [ ] waveforms_export
  - [x] write                          **re-export**
- [-] SigpyPulseOpts                   **no sigpy support**
- [x] add_gradients                     *grad_funcs.py*
- [x] align                             *helpers.py*
- [-] calc_SAR                         **deprecated**
- [x] calc_adc_segments                **re-exported**
- [x] calc_duration                     *helpers.py*
- [ ] calc_ramp
- [x] calc_rf_bandwidth                 *helpers.py* **non-differentiable re-export**
- [x] calc_rf_center                    *helpers.py*
- [-] disable_trace                    **no trace support**
- [-] enable_trace                     **no trace support**
- [x] eps                              **re-exported**
- [x] get_supported_labels             **re-exported**
- [x] get_supported_rf_uses            **re-exported**
- [x] make_adc                          *make_adc.py*
- [-] make_adiabatic_pulse             **no adiabatic support**
- [x] make_arbitrary_grad               *make_grad.py*
- [x] make_arbitrary_rf                 *make_pulse.py*
- [x] make_block_pulse                  *make_pulse.py*
- [x] make_delay                        *make_basic.py*
- [x] make_digital_output_pulse         *make_basic.py* **acts as delay in mr0**
- [x] make_extended_trapezoid           *make_grad.py*
- [x] make_extended_trapezoid_area      *make_grad.py*
- [x] make_gauss_pulse                  *make_pulse.py*
- [x] make_label                        *make_basic.py*
- [x] make_sinc_pulse                   *make_pulse.py*
- [-] make_slr                         **no sigpy support**
- [-] make_sms                         **no sigpy support**
- [x] make_soft_delay                   *make_basic.py*
- [x] make_trapezoid                    *make_grad.py*
- [x] make_trigger                      *make_basic.py* **acts as delay in mr0**
- [x] points_to_waveform                *grad_funcs.py*
- [ ] rotate
- [x] round_half_up                     *math.py*
- [x] scale_grad                        *grad_funcs.py*
- [-] sigpy_n_seq                      **no sigpy support**
- [x] split_gradient                    *grad_funcs.py*
- [x] split_gradient_at                 *grad_funcs.py*
- [x] traj_to_grad                      *helpers.py*
- pulseq-zero math
  - [x] ceil
  - [x] floor
  - [x] round
  - [x] interp
