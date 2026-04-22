# Changelog

All notable changes to pulseq-zero are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

The 1.0 release unifies pulseq-zero into a single, always-on facade. The
mode-switching surface (`pp_impl`, `mr0_mode()`, `is_mr0_mode()`) is gone
— the same script now writes `.seq` files *and* feeds MR-zero, with no
context manager and no mode flag. Targets PyPulseq 1.5.0.post1.

### Breaking changes

- Imports: `import pulseqzero; pp = pulseqzero.pp_impl` → `import pulseqzero as pp`.
- The `with pulseqzero.mr0_mode():` context manager has been removed.
  `seq.to_mr0()` and `seq.write()` both work unconditionally.
- `pulseqzero.is_mr0_mode()`, `pulseqzero.pp_impl`, `use_pypulseq()`,
  `use_pulseqzero()`, `torch_to_numpy`, and `convert_tensor` are gone.
  Unsupported PyPulseq functions now raise `NotImplementedError` with a
  named workaround instead of asking user code to branch on the mode.
- There is no deprecation shim. Scripts migrating from 0.3 need the two
  mechanical edits above (import swap + drop the `mr0_mode` wrapper).

### Added

- `Sequence.write(path)` / `Sequence.to_pypulseq()` on the adapter. Both
  lazily translate the in-memory event graph through PyPulseq and emit
  `warnings.warn(...)` on every call so usage inside a hot optimization
  loop is visible. `seq.to_pypulseq()` is the explicit escape hatch for
  one-off PyPulseq-native calls.
- Forwarding stubs on `Sequence`: `calculate_pns`, `test_report`,
  `paper_plot` all route through `self.to_pypulseq().foo(...)`.
- Clear `NotImplementedError` raisers for PyPulseq entry points we
  don't (yet) wrap: `make_adiabatic_pulse`, `sigpy_n_seq`, `make_slr`,
  `make_sms`, `SigpyPulseOpts`, `align`, `calc_ramp`, `rotate`,
  `points_to_waveform`, `traj_to_grad`. Each message names the function
  and points at the workaround.
- [demo/](demo/): a `uv`-managed workspace member targeting PyPulseq
  1.5.0.post1.
  - [demo/write_tse.py](demo/write_tse.py): TSE sequence ported from the
    PyPulseq 1.5 upstream example. The refocusing flip angles are an
    input (`refoc_flips`, length `n_echo`, radians) and the refocusing
    pulse is rebuilt per echo.
  - [demo/main.py](demo/main.py): end-to-end demo that simulates a
    target image from a fully-180° TSE refocusing train and optimizes
    the refocusing flip-angle vector with an RMS image-fidelity loss
    plus a `sum(flips**2)` SAR penalty.

### Changed

- Pulse-shape generation is delegated to PyPulseq. `make_sinc_pulse`,
  `make_gauss_pulse`, `make_block_pulse`, and `make_arbitrary_rf` now
  call the corresponding `pypulseq.make_*` factory once at construction
  time and store the returned `(t, signal)` tuple on the adapter
  `Pulse`. The hand-rolled envelope math and companion `gz`/`gzr`
  trapezoid construction are gone (~150 LoC deleted).
  - `flip_angle`, `phase_offset`, `freq_offset`, and `delay` stay as
    live torch tensors on `Pulse`; autograd on `flip_angle` is
    re-established in `seq_convert.integrate_pulse` via the
    `window_area / full_area` ratio trick.
  - Pulse-shape parameters (`duration` when used to shape the envelope,
    `time_bw_product`, `apodization`, `center_pos`, `slice_thickness`)
    are no longer differentiable — they flow through PyPulseq as numpy.
    These were never optimized in practice; see README §4 for the full
    differentiability list.
- `pulseqzero.Opts` is now a direct re-export of `pypulseq.Opts`. All
  fields (including 1.5 additions: `B0`, `adc_samples_limit`,
  `adc_samples_divisor`) come along for free.
- `Sequence.check_timing()` is a `(True, [])` stub. Real validation
  happens on `write()` — PyPulseq runs `check_timing` internally. Users
  wanting on-demand validation call `seq.to_pypulseq().check_timing()`
  explicitly. The stub keeps hot-loop callers warning-free.

### Fixed

- `make_sinc_pulse` / `make_gauss_pulse`: the old `generate_shape`
  closures called `float(flip_angle)` on a torch tensor, raising a
  scalar-conversion warning and silently breaking autograd. With the
  PyPulseq delegation in this release the code path is gone entirely.
- `seq_convert.integrate_pulse`: previously funneled the flip through
  numpy (`np.trapz` / `np.interp` / `.tolist()`), which severed
  gradients and triggered `RuntimeError: element 0 of tensors does not
  require grad and does not have a grad_fn` on `loss.backward()`. The
  windowed-flip computation now forms a grad-free `window_area /
  full_area` ratio and multiplies it by the live `rf.flip_angle`
  tensor. Switched the deprecated `np.trapz` to `np.trapezoid` at the
  same time.

### Acceptance

- `uv run demo/main.py` completes 30 Adam iterations end-to-end with
  monotonically-decreasing SAR and non-NaN data loss (matches
  pre-unification behavior).
- `demo/write_tse.py`, run once under `import pulseqzero as pp` and
  once under `import pypulseq as pp`, produces byte-identical `.seq`
  output (after stripping the `[SIGNATURE]` block). The translator
  round-trips.
