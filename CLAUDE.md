# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Pulseq-zero is a thin, always-on facade around PyPulseq. The same sequence script doubles as:
1. a normal PyPulseq script — `seq.write("scan.seq")`, `seq.plot()`, etc.
2. a differentiable sequence definition consumable by MR-zero / MRzeroCore — `seq.to_mr0()` for PDG simulation and gradient-descent optimization of any sequence parameter through PyTorch autograd.

No mode flag, no context manager, no import swap at runtime. The adapter *is* the library; PyPulseq is invoked under the hood for pulse-shape generation at construction time and for `.seq` translation at export time.

Targets **PyPulseq 1.5.0.post1**. No declared runtime dependencies — `pypulseq`, `torch`, `MRzeroCore`, `numpy`, `matplotlib` must already be in the environment.

## Install / run

```bash
# Editable install into a venv that inherits system-site-packages
python -m venv --system-site-packages .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install --editable .
```

There is no test runner, linter, or CI for correctness — `.github/workflows/python-publish.yml` only publishes to PyPI on release. The runnable examples live in [demo/](demo/), a `uv`-managed workspace member:

- `uv run demo/write_tse.py` — build a TSE sequence and write a `.seq` file.
- `uv run demo/main.py` — end-to-end optimization demo that shrinks SAR of the TSE refocusing train while keeping the reconstruction close to a target image. Needs [demo/brain.npz](demo/brain.npz).

`demo/write_tse.py` exposes `main(refoc_flips, plot, write_seq)` so `demo/main.py` imports and re-runs it inside an Adam loop without triggering file writes.

## Architecture

### Public surface ([src/pulseqzero/__init__.py](src/pulseqzero/__init__.py))

A thin module of direct re-exports from the adapter. `import pulseqzero as pp` gives you a drop-in PyPulseq-like module whose events hold torch tensors. There is no `Impl` singleton, no `pp_impl`, no `use_pypulseq()` / `use_pulseqzero()` mode flip — the whole facade is the adapter.

Unsupported-but-plannable PyPulseq entry points raise `NotImplementedError` with a named workaround instead of asking user code to branch. README §4 tracks per-function coverage with ✅/➡️/🚫/⚠️ glyphs.

### The adapter ([src/pulseqzero/adapter/](src/pulseqzero/adapter/))

Each file mirrors a chunk of the PyPulseq API and returns dataclass-style event objects that hold **torch tensors** so gradients flow through:

- `pulses.py` → `Pulse` (+ `make_sinc_pulse` / `make_gauss_pulse` / `make_block_pulse` / `make_arbitrary_rf`). Each factory delegates shape generation to the corresponding `pypulseq.make_*` factory (called with detached scalars), then wraps the returned `(t, signal)` on the `Pulse` as a numpy tuple. Live torch tensors for `flip_angle` / `phase_offset` / `freq_offset` / `delay` stay on the dataclass; pulse-shape parameters (TBW, apodization, slice_thickness, duration-when-used-for-shape) are eagerly numeric.
- `grads.py` → `TrapGrad`, `FreeGrad` (+ `make_trapezoid`, `make_arbitrary_grad`, `make_extended_trapezoid`, `scale_grad`, `split_gradient`, `split_gradient_at`, `add_gradients`).
- `adc.py`, `delay.py`, `opts.py` — remaining event / config types. `opts.py` is a one-line re-export of `pypulseq.Opts` (with a `.default` class attr added for convenience).
- `sequence.py` → `Sequence`. Stores `blocks: list[list[event]]` and `definitions: dict`. `to_mr0()` is the simulation bridge; `to_pypulseq()` / `write()` is the `.seq` export bridge. `to_pypulseq()` emits `warnings.warn(...)` on every call so hot-loop usage is visible.
- `seq_convert.py` — `convert(pp0, samples_offres, samples_slicesel) -> mr0.Sequence`. Walks `blocks`, classifies each block as pulse / adc / spoiler, starts a new MR-zero repetition on every pulse, and analytically integrates gradients across each sub-interval. `integrate_pulse` reads the stashed `rf.shape` numpy tuple and reconnects `flip_angle` autograd through a `window_area / full_area` ratio multiplied by the live tensor. `integrate()` hand-codes the trapezoid area as a Heaviside-gated closed form so autograd differentiates through it.
- `to_pypulseq.py` — per-event translators used by `Sequence.to_pypulseq()`. Pulse translator re-calls the stashed `_pp_factory` with `_pp_kwargs` and overrides mutable fields (`flip_angle`, `phase_offset`, `freq_offset`, `delay`) from the live `Pulse` so post-construction edits are honored. FreeGrad translator picks `make_extended_trapezoid` vs `make_arbitrary_grad` by checking whether the time axis starts at 0.
- `extended_trap_grad.py` — copied verbatim from PyPulseq, **not differentiable** (flagged in README).

### Differentiable rounding ([src/pulseqzero/math.py](src/pulseqzero/math.py))

PyPulseq rounds timings to raster grids, which kills gradients. `pp.round` / `pp.ceil` / `pp.floor` are `torch.autograd.Function` subclasses whose backward pass is the identity — use these on any timing derived from a parameter being optimized. They fall back to numpy for plain Python numbers.

### Differentiability guarantees

Gradients flow end-to-end through: RF `flip_angle` / `phase_offset` / `freq_offset` / `delay`, ADC `phase_offset` / `freq_offset` / `delay` / `dwell`, gradient `amplitude` / `rise_time` / `flat_time` / `fall_time` / `delay`, and block / repetition / TR / TE durations.

Explicitly **not** differentiable (intentional): RF pulse-shape samples and the parameters that feed shape generation (`time_bw_product`, `apodization`, `center_pos`, `slice_thickness`, duration-used-for-shape), arbitrary-gradient waveform samples (only the amplitude scale is differentiable), and `Opts` fields (max_grad, rasters, dead times).

### What the adapter deliberately stubs or raises

- `Sequence.check_timing()` returns `(True, [])` unconditionally. Real timing validation happens inside `write()` (PyPulseq runs its own `check_timing` during export). Users wanting on-demand validation call `seq.to_pypulseq().check_timing()` explicitly. The stub keeps hot-loop callers warning-free (`write_tse.py` calls `seq.check_timing()` on every build).
- `calc_rf_bandwidth`, `calc_rf_center`, `calc_SAR`, `make_label` are numeric stubs — good enough for the TSE demo, not for exotic pipelines.
- `make_adiabatic_pulse`, `sigpy_n_seq`, `make_slr`, `make_sms`, `SigpyPulseOpts`, `align`, `calc_ramp`, `rotate`, `points_to_waveform`, `traj_to_grad` all raise `NotImplementedError` with a named workaround. The escape hatch for any of them is `seq.to_pypulseq()` — returns a native `pypulseq.Sequence` for one-off exotic calls.

Before "restoring" a missing attribute on the adapter, check whether it is deliberately stubbed — README §4 is the source of truth.

### Tensor/numpy boundary rule

Tensors stay live on the adapter event dataclasses. The boundary where torch → numpy happens is **at export time**, inside `adapter/to_pypulseq._n(x)` — tensors are detached before reaching PyPulseq, so gradients are lost at the `write()` / `to_pypulseq()` boundary. That is intentional: `.seq` export is for after optimization has finished. Gradients are meaningful everywhere else — through the adapter events and into `mr0.Sequence` via `to_mr0()`.
