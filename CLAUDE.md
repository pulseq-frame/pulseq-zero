# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Pulseq-zero is a thin facade that lets the same PyPulseq sequence script double as:
1. a normal PyPulseq script (writes `.seq` files, plots, etc.), and
2. a differentiable sequence definition consumable by MR-zero / MRzeroCore for PDG simulation and gradient-descent optimization of any sequence parameter through PyTorch autograd.

It targets PyPulseq 1.4.2 specifically. It has no declared runtime dependencies — `pypulseq`, `torch`, `MRzeroCore`, `numpy`, `matplotlib` must already be in the environment.

## Install / run

```bash
# Editable install into a venv that inherits system-site-packages
python -m venv --system-site-packages .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
pip install --editable .
```

There is no test runner, linter, or CI for correctness — `.github/workflows/python-publish.yml` only publishes to PyPI on release. The files under [tests/](tests/) are runnable example scripts, not a pytest suite:

- `python tests/write_gre.py` / `python tests/write_epi.py` — build a sequence and write a `.seq` file.
- `python tests/optimize.py` — end-to-end optimization demo (needs `tests/quantalized.npz` phantom).

Sequence scripts expose a `main(...)` function with `plot` and `write_seq` flags so [tests/optimize.py](tests/optimize.py) can import and re-run them in mr0 mode without triggering file writes.

## Architecture

### The mode switch ([src/pulseqzero/__init__.py](src/pulseqzero/__init__.py))

The entire public surface is a single module-level singleton `pp_impl = Impl()`. User code does `pp = pulseqzero.pp_impl` and then treats `pp` exactly like the `pypulseq` module.

`Impl` has **two population methods** that rebind the same set of attributes (`Sequence`, `make_sinc_pulse`, `calc_duration`, …):

- `use_pypulseq()` — forwards every attribute to real `pypulseq`, wrapped in `torch_to_numpy` so torch tensors get `.detach().cpu().numpy()`-converted at the boundary. Also defines a `WrappedSequence(pp.Sequence)` subclass that wraps `add_block` / `set_definition` for the same reason.
- `use_pulseqzero()` — forwards every attribute to the differentiable reimplementation in [src/pulseqzero/adapter/](src/pulseqzero/adapter/). Sets `self.mr0_mode = True`.

The `mr0_mode()` context manager flips to `use_pulseqzero()` on enter and back to `use_pypulseq()` on exit. `is_mr0_mode()` exposes the flag so user scripts can branch on features that don't exist in mr0 mode (`calculate_pns`, sigpy, etc.).

Consequence: there is no inheritance and no abstract base. Keeping the two modes in sync is manual — when adding a pypulseq function, edit both `use_pypulseq` and `use_pulseqzero` (or leave the latter commented out, matching the pattern for unsupported functions). README §4 tracks coverage.

### The adapter ([src/pulseqzero/adapter/](src/pulseqzero/adapter/))

Each file mirrors a chunk of the pypulseq API and returns dataclass-style event objects that hold **torch tensors** (so gradients flow through):

- `pulses.py` → `Pulse` (+ `make_sinc_pulse` / `make_gauss_pulse` / `make_block_pulse` / `make_arbitrary_rf`). Pulses carry a `_generate_shape()` closure, not a precomputed waveform.
- `grads.py` → `TrapGrad`, `FreeGrad` (+ the `make_trapezoid`, `make_arbitrary_grad`, `make_extended_trapezoid`, `scale_grad`, `split_gradient` factories).
- `adc.py`, `delay.py`, `opts.py` — the remaining event / config types.
- `sequence.py` → `Sequence`, the mr0-mode replacement. Stores `blocks: list[list[event]]`. Its `.to_mr0()` method is the central bridge.
- `seq_convert.py` — `convert(pp0, samples_offres, samples_slicesel) -> mr0.Sequence`. Walks `blocks`, classifies each block as pulse / adc / spoiler, starts a new MR-zero repetition on every pulse, and analytically integrates gradients across each sub-interval. The `integrate()` function hand-codes the trapezoid area as a Heaviside-gated closed form so autograd can differentiate through it.
- `extended_trap_grad.py` — copied verbatim from pypulseq, **not differentiable** (flagged in README).

### Differentiable rounding ([src/pulseqzero/math.py](src/pulseqzero/math.py))

PyPulseq rounds timings to raster grids, which kills gradients. `pp.round` / `pp.ceil` / `pp.floor` are `torch.autograd.Function` subclasses whose backward pass is the identity — use these (not `torch.round` or `np.ceil`) on any timing derived from a parameter being optimized. They fall back to numpy for plain Python numbers.

### What mr0 mode intentionally drops

The adapter is **not** a full pypulseq reimplementation. It aims to cover everything needed for simulation/optimization and little more. README §4 lists every pypulseq entry point and its support status. Common omissions: pulse waveforms aren't generated eagerly (`calc_rf_bandwidth`, `calc_rf_center` return stubs), `check_timing` always passes, `write()` warns and does nothing, sigpy/adiabatic pulses raise. When touching the adapter, check whether the missing attribute is deliberately stubbed (see the "Disabled" / "Altered behaviour" tables in the README) before restoring it.

### Tensor/numpy boundary rule

In pypulseq mode, every call goes through `torch_to_numpy` — tensors are detached before reaching pypulseq, so any gradient is lost at the boundary. That is intentional: pypulseq mode is for final `.seq` export, after optimization has finished. Gradients are only meaningful inside a `with mr0_mode():` block, where values stay as torch tensors end-to-end through the adapter and into `mr0.Sequence`.
