# `pulseqzero.math`: differentiable math helpers

This documents [src/pulseqzero/math.py](src/pulseqzero/math.py), the small set of differentiable
helpers pulseq-zero uses internally (and exposes to users) wherever plain `torch`/`numpy`
would otherwise break the optimization pipeline: rounding operations that would zero out
gradients, and gradient/waveform interpolation that pypulseq itself does with `numpy`.

`ceil`, `floor`, `round`, and `round_half_up` are re-exported at the top level
(`pulseqzero.ceil`, etc. — see the README's "Differentiable rounding" section). `interp` is
an internal helper used by `wrapper/grad_funcs.py` and is not re-exported; import it
explicitly with `from pulseqzero.math import interp` if you need it directly.

---

## Straight-through rounding: `ceil`, `floor`, `round`

* **Forward pass**: behaves exactly like the corresponding `torch`/`numpy` rounding operation.
* **Backward pass**: returns the incoming gradient unchanged (a straight-through estimator) —
  these functions act as the identity for autograd, even though the forward value is discrete.
* **Use case**: rounding a timing (or any other sequence quantity) that is derived from an
  optimized `requires_grad=True` parameter, without killing its gradient.
* Plain Python numbers / `numpy` arrays are also accepted: pulseq-zero falls back to
  `np.ceil` / `np.floor` / `np.round` when the input isn't a tensor autograd can attach to.

```python
import pulseqzero as pp

y = pp.ceil(x)   # differentiable version of torch.ceil
y = pp.floor(x)  # differentiable version of torch.floor
y = pp.round(x)  # differentiable version of torch.round
```

For plain numeric rounding outside of an optimization (nothing needs a gradient through it),
`np.round` / `torch.round` are simpler and fine to use instead.

### `round_half_up(n, decimals=0)`

Differentiable rounding that rounds halves away from zero instead of using `torch.round`'s /
`np.round`'s banker's rounding (round-half-to-even). Implemented as
`floor(abs(n) * 10**decimals + 0.5) / 10**decimals`, so it inherits `floor`'s straight-through
backward pass.

> **Known bug (matches pypulseq):** because the sign is stripped before rounding and never
> restored, `round_half_up` returns the wrong sign for negative inputs (e.g.
> `round_half_up(-2.5)` does not give `-3`). This mirrors a bug in pypulseq's own
> `round_half_up`, kept here intentionally for parity — do not rely on this function for
> negative values.

---

## `interp(x, xp, fp, left=None, right=None, tol=None)`

Autograd-compatible 1D linear interpolation mirroring `numpy.interp` — differentiable in `x`,
`xp`, **and** `fp` (unlike `numpy.interp`/`torch`, which have no autograd support for this at
all). Unlike `ceil`/`floor`/`round` above, this is **not** a straight-through estimator: the
forward value and its gradient both come from the real linear-interpolation formula.

* Linear within `[xp[0], xp[-1]]`; outside that range it returns `left` (for `x < xp[0]`) and
  `right` (for `x > xp[-1]`), defaulting to the edge values `fp[0]` / `fp[-1]` like `numpy`.
  Pass `left=right=0` to zero-fill outside the support.
* `xp` must be sorted ascending.
* `tol` is a slack on the boundary test, in addition to what `numpy.interp` does: the default
  (a few ULP relative to `xp`'s magnitude) lets a query landing exactly on a boundary survive
  float rounding, e.g. when `xp` and `x` are computed in different dtypes. Pass `tol=0` for a
  strict boundary when the query grid is built from the exact `xp`.

This is what lets pulseq-zero superimpose and re-sample gradient waveforms (arbitrary
gradients, `add_gradients`, `split_gradient_at`, ...) while keeping gradients flowing through
both the waveform amplitudes and the timing.
