# Test suite

What is checked, and why each test exists:

| file | question |
| --- | --- |
| `test_export_fidelity.py` | Does a PyPulseq script produce the *same* `.seq` file when run against pulseq-zero? |
| `test_simulation_agreement.py` | Does `seq.to_mr0()` simulate the same signal as writing the `.seq` and importing it with MR-zero? |
| `test_differentiability.py` | Do derivatives reach every parameter documented as differentiable, and are they right? |

The first two are the questions a user of the adapter actually has: the sequence
that is exported to the scanner must be the one PyPulseq would have written, and
the sequence that is optimized must be the one that is exported. They are run
over the sequences in `sequences.py`, which include six **unmodified** upstream
PyPulseq example scripts (see `pypulseq_examples/README.md`), executed against
either backend through the `sys.modules` redirect in `redirect.py`.

Run them:

```bash
uv run --group dev pytest tests
```

The whole suite takes well under a minute; the simulations run on a
12-compartment analytic phantom, and nothing is downloaded.

The suite needs `torch` and `MRzeroCore` in addition to `pypulseq`; a CPU-only
torch build is enough.
