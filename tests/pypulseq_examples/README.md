# Vendored PyPulseq example scripts

Verbatim copies of the example scripts shipped with PyPulseq 1.5.0, from
<https://github.com/imr-framework/pypulseq/tree/v1.5.0/examples/scripts>:

| file | upstream |
| --- | --- |
| `write_gre.py` | `examples/scripts/write_gre.py` |
| `write_haste.py` | `examples/scripts/write_haste.py` |
| `write_radial_gre.py` | `examples/scripts/write_radial_gre.py` |
| `write_ute.py` | `examples/scripts/write_ute.py` |
| `write_epi_se_rs.py` | `examples/scripts/write_epi_se_rs.py` |
| `write_tse.py` | `examples/scripts/write_tse.py` |

They are **not modified**: every one of them says `import pypulseq as pp`, and
nothing in them knows that pulseq-zero exists. That is the point of the export
fidelity test — the same unmodified script has to produce the same `.seq` file
whichever module the name `pypulseq` is bound to (see `tests/redirect.py`).
Modifying a script here would weaken the claim; adapt the test instead.

PyPulseq is licensed under the AGPL-3.0, as is pulseq-zero.

`demo/write_tse.py` is a *different*, modified copy of the TSE example (it
exposes the refocusing flip angles for the optimization demo) and is not used
by the tests.
