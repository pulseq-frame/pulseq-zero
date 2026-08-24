"""The same script, run against pypulseq and against pulseq-zero, must write
the same ``.seq`` file.

This is the property the whole adapter rests on: whatever an optimization does
with the sequence, what reaches the scanner is what PyPulseq would have
written. Comparison is byte-wise over the whole file with the ``[SIGNATURE]``
block removed, which is a hash over the remainder and would only report the
same difference twice.
"""

import pytest

import pulseqzero
import pypulseq

from sequences import ALL


def strip_signature(text):
    idx = text.find("[SIGNATURE]")
    return text if idx < 0 else text[:idx]


def first_difference(a, b):
    """A line number and the two lines, for a readable failure message."""
    lines_a, lines_b = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(lines_a, lines_b)):
        if x != y:
            return f"line {i + 1}:\n  pypulseq:   {x}\n  pulseqzero: {y}"
    return f"identical up to line {min(len(lines_a), len(lines_b))}, " \
           f"then {len(lines_a)} vs {len(lines_b)} lines"


@pytest.mark.parametrize("name", sorted(ALL))
def test_seq_file_is_identical(name, tmp_path):
    builder = ALL[name]

    reference = tmp_path / f"{name}_pypulseq.seq"
    written = tmp_path / f"{name}_pulseqzero.seq"
    builder(pypulseq).write(str(reference))
    builder(pulseqzero).write(str(written))

    a = strip_signature(reference.read_text())
    b = strip_signature(written.read_text())
    assert a == b, first_difference(a, b)


@pytest.mark.parametrize("name", sorted(ALL))
def test_seq_file_is_not_empty(name, tmp_path):
    """Guards the test above: an exporter that writes nothing would pass it."""
    path = tmp_path / f"{name}.seq"
    ALL[name](pulseqzero).write(str(path))
    text = path.read_text()
    assert "[BLOCKS]" in text
    assert len(strip_signature(text)) > 1000
