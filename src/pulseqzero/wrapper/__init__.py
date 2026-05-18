"""The meat of pulseq-zero: functions that mimic what's available in pypulseq
but internally store the inputs to forward them either to MR-zero or to
pypulseq. This allows pulseq-zero to provide functions like seq.write() where
the pulseq sequence is built on the fly, as well as seq.to_mr0() where the
sequence is converted to MR-zero while keeping the gradients intact."""
