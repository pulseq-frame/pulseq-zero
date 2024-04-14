def calc_rf_center(rf):
    # The pulseq function returns (time_center, id_center)
    # We don't generate shapes with sample indices, scritps that rely on the
    # second value currently would not work.
    return rf.t_center, None