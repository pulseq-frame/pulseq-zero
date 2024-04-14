def calc_duration(*events):
    return max(ev.duration for ev in events)
