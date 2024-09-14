def convert(from_value, from_unit, gamma, to_unit=""):
    from math import pi

    value_SI = from_value * {
        "Hz/m": 1,
        "mT/m": 1e-3 * gamma,
        "rad/ms/mm": 1e6 / (2 * pi),
        "Hz/m/s": 1,
        "mT/m/ms": gamma,
        "T/m/s": gamma,
        "rad/ms/mm/ms": 1e9 / (2 * pi),
    }[from_unit]

    return value_SI * {
        "Hz/m": 1,
        "mT/m": 1e3 / gamma,
        "rad/ms/mm": 1e-6 * 2 * pi,
        "Hz/m/s": 1,
        "mT/m/ms": 1 / gamma,
        "T/m/s": 1 / gamma,
        "rad/ms/mm/ms": 1e-9 * 2 * pi
    }[to_unit]


class Opts:
    def __init__(
        self,
        adc_dead_time=None,
        adc_raster_time=None,
        block_duration_raster=None,
        gamma=None,
        grad_raster_time=None,
        grad_unit="Hz/m",
        max_grad=None,
        max_slew=None,
        rf_dead_time=None,
        rf_raster_time=None,
        rf_ringdown_time=None,
        rise_time=None,
        slew_unit="Hz/m/s",
        B0=None,
    ):
        def select(a, b):
            if a is not None:
                return a
            else:
                return getattr(self.default, b)
            return a if a is not None else b

        self.gamma = select(gamma, "gamma")
        self.max_grad = select(convert(max_grad, grad_unit, self.gamma, "Hz/m"), "max_grad")
        self.max_slew = select(convert(max_slew, slew_unit, self.gamma, "Hz/m/s"), "max_slew")
        # Rise time seems to overwrite
        if rise_time is not None:
            self.max_slew = self.max_grad / rise_time
        self.adc_dead_time = select(adc_dead_time, "adc_dead_time")
        self.adc_raster_time = select(adc_raster_time, "adc_raster_time")
        self.block_duration_raster = select(block_duration_raster, "block_duration_raster")
        self.rf_dead_time = select(rf_dead_time, "rf_dead_time")
        self.rf_raster_time = select(rf_raster_time, "rf_raster_time")
        self.grad_raster_time = select(grad_raster_time, "grad_raster_time")
        self.rf_ringdown_time = select(rf_ringdown_time, "rf_ringdown_time")
        self.B0 = select(B0, "B0")

    def set_as_default(self):
        from copy import copy
        Opts.default = copy(self)

    @classmethod
    def reset_default(cls):
        cls.default = Opts(
            gamma=42.576e6,
            max_grad=40,
            grad_unit="mT/m",
            max_slew=170,
            slew_unit="T/m/s",
            rf_dead_time=0,
            rf_ringdown_time=0,
            adc_dead_time=0,
            adc_raster_time=100e-9,
            rf_raster_time=1e-6,
            grad_raster_time=10e-6,
            block_duration_raster=10e-6,
            B0=1.5
        )

    def __str__(self) -> str:
        """
        Print a string representation of the system limits objects.
        """
        variables = vars(self)
        s = [f"{key}: {value}" for key, value in variables.items()]
        s = "\n".join(s)
        s = "System limits:\n" + s
        return s


Opts.reset_default()
