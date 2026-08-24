"""Run an unmodified PyPulseq script against either backend.

The scripts in ``tests/pypulseq_examples/`` are verbatim copies of the upstream
PyPulseq examples: they say ``import pypulseq as pp``, and nothing in them is
aware of pulseq-zero. ``build(script, backend)`` executes such a script with the
name ``pypulseq`` bound to `backend`, which is the adoption path the README
calls "reuse an unmodified PyPulseq script".

Two forms of import have to be served:

    import pypulseq as pp                        -> ``sys.modules["pypulseq"]``
    from pypulseq.calc_rf_center import ...      -> ``sys.modules["pypulseq.X"]``

The second is what ``write_haste.py`` uses. Pulseq-zero exposes the whole API
from its top-level namespace and has no submodule per function, so the finder
below fabricates ``pypulseq.X`` on demand and resolves attribute access on it
against that flat namespace.

``sys.modules["pypulseq"]`` is a proxy rather than the backend module itself,
because importing ``pypulseq.Sequence.sequence`` makes the import machinery
bind the *submodule* ``Sequence`` on its parent -- which, on the backend
itself, would overwrite the ``Sequence`` class for the rest of the process.
The proxy refuses exactly those assignments and forwards everything else.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from contextlib import contextmanager
from pathlib import Path

EXAMPLES = Path(__file__).parent / "pypulseq_examples"


class _Forwarding(types.ModuleType):
    """A module whose attributes come from `backend`'s flat namespace."""

    def __init__(self, name, backend):
        super().__init__(name)
        self.__dict__["_backend"] = backend

    def __getattr__(self, name):  # only consulted when not in __dict__
        try:
            return getattr(self.__dict__["_backend"], name)
        except AttributeError:
            raise AttributeError(f"{self.__name__} has no attribute {name!r}") from None

    def __setattr__(self, name, value):
        # ``import pypulseq.Sequence.sequence`` binds the fabricated submodule
        # as ``pypulseq.Sequence``; letting that through would shadow the
        # backend's ``Sequence`` class.
        if isinstance(value, types.ModuleType) and hasattr(self.__dict__["_backend"], name):
            return
        self.__dict__[name] = value


class _SubmoduleShim(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fabricate ``pypulseq.<anything>`` from the flat namespace of `backend`."""

    def __init__(self, backend):
        self.backend = backend

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("pypulseq."):
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _Forwarding(spec.name, self.backend)
        module.__path__ = []  # so that ``pypulseq.Sequence.sequence`` resolves
        return module

    def exec_module(self, module):
        pass


@contextmanager
def as_pypulseq(backend):
    """Bind the name ``pypulseq`` to `backend` for the duration of the block."""
    saved = {
        name: mod for name, mod in sys.modules.items()
        if name == "pypulseq" or name.startswith("pypulseq.")
    }
    # ``from pypulseq.make_delay import make_delay`` rebinds the *submodule* as
    # ``pypulseq.make_delay``, overwriting the function that pypulseq's
    # ``__init__`` had bound there -- which breaks pulseq-zero's own internal
    # ``pp.make_delay`` calls for the rest of the process. Snapshot the
    # namespace and put it back afterwards.
    native_pp = saved.get("pypulseq")
    native_attrs = dict(vars(native_pp)) if native_pp is not None else None
    for name in saved:
        del sys.modules[name]

    native = backend.__name__ == "pypulseq"
    finder = None if native else _SubmoduleShim(backend)
    if native:
        sys.modules["pypulseq"] = backend
    else:
        proxy = _Forwarding("pypulseq", backend)
        proxy.__path__ = []
        sys.modules["pypulseq"] = proxy
        sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
        for name in [n for n in sys.modules
                     if n == "pypulseq" or n.startswith("pypulseq.")]:
            del sys.modules[name]
        sys.modules.update(saved)
        if native_attrs is not None:
            native_pp.__dict__.clear()
            native_pp.__dict__.update(native_attrs)


def build(script, backend, **kwargs):
    """Import ``tests/pypulseq_examples/<script>.py`` against `backend` and
    return the ``Sequence`` that its ``main()`` builds."""
    with as_pypulseq(backend):
        sys.path.insert(0, str(EXAMPLES))
        sys.modules.pop(script, None)
        try:
            module = importlib.import_module(script)
            return module.main(plot=False, write_seq=False, **kwargs)
        finally:
            sys.path.remove(str(EXAMPLES))
            sys.modules.pop(script, None)
