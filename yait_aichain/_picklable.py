"""
An exception that survives ``pickle`` and ``copy``.

Python rebuilds an exception by calling its class with ``self.args``, which
fails for the library's exceptions: they take fields in ``__init__`` and pass
one formatted message up. Rebuilt from its message and fields instead, without
calling ``__init__`` again, so a process pool, Celery or Ray hands back the
error that happened rather than a ``TypeError`` from the unpickler.
"""

from __future__ import annotations


def _restore(cls: type, args: tuple, state: dict) -> BaseException:
    exc = cls.__new__(cls)
    BaseException.__init__(exc, *args)
    exc.__dict__.update(state)
    return exc


class Picklable:
    """Mixin for an exception whose ``__init__`` does not take its ``args``."""

    def __reduce__(self):
        return _restore, (type(self), self.args, dict(self.__dict__))
