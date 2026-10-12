"""
An exception that survives ``pickle`` and ``copy``.

Python rebuilds an exception by calling its class with ``self.args``. The
library's exceptions take their fields in ``__init__`` and pass one formatted
message up, so that call either fails (``BudgetExceeded("budget exhausted…")``
is missing two arguments) or formats the message a second time. Through a
process pool, Celery or Ray the caller then receives a ``TypeError`` from the
unpickler instead of the error that happened. Rebuilt from its message and its
fields instead, without calling ``__init__`` again.
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
