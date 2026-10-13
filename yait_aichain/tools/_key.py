"""
tools._key — a tool's API key, resolved for the request being made.

``Model(api_key=callable)`` resolves its key per request from the ambient
``RunContext``. A tool froze whatever it was given: a callable was sent as
``Bearer <function …>`` and answered 401, so a multi-tenant RAG had to build
its tools per request or bill every tenant to one key. ``ApiKey`` is the
model clients' rule for tools: a string is the key, a callable is asked each
time with the current context, and an empty answer is an error rather than a
fall-back to another tenant's key.
"""

from __future__ import annotations


class ApiKey:
    """A descriptor: assign a key or a resolver; read the key for now."""

    def __set_name__(self, owner, name):
        self._slot = "_key_source" + name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        source = obj.__dict__.get(self._slot)
        if not callable(source):
            return source
        from ..clients._base import _clean_key
        from ..state import current
        context = current()
        key = _clean_key(source(context))
        if not key:
            raise ValueError(f"no API key resolved for tenant {getattr(context, 'tenant', None)!r}: "
                             f"the resolver given to {type(obj).__name__}(api_key=…) returned nothing, "
                             "and there is no process-wide key to fall back on")
        return key

    def __set__(self, obj, value):
        if value is not None and not isinstance(value, str) and not callable(value):
            raise TypeError(f"{type(obj).__name__}: api_key is a string or a callable that takes "
                            f"the run's context; got {type(value).__name__}")
        obj.__dict__[self._slot] = value
