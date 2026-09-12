"""
state._current — which run is in flight, reachable without threading it through.

`RunContext` carries the tenant and the request's metadata. Getting it to the
place that needs it is the awkward part: the code that needs a tenant's API
key is a client's `_auth_headers`, four layers below the `Chain.run()` that
knows the tenant, and passing it down by parameter would mean a `context=` on
`Skill`, on `Pool`, on every `Tool` — the environment leaking into the
scenario, which `VISION.md` names as the signal that the design has gone
wrong. The scenario is what you write to test a hypothesis; it must not grow a
parameter because production has tenants.

So the context is ambient for the duration of a run, and `contextvars` rather
than a thread-local: a `ContextVar` is copied into a task when
`ThreadPoolExecutor` is handed a copied context, which is how `Pool` keeps
each item under the same run — and, unlike a thread-local, it does not leak
into an unrelated thread that happens to be reused.

**Nothing secret goes in here.** `RunContext` is serialised into the run
document and `FileStore` writes that to disk, so a tenant's key placed in it
would be written out in plaintext. Secrets are resolved *from* the context by
a callable the caller supplies — see `Model(api_key=...)` — and never stored
in it.
"""

from __future__ import annotations

import contextvars

_CURRENT: contextvars.ContextVar = contextvars.ContextVar(
    "yait_aichain_run_context", default=None)


def current():
    """The `RunContext` of the run in flight, or None outside one."""
    return _CURRENT.get()


class using:
    """Make *context* the current one for the duration of a block.

    ::

        with using(RunContext(tenant="acme")):
            chain.run()
    """

    def __init__(self, context) -> None:
        self._context = context
        self._token = None

    def __enter__(self):
        self._token = _CURRENT.set(self._context)
        return self._context

    def __exit__(self, *exc) -> bool:
        # Reset by token rather than to None: runs nest — an Agent inside a
        # Chain step — and clearing on the way out of the inner one would
        # leave the outer run without its own tenant for the rest of its life.
        _CURRENT.reset(self._token)
        return False
