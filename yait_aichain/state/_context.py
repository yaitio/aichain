"""
state._context — RunContext
===========================

Non-secret per-request context, injected into a single ``run()`` and threaded
down to steps and tools. Carries the tenant, tracing ids, and arbitrary
metadata — NOT secrets. API keys stay on ``Model`` / ``Tool`` (``api_key=`` or
env), supplied when the scenario is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RunContext:
    """
    Per-request context for one ``run()`` / ``resume()``.

    Parameters
    ----------
    tenant : str | None
        Logical tenant / account this run belongs to.
    metadata : dict
        Arbitrary non-secret data (user id, request id, locale, trace ids …)
        available to steps and tools for logging and routing.
    """

    tenant:   "str | None" = None
    metadata: dict          = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Convenience read from ``metadata``."""
        return self.metadata.get(key, default)
