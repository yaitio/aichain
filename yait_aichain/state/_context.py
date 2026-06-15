"""
state._context — RunContext
===========================

Non-secret per-request context for a single ``run()`` / ``resume()``. Carries
the tenant, tracing ids, and arbitrary metadata — NOT secrets (API keys stay on
``Model`` / ``Tool``). It is exposed as ``chain.context`` / ``agent.context``
for the duration of the run and is persisted in the run document, so it is
restored on ``resume`` (even in another process).
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
        Arbitrary non-secret data (user id, request id, locale, trace ids …),
        available via ``chain.context`` / ``agent.context`` for logging and
        routing.
    """

    tenant:   "str | None" = None
    metadata: dict          = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Convenience read from ``metadata``."""
        return self.metadata.get(key, default)

    def to_dict(self) -> dict:
        return {"tenant": self.tenant, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: "dict | None") -> "RunContext":
        if not data:
            return cls()
        return cls(tenant=data.get("tenant"),
                   metadata=dict(data.get("metadata") or {}))
