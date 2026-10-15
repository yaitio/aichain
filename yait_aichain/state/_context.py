"""
state._context — RunContext, the non-secret per-request context of one run.

Persisted in the run document, so it is restored on ``resume`` in another
process; API keys never go in it. See docs/production.md.
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

    tenant:   str | None = None
    metadata: dict          = field(default_factory=dict)

    def get(self, key: str, default=None):
        """Convenience read from ``metadata``.

        Parameters
        ----------
        key : str
            A key of ``metadata``.
        default : any, optional
            Returned when it is absent.
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> dict:
        return {"tenant": self.tenant, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict | None) -> RunContext:
        if not data:
            return cls()
        return cls(tenant=data.get("tenant"),
                   metadata=dict(data.get("metadata") or {}))
