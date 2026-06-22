"""
tools._permissions — risk classes & permission policy
=====================================================

The governance side of the step boundary (1.4.4). Every tool carries a **risk
class** (data on the tool, like the provider registry is data); a
:class:`PermissionPolicy` maps a risk class to a runtime **decision** the harness
enforces *before* a tool executes — outside the model.

Decisions:

* ``"allow"``   — run the tool.
* ``"approve"`` — pause the run for an external approval (reuses suspend/resume,
  exactly like :class:`~yait_aichain.tools.Gate`); resume with
  ``{"approved": True}`` to proceed.
* ``"deny"``    — never run; the tool call still returns a (denial) result, so
  the "every tool call returns a result" invariant holds.

Enforcement is **opt-in**: an ``Agent`` without ``permissions=`` behaves exactly
as before. With a policy attached, the shipped default lets ``read``/``draft``/
``write`` run, gates ``external``/``financial``/``privileged`` behind approval,
and denies ``destructive`` — so existing (unmarked, ``write``) tools keep
working and you opt risky tools into gating by tagging their ``risk``.
"""

from __future__ import annotations

# ── Risk classes (a tool's ``risk`` attribute) ─────────────────────────────────

READ        = "read"          # pure reads, no side effects
DRAFT       = "draft"         # produces a draft/proposal, commits nothing
WRITE       = "write"         # internal write / state change
EXTERNAL    = "external"      # sends something outside (email, message, post)
FINANCIAL   = "financial"     # moves money / incurs charges
DESTRUCTIVE = "destructive"   # deletes / irreversibly overwrites
PRIVILEGED  = "privileged"    # identity / access / security changes

RISK_CLASSES = frozenset({
    READ, DRAFT, WRITE, EXTERNAL, FINANCIAL, DESTRUCTIVE, PRIVILEGED,
})

# ── Decisions ──────────────────────────────────────────────────────────────────

ALLOW   = "allow"
APPROVE = "approve"
DENY    = "deny"

DECISIONS = frozenset({ALLOW, APPROVE, DENY})

# Shipped default: non-breaking for existing (write) tools; gate the risky ones.
_DEFAULT_RULES = {
    READ:        ALLOW,
    DRAFT:       ALLOW,
    WRITE:       ALLOW,
    EXTERNAL:    APPROVE,
    FINANCIAL:   APPROVE,
    PRIVILEGED:  APPROVE,
    DESTRUCTIVE: DENY,
}


class PermissionPolicy:
    """
    Maps a tool's risk class to a runtime decision (``allow``/``approve``/
    ``deny``). The model never decides its own permission — the harness consults
    this policy, which lives outside the model.

    Parameters
    ----------
    rules : dict[str, str] | None
        Overrides merged over the shipped defaults, e.g.
        ``{"external": "allow", "financial": "deny"}``.
    default : str
        Decision for a risk class not present in the rules (default
        ``"approve"`` — unknown/unclassified risk is gated, not silently run).

    Examples
    --------
    >>> policy = PermissionPolicy({"financial": "approve", "destructive": "deny"})
    >>> policy.decide_risk("financial")
    'approve'
    """

    def __init__(self, rules: "dict | None" = None, *, default: str = APPROVE) -> None:
        merged = {**_DEFAULT_RULES, **(rules or {})}
        bad = {d for d in merged.values() if d not in DECISIONS} | (
            {default} if default not in DECISIONS else set()
        )
        if bad:
            raise ValueError(
                f"Invalid permission decision(s) {sorted(bad)}; "
                f"must be one of {sorted(DECISIONS)}."
            )
        self.rules   = merged
        self.default = default

    def decide_risk(self, risk: str) -> str:
        """Return the decision for a bare *risk* class string."""
        return self.rules.get(risk, self.default)

    def decide(self, tool) -> str:
        """Return the decision for *tool* based on its ``risk`` attribute."""
        return self.decide_risk(getattr(tool, "risk", WRITE))

    def __repr__(self) -> str:
        return f"PermissionPolicy(rules={self.rules}, default={self.default!r})"


__all__ = [
    "PermissionPolicy",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
]
