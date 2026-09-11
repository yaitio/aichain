"""
tools._permissions — risk classes & permission policy
=====================================================

The governance side of the step boundary (1.4.4). Every tool carries a **risk
class** (data on the tool, like the provider registry is data); a
:class:`PermissionPolicy` maps a risk class to a runtime **decision** the harness
enforces *before* a tool executes — outside the model.

Decisions:

* ``"allow"``   — run the tool.
* ``"approve"`` — do not run the tool until someone outside the model says so.
  The Agent asks its ``approve=`` callable and runs the tool only on a yes.
  **With no approver attached the call is refused**, because the alternative is
  worse in exactly the case this exists for: a policy whose whole content is
  "ask a human" cannot mean "go ahead" when there is no human.
* ``"deny"``    — never run; the tool call still returns a (denial) result, so
  the "every tool call returns a result" invariant holds.

Enforcement is **opt-in**: an ``Agent`` without ``permissions=`` behaves exactly
as before. With a policy attached, the shipped default lets ``read``/``draft``/
``write`` run, gates ``external``/``financial``/``privileged`` behind approval,
and denies ``destructive`` — so existing (unmarked, ``write``) tools keep
working and you opt risky tools into gating by tagging their ``risk``.

**What this module promised and did not do, until 2026-09-11.** Everything
above was written when the module was; the Agent consulted it for ``deny`` and
for nothing else, so ``approve`` fell through to running the tool. A policy
left at its defaults — and the default for an unclassified risk is
``approve`` — therefore gated nothing at all, including ``financial``, while
reading in every docstring and every test as protection. The tests are part of
the story: they asserted what :meth:`PermissionPolicy.decide` *returns* and
never once that the returned decision *happened*. An opinion nobody acts on is
not a policy, and this is the shape that defect takes in a permission layer.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ApprovalRequest:
    """What an approver is shown before the tool runs.

    Enough to decide with, and no more: the tool's own name and risk class,
    the arguments it would run with, and the call id so a request can be
    paired with the event stream and with the result that follows. Deliberately
    not the Tool object — an approver that can reach in and run it is not a
    gate.

    Attributes
    ----------
    tool      : the tool's name.
    risk      : its risk class, which is what the policy actually judged.
    arguments : what it would be called with.
    call_id   : the provider's id for this call, when there is one.
    agent     : the name of the agent asking, which matters once a team is
                delegating — the approver needs to know who wants this.
    """

    tool:      str
    risk:      str
    arguments: dict
    call_id:   str = ""
    agent:     str = ""


@dataclass(frozen=True)
class ApprovalDecision:
    """An approver's answer, when a bare yes/no is not enough.

    Returning ``True`` or ``False`` stays the short form and is what most
    approvers will use. This exists for the other half of the requirement: a
    refusal that says *why*. A UI that shows "the tool was not approved" and
    nothing else has thrown away the only part a person can act on, and the
    reason cannot be recovered afterwards — it lived in the head of whoever
    clicked no.

    Attributes
    ----------
    granted : whether the call may run.
    reason  : one sentence, carried into the event stream and into the
              denial the model is told about.
    """

    granted: bool
    reason:  str = ""

    def __bool__(self) -> bool:
        return bool(self.granted)


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
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
]
