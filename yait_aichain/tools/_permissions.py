"""Risk classes and the permission policy.

Every tool carries a risk class; a ``PermissionPolicy`` maps it to ``allow``,
``approve`` or ``deny``, decided by the harness before the tool runs, outside
the model. See docs/control/permissions.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Risk classes (a tool's ``risk`` attribute) ─────────────────────────────────

READ        = "read"          # pure reads, no side effects
DRAFT       = "draft"         # produces a draft/proposal, commits nothing
WRITE       = "write"         # internal write / state change
EXTERNAL    = "external"      # sends something outside (email, message, post)
FINANCIAL   = "financial"     # moves money: a payment, a refund, a transfer
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
    run_id    : the run asking: one agent serves many runs, and an approver
                answering several at once pairs each answer with its run.
    """

    tool:      str
    risk:      str
    arguments: dict
    call_id:   str = ""
    agent:     str = ""
    run_id:    str = ""


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

    def __init__(self, rules: dict | None = None, *, default: str = APPROVE) -> None:
        # A misspelt class is a rule that matches nothing: `{"destructve":
        # "deny"}` would leave `destructive` at its default.
        unknown = sorted(set(rules or {}) - RISK_CLASSES)
        if unknown:
            raise ValueError(f"PermissionPolicy: {unknown} is not a risk class; the classes are "
                             f"{sorted(RISK_CLASSES)}")
        merged = {**_DEFAULT_RULES, **(rules or {})}
        bad = {d for d in merged.values() if d not in DECISIONS} | (
            {default} if default not in DECISIONS else set()
        )
        if bad:
            raise ValueError(
                f"Invalid permission decision(s) {sorted(bad)}; "
                f"must be one of {sorted(DECISIONS)}."
            )
        # Read-only: the rules are checked here, and a change after the check
        # — `policy.rules["destructive"] = "alow"` — would be a decision no
        # gate knows.
        from types import MappingProxyType
        self.rules   = MappingProxyType(merged)
        self.default = default

    def decide_risk(self, risk: str) -> str:
        """Return the decision for a bare *risk* class string.

        Parameters
        ----------
        risk : str
            A risk class.
        """
        return self.rules.get(risk, self.default)

    def decide(self, tool, arguments: dict | None = None) -> str:
        """Return the decision for one call of *tool*.

        The class is the call's, when the tool can tell: a tool with
        ``risk_for(arguments)`` names the class of this call — ``git status``
        and ``git push`` through one shell were one decision, and an ungated
        shell walked round a gated ``git`` tool. Otherwise the tool's own
        ``risk``.

        Parameters
        ----------
        tool : Tool
            The tool about to run.
        arguments : dict | None, optional
            The call's arguments.
        """
        return self.decide_risk(risk_of(tool, arguments))

    def __repr__(self) -> str:
        return f"PermissionPolicy(rules={dict(self.rules)}, default={self.default!r})"



def risk_of(tool, arguments: dict | None = None) -> str:
    """The risk class of one call of *tool*: what its ``risk_for`` says of
    these arguments, else its ``risk``. A class the tool names that is not a
    class is its own ``risk`` — never a lower one."""
    base = getattr(tool, "risk", WRITE)
    judge = getattr(tool, "risk_for", None)
    if callable(judge) and arguments is not None:
        try:
            risk = judge(arguments)
        except Exception:
            return base
        if risk in RISK_CLASSES:
            return risk
    return base

__all__ = [
    "PermissionPolicy", "ApprovalRequest", "ApprovalDecision", "risk_of",
    "RISK_CLASSES", "DECISIONS",
    "READ", "DRAFT", "WRITE", "EXTERNAL", "FINANCIAL", "DESTRUCTIVE", "PRIVILEGED",
    "ALLOW", "APPROVE", "DENY",
]
