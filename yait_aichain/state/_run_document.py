"""
state._run_document — RunDocument
=================================

A self-contained, JSON-serialisable snapshot of one run. Holds three things:

* ``variables`` — the key-value data, the single source of truth (filled as
  steps produce named outputs);
* ``steps`` — per-step control state: ``status`` (+ optional ``suspend`` and an
  optional observability ``log``); the value of any step's output already lives
  in ``variables``;
* ``definition`` — the serialised scenario (e.g. ``Chain.save`` output) so a run
  can be resumed from this one artifact alone.

Resume continues from the first non-``done`` step; ``done`` steps are never
re-run.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


class StepStatus:
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    SUSPENDED = "suspended"
    FAILED    = "failed"


@dataclass
class RunDocument:
    """One run's durable state (see module docstring)."""

    run_id:     str
    kind:       str                      # "chain" | "agent"
    variables:  dict           = field(default_factory=dict)
    steps:      list[dict]     = field(default_factory=list)
    usage:      dict           = field(default_factory=dict)
    definition: "dict | None"  = None

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    def new(cls, kind: str, step_names: list[str], *,
            variables: "dict | None" = None,
            definition: "dict | None" = None) -> "RunDocument":
        """Start a fresh document: all steps ``pending``, seeded variables."""
        steps = [{"id": i, "name": name, "status": StepStatus.PENDING}
                 for i, name in enumerate(step_names)]
        return cls(
            run_id     = f"run-{uuid.uuid4().hex[:12]}",
            kind       = kind,
            variables  = dict(variables or {}),
            steps      = steps,
            definition = definition,
        )

    # ── status / cursor ──────────────────────────────────────────────
    @property
    def status(self) -> str:
        """Aggregate run status, derived from the step statuses."""
        states = {s["status"] for s in self.steps}
        if StepStatus.FAILED in states:
            return StepStatus.FAILED
        if StepStatus.SUSPENDED in states:
            return StepStatus.SUSPENDED
        if states and states <= {StepStatus.DONE}:
            return StepStatus.DONE
        return StepStatus.RUNNING

    def first_pending(self) -> "int | None":
        """Index of the first step not yet ``done`` (the resume cursor)."""
        for s in self.steps:
            if s["status"] != StepStatus.DONE:
                return s["id"]
        return None

    def step(self, idx: int) -> dict:
        return self.steps[idx]

    def suspended_step(self) -> "dict | None":
        return next((s for s in self.steps
                     if s["status"] == StepStatus.SUSPENDED), None)

    # ── serialisation ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "run_id":     self.run_id,
            "kind":       self.kind,
            "status":     self.status,        # derived, stored for readers
            "variables":  self.variables,
            "steps":      self.steps,
            "usage":      self.usage,
            "definition": self.definition,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunDocument":
        return cls(
            run_id     = data["run_id"],
            kind       = data.get("kind", "chain"),
            variables  = data.get("variables", {}),
            steps      = data.get("steps", []),
            usage      = data.get("usage", {}),
            definition = data.get("definition"),
        )
