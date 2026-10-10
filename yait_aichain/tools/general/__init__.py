"""
tools.general
=============

Tools that belong to no provider: ``wait`` parks a run until a signal arrives,
``gate`` holds another tool until its call is approved. The classes behind
them are :class:`Wait` and :class:`Gate`.
"""

from ._wait import Gate, Wait


def wait(reason: str = "", resume_with: "dict | None" = None, *, name: str = "wait",
         description: "str | None" = None, hint: "dict | None" = None) -> Wait:
    """
    A pause: the run parks here until ``resume(run_id, signal=…)``.

    ::

        Chain(steps=[(draft, "post"), (wait(reason="Approve the post?"), "approval"), (publish, "done")])
    """
    from .._callable import named
    return named(Wait(reason, resume_with, name=name, hint=hint), None, description)


def gate(tool, *, reason: str = "", resume_with: "dict | None" = None, decision_key: str = "approved",
         name: "str | None" = None, description: "str | None" = None, hint: "dict | None" = None) -> Gate:
    """
    *tool*, held until its call is approved: the run parks with the call, and
    the action runs only on a signal whose *decision_key* is true.

    ::

        Agent(model, tools=[gate(publish, reason="Publish this?")], store=FileStore("runs/"))
    """
    from .._callable import named
    return named(Gate(tool, reason=reason, resume_with=resume_with, decision_key=decision_key,
                      name=name, hint=hint), None, description)


__all__ = ["wait", "gate", "Wait", "Gate"]
