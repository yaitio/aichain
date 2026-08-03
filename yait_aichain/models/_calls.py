"""
models._calls
=============

Native tool calling — the value objects and the two conversation turns.

A model given tool declarations answers in one of two ways: text (the answer)
or a request to call tools. This module is the typed form of the second, plus
the helpers that put a call and its result back into the universal message
list, so a conversation with tools stays expressible in the same
``{"role", "parts"}`` schema the rest of the library speaks.

Two turn kinds extend that schema:

* an ``assistant`` turn may carry ``tool_calls`` — the model asked to act::

      {"role": "assistant", "parts": [...optional text...],
       "tool_calls": [{"id": "...", "name": "...", "arguments": {...}}]}

* a ``tool`` turn carries one call's result, keyed to the call id::

      {"role": "tool", "call_id": "...", "parts": [{"type": "text", "text": ...}]}

Providers require results to reference the call id in their own wire format —
a result faked as a user turn ("TOOL RESULT: ...") is not part of the native
protocol and models treat it as conversation, not as ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    """One requested call: which tool, with which arguments."""

    id:        str
    name:      str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallRequest:
    """
    A model reply that asks for tool calls.

    Returned by ``from_response`` (and, from phase 2 on, by ``Skill.run``)
    *instead of* a string when the model chose to act. ``text`` carries any
    prose the model sent alongside — some providers emit both.

    Providers may request **several** calls in one turn. Honouring all of them
    is deliberate: it is free parallelism, and executing only the first would
    silently narrow the channel this feature exists to widen.
    """

    calls: "tuple[ToolCall, ...]"
    text:  str = ""

    def __bool__(self) -> bool:
        return bool(self.calls)

    def as_turn(self) -> dict:
        """This reply as a universal assistant turn, for appending."""
        turn: dict = {
            "role": "assistant",
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in self.calls
            ],
        }
        if self.text:
            turn["parts"] = [{"type": "text", "text": self.text}]
        return turn


def tool_result_turn(call_id: str, result) -> dict:
    """One call's result as a universal ``tool`` turn."""
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False, default=str)
    return {"role": "tool", "call_id": call_id,
            "parts": [{"type": "text", "text": result}]}
