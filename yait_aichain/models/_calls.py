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


#: Part types a tool may hand back instead of text. Anything else is data and
#: is serialised, as it always was.
MEDIA_PARTS = frozenset({"image", "audio", "video", "document"})


def _is_part(value) -> bool:
    return (isinstance(value, dict)
            and value.get("type") in MEDIA_PARTS
            and isinstance(value.get("source"), dict))


def carries_media(result) -> bool:
    """True when a tool result is, or contains, a media part."""
    return _is_part(result) or (isinstance(result, (list, tuple))
                                and any(_is_part(v) for v in result))


def split_media_result(msg: dict) -> "tuple[dict, dict | None]":
    """
    Split a tool turn into the part a text-only provider accepts and the rest.

    Returns ``(tool_turn, follow_up)``. Only Anthropic takes media inside a
    tool result; OpenAI and Google require the result to be text, so the media
    is sent as a user message immediately after it — the model still sees the
    image on the same turn, one message later. ``follow_up`` is ``None`` when
    there was no media to move, and the tool turn is then returned unchanged.
    """
    parts = msg.get("parts") or []
    media = [p for p in parts if p.get("type") in MEDIA_PARTS]
    if not media:
        return msg, None

    text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "text")
    kinds = ", ".join(sorted({p["type"] for p in media}))
    tool_turn = dict(msg)
    tool_turn["parts"] = [{"type": "text",
                           "text": (text + "\n" if text else "")
                                   + f"[{len(media)} {kinds} attachment(s) follow "
                                     "in the next message]"}]
    # The follow-up carries a line of text as well as the media, naming the
    # call it came from: a message that is nothing but an image arrives with
    # no stated relation to the conversation. Whether the caption also makes
    # the image easier to attend to is not established — measured on
    # gpt-4o-mini it moved 2 of 6 to 3 of 6, which on six samples is noise.
    label = {"type": "text",
             "text": f"Result of the `{msg.get('call_id') or 'tool'}` call:"}
    return tool_turn, {"role": "user", "parts": [label, *media]}


def tool_result_turn(call_id: str, result) -> dict:
    """
    One call's result as a universal ``tool`` turn.

    A result is normally text, and anything that is not a string is
    serialised. The exception is media: a tool that returns a media part — or
    a list mixing media parts and strings — hands those through as parts
    instead of describing them as JSON. Without this an agent that renders
    something cannot look at what it rendered, and a vision loop has to be
    lifted out of the agent into a separate Skill.

    Providers disagree about whether a tool result may carry an image, so the
    parts are carried here and each family decides on the wire: Anthropic puts
    them inside the tool result, the others send them as a following user
    message.
    """
    if _is_part(result):
        return {"role": "tool", "call_id": call_id, "parts": [result]}

    if isinstance(result, (list, tuple)) and any(_is_part(v) for v in result):
        parts = [v if _is_part(v)
                 else {"type": "text",
                       "text": v if isinstance(v, str)
                       else json.dumps(v, ensure_ascii=False, default=str)}
                 for v in result]
        return {"role": "tool", "call_id": call_id, "parts": parts}

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False, default=str)
    return {"role": "tool", "call_id": call_id,
            "parts": [{"type": "text", "text": result}]}
