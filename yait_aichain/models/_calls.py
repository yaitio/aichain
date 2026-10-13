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
    """One requested call: which tool, with which arguments.

    Attributes
    ----------
    id : str
        The provider's id for this call; a tool result answers it by this id.
    name : str
        The tool's name.
    arguments : dict
        The arguments, parsed from the provider's JSON.
    """

    id:        str
    name:      str
    # Out of the hash: a dict has none, and a frozen call whose hash raised
    # could not be put in a set or used as a key.
    arguments: dict = field(default_factory=dict, hash=False)


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

    Attributes
    ----------
    calls : tuple[ToolCall, ...]
        Every call the model asked for in this turn, in order.
    text : str
        Prose the model sent beside them, if any.
    opaque : dict
        What a provider needs back verbatim on the next turn, keyed by the
        provider family — Anthropic's thinking blocks, which must open the
        assistant turn before a tool result when extended thinking is on.
        Not for reading; carried so the history stays one the provider takes.
    """

    calls: tuple[ToolCall, ...]
    text:  str = ""
    opaque: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        # Every call gets an id, and no two share one. harmony carries none,
        # Google none on most models, and a result is matched to its call by
        # id alone — two calls answering to one id, or to "", leave a result
        # nobody can place and a dangling call nobody notices.
        seen: set = set()
        fixed = []
        for i, call in enumerate(self.calls):
            base = call.id or f"call_{i}"
            new, n = base, 1
            while new in seen:
                n += 1
                new = f"{base}_{n}"
            seen.add(new)
            fixed.append(call if new == call.id else ToolCall(new, call.name, call.arguments))
        object.__setattr__(self, "calls", tuple(fixed))

    def __bool__(self) -> bool:
        return bool(self.calls)

    @classmethod
    def from_turn(cls, turn: dict) -> ToolCallRequest:
        """The reply an assistant turn holds — the inverse of `as_turn`, for a
        driver that kept the conversation as JSON between invocations and has
        to act on its last turn.

        Parameters
        ----------
        turn : dict
            An assistant turn with ``tool_calls``, as ``as_turn`` wrote it.
        """
        calls = tuple(ToolCall(id=c.get("id", ""), name=c["name"], arguments=dict(c.get("arguments") or {}))
                      for c in turn.get("tool_calls") or [])
        text = "".join(p.get("text", "") for p in turn.get("parts") or [] if p.get("type") == "text")
        return cls(calls=calls, text=text, opaque=dict(turn.get("opaque") or {}))

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
        if self.opaque:
            turn["opaque"] = dict(self.opaque)
        return turn


#: Part types a tool may hand back instead of text. Anything else is data and
#: is serialised, as it always was.
MEDIA_PARTS = frozenset({"image", "audio", "video", "document"})


#: The sources a tool result may carry media in: the bytes, or a URL. Not a
#: local path — any server's JSON shaped like ``{"kind": "file", "path": …}``
#: had the library read that path on the next turn and send it to the model,
#: ``/proc/self/environ`` included. A tool that has a file reads it once, with
#: `media_part`, and hands over the bytes.
TOOL_MEDIA_SOURCES = frozenset({"base64", "url"})


def _is_part(value) -> bool:
    return (isinstance(value, dict)
            and value.get("type") in MEDIA_PARTS
            and isinstance(value.get("source"), dict)
            and value["source"].get("kind") in TOOL_MEDIA_SOURCES)


#: The largest file `media_part` reads: past it a model would refuse the
#: part anyway, and reading it is how a tool exhausts a Lambda's memory.
MEDIA_PART_MAX_BYTES = 20 * 1024 * 1024


def media_part(path: str, mime: str | None = None) -> dict:
    """A file as a media part a tool can return: read once, now, as base64.

    The type follows the MIME type — ``image``, ``audio``, ``video``, else
    ``document``. Self-contained, so the history it enters means the same
    thing in another process and after the file changes.

    Parameters
    ----------
    path : str
        The file.
    mime : str | None, optional
        Its MIME type; guessed from the name and the first bytes when omitted.
    """
    import base64
    import pathlib

    from ..skills._adapters import _guess_mime

    file = pathlib.Path(path)
    if not file.is_file():
        raise ValueError(f"media_part needs a regular file; {path!r} is not one")
    size = file.stat().st_size
    if size > MEDIA_PART_MAX_BYTES:
        raise ValueError(f"{path!r} is {size:,} bytes; a media part takes at most "
                         f"{MEDIA_PART_MAX_BYTES:,}")
    raw = file.read_bytes()
    mime = mime or _guess_mime(file, raw[:16])
    kind = mime.split("/", 1)[0]
    return {"type": kind if kind in ("image", "audio", "video") else "document",
            "source": {"kind": "base64", "mime": mime,
                       "data": base64.b64encode(raw).decode("ascii")}}


def carries_media(result) -> bool:
    """True when a tool result is, or contains, a media part."""
    return _is_part(result) or (isinstance(result, (list, tuple))
                                and any(_is_part(v) for v in result))


def split_media_result(msg: dict) -> tuple[dict, dict | None]:
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


def dangling_calls(messages: list) -> list:
    """Ids of tool calls in *messages* that no later turn answers.

    A conversation in which the model asked for a tool and never heard back
    is malformed, and providers disagree about it in the worst possible way:
    OpenAI's chat completions, the Responses API and Anthropic all reject the
    request; Google keys results by name and is looser; a self-hosted
    OpenAI-compatible server — vLLM, llama.cpp, mlx — validates nothing at
    all and templates whatever it was handed, so the model meets its own
    unanswered call and improvises. It repeats it, apologises, or invents the
    result, differently each time.

    That is run-to-run variance manufactured by us, and it is worst where it
    is least visible: the condition that produces a dangling call — a tool
    that raised, a result lost across an invocation boundary — is itself
    intermittent, so the same input behaves differently on different trials.

    ``run()`` cannot produce one: it appends a result turn for every call in
    a reply, failed ones included. The externally driven seam can, because
    ``step()`` hands the obligation to the caller and nothing checked that
    they met it — and that seam is precisely what a benchmark harness and a
    serverless driver use, which is where a result crosses a process boundary
    and can be lost.

    Parameters
    ----------
    messages : list
        A conversation, as ``Agent.step()`` takes it.
    """
    # Counted, not collected into a set: one id asked twice and answered once
    # is a call left without its result, and a set saw both as answered.
    # Walked in order, so a result answers a call made before it.
    from collections import Counter
    owed: Counter = Counter()
    order: list = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = (call.get("id") if isinstance(call, dict) else None) or ""
                owed[call_id] += 1
                order.append(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("call_id") or ""
            if owed[call_id] > 0:
                owed[call_id] -= 1
    dangling: list = []
    for call_id in reversed(order):              # the latest calls are the unanswered ones
        if owed[call_id] > 0:
            owed[call_id] -= 1
            dangling.append(call_id)
    return list(reversed(dangling))


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

    Parameters
    ----------
    call_id : str
        The ``ToolCall.id`` this result answers.
    result : any
        What the tool returned; a non-string is rendered as text.
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
