"""
yait_aichain.testing — models for tests and offline examples.

`ScriptedModel` replays a script and works wherever a `Model` does;
`recorded()` puts a real `Model` on a canned transport and keeps what it sent.
Neither touches the network. See ``docs/api/testing.md``.
"""

from __future__ import annotations

import contextlib
import json
import threading
from dataclasses import dataclass, field

from .models import Model

__all__ = ["ScriptedModel", "call", "recorded", "Request"]


@dataclass(frozen=True)
class _Call:
    name: str
    arguments: dict


def call(name: str, **arguments) -> _Call:
    """A tool call in a `ScriptedModel`'s script: ``call("refund", amount=42)``.

    Parameters
    ----------
    name : str
        The tool the model calls.
    **arguments
        The arguments it calls it with.
    """
    return _Call(name, dict(arguments))


@dataclass
class Request:
    """One request a model sent.

    For a `ScriptedModel`: ``messages`` and ``tools`` as the loop handed them.
    Under `recorded()`: ``path``, ``body`` and ``headers`` as they would have
    gone on the wire, an auth header as a fingerprint (``sk-…a1b2``).

    Parameters
    ----------
    messages : list
        The universal messages of the turn.
    tools : list
        The tool schemas offered.
    path : str
        The request path.
    body : dict
        The request body.
    headers : dict
        The request headers.
    """
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    path: str = ""
    body: dict = field(default_factory=dict)
    headers: dict = field(default_factory=dict)


def _turn(reply) -> list:
    """A script item as a list of calls, or ``None`` for an answer."""
    if isinstance(reply, _Call):
        return [reply]
    if isinstance(reply, (list, tuple)) and reply and all(isinstance(c, _Call) for c in reply):
        return list(reply)
    return None


class ScriptedModel(Model):
    """
    A model that replays a script, for tests and offline examples.

    Each reply is a string (an answer), a dict (a JSON answer),
    ``call(name, **arguments)`` (a tool call) or a list of calls (several in
    one turn). The last reply repeats if the loop asks for more. It goes
    through the library's own request and response path, so `Skill`,
    `Chain`, `Pool` and `Agent` treat it as any other model — ``run()`` and
    ``stream()`` both, a streamed answer arriving word by word.

    Parameters
    ----------
    replies : list
        The script, one item per model call.
    usage : dict, optional
        Tokens each call reports: ``{"input": 1000, "output": 200}``. Zero when
        not given.
    price : dict, optional
        USD per million tokens: ``{"input": 1.0, "output": 3.0}``. With it,
        each call reports its cost and a budget has something to count;
        without it the cost is ``None``, as for any unpriced model.
    name : str, optional
        The model's name in events and records. Default ``"scripted"``.

    Attributes
    ----------
    requests : list[Request]
        What the loop sent each turn — messages and tool schemas.

    Examples
    --------
    >>> from yait_aichain import Agent
    >>> model = ScriptedModel([call("refund", amount=42), "Refunded."])
    >>> Agent(model, tools=[refund]).run("refund 42").output
    'Refunded.'
    """

    def __init__(self, replies: list, *, usage: dict | None = None,
                 price: dict | None = None, name: str = "scripted") -> None:
        if not isinstance(replies, (list, tuple)) or not replies:
            raise ValueError("replies is a non-empty list of answers and call(...) items")
        if price is not None and not {"input", "output"} <= set(price):
            raise ValueError("price takes {'input': $/M, 'output': $/M}")
        super().__init__(f"private/{name}", api_key="")
        self.replies = list(replies)
        self.usage = {"input": 0, "output": 0, **(usage or {})}
        self.price = dict(price) if price else None
        self.requests: list = []
        self._next = 0
        self._lock = threading.Lock()
        self._turn = threading.local()
        self.client.send = self._send
        self.client._post_sse = self._post_sse

    # What the loop sent, in the universal form it was handed — kept when it
    # is sent, not when it is built: a budget refuses between the two.
    def build(self, messages, output, tools=None):
        self._turn.request = Request(messages=json.loads(json.dumps(messages, default=str)),
                                     tools=list(tools or []))
        return super().build(messages, output, tools)

    def stream(self, messages, output, tools=None):
        self._turn.request = Request(messages=json.loads(json.dumps(messages, default=str)),
                                     tools=list(tools or []))
        return (yield from super().stream(messages, output, tools))

    def _sent(self) -> None:
        request = getattr(self._turn, "request", None) or Request()
        self._turn.request = None
        with self._lock:
            self.requests.append(request)

    def _reply(self):
        with self._lock:
            reply = self.replies[min(self._next, len(self.replies) - 1)]
            self._next += 1
            n = self._next
        return reply, n

    def _usage(self) -> dict:
        u = {"prompt_tokens": self.usage["input"], "completion_tokens": self.usage["output"],
             "total_tokens": self.usage["input"] + self.usage["output"]}
        if self.price:
            u["cost"] = (self.usage["input"] * self.price["input"]
                         + self.usage["output"] * self.price["output"]) / 1_000_000
        return u

    @staticmethod
    def _calls(calls, n) -> list:
        return [{"id": f"call-{n}-{i + 1}", "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for i, c in enumerate(calls)]

    def _send(self, path, body, headers):
        self._sent()
        reply, n = self._reply()
        calls = _turn(reply)
        if calls:
            message = {"role": "assistant", "content": None, "tool_calls": self._calls(calls, n)}
            finish = "tool_calls"
        else:
            text = reply if isinstance(reply, str) else json.dumps(reply)
            message, finish = {"role": "assistant", "content": text}, "stop"
        return json.dumps({"choices": [{"message": message, "finish_reason": finish}],
                           "usage": self._usage()}).encode()

    def _post_sse(self, path, body, headers):
        self._sent()
        reply, n = self._reply()
        calls = _turn(reply)
        if calls:
            for i, c in enumerate(self._calls(calls, n)):
                yield {"choices": [{"delta": {"tool_calls": [{"index": i, **c}]}}]}
            finish = "tool_calls"
        else:
            text = reply if isinstance(reply, str) else json.dumps(reply)
            words = text.split(" ")
            for i, word in enumerate(words):
                yield {"choices": [{"delta": {"content": word + (" " if i < len(words) - 1 else "")}}]}
            finish = "stop"
        yield {"choices": [{"delta": {}, "finish_reason": finish}]}
        yield {"choices": [], "usage": self._usage()}


# ── recorded() ────────────────────────────────────────────────────────────────

_AUTH = ("authorization", "x-api-key", "x-goog-api-key", "api-key")
_USAGE_IN, _USAGE_OUT = 10, 5


def _fingerprint(value: str) -> str:
    scheme, _, secret = value.rpartition(" ")
    if len(secret) < 16:
        hidden = "…"
    else:
        hidden = f"{secret[:3]}…{secret[-4:]}"
    return f"{scheme} {hidden}" if scheme else hidden


def _canned(model, path: str, reply: str) -> dict:
    family = type(model.client).__name__.lower()
    if "anthropic" in family:
        return {"content": [{"type": "text", "text": reply}], "stop_reason": "end_turn",
                "usage": {"input_tokens": _USAGE_IN, "output_tokens": _USAGE_OUT}}
    if "google" in family:
        return {"candidates": [{"content": {"role": "model", "parts": [{"text": reply}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": _USAGE_IN, "candidatesTokenCount": _USAGE_OUT,
                                  "totalTokenCount": _USAGE_IN + _USAGE_OUT}}
    if path.rstrip("/").endswith("/responses"):
        return {"status": "completed",
                "output": [{"type": "message", "role": "assistant",
                            "content": [{"type": "output_text", "text": reply}]}],
                "usage": {"input_tokens": _USAGE_IN, "output_tokens": _USAGE_OUT,
                          "total_tokens": _USAGE_IN + _USAGE_OUT}}
    if "openai" in family or "perplexity" in family or "qwen" in family:
        return {"choices": [{"message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": _USAGE_IN, "completion_tokens": _USAGE_OUT,
                          "total_tokens": _USAGE_IN + _USAGE_OUT}}
    raise NotImplementedError(f"recorded() answers text requests; {type(model.client).__name__} "
                              "is not one of the families it can stand in for")


class _Recording:
    def __init__(self):
        self.requests: list = []


@contextlib.contextmanager
def recorded(model: Model, reply: str = "ok"):
    """
    Put a real `Model` on a canned transport and record what it sends.

    Inside the block every call answers *reply* in the provider's own response
    shape, with 10 input and 5 output tokens, and nothing reaches the network.
    The model keeps its name, options, adaptations and price, so
    ``effective_options``, ``last_adaptations`` and the cost are the library's
    real answers. Buffered calls only; a stream inside the block raises.

    Parameters
    ----------
    model : Model
        The model whose transport is replaced for the duration of the block.
    reply : str, optional
        The text every call answers. Default ``"ok"``.

    Yields
    ------
    An object whose ``requests`` lists one `Request` per call: ``path``,
    ``body`` and ``headers``, an auth header as a fingerprint, never the key.

    Examples
    --------
    >>> model = Model("gpt-5.5", options={"temperature": 0.2})
    >>> with recorded(model) as rec:
    ...     Skill(model, prompt="Hi").run()
    >>> rec.requests[0].body["model"]
    'gpt-5.5'
    """
    rec = _Recording()
    client = model.client
    saved = {k: client.__dict__[k] for k in ("send", "_post_sse") if k in client.__dict__}

    def send(path, body, headers):
        safe = {k: (_fingerprint(v) if k.lower() in _AUTH and isinstance(v, str) else v)
                for k, v in (headers or {}).items()}
        rec.requests.append(Request(path=path, body=json.loads(json.dumps(body, default=str)),
                                    headers=safe))
        return json.dumps(_canned(model, path, reply)).encode()

    def no_stream(path, body, headers):
        raise NotImplementedError("recorded() answers buffered calls; use run(), not stream()")
        yield  # pragma: no cover

    client.send, client._post_sse = send, no_stream
    try:
        yield rec
    finally:
        for k in ("send", "_post_sse"):
            client.__dict__.pop(k, None)
        client.__dict__.update(saved)
