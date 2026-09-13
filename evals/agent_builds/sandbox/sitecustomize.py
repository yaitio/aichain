"""
The sandbox a model-written program runs in: no network, a record of what it built.

Loaded by Python at start-up because its directory is first on PYTHONPATH.
Every `Model` gets a client whose `send` returns an empty body and whose
`from_response` is scripted, so a program that builds requests the real way —
options checked, tools declared, schemas validated — never reaches a provider.
What the program constructed and called is written to `$AICHAIN_TRACE` at exit,
and the task's check reads that record rather than trusting what it printed.

Scripted replies:
  * tools offered and the last turn is not a tool result → one tool call, to
    the tool whose name contains `$AICHAIN_STUB_PREFER`, else the first one;
    arguments filled from its schema;
  * `json_schema` → an object with every property filled; `json` → {"result": …};
  * otherwise text `STUB-<n>: <start of the last user turn>`, so a check can
    see one step's output arrive in the next step's prompt.
"""

import atexit
import functools
import itertools
import json
import os
import sys

TRACE_PATH = os.environ.get("AICHAIN_TRACE")
PREFER = os.environ.get("AICHAIN_STUB_PREFER", "")
BUILTIN_TOOLS = {"pool", "delegate", "write_plan"}

trace = {"models": [], "calls": [], "skills": [], "chains": [], "pools": [],
         "agents": [], "tool_runs": [], "approvals": [], "methods": {},
         "risks": {}, "errors": []}
_counter = itertools.count(1)


def _count(name):
    trace["methods"][name] = trace["methods"].get(name, 0) + 1


def _internal(depth=2):
    try:
        return sys._getframe(depth).f_globals.get("__name__", "").startswith("yait_aichain.")
    except ValueError:
        return False


def _text_of(messages):
    out = []
    for m in messages or []:
        for p in (m.get("parts") or []) if isinstance(m, dict) else []:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                out.append(p.get("text", ""))
    return "\n".join(out)


def _last_user_text(messages):
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return _text_of([m])
    return ""


def _fill(schema, n):
    kind = (schema or {}).get("type")
    if "enum" in (schema or {}):
        return schema["enum"][0]
    if kind == "string":
        return f"STUB-{n}"
    if kind == "integer":
        return 3
    if kind == "number":
        return 3.0
    if kind == "boolean":
        return True
    if kind == "array":
        return [_fill(schema.get("items", {"type": "string"}), n)]
    if kind == "object":
        props = schema.get("properties") or {}
        return {k: _fill(v, n) for k, v in props.items()}
    return f"STUB-{n}"


def _arguments(parameters):
    props = (parameters or {}).get("properties") or {}
    required = set((parameters or {}).get("required") or [])
    chosen = [k for k in props if k in required or k == "input"] or list(props)[:1]
    return {k: _fill(props[k], 0) if props[k].get("type") != "string" else "3"
            for k in chosen}


def _install():
    import yait_aichain.models._base as mb
    from yait_aichain.models._calls import ToolCall, ToolCallRequest
    from yait_aichain.skills._skill import Skill
    from yait_aichain.chain._chain import Chain
    from yait_aichain.pool._pool import Pool
    from yait_aichain.agent._agent import Agent
    from yait_aichain.tools._base import Tool

    # ── Model: real request building, scripted transport ──────────────────
    init = mb.Model.__init__

    @functools.wraps(init)
    def model_init(self, *a, **k):
        init(self, *a, **k)
        trace["models"].append({"name": self.name, "provider": getattr(self, "_provider", None)})
        self.client.send = lambda path, body, headers: b"{}"

    mb.Model.__init__ = model_init

    to_request = mb.Model.to_request

    @functools.wraps(to_request)
    def model_to_request(self, messages, output, tools=None):
        result = to_request(self, messages, output, tools=tools) if tools else to_request(self, messages, output)
        names = [((t.get("function") or t).get("name")) for t in (tools or [])]
        last = messages[-1].get("role") if messages and isinstance(messages[-1], dict) else None
        self._stub = {"tools": tools or [], "last_role": last, "messages": messages}
        trace["calls"].append({"model": self.name, "provider": getattr(self, "_provider", None),
                               "tools": names, "last_role": last,
                               "format": ((output or {}).get("format") or {}).get("type"),
                               "text": _text_of(messages)[-3000:]})
        return result

    mb.Model.to_request = model_to_request

    def _reply(self, output):
        stub = getattr(self, "_stub", {}) or {}
        n = next(_counter)
        tools = [t for t in stub.get("tools", [])
                 if ((t.get("function") or t).get("name")) not in BUILTIN_TOOLS]
        if tools and stub.get("last_role") != "tool":
            pick = next((t for t in tools if PREFER and PREFER in (t.get("function") or t).get("name", "")), tools[0])
            fn = pick.get("function") or pick
            reply = ToolCallRequest(calls=(ToolCall(id=f"call-{n}", name=fn["name"],
                                                    arguments=_arguments(fn.get("parameters"))),))
            kind = "tool_call"
        else:
            fmt = (output or {}).get("format") or {}
            if fmt.get("type") == "json_schema":
                reply = _fill(dict(fmt.get("schema") or {}, type="object"), n)
                kind = "json"
            elif fmt.get("type") == "json":
                reply = {"result": f"STUB-{n}"}
                kind = "json"
            else:
                reply = f"STUB-{n}: {_last_user_text(stub.get('messages'))[:60]}"
                kind = "text"
        if trace["calls"]:
            trace["calls"][-1]["reply_kind"] = kind
            trace["calls"][-1]["reply"] = reply if isinstance(reply, (str, dict)) else "tool_call"
        return reply

    mb.Model.from_response = lambda self, response, output: _reply(self, output)

    def model_stream(self, messages, output, tools=None):
        model_to_request(self, messages, output, tools=tools)
        self.last_stream_usage = None
        reply = _reply(self, output)
        self.last_stream_result = reply
        if isinstance(reply, str):
            yield reply
            self.last_stream_result = None

    mb.Model.stream = model_stream

    # ── What the program built ────────────────────────────────────────────
    s_init = Skill.__init__

    @functools.wraps(s_init)
    def skill_init(self, *a, **k):
        internal = _internal()
        s_init(self, *a, **k)
        models = self.model if isinstance(getattr(self, "model", None), list) else [getattr(self, "model", None)]
        trace["skills"].append({
            "internal": internal, "prompt": k.get("prompt") is not None,
            "models": [getattr(m, "name", None) for m in models],
            "messages": len((getattr(self, "_input", {}) or {}).get("messages", [])),
            "roles": [m.get("role") for m in (getattr(self, "_input", {}) or {}).get("messages", [])],
            "format": ((getattr(self, "_output", {}) or {}).get("format") or {}).get("type"),
            "max_cost": getattr(self, "max_cost", None) is not None})

    Skill.__init__ = skill_init

    c_init = Chain.__init__

    @functools.wraps(c_init)
    def chain_init(self, *a, **k):
        c_init(self, *a, **k)
        trace["chains"].append({
            "kinds": [st[3] for st in self._steps],
            "runners": [type(st[0]).__name__ for st in self._steps],
            "keys": [st[1] for st in self._steps],
            "on_step_error": self.on_step_error,
            "store": type(self._store).__name__, "hooks": len(self.hooks),
            "max_cost": self.max_cost is not None})

    Chain.__init__ = chain_init

    p_init = Pool.__init__

    @functools.wraps(p_init)
    def pool_init(self, *a, **k):
        p_init(self, *a, **k)
        trace["pools"].append({"items": len(self._items), "max_flows": self._max_flows,
                               "on_error": self._on_error,
                               "runner": type(self._runner).__name__,
                               "max_cost": self.max_cost is not None})

    Pool.__init__ = pool_init

    a_init = Agent.__init__

    @functools.wraps(a_init)
    def agent_init(self, *a, **k):
        internal = _internal()
        a_init(self, *a, **k)
        trace["agents"].append({
            "internal": internal, "model": getattr(self.model, "name", None),
            "tools": [getattr(t, "name", type(t).__name__) for t in self.tools],
            "stop_when": [{"kind": getattr(c, "kind", None), "name": getattr(c, "name", None),
                           "spec": getattr(c, "spec", None)} for c in self.stop_when],
            "mode": self.mode, "permissions": self.permissions is not None,
            "approve": self.approve is not None, "hooks": len(self.hooks),
            "max_cost": self.max_cost is not None})
        for t in self.tools:
            trace["risks"][getattr(t, "name", type(t).__name__)] = getattr(t, "risk", None)
        if self.approve is not None:
            inner = self.approve

            def approve(request, _inner=inner):
                answer = _inner(request)
                trace["approvals"].append({"tool": getattr(request, "tool", None),
                                           "granted": bool(getattr(answer, "granted", answer))})
                return answer

            self.approve = approve

    Agent.__init__ = agent_init

    # ── What the program called ───────────────────────────────────────────
    for cls, names in ((Chain, ("run", "save", "load", "resume")), (Pool, ("run",)),
                       (Agent, ("run", "stream", "step", "execute")), (Skill, ("run",))):
        for name in names:
            original = cls.__dict__[name]
            is_cm = isinstance(original, classmethod)
            fn = original.__func__ if is_cm else original

            def make(fn=fn, label=f"{cls.__name__}.{name}"):
                @functools.wraps(fn)
                def wrapper(*a, **k):
                    if not _internal():
                        _count(label)
                    return fn(*a, **k)
                return wrapper

            setattr(cls, name, classmethod(make()) if is_cm else make())

    # Tool runs, including subclasses defined after this point.
    def wrap_run(klass):
        if "run" in klass.__dict__ and not getattr(klass.__dict__["run"], "_traced", False):
            inner = klass.__dict__["run"]

            @functools.wraps(inner)
            def run(self, *a, **k):
                trace["tool_runs"].append(getattr(self, "name", None) or type(self).__name__)
                return inner(self, *a, **k)

            run._traced = True
            klass.run = run

    previous = Tool.__dict__.get("__init_subclass__")

    def __init_subclass__(klass, **kw):
        if previous is not None:
            previous.__func__(klass, **kw)
        wrap_run(klass)

    Tool.__init_subclass__ = classmethod(__init_subclass__)


def _write():
    if TRACE_PATH:
        with open(TRACE_PATH, "w") as fh:
            json.dump(trace, fh, default=str)


if TRACE_PATH:
    try:
        _install()
    except Exception as exc:                          # the harness must say so
        trace["errors"].append(f"sandbox install failed: {type(exc).__name__}: {exc}")
    atexit.register(_write)
