"""
What kind of runner a step or a pool item is, what it reads, and running it —
one answer for Chain and Pool. See docs/pipelines/chain.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


def is_tool(runner) -> bool:
    from .tools._base import Tool
    return isinstance(runner, Tool)


def is_agent(runner) -> bool:
    from .agent import Agent
    return isinstance(runner, Agent)


def is_chain(runner) -> bool:
    from .chain import Chain
    return isinstance(runner, Chain)


def kind_of(runner) -> str:
    """``"tool"``, ``"agent"``, ``"call"`` (a plain function) or ``"run"``
    (a Skill, a Chain, anything with ``run(variables=...)``)."""
    if is_tool(runner):
        return "tool"
    if is_agent(runner):
        return "agent"
    if callable(runner) and not hasattr(runner, "run"):
        return "call"
    if callable(getattr(runner, "run", None)):
        return "run"
    raise TypeError(f"{type(runner).__name__} is not a runner: give a Skill, a Tool, a Chain, "
                    "an Agent or a function")


def inputs(runner, task_key: str = "task") -> dict:
    """What *runner* reads, by name: ``{"url": {"type": "string", "description":
    "Article URL", "required": True}}`` — from a function's signature and
    docstring, a tool's schema, an agent's task, a chain's own schema or a
    Skill's template."""
    import inspect
    from ._template import placeholder_names
    from .tools._base import _param_docs
    from .chain._inputs import _type_name
    kind = kind_of(runner)
    out: dict = {}
    if kind == "call":
        docs = _param_docs(inspect.getdoc(runner) or "")
        for name, p in inspect.signature(runner).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            spec = {"type": _type_name(p.annotation)}
            if docs.get(name):
                spec["description"] = docs[name]
            spec.update({"required": True} if p.default is p.empty else {"default": p.default})
            out[name] = spec
    elif kind == "tool":
        params = getattr(runner, "parameters", {}) or {}
        required = set(params.get("required") or ())
        for name, prop in (params.get("properties") or {}).items():
            spec = {k: v for k, v in prop.items() if k in ("type", "description", "enum")}
            out[name] = {**spec, "required": True} if name in required else spec
    elif kind == "agent":
        label = getattr(runner, "name", None) or "the agent"
        out[task_key] = {"type": "string", "description": f"the task for {label}",
                         "required": True}
    elif is_chain(runner):
        out = {name: dict(spec) for name, spec in runner.input_schema().items()}
    else:                                              # a Skill: its template's placeholders
        defaults = getattr(runner, "variables", {}) or {}
        names: list = []
        for message in (getattr(runner, "_input", {}) or {}).get("messages", []):
            for part in message.get("parts", []):
                text = part if isinstance(part, str) else (part.get("text") or "") \
                    if isinstance(part, dict) else ""
                names += placeholder_names(text)
        for name in dict.fromkeys(names):
            out[name] = ({"type": "string", "default": defaults[name]} if name in defaults
                         else {"type": "string", "required": True})
    return out


def outputs(runner) -> set | None:
    """The fields a runner's output is known to have before it runs, or None:
    a Skill answering to a JSON schema, a function returning a TypedDict or a
    dataclass."""
    import dataclasses
    import inspect
    import typing
    fmt = ((getattr(runner, "_output", None) or {}).get("format") or {}) \
        if not callable(runner) or hasattr(runner, "run") else {}
    if fmt.get("type") == "json_schema":
        props = (fmt.get("schema") or {}).get("properties")
        return set(props) if isinstance(props, dict) else None
    if kind_of(runner) == "call":
        try:
            ret = typing.get_type_hints(runner).get("return")
        except Exception:
            ret = inspect.signature(runner).return_annotation
        if dataclasses.is_dataclass(ret):
            return {f.name for f in dataclasses.fields(ret)}
        if isinstance(ret, type) and issubclass(ret, dict) and getattr(ret, "__annotations__", None):
            return set(typing.get_type_hints(ret))
    return None


@dataclass
class Outcome:
    """What running a step gave: its output and usage, or the run it parked."""
    output: Any = None
    usage: Any = None
    parked: Any = None


def invoke(runner, values: dict, *, task_key: str = "task", output_field: str = "output",
           mirror: bool = False, label: str = "") -> Outcome:
    """Run an agent, a Skill or a Chain on *values* — the part Chain and Pool
    share; a tool and a function stay with their caller, which decides and
    signals them. A failed agent or chain raises; a parked one is returned as
    ``parked`` for the caller to carry or refuse. *mirror* copies what the run
    recorded back onto the runner the caller holds."""
    from .chain._result import ChainResult
    from .models._usage import Usage
    from .state import SuspendedResult
    what = label or getattr(runner, "name", None) or type(runner).__name__
    if is_agent(runner):
        task = values.get(task_key, "")
        if not task:
            raise ValueError(f"{what}: an agent reads its task from {task_key!r}, which is empty "
                             f"or missing — give it under {task_key!r}")
        result = runner.run(task=task, variables=values)
        if isinstance(result, SuspendedResult):
            return Outcome(parked=result)
        if not result:
            raise RuntimeError(f"{what} failed: {getattr(result, 'error', None) or 'no result'}")
        usage = getattr(result, "usage", None)
        tokens, cost = getattr(result, "tokens_used", 0) or 0, getattr(result, "cost", None)
        if usage is None and (tokens or cost):
            usage = Usage(total_tokens=tokens, cost=cost)
        return Outcome(getattr(result, output_field), usage)
    # A copy per run: one Skill shared by a Pool's threads would have its
    # `last_usage` overwritten by another before this one read it.
    own = copy.copy(runner)
    output = own.run(variables=values)
    if mirror:
        for key, value in vars(own).items():
            if vars(runner).get(key, _MISSING) is not value:
                setattr(runner, key, value)
    if isinstance(output, SuspendedResult):
        return Outcome(parked=output)
    if isinstance(output, ChainResult):
        if not output:
            raise RuntimeError(f"{what} failed: {output.error}")
        return Outcome(output.output, output.usage)
    return Outcome(output, getattr(own, "last_usage", None))


_MISSING = object()
