"""
chain._flow — what a chain needs from its caller, and what can be checked
about its data flow before a run. A mixin of `Chain`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import logging
import warnings

_skip_log = logging.getLogger("yait_aichain.chain")

from .._template import placeholder_names

# Default options applied to every Agent step unless overridden.
_AGENT_DEFAULT_OPTIONS: dict = {
    "task_key":     "task",
    "output_field": "output",
}


from ._inputs import _runner_name, _sources, _type_name


def _chain_class():
    from ._chain import Chain                  # here, not above: it imports this module
    return Chain


class _Flow:
    """`input_schema()` and the checks made when a chain is built."""

    def input_schema(self) -> dict:
        """
        What a run needs from its caller: every input some step reads that no
        earlier step writes, with its type, its description, and whether it is
        required or has a default.

        Read from what the steps already say — a function's signature and
        docstring, a tool's schema, a Skill's template placeholders, an Agent's
        task, a chain's own schema — with ``variables=`` as defaults and
        ``inputs=`` as descriptions. An input nothing describes is said, since
        a caller or a model filling it has only its name to go on::

            {"url": {"type": "string", "description": "Article URL", "required": True},
             "language": {"type": "string", "default": "English"}}
        """
        written: set = set()
        schema: dict = {}
        unread: dict = {}
        for runner, key, input_map, kind, options in self._steps:
            for name, spec, where in self._needs(runner, kind, input_map, options):
                if name in written:
                    continue
                known = schema.setdefault(name, {})
                for field_, value in spec.items():
                    known.setdefault(field_, value)
                if not known.get("description"):
                    unread.setdefault(name, where)
            written.add(key)
        for name, spec in schema.items():
            if name in self.variables:
                spec.pop("required", None)
                spec["default"] = self.variables[name]
            if self.inputs.get(name):
                spec["description"] = self.inputs[name]
        for name, where in unread.items():
            if not schema[name].get("description"):
                warnings.warn(f"input {name!r} has no description (read only by {where}); "
                              f"give one: Chain(..., inputs={{{name!r}: '...'}})", stacklevel=2)
        return schema

    @staticmethod
    def _needs(runner, kind: str, input_map: dict, options: dict) -> list:
        """``(name, spec, where)`` for each input one step reads."""
        import inspect
        from ..tools._base import _param_docs
        out: list = []

        def mapped(param: str, spec: dict, where: str) -> None:
            """A parameter the input map fills reads its sources instead."""
            source = input_map[param]
            for name in _sources({param: source}):
                head, _, rest = name.partition(".")
                out.append((head, {"type": "object"} if rest else dict(spec), where))

        label = _runner_name(runner) or type(runner).__name__
        if kind == "call":
            docs = _param_docs(inspect.getdoc(runner) or "")
            for name, p in inspect.signature(runner).parameters.items():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                spec = {"type": _type_name(p.annotation)}
                if docs.get(name):
                    spec["description"] = docs[name]
                if name in input_map:
                    mapped(name, spec, f"{label}")
                elif p.default is p.empty:
                    out.append((name, {**spec, "required": True}, label))
                else:
                    out.append((name, {**spec, "default": p.default}, label))
        elif kind == "tool":
            params = getattr(runner, "parameters", {}) or {}
            required = set(params.get("required") or ())
            for name, prop in (params.get("properties") or {}).items():
                spec = {k: v for k, v in prop.items() if k in ("type", "description", "enum")}
                if name in input_map:
                    mapped(name, spec, label)
                else:
                    out.append((name, {**spec, "required": True} if name in required else spec,
                                label))
            for name in input_map:
                if name not in (params.get("properties") or {}):
                    mapped(name, {}, label)
        elif kind == "agent":
            out.append((options.get("task_key", "task"),
                        {"type": "string", "description": f"the task for {label}", "required": True},
                        label))
        elif isinstance(runner, _chain_class()):
            for name, spec in runner.input_schema().items():
                out.append((name, dict(spec), label))
        else:                                         # a Skill: its template's placeholders
            defaults = getattr(runner, "variables", {}) or {}
            names: list = []
            for message in (getattr(runner, "_input", {}) or {}).get("messages", []):
                for part in message.get("parts", []):
                    text = part if isinstance(part, str) else (part.get("text") or "") \
                        if isinstance(part, dict) else ""
                    names += placeholder_names(text)
            for name in dict.fromkeys(names):
                if name in input_map:
                    mapped(name, {"type": "string"}, f"a Skill template ({label})")
                elif name in defaults:
                    out.append((name, {"type": "string", "default": defaults[name]},
                                f"a Skill template ({label})"))
                else:
                    out.append((name, {"type": "string", "required": True},
                                f"a Skill template ({label})"))
        return out

    def _check_flow(self) -> None:
        """What can be known before a run: a step reading what only a later step writes
        is an error, and a name no step writes that is one letter from a key is said."""
        writes: dict = {}
        for i, (_r, key, *_rest) in enumerate(self._steps):
            writes.setdefault(key, i)
        for i, (runner, _key, input_map, _kind, _opts) in enumerate(self._steps):
            for source in _sources(input_map):
                head = source.partition(".")[0]
                if head in self.variables:
                    continue
                first = writes.get(head)
                if first is not None and first >= i:
                    raise ValueError(
                        f"step {i} ({_runner_name(runner) or type(runner).__name__}) reads "
                        f"{source!r}, which only step {first} writes — it has not run yet")
                if first is None:
                    import difflib
                    earlier = [k for k, j in writes.items() if j < i]
                    if close := difflib.get_close_matches(head, earlier, n=1, cutoff=0.8):
                        warnings.warn(f"step {i} reads {source!r}, which no step writes; unless it "
                                      f"is a run variable, did you mean {close[0]!r}?", stacklevel=3)

