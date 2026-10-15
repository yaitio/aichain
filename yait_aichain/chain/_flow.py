"""
chain._flow — what a chain needs from its caller, and what can be checked
about its data flow before a run. A mixin of `Chain`.
"""

from __future__ import annotations

import warnings

from ._inputs import _runner_name, _sources


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
        """``(name, spec, where)`` for each input one step reads: what the runner
        reads (`_runners.inputs`), its mapped parameters reading their sources."""
        from .._runners import inputs
        out: list = []
        label = _runner_name(runner) or type(runner).__name__
        reads = inputs(runner, task_key=options.get("task_key", "task"))
        if isinstance(runner, _chain_class()):
            return [(name, spec, label) for name, spec in reads.items()]
        where = label if kind in ("call", "tool", "agent") else f"a Skill template ({label})"

        def mapped(param: str, spec: dict) -> None:
            for name in _sources({param: input_map[param]}):
                head, _, rest = name.partition(".")
                out.append((head, {"type": "object"} if rest else dict(spec), where))

        for name, spec in reads.items():
            if name in input_map:
                mapped(name, {k: v for k, v in spec.items() if k not in ("required", "default")})
            else:
                out.append((name, spec, where))
        if kind == "tool":
            for name in input_map:
                if name not in reads:
                    mapped(name, {})
        return out

    def _check_flow(self) -> None:
        """What can be known before a run: a step reading what only a later step writes
        is an error, and a name no step writes that is one letter from a key is said."""
        from .._runners import outputs
        writes: dict = {}
        fields: dict = {}
        for i, (r, key, *_rest) in enumerate(self._steps):
            writes.setdefault(key, i)
            fields.setdefault(key, outputs(r))
        for i, (runner, _key, input_map, _kind, _opts) in enumerate(self._steps):
            for source in _sources(input_map):
                head = source.partition(".")[0]
                if head in self.variables:
                    continue
                first = writes.get(head)
                field = source.partition(".")[2].partition(".")[0]
                known = fields.get(head)
                if first is not None and first < i and field and known is not None \
                        and field not in known:
                    import difflib
                    close = difflib.get_close_matches(field, sorted(known), n=1)
                    raise ValueError(
                        f"step {i} ({_runner_name(runner) or type(runner).__name__}) reads "
                        f"{source!r}, and step {first}'s output has no field {field!r}"
                        + (f"; did you mean {head + '.' + close[0]!r}?" if close
                           else f"; it has {sorted(known)}"))
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

