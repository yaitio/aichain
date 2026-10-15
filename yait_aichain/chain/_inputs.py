"""
chain._inputs — how a step's inputs are filled from the chain's variables:
a name, a path into one (`page.body`), a template; and what a runner is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import logging
from collections.abc import Mapping

_skip_log = logging.getLogger("yait_aichain.chain")

from .._template import _PLACEHOLDER_RE, placeholder_names, substitute_placeholders

# One vocabulary with Pool — see `_errors_policy`. `collect` is new here and
# is a silent `skip`: a chain of a hundred generated steps that expects a few
# to fail should not have to choose between a warning per failure and no
# record at all.

# Default options applied to every Agent step unless overridden.
_AGENT_DEFAULT_OPTIONS: dict = {
    "task_key":     "task",
    "output_field": "output",
}




_MISSING = object()


def _usage_to_dict(usage) -> dict:
    """Serialise a ``Usage`` (or None) for the run document."""
    if not usage:
        return {}
    # Every field: four of the seven were kept, and the cache and image counts
    # were lost at a pause.
    from dataclasses import asdict
    return asdict(usage)




def _build_tool_kwargs(tool, accumulated: dict, input_map: dict) -> dict:
    """
    Build the ``**kwargs`` dict to pass to ``tool.run()``.

    A parameter the step's *input_map* does not mention takes the variable of
    the same name, when there is one, and is otherwise left to the tool's own
    default. A parameter it does mention is filled from its value:

      * ``"draft"`` — a variable's name: that variable's value;
      * ``"Draw {topic} in watercolor"`` — a template: the variables
        substituted, as a Skill substitutes its prompt;
      * ``"{candidates}"`` — one placeholder alone: the variable's value itself,
        so a list stays a list;
      * ``{"query": "{question}", "top_n": 5}`` — a dict, resolved key by key:
        templates are filled, and anything else — a number, a plain string — is
        a literal.

    A name or a template naming a variable the chain does not have, and a
    parameter the tool does not take, are errors that say which.

    Parameters
    ----------
    tool       : Tool instance whose ``.parameters`` schema is inspected.
    accumulated: Current accumulated variable dict.
    input_map  : ``{"tool_param": value}`` — see above.
    """
    from ..tools._base import _open
    props  = tool.parameters.get("properties", {})
    name   = getattr(tool, "name", None) or type(tool).__name__
    # A tool with no properties took any mapping and dropped it; an open
    # schema takes what it is given, and anything else refuses a stranger.
    if not _open(tool.parameters) and (unknown := [k for k in input_map if k not in props]):
        raise ValueError(f"the step maps {unknown} into {name!r}, which takes {sorted(props)}")
    kwargs = {}
    for param in props:
        if param in input_map:
            kwargs[param] = _resolve(input_map[param], accumulated, f"{name}.{param}", top=True)
        elif param in accumulated:
            kwargs[param] = accumulated[param]
    for param in input_map:
        if param not in props:
            kwargs[param] = _resolve(input_map[param], accumulated, f"{name}.{param}", top=True)
    return kwargs


def _callable_name(f) -> str:
    owner = getattr(f, "__self__", None)
    return (f"{type(owner).__name__}.{f.__name__}" if owner is not None and hasattr(f, "__name__")
            else getattr(f, "__name__", None) or type(f).__name__)


def _build_call_kwargs(f, accumulated: dict, input_map: dict) -> tuple[list, dict]:
    """
    The arguments for a function or a method step: each parameter its
    signature names, filled as a tool's are (see `_build_tool_kwargs`); a
    required one the chain cannot fill is an error naming it. Positional-only
    parameters — `len(obj, /)` — go positionally, where they went by name and
    the call failed.
    """
    import inspect
    params = {n: p for n, p in inspect.signature(f).parameters.items()
              if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
    where = _callable_name(f)
    if unknown := [k for k in input_map if k not in params]:
        raise ValueError(f"the step maps {unknown} into {where!r}, which takes {list(params)}")
    args: list = []
    kwargs = {}
    for name, p in params.items():
        if name in input_map:
            value = _resolve(input_map[name], accumulated, f"{where}.{name}", top=True)
        elif name in accumulated:
            value = accumulated[name]
        elif p.default is p.empty:
            raise ValueError(f"{where} needs {name!r}, and the chain has no variable of that name — "
                             f"map one: ({where}, key, {{{name!r}: 'variable'}}); it has {sorted(accumulated)}")
        else:
            continue
        if p.kind is p.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[name] = value
    return args, kwargs


def _runner_name(runner) -> str | None:
    """What a step's output is called when no key is given: the runner's own
    name — a Skill's, an Agent's, a tool's, a Chain's, a function's. None for
    one with no name to give, a lambda or an unnamed Skill."""
    name = getattr(runner, "name", None)
    if not name and callable(runner) and not hasattr(runner, "run"):
        name = getattr(runner, "__name__", None)
    if not isinstance(name, str) or not name.isidentifier():
        return None
    return name


def _lookup(path: str, accumulated: dict, where: str):
    """A variable, or a path into one: ``"page.body"`` is the ``body`` field
    of the dict stored under ``page``. A wrong step names what is there."""
    if path in accumulated:
        return accumulated[path]
    head, _, rest = path.partition(".")
    if not rest or head not in accumulated:
        import difflib
        close = difflib.get_close_matches(head, list(accumulated), n=1)
        hint = f"; did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"{where} is mapped to the variable {path!r}, which the chain does "
                         f"not have; it has {sorted(accumulated)}{hint}")
    value, walked = accumulated[head], head
    for part in rest.split("."):
        if not isinstance(value, Mapping) or part not in value:
            import difflib
            fields = sorted(value) if isinstance(value, Mapping) else []
            close = difflib.get_close_matches(part, [str(f) for f in fields], n=1)
            hint = (f"; did you mean {walked + '.' + close[0]!r}?" if close else
                    f"; {walked!r} has {fields}" if fields else
                    f"; {walked!r} is a {type(value).__name__}, not a dict")
            raise ValueError(f"{where} reads {path!r}, and there is no {part!r} in {walked!r}{hint}")
        value, walked = value[part], f"{walked}.{part}"
    return value


def _resolve(value, accumulated: dict, where: str, top: bool = False):
    """One value of a step's input_map, filled from the chain's variables."""
    if isinstance(value, str):
        names = placeholder_names(value)
        if not names:
            if not top:
                return value                                  # a literal inside a dict
            return _lookup(value, accumulated, where)
        if missing := [n for n in dict.fromkeys(names) if n not in accumulated]:
            raise ValueError(f"{where}: the template {value!r} names {missing}, which the chain does "
                             f"not have; it has {sorted(accumulated)}")
        if (alone := _PLACEHOLDER_RE.fullmatch(value)) is not None:
            return accumulated[alone.group(1)]                # the value itself, not its text
        return substitute_placeholders(value, accumulated)
    if isinstance(value, dict):
        return {k: _resolve(v, accumulated, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, accumulated, where) for v in value]
    return value                                              # a number, a bool: a literal


def _type_name(annotation) -> str:
    """A JSON type for a Python annotation; "string" when it says nothing."""
    import inspect
    return {int: "integer", float: "number", bool: "boolean", dict: "object", list: "array",
            str: "string"}.get(annotation if annotation is not inspect.Parameter.empty else str,
                               "string")


def _sources(input_map: dict) -> list:
    """The variable names an input map reads: a bare name or path, and every
    placeholder of a template, at any depth."""
    out: list = []
    for value in (input_map or {}).values():
        if isinstance(value, str):
            names = placeholder_names(value)
            out += names if names else ([value] if value.replace(".", "_").isidentifier() else [])
        elif isinstance(value, dict):
            out += [n for v in value.values() if isinstance(v, str) for n in placeholder_names(v)]
    return out


