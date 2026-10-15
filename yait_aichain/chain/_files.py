"""
chain._files — a chain written to a file and read back. A mixin of `Chain`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import importlib
import logging
import warnings

_skip_log = logging.getLogger("yait_aichain.chain")




from ._inputs import _MISSING, _usage_to_dict, _callable_name  # noqa: F401


def _import_tool(class_path: str, init_args: dict, trusted: bool):
    """Build the tool a chain file names. Unless *trusted*, it must be a package
    `Tool` declaring `_safe_for_untrusted_load`, its refusal hook must accept the
    arguments, and what it builds must be `read` or `draft`."""
    from ..tools._base import Tool

    module_path, _, class_name = class_path.rpartition(".")
    if not trusted and module_path != "yait_aichain" and not module_path.startswith("yait_aichain."):
        raise ValueError(
            f"Refusing to import tool class {class_path!r} from outside 'yait_aichain' while "
            "loading an untrusted chain file. Pass trusted=True to Chain.load() only for files "
            "you fully control.")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Cannot import tool module '{module_path}': {exc}") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module '{module_path}' has no class '{class_name}'")
    if trusted:
        return cls(**init_args)

    if not (isinstance(cls, type) and issubclass(cls, Tool)):
        raise ValueError(f"Refusing to load {class_path}: it is not a Tool")
    if not getattr(cls, "_safe_for_untrusted_load", False):
        raise ValueError(
            f"Refusing to load {class_path} from an untrusted chain file: it runs what the file "
            "tells it to — a command, a server, an address or a folder. Pass trusted=True to "
            "Chain.load() for a file you wrote, or build this step in code.")
    refusal = getattr(cls, "_untrusted_load_refusal", None)
    if refusal and (why := refusal(init_args)):
        raise ValueError(f"Refusing to load {class_path}: {why}")
    tool = cls(**init_args)
    if getattr(tool, "risk", "write") not in ("read", "draft"):
        raise ValueError(f"Refusing to load {class_path} from an untrusted chain file: its risk "
                         f"is {tool.risk!r}. Pass trusted=True for a file you wrote.")
    return tool



def _tool_spec(tool) -> dict:
    """A tool as a file holds it: its class and its settings, or a refusal naming
    why it cannot be, rather than a file that fails only on load."""
    from ..tools.general._wait import Gate
    from ..utilities.mcp._tools import MCPTool
    _refuse_function_tool(tool)
    if isinstance(tool, (Gate, MCPTool)):
        raise TypeError(f"tool {tool.name!r} ({type(tool).__name__}) cannot be saved: "
                        + ("it wraps another tool, which is an object, not a setting"
                           if isinstance(tool, Gate) else
                           "its server and headers may carry credentials")
                        + ". Build this step in code.")
    cls = type(tool)
    spec = {"class": f"{cls.__module__}.{cls.__qualname__}",
            "init_args": tool._serialise_init_args() if hasattr(tool, "_serialise_init_args") else {}}
    for attr in ("name", "description"):
        if getattr(tool, attr, None) and getattr(tool, attr) != getattr(cls, attr, None):
            spec[attr] = getattr(tool, attr)
    return spec


def _refuse_function_tool(t) -> None:
    """A @tool function is code: a saved chain could name it and never bring it back."""
    from ..tools._base import FunctionTool
    if isinstance(t, FunctionTool):
        raise TypeError(f"tool {t.name!r} is a @tool function, and a function cannot be saved — "
                        "it is code, not configuration. Build this chain in code, or make the "
                        "tool a Tool subclass whose settings can be written down.")




if TYPE_CHECKING:
    from ._chain import Chain


class _Files:
    """`save()` and `load()`."""

    def save(self, path: str) -> None:
        """
        Serialise this chain to a YAML file at *path* — metadata, each step's key and
        input map, a Skill's model and template, a tool's class path and init args.
        API keys are never written; :meth:`load` reads them from the environment.

        Parameters
        ----------
        path : str
            Destination file path; parent directories are created.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        """
        if self.policy is not None and not (hasattr(self.policy, "rules")
                                            and hasattr(self.policy, "default")):
            warnings.warn("this chain's policy is not a PermissionPolicy and is not saved; give it "
                          "to Chain.load(...)'s result again before running it", stacklevel=2)
        if self.approve is not None or self.interceptors:
            # Code, not settings: a file cannot carry a function. A chain
            # loaded back without them refuses what needed a person, and runs
            # what an interceptor would have changed.
            warnings.warn("this chain's approver and interceptors are code and are not saved; "
                          "give them to Chain.load(...)'s result again", stacklevel=2)
        steps_data = []
        for runner, output_key, input_map, kind, options in self._steps:
            if kind == "call":
                raise TypeError(f"step {_callable_name(runner)!r} is a function or a utility's method, and "
                                "cannot be saved — it is code, or a method of a live object, not "
                                "configuration. Build this chain in code.")
            if kind == "skill":
                skill_data = runner.to_dict()

                step: dict = {
                    "kind":       "skill",
                    "output_key": output_key,
                    "skill":      skill_data,
                }

            elif kind == "agent":
                cls        = type(runner)
                class_path = f"{cls.__module__}.{cls.__qualname__}"
                # Serialise the Agent's constructor fields, read directly
                # rather than through `getattr(..., default)`: a fallback
                # turns a missing attribute into a plausible value and a
                # saved chain into a different one, in silence.
                agent_data: dict = {
                    "class":       class_path,
                    "model":       runner.model.name,
                    "mode":        runner.mode,
                    "verbose":     runner.verbose,
                }
                if runner.instructions:
                    agent_data["instructions"] = runner.instructions
                if runner.name:
                    agent_data["name"] = runner.name
                if runner.description:
                    agent_data["description"] = runner.description
                # `stop_when` holds closures — a step ceiling is a function, not
                # a number — so what survives is what can be named. A condition
                # that cannot be reconstructed is dropped **loudly**: a chain
                # that silently comes back without its budget is the quiet
                # failure to avoid.
                kept, lost = [], []
                for cond in (runner.stop_when or []):
                    spec = getattr(cond, "spec", None)
                    (kept if spec else lost).append(spec or cond)
                if kept:
                    agent_data["stop_when"] = kept
                if lost:
                    import warnings as _w
                    _w.warn(
                        f"Agent {runner.name or class_path!r}: "
                        f"{len(lost)} stop condition(s) cannot be serialised "
                        "and will not come back on load; rebuild them in code.",
                        RuntimeWarning, stacklevel=2)
                # Tool list — store class paths only (no API keys)
                if getattr(runner, "tools", None):
                    # Each with its settings: a class path alone would rebuild
                    # a Shell tool in the loader's working directory.
                    agent_data["tools"] = [_tool_spec(t) for t in runner.tools]
                # The policy comes back with the agent, or reloading would lose
                # the gate quietly. An approver is code and does not: reloaded,
                # an `approve` rule refuses until one is given.
                if runner.permissions is not None:
                    agent_data["permissions"] = {"rules": dict(runner.permissions.rules),
                                                 "default": runner.permissions.default}
                if runner.approve is not None or runner.hooks:
                    import warnings as _w
                    _w.warn(f"Agent {runner.name or class_path!r}: its approver and hooks are code "
                            "and are not saved; pass them again after Chain.load — until then an "
                            "'approve' rule refuses.", RuntimeWarning, stacklevel=2)

                step = {
                    "kind":       "agent",
                    "output_key": output_key,
                    "agent":      agent_data,
                }

            else:  # tool
                step = {
                    "kind":       "tool",
                    "output_key": output_key,
                    "tool":       _tool_spec(runner),
                }
                # A name or a description given to the factory, over the
                # class's: kept for every tool, whatever its constructor takes.

            if input_map:
                step["input_map"] = input_map
            if options:
                step["options"] = options

            steps_data.append(step)

        data: dict = {"steps": steps_data}
        if self.name:
            data["name"] = self.name
        if self.description:
            data["description"] = self.description
        if self.variables:
            data["variables"] = self.variables
        if self.on_step_error != "raise":
            data["on_step_error"] = self.on_step_error
        if hasattr(self.policy, "rules") and hasattr(self.policy, "default"):
            data["policy"] = {"rules": dict(self.policy.rules), "default": self.policy.default}
        if self.inputs:
            data["inputs"] = dict(self.inputs)
        # Reorder so name/description/variables appear before steps
        ordered: dict = {}
        for k in ("name", "description", "on_step_error", "policy", "inputs", "variables", "steps"):
            if k in data:
                ordered[k] = data[k]

        from .. import _yaml
        _yaml.write(path, ordered, "Chain.save()")

    @classmethod
    def load(
        cls,
        path: str,
        api_key: str | None = None,
        *,
        trusted: bool = False,
    ) -> Chain:
        """
        Load a chain from a YAML file :meth:`save` wrote; loading imports and runs the
        tool classes it names, so only ``yait_aichain`` classes unless *trusted*.

        Parameters
        ----------
        path : str
            The YAML file ``save()`` wrote.
        api_key : str | None, optional
            A key for every Skill step's model; otherwise each reads its own from the environment.
        trusted : bool, optional
            Allow tool classes outside ``yait_aichain`` — only for files you wrote or control.

        Returns
        -------
        Chain
            The chain, ready to :meth:`run`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML is missing, or a tool's module cannot be imported.
        ValueError
            If a step's ``kind`` is unrecognised, or a required API key is missing.
        """
        from .. import _yaml
        data = _yaml.read(path, "Chain.load()", ("steps",))

        # Local imports to avoid circular references at module load time
        from ..models._base import Model as _Model
        from ..skills._skill import Skill as _Skill

        steps = []
        for step_data in data.get("steps", []):
            kind       = step_data["kind"]
            output_key = step_data.get("output_key", "result")
            input_map  = step_data.get("input_map", {})
            options    = step_data.get("options", {})

            if kind == "skill":
                skill = _Skill.from_dict(step_data["skill"], api_key)
                entry: tuple | object = (skill, output_key, input_map, options)

            elif kind == "agent":
                from ..agent._agent import Agent as _Agent

                ad = step_data["agent"]

                # Reconstruct tools from class paths
                tool_instances = []
                for spec in ad.get("tools", []):
                    if isinstance(spec, str):            # written before tools kept their settings
                        spec = {"class": spec}
                    t = _import_tool(spec["class"], spec.get("init_args") or {}, trusted)
                    for attr in ("name", "description"):
                        if spec.get(attr):
                            setattr(t, attr, spec[attr])
                    tool_instances.append(t)

                from ..agent import cost_budget, step_count, token_budget
                _BUILDERS = {"step_count": step_count,
                             "token_budget": token_budget,
                             "cost_budget": cost_budget}
                stop_when = []
                for spec in ad.get("stop_when", []):
                    build = _BUILDERS.get(spec.get("kind"))
                    if build is None:                # a `check` is a closure
                        continue                     # and cannot come back
                    stop_when.append(build(spec["value"]))

                # An agent step with no `model` cannot be rebuilt; refuse it
                # rather than guess one.
                if "model" not in ad:
                    raise ValueError(
                        "This chain file has an agent step with no model name. "
                        "Rebuild the chain in code and save it again.")
                agent = _Agent(
                    _Model(ad["model"], api_key=api_key),
                    tools        = tool_instances or None,
                    mode         = ad.get("mode", "agile"),
                    stop_when    = stop_when or None,
                    verbose      = ad.get("verbose", 0),
                    instructions = ad.get("instructions", ""),
                    name         = ad.get("name"),
                    description  = ad.get("description"),
                    policy       = ({"rules": dict(ad["permissions"].get("rules") or {}),
                                     "default": ad["permissions"].get("default", "approve")}
                                    if ad.get("permissions") else None),
                )
                entry = (agent, output_key, input_map, options)

            elif kind == "tool":
                td         = step_data["tool"]
                class_path = td["class"]
                init_args  = td.get("init_args") or {}

                tool = _import_tool(class_path, init_args, trusted)
                for attr in ("name", "description"):
                    if td.get(attr):
                        setattr(tool, attr, td[attr])
                entry = (tool, output_key, input_map, options)

            else:
                raise ValueError(
                    f"Unknown step kind {kind!r}. "
                    f"Expected 'skill', 'tool', or 'agent'."
                )

            steps.append(entry)

        return cls(
            steps          = steps,
            variables      = data.get("variables"),
            name           = data.get("name"),
            description    = data.get("description"),
            on_step_error  = data.get("on_step_error", "raise"),
            policy         = data.get("policy"),
            inputs         = data.get("inputs"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

