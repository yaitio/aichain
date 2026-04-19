"""
chain._chain
============

``Chain`` — a sequential pipeline of :class:`~skills.Skill`,
:class:`~tools.Tool`, and :class:`~agent.Agent` instances.

A Chain wires steps together so each step's output automatically feeds the
next step's input as a named variable.  Skills, Tools, and Agents can be
freely mixed in any order.

Step specification
------------------
Each element of *steps* may be any of the following forms:

  * ``runner``
        Bare Skill, Tool, or Agent.  String output stored under ``"result"``.

  * ``(runner, output_key: str)``
        String output stored under *output_key*.

  * ``(runner, output_key: str, input_map: dict)``
        Same as above **plus** an *input_map* that renames accumulated
        variables before they are injected into the step.

        ``input_map`` has the form ``{"param_name": "accumulated_var"}``.
        For example ``{"source": "url"}`` tells the step to pass
        ``accumulated["url"]`` as the ``source`` argument.
        Useful when the accumulated variable name differs from the parameter
        name the runner expects.

  * ``(runner, output_key: str, input_map: dict, options: dict)``
        Same as above **plus** an *options* dict for step-level settings.
        Currently meaningful only for Agent steps:

        ``task_key``     (str,  default ``"task"``)
            Name of the accumulated variable that holds the agent's task
            string.  The task is read from ``accumulated[task_key]``
            before the agent runs.

        ``output_field`` (str,  default ``"output"``)
            Which field of the :class:`~agent.AgentResult` to extract as
            the step's output.  Use ``"output"`` for the final answer
            (default), ``"memory"`` for the agent's memory snapshot, or
            any other ``AgentResult`` attribute name.

Variable flow
-------------
1. The chain starts with *initial variables* — merged from
   ``Chain.variables`` and ``run(variables=...)``, call-time values win.

2. Before each step the *accumulated variable dict* is inspected to build
   the step's input:

   * **Skill** — the full accumulated dict is passed as
     ``skill.run(variables=accumulated)``.  The skill's template engine
     picks the placeholders it needs.

   * **Tool**  — kwargs are assembled by matching the tool's declared
     ``parameters["properties"]`` keys against the accumulated dict
     (with optional *input_map* renaming).  Only declared parameters that
     have a matching accumulated variable are forwarded, so extra variables
     are silently ignored and never cause unexpected keyword errors.

   * **Agent** — ``agent.run(task=accumulated[task_key],
     variables=accumulated)`` is called.  The task string is read from
     ``accumulated[task_key]`` (default key: ``"task"``).  The
     ``AgentResult`` field named by ``output_field`` (default:
     ``"output"``) is extracted as the step's output value.

3. After each step the output is merged back into the accumulated dict:

     * ``str`` output  →  ``accumulated[output_key] = output``
     * ``dict`` output →  ``accumulated.update(output)``

4. The final step's output is returned by ``run()``.

Memory / history
----------------
After each ``run()`` call, ``chain.history`` holds one record per step::

    {
        "step":       int,                    # 0-based index
        "kind":       "skill"|"tool"|"agent", # type of runner
        "name":       str,                    # runner.name or "step_N"
        "input":      dict,                   # accumulated vars before step
        "output":     str|dict,               # raw output from the runner
        "output_key": str,                    # variable name for str output
        "options":    dict,                   # step-level options (agents)
    }
"""

from __future__ import annotations

import importlib
import os
import warnings

_VALID_ON_STEP_ERROR: frozenset[str] = frozenset({"raise", "stop", "skip"})

# Default options applied to every Agent step unless overridden.
_AGENT_DEFAULT_OPTIONS: dict = {
    "task_key":     "task",
    "output_field": "output",
}


def _is_tool(runner) -> bool:
    """Return True if *runner* is a Tool instance (lazy, import-free check)."""
    # Avoid a hard import cycle; check the MRO by class name instead of
    # isinstance so that neither module needs to import the other at load time.
    return type(runner).__module__.startswith("tools")


def _is_agent(runner) -> bool:
    """Return True if *runner* is an Agent instance (lazy, import-free check)."""
    return type(runner).__module__.startswith("agent")


def _build_tool_kwargs(tool, accumulated: dict, input_map: dict) -> dict:
    """
    Build the ``**kwargs`` dict to pass to ``tool.run()``.

    For each parameter declared in ``tool.parameters["properties"]``:
      * Check whether *input_map* remaps it to a different accumulated key.
      * Fall back to the parameter name itself.
      * Include the value only when the resolved key is present in *accumulated*.

    Parameters that have no matching accumulated variable are omitted (the
    tool's own default or required-parameter validation will handle them).

    Parameters
    ----------
    tool       : Tool instance whose ``.parameters`` schema is inspected.
    accumulated: Current accumulated variable dict.
    input_map  : ``{"tool_param": "accumulated_var"}`` renaming rules.
    """
    props  = tool.parameters.get("properties", {})
    kwargs = {}
    for param in props:
        src = input_map.get(param, param)   # honour explicit remap, else same name
        if src in accumulated:
            kwargs[param] = accumulated[src]
    return kwargs


class Chain:
    """
    Sequential pipeline of :class:`~skills.Skill` and
    :class:`~tools.Tool` instances.

    Both kinds of runner are accepted in any order and can be freely mixed.
    Skills receive the full accumulated variable dict for template
    substitution; Tools receive only the kwargs that match their declared
    parameters (with optional renaming via *input_map*).

    Parameters
    ----------
    steps : list
        Ordered list of runners.  Each element may be:

        * A bare ``Skill``, ``Tool``, or ``Agent``.
        * A ``(runner, output_key)`` tuple.
        * A ``(runner, output_key, input_map)`` tuple — *input_map* is a
          ``dict`` that renames accumulated variables before they are passed
          to a Tool (e.g. ``{"source": "url"}`` passes ``accumulated["url"]``
          as the tool's ``source`` argument).
        * A ``(runner, output_key, input_map, options)`` tuple — *options*
          is a ``dict`` of step-level settings.  For Agent steps the
          recognised keys are:

          ``task_key`` (str, default ``"task"``)
              Name of the accumulated variable that holds the agent's task
              string.

          ``output_field`` (str, default ``"output"``)
              Which ``AgentResult`` attribute to extract as the step output.
              Typically ``"output"`` (the final answer) or ``"memory"``.

    variables : dict | None, optional
        Default variable values applied to every ``run()`` call.
        Call-time variables take precedence.

    name : str | None, optional
        Human-readable label for this chain.

    description : str | None, optional
        Short description of what the chain does.

    Raises
    ------
    ValueError
        If *steps* is empty or any tuple element has an unsupported shape.

    Examples
    --------
    Mixed Skill + Tool pipeline — fetch a URL, clean it, translate it::

        from models import Model
        from skills import Skill
        from tools  import MarkItDownTool
        from chain  import Chain

        fetch      = MarkItDownTool()
        cleaner    = Skill(model=Model("gpt-4o"), ...)
        translator = Skill(model=Model("gpt-4o"), ...)

        chain = Chain(
            steps=[
                (fetch,      "raw_markdown", {"source": "url"}),
                (cleaner,    "article"),
                (translator, "translation"),
            ],
        )

        result = chain.run(variables={
            "url":      "https://example.com/article",
            "language": "French",
        })

    Two-step Skill-only pipeline::

        chain = Chain([analyst, translator])
        output = chain.run(variables={"topic": "quantum computing",
                                      "language": "Spanish"})
    """

    def __init__(
        self,
        steps,
        variables:      dict | None = None,
        name:           str  | None = None,
        description:    str  | None = None,
        on_step_error:  str         = "raise",
    ) -> None:
        if not steps:
            raise ValueError("Chain requires at least one step.")
        if on_step_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {on_step_error!r}"
            )
        self._steps         = self._normalise(steps)
        self.variables      = variables or {}
        self.name           = name
        self.description    = description
        self.on_step_error  = on_step_error
        self._history: list[dict] = []
        self._accumulated: dict   = {}     # snapshot of accumulated vars after last run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        variables:     dict | None = None,
        on_step_error: str  | None = None,
    ) -> "str | dict | None":
        """
        Execute all steps in order and return the final step's output.

        Parameters
        ----------
        variables : dict | None, optional
            Variables for this invocation.  Merged with ``self.variables``;
            call-time values win.
        on_step_error : str | None, optional
            Override the instance-level ``on_step_error`` for this call only.
            Must be ``"raise"``, ``"stop"``, or ``"skip"`` when provided.

        Returns
        -------
        str | dict | None
            Output of the last successful step.  ``None`` when
            ``on_step_error`` is ``"stop"`` or ``"skip"`` and the pipeline
            had no successful steps (e.g. step 0 failed).

        Raises
        ------
        RuntimeError, clients._base.APIError, …
            Any step exception, when ``on_step_error`` is ``"raise"``
            (the default).
        ValueError
            If *on_step_error* override value is not recognised.
        """
        _on_error = self.on_step_error if on_step_error is None else on_step_error
        if _on_error not in _VALID_ON_STEP_ERROR:
            raise ValueError(
                f"on_step_error must be one of {sorted(_VALID_ON_STEP_ERROR)}; "
                f"got {_on_error!r}"
            )

        self._history = []
        accumulated   = {**self.variables, **(variables or {})}
        last_output: "str | dict | None" = None

        for idx, (runner, output_key, input_map, kind, options) in enumerate(self._steps):
            step_input = dict(accumulated)          # snapshot before the step
            name       = getattr(runner, "name", None) or f"step_{idx}"

            try:
                if kind == "tool":
                    kwargs = _build_tool_kwargs(runner, accumulated, input_map)
                    output = runner.run(**kwargs)       # Tool: direct kwargs call

                elif kind == "agent":
                    # Resolve task string from accumulated dict
                    task_key = options.get("task_key", _AGENT_DEFAULT_OPTIONS["task_key"])
                    task     = accumulated.get(task_key, "")
                    if not task:
                        raise ValueError(
                            f"Agent step {idx} ({name!r}): accumulated variable "
                            f"{task_key!r} is empty or missing.  Set it in a "
                            f"prior step or in the initial variables."
                        )
                    agent_result = runner.run(task=task, variables=accumulated)

                    if not agent_result:
                        raise RuntimeError(
                            f"Agent step {idx} ({name!r}) failed: "
                            f"{agent_result.error}"
                        )

                    # Extract the requested field from AgentResult
                    output_field = options.get(
                        "output_field", _AGENT_DEFAULT_OPTIONS["output_field"]
                    )
                    output = getattr(agent_result, output_field, None)

                else:
                    # Skill: pass full accumulated dict, honouring any input_map renames.
                    # input_map for skills works as an *alias* layer: the mapped key is
                    # added/overwritten in the variables copy passed to the skill, without
                    # touching the real accumulated dict.
                    if input_map:
                        skill_vars = dict(accumulated)
                        for dst, src in input_map.items():
                            if src in accumulated:
                                skill_vars[dst] = accumulated[src]
                        output = runner.run(variables=skill_vars)
                    else:
                        output = runner.run(variables=accumulated)

            except Exception as exc:
                self._history.append({
                    "step":       idx,
                    "kind":       kind,
                    "name":       name,
                    "input":      step_input,
                    "output":     None,
                    "output_key": output_key,
                    "options":    options,
                    "error":      str(exc),
                })

                if _on_error == "raise":
                    raise

                if _on_error == "stop":
                    return last_output

                # "skip" — warn and continue; downstream steps may lack a
                # variable they depend on, but the caller owns that trade-off.
                warnings.warn(
                    f"Chain step {idx} ({name!r}) failed and was skipped: {exc}. "
                    f"Downstream steps that read {output_key!r} will receive a "
                    f"stale or absent value.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            self._history.append({
                "step":       idx,
                "kind":       kind,
                "name":       name,
                "input":      step_input,
                "output":     output,
                "output_key": output_key,
                "options":    options,
            })

            # Merge output into accumulated vars for downstream steps
            if isinstance(output, dict):
                accumulated.update(output)
            else:
                accumulated[output_key] = output

            last_output = output

        self._accumulated = dict(accumulated)
        return last_output

    @property
    def accumulated(self) -> dict:
        """
        Snapshot of the full accumulated variable dict after the most recent
        ``run()`` call.  Includes all initial variables, step outputs, and any
        keys written by Tools that return dicts.

        Returns a shallow copy; modifying it does not affect internal state.
        Returns an empty dict before the first ``run()``.

        Typical use — assembling a sectional document after all steps complete::

            chain.run(variables=initial_vars)
            document = assemble_document(chain.accumulated, sections)
        """
        return dict(self._accumulated)

    @property
    def history(self) -> list[dict]:
        """
        Step records from the most recent ``run()`` call.

        Returns a shallow copy; modifying it does not affect the chain's state.
        Returns an empty list before the first ``run()``.

        Each record contains:

        ``step``       — 0-based step index
        ``kind``       — ``"skill"`` or ``"tool"``
        ``name``       — runner name (or ``"step_N"`` if unnamed)
        ``input``      — accumulated variable dict *before* this step ran
        ``output``     — raw output (``str`` or ``dict``)
        ``output_key`` — variable name used to store a ``str`` output
        """
        return list(self._history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise this chain to a YAML file at *path*.

        The file stores everything needed to recreate the chain:

        * Chain-level metadata — ``name``, ``description``, ``variables``
        * Each step's ``output_key`` and ``input_map``
        * For **Skill** steps — the model name, input template, output spec,
          variables, and options (same fields as ``Skill.save()``)
        * For **Tool** steps — the fully-qualified class path
          (``"module.ClassName"``) and any serialisable ``init_args``

        API keys are **never** written to the file; :meth:`load` resolves
        them from environment variables at load time.

        Parameters
        ----------
        path : str
            Destination file path.  Parent directories are created
            automatically if they do not exist.

        Raises
        ------
        ImportError
            If PyYAML is not installed.

        Example
        -------
        ::

            chain.save("chains/analyse_and_translate.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for Chain.save(). "
                "Install it with: pip install pyyaml"
            )

        steps_data = []
        for runner, output_key, input_map, kind, options in self._steps:
            if kind == "skill":
                skill_data: dict = {
                    "model_name": runner.model.name,
                }
                if runner.name:
                    skill_data["name"] = runner.name
                if runner.description:
                    skill_data["description"] = runner.description
                skill_data["input"]  = runner._input
                skill_data["output"] = runner._output
                if runner.variables:
                    skill_data["variables"] = runner.variables
                if runner.options:
                    skill_data["options"] = runner.options

                step: dict = {
                    "kind":       "skill",
                    "output_key": output_key,
                    "skill":      skill_data,
                }

            elif kind == "agent":
                cls        = type(runner)
                class_path = f"{cls.__module__}.{cls.__qualname__}"
                # Serialise key Agent constructor fields
                agent_data: dict = {
                    "class":       class_path,
                    "orchestrator": runner.orchestrator.name
                        if hasattr(runner, "orchestrator") else None,
                    "mode":        getattr(runner, "mode",        "agile"),
                    "max_steps":   getattr(runner, "max_steps",   10),
                    "max_attempts":getattr(runner, "max_attempts", 3),
                    "max_tokens":  getattr(runner, "max_tokens",  50_000),
                    "verbose":     getattr(runner, "verbose",     0),
                }
                if getattr(runner, "persona", None):
                    agent_data["persona"] = runner.persona
                # Tool list — store class paths only (no API keys)
                if getattr(runner, "tools", None):
                    agent_data["tools"] = [
                        f"{type(t).__module__}.{type(t).__qualname__}"
                        for t in runner.tools
                    ]

                step = {
                    "kind":       "agent",
                    "output_key": output_key,
                    "agent":      agent_data,
                }

            else:  # tool
                cls       = type(runner)
                class_path = f"{cls.__module__}.{cls.__qualname__}"
                init_args  = runner._serialise_init_args() \
                    if hasattr(runner, "_serialise_init_args") else {}

                step = {
                    "kind":       "tool",
                    "output_key": output_key,
                    "tool": {
                        "class":     class_path,
                        "init_args": init_args,
                    },
                }

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
        # Reorder so name/description/variables appear before steps
        ordered: dict = {}
        for k in ("name", "description", "on_step_error", "variables", "steps"):
            if k in data:
                ordered[k] = data[k]

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(ordered, fh, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)

    @classmethod
    def load(cls, path: str, api_key: "str | None" = None) -> "Chain":
        """
        Load a chain from a YAML file previously created by :meth:`save`.

        Each step is reconstructed from the serialised data:

        * **Skill** steps — the model is re-created via
          :class:`~models.Model` (API key resolved from the environment
          unless *api_key* is supplied), then a :class:`~skills.Skill` is
          instantiated from the stored template.
        * **Tool** steps — the class is imported by its stored module path
          and instantiated with the stored ``init_args``; API keys are
          resolved from environment variables automatically.

        Parameters
        ----------
        path : str
            Path to a YAML file previously written by :meth:`save`.
        api_key : str | None, optional
            Override the API key used for **all** Skill models in this chain.
            When omitted, each model's key is read from its provider's
            environment variable.

        Returns
        -------
        Chain
            A fully initialised :class:`Chain` ready to call :meth:`run`.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If PyYAML is not installed, or a tool's module cannot be imported.
        ValueError
            If a step's ``kind`` is unrecognised, or a required API key is
            missing.

        Example
        -------
        ::

            from chain import Chain

            chain  = Chain.load("chains/analyse_and_translate.yaml")
            result = chain.run(variables={"topic": "AI safety",
                                          "language": "French"})
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for Chain.load(). "
                "Install it with: pip install pyyaml"
            )

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        # Local imports to avoid circular references at module load time
        from models._base import Model as _Model
        from skills._skill import Skill as _Skill

        steps = []
        for step_data in data.get("steps", []):
            kind       = step_data["kind"]
            output_key = step_data.get("output_key", "result")
            input_map  = step_data.get("input_map", {})
            options    = step_data.get("options", {})

            if kind == "skill":
                sd    = step_data["skill"]
                model = _Model(sd["model_name"], api_key=api_key)
                skill = _Skill(
                    model       = model,
                    input       = sd["input"],
                    output      = sd["output"],
                    variables   = sd.get("variables"),
                    options     = sd.get("options"),
                    name        = sd.get("name"),
                    description = sd.get("description"),
                )
                entry: "tuple | object" = (skill, output_key, input_map, options)

            elif kind == "agent":
                from models._base  import Model as _Model2
                from agent._agent  import Agent as _Agent

                ad = step_data["agent"]

                # Reconstruct tools from class paths
                tool_instances = []
                for tool_path in ad.get("tools", []):
                    t_mod, t_cls = tool_path.rsplit(".", 1)
                    try:
                        t_module = importlib.import_module(t_mod)
                    except ModuleNotFoundError as exc:
                        raise ImportError(
                            f"Cannot import tool module '{t_mod}': {exc}"
                        ) from exc
                    tool_instances.append(getattr(t_module, t_cls)())

                agent = _Agent(
                    orchestrator = _Model2(ad["orchestrator"],
                                          api_key=api_key),
                    tools        = tool_instances or None,
                    mode         = ad.get("mode",         "agile"),
                    max_steps    = ad.get("max_steps",    10),
                    max_attempts = ad.get("max_attempts", 3),
                    max_tokens   = ad.get("max_tokens",   50_000),
                    verbose      = ad.get("verbose",      0),
                    persona      = ad.get("persona"),
                )
                entry = (agent, output_key, input_map, options)

            elif kind == "tool":
                td         = step_data["tool"]
                class_path = td["class"]
                init_args  = td.get("init_args") or {}

                # Import the tool class dynamically
                module_path, class_name = class_path.rsplit(".", 1)
                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError as exc:
                    raise ImportError(
                        f"Cannot import tool module '{module_path}': {exc}"
                    ) from exc
                try:
                    tool_cls = getattr(module, class_name)
                except AttributeError:
                    raise ImportError(
                        f"Module '{module_path}' has no class '{class_name}'"
                    )

                tool  = tool_cls(**init_args)
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(steps) -> list:
        """
        Validate and normalise *steps* into a list of
        ``(runner, output_key, input_map, kind, options)`` 5-tuples.
        """
        result = []
        for item in steps:
            if isinstance(item, tuple):
                if len(item) == 2:
                    runner, key = item
                    input_map   = {}
                    options     = {}
                elif len(item) == 3:
                    runner, key, input_map = item
                    options = {}
                    if not isinstance(input_map, dict):
                        raise ValueError(
                            "The third element of a step tuple must be a dict "
                            f"(input_map); got {input_map!r}"
                        )
                elif len(item) == 4:
                    runner, key, input_map, options = item
                    if not isinstance(input_map, dict):
                        raise ValueError(
                            "The third element of a step tuple must be a dict "
                            f"(input_map); got {input_map!r}"
                        )
                    if not isinstance(options, dict):
                        raise ValueError(
                            "The fourth element of a step tuple must be a dict "
                            f"(options); got {options!r}"
                        )
                else:
                    raise ValueError(
                        "Step tuples must have 2, 3, or 4 elements: "
                        "(runner, output_key), "
                        "(runner, output_key, input_map), or "
                        "(runner, output_key, input_map, options); "
                        f"got a {len(item)}-element tuple"
                    )
                if not isinstance(key, str):
                    raise ValueError(
                        f"output_key must be a str; got {key!r}"
                    )
            else:
                runner    = item
                key       = "result"
                input_map = {}
                options   = {}

            if _is_tool(runner):
                kind = "tool"
            elif _is_agent(runner):
                kind = "agent"
            else:
                kind = "skill"

            result.append((runner, key, input_map, kind, options))

        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        step_names = [
            f"[{kind[0].upper()}]{getattr(runner, 'name', None) or type(runner).__name__}→{key}"
            for runner, key, _, kind, _opts in self._steps
        ]
        parts = [f"steps=[{', '.join(step_names)}]"]
        if self.name:
            parts.append(f"name={self.name!r}")
        if self.on_step_error != "raise":
            parts.append(f"on_step_error={self.on_step_error!r}")
        return f"Chain({', '.join(parts)})"
