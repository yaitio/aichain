"""
agent._agent
============

``Agent`` — the autonomous execution engine.

An Agent uses an *orchestrator* model to plan, direct, and reflect on tasks.
It executes steps using a combination of :class:`~tools.Tool` instances
(for external actions) and dynamically-constructed LLM skills (for reasoning
and generation).

Two execution modes
-------------------
waterfall
    The orchestrator creates a fixed plan upfront.  Steps execute in order.
    After each step a reflection decides: continue → retry → stop (fatal).
    Replanning is not allowed.  If a step fails beyond ``max_attempts`` the
    agent stops and reports failure.

agile
    Same structure but the reflection phase can also trigger replanning.
    The orchestrator may revise the remaining steps based on what it has
    learned, jump back to an earlier step, or decide early termination.
    The total token budget caps overall spend.

Orchestrator call sequence (per step)
--------------------------------------
1. **plan**   — once at the start; produces an ordered list of steps
2. **action** — before each step; produces exact tool kwargs or skill prompt
3. **reflect**— after each step; produces assessment + next decision

All three call types return structured JSON parsed by ``_parse_json()``.
Token usage is extracted from every raw API response and accumulated in
``tokens_used``.

Memory
------
``agent.memory`` is an :class:`~agent.AgentMemory` instance that persists
for the duration of one ``run()`` call.  Every step can read from it and
write to it.  Initial variables passed to ``run()`` are seeded into memory
before the first step.

Verbosity
---------
``verbose=0``  (default) — silent; no console output.
``verbose=1``  — progress: shows the plan, one status line per step, and a
                 final summary.  Good for monitoring long-running agents.
``verbose=2``  — detailed: adds full action payloads (tool kwargs / skill
                 prompts), output previews, per-call token breakdowns, and
                 the orchestrator's reflection reasoning.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

from ._memory      import AgentMemory
from ._result      import AgentResult
from .             import _prompts as prompts
from tools._base   import Tool


# ---------------------------------------------------------------------------
# _SpawnTool — exposes Agent.spawn() to the orchestrator LLM as a tool
# ---------------------------------------------------------------------------

class _SpawnTool(Tool):
    """
    Internal tool that lets the orchestrator LLM dynamically spawn child agents.

    Added to an agent's tool list when ``allow_spawn=True``.  The LLM calls it
    with a ``task`` string and optional ``tools`` / ``model`` overrides; the
    parent :meth:`Agent.spawn` method creates and runs the child, then returns
    its output as a plain string.

    Users never instantiate this directly — set ``allow_spawn=True`` on the
    parent :class:`Agent` and the framework adds it automatically.
    """

    name: str = "spawn_agent"
    description: str = (
        "Spawn a child agent to execute a focused, self-contained sub-task. "
        "The child runs its own full plan-execute-reflect loop and returns its output. "
        "Use this to delegate research, analysis, writing, or any task that benefits "
        "from autonomous multi-step execution."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {
                "type":        "string",
                "description": "The specific, self-contained task for the child agent.",
            },
            "tools": {
                "type":  "array",
                "items": {"type": "string"},
                "description": (
                    "Names of tools from the parent's tool list to give the child. "
                    "Omit to give the child all parent tools (excluding spawn_agent)."
                ),
            },
            "model": {
                "type":        "string",
                "description": (
                    "Name of an executor model to use as the child's orchestrator. "
                    "Omit to inherit the parent's orchestrator."
                ),
            },
        },
        "required": ["task"],
    }

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent

    def run(  # type: ignore[override]
        self,
        task:  str,
        tools: list | None = None,
        model: str  | None = None,
    ) -> str:
        # Resolve tool names → Tool objects from parent's list
        resolved_tools: list | None = None
        if tools is not None:
            parent_map     = {t.name: t for t in self._agent.tools if t.name != "spawn_agent"}
            resolved_tools = [parent_map[n] for n in tools if n in parent_map]

        # Resolve model name → Model object from parent's executor map
        resolved_model = None
        if model is not None:
            resolved_model = self._agent._executor_map.get(model)

        return self._agent.spawn(task, tools=resolved_tools, model=resolved_model)


# ---------------------------------------------------------------------------
# Safe string formatter (missing keys → left as {key})
# ---------------------------------------------------------------------------

class _SafeFormatMap(dict):
    """Return the literal ``{key}`` for any key not present in the dict."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    Autonomous task-execution engine.

    Parameters
    ----------
    orchestrator : Model
        The model responsible for planning, action determination, and
        reflection.  Should be a capable reasoning model
        (e.g. ``claude-opus-4-6``, ``gpt-4o``, ``o3``).

    tools : list[Tool] | None, optional
        Tools the agent may call.  Each tool is identified by its
        ``tool.name`` attribute in orchestrator prompts.

    executors : list[Model] | None, optional
        Models available for LLM skill steps.  When omitted, the
        orchestrator itself is used for all execution.  The orchestrator
        can request a specific executor by name in its action response.

    mode : "waterfall" | "agile"
        Execution mode.  See module docstring for details.

    max_steps : int, optional
        Maximum number of plan steps.  Default 10.

    max_attempts : int, optional
        Maximum retry attempts per individual step.  Default 3.

    max_tokens : int, optional
        Total token budget across all LLM calls (planning + execution +
        reflection).  The agent stops when this is exceeded.  Default 50 000.

    memory : AgentMemory | None, optional
        Custom memory instance.  When omitted a fresh one is created per
        ``run()`` call.  Pass a pre-populated instance to seed the agent
        with prior knowledge.

    verbose : int, optional
        Console verbosity level:

        * ``0`` (default) — silent.
        * ``1`` — progress: plan overview, one status line per step, final
          summary with total tokens used.
        * ``2`` — detailed: everything from level 1 plus action payloads
          (tool kwargs / skill prompts), output previews, per-call token
          breakdowns, and the orchestrator's reflection reasoning.

    name : str | None, optional
        Human-readable label for this agent.

    description : str | None, optional
        Short description of the agent's purpose.

    persona : str | None, optional
        Optional identity / domain context injected at the start of every
        orchestrator system prompt (planning, action, and reflection).
        Use this to give the orchestrator a specific role, area of
        expertise, or behavioural constraints.  When ``None`` (default),
        only the built-in system prompts are used.

    allow_spawn : bool, optional
        When ``True``, adds a ``spawn_agent`` tool to the agent's tool list.
        The orchestrator LLM can then call ``spawn_agent(task=..., tools=[...],
        model="...")`` to dynamically create and run child agents at runtime.
        Children inherit the parent's config by default; any field passed to
        ``spawn_agent`` overrides the inherited value.  Default ``False``.

    Dynamic sub-agents (``allow_spawn=True``)
    ------------------------------------------
    Enable spawn so the orchestrator can delegate sub-tasks at runtime::

        orchestrator = Agent(
            orchestrator = Model("claude-opus-4-6"),
            tools        = [PerplexitySearchTool()],
            mode         = "agile",
            allow_spawn  = True,          # ← adds spawn_agent tool
            name         = "orchestrator",
            persona      = (
                "You are a chief analyst.  Delegate research to a child agent "
                "via spawn_agent(task=..., tools=['perplexity_search']).  "
                "Write the final briefing yourself."
            ),
        )
        result = orchestrator.run("Write a briefing on fusion energy breakthroughs.")

    The LLM will call ``spawn_agent`` as a tool — passing a focused ``task``
    and optionally filtering the ``tools`` list or choosing a different
    ``model``.  :meth:`spawn` creates and runs the child, then returns its
    output as a plain string that the parent stores in memory like any other
    tool result.  Nesting depth is capped at ``_max_depth`` (default 5).

    Examples
    --------
    Silent execution (default)::

        result = agent.run("Summarise the top AI news in French.")

    Progress view::

        agent = Agent(orchestrator=Model("gpt-4o"), verbose=1, ...)
        result = agent.run("...")
        # [Agent] fusion_research · waterfall · tools: [brave_search]
        # [Plan]  3 steps
        #   1  ⚙  brave_search   Search for recent fusion energy news
        #   2  ◆  gpt-4o         Synthesise and summarise the findings
        #   3  ◆  gpt-4o         Translate the summary into French
        # [Step 1/3] Search for recent fusion energy news
        #   ⚙  brave_search · query="fusion energy 2025" · count=5
        #   ✓  success · continue · stored as 'search_results' · +340 tokens
        # ...

    Agent with a custom persona::

        agent = Agent(
            orchestrator = Model("gpt-4o"),
            persona      = (
                "You are a senior financial analyst specialising in tech equities. "
                "Always cite data sources and flag information older than 30 days."
            ),
        )

    Detailed view::

        agent = Agent(orchestrator=Model("gpt-4o"), verbose=2, ...)
        # Adds full prompts, output previews, and per-call token breakdowns.
    """

    MODES = ("waterfall", "agile")

    def __init__(
        self,
        orchestrator,
        tools:        list | None        = None,
        executors:    list | None        = None,
        mode:         str                = "waterfall",
        max_steps:    int                = 10,
        max_attempts: int                = 3,
        max_tokens:   int                = 50_000,
        memory:       AgentMemory | None = None,
        verbose:      int                = 0,
        name:         str | None         = None,
        description:  str | None         = None,
        persona:      str | None         = None,
        allow_spawn:  bool               = False,
        _depth:       int                = 0,
        _max_depth:   int                = 5,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(
                f"mode must be one of {self.MODES}; got {mode!r}"
            )
        if verbose not in (0, 1, 2):
            raise ValueError(
                f"verbose must be 0, 1, or 2; got {verbose!r}"
            )

        self.orchestrator = orchestrator
        self.executors    = executors or [orchestrator]
        self.mode         = mode
        self.max_steps    = max_steps
        self.max_attempts = max_attempts
        self.max_tokens   = max_tokens
        self.memory       = memory if memory is not None else AgentMemory()
        self.verbose      = verbose
        self.name         = name
        self.description  = description
        self.persona      = persona
        self.allow_spawn  = allow_spawn
        self._depth       = _depth
        self._max_depth   = _max_depth

        # ── Build tool list ───────────────────────────────────────────────
        tool_list: list = list(tools or [])
        if allow_spawn:
            # Prepend spawn_agent so the LLM sees it first in the schema list
            tool_list = [_SpawnTool(self)] + tool_list
        self.tools = tool_list

        # Fast lookup maps
        self._tool_map:     dict = {t.name: t for t in self.tools}
        self._executor_map: dict = {m.name: m for m in self.executors}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task:      str,
        variables: dict | None = None,
    ) -> AgentResult:
        """
        Execute *task* autonomously and return an :class:`AgentResult`.

        Parameters
        ----------
        task : str
            Natural-language description of what the agent should accomplish.

        variables : dict | None, optional
            Initial values seeded into memory before the first step.
            The orchestrator will see these as available context variables.

        Returns
        -------
        AgentResult
            Always returns — never raises.  Check ``result.success`` or
            ``bool(result)`` to determine outcome.
        """
        # ── Initialise state ─────────────────────────────────────────────
        self.memory.clear()
        if variables:
            self.memory.update(variables)

        history:      list[dict] = []
        tokens_used:  int        = 0
        current_plan: list[dict] = []

        # ── Header ───────────────────────────────────────────────────────
        self._log(1, "")
        self._log(1,
            f"╔══ Agent: {self.name or 'unnamed'} "
            f"│ mode={self.mode} "
            f"│ budget={self.max_tokens:,} tokens "
            + ("=" * max(0, 46 - len(self.name or "unnamed")))
        )
        self._log(1,
            f"║  Task   : "
            + textwrap.shorten(task, width=90, placeholder="…")
        )
        tool_names = [t.name for t in self.tools] or ["(none)"]
        exec_names = [m.name for m in self.executors]
        self._log(1, f"║  Tools  : {tool_names}")
        self._log(1, f"║  Execs  : {exec_names}")
        self._log(1, f"╚{'═' * 70}")

        try:
            # ── Phase 1: Planning ─────────────────────────────────────────
            self._log(1, "\n[Plan] Generating plan…")

            plan_msgs = prompts.planning_messages(
                task           = task,
                tools          = self.tools,
                executor_names = [m.name for m in self.executors],
                memory         = self.memory.all(),
                max_steps      = self.max_steps,
                persona        = self.persona,
            )
            plan_raw, plan_tokens = self._llm_call(self.orchestrator, plan_msgs)
            tokens_used += plan_tokens

            plan_data    = self._parse_json(plan_raw)
            current_plan = plan_data.get("steps", [])[:self.max_steps]

            if not current_plan:
                self._log(1, "       ✗ Orchestrator produced an empty plan.")
                return AgentResult(
                    success=False, output=None, mode=self.mode,
                    steps_taken=0, tokens_used=tokens_used,
                    plan=[], history=[], memory=self.memory.all(),
                    error="Orchestrator produced an empty plan.",
                )

            # Print plan
            self._log(1, f"[Plan] {len(current_plan)} step(s) · +{plan_tokens:,} tokens")
            if self.verbose >= 2:
                reasoning = plan_data.get("reasoning", "")
                if reasoning:
                    self._log(2, f"       reasoning: {textwrap.shorten(reasoning, 120, placeholder='…')}")
            for s in current_plan:
                icon   = "⚙" if s.get("type") == "tool" else "◆"
                target = s.get("tool_name") or s.get("model") or "default"
                goal   = textwrap.shorten(s.get("goal", ""), 60, placeholder="…")
                self._log(1, f"       {s['id']:>2}.  {icon}  {target:<22}  {goal}")

            # ── Phase 2: Step execution loop ──────────────────────────────
            step_idx = 0
            advance  = True

            while step_idx < len(current_plan):

                if tokens_used >= self.max_tokens:
                    self._log(1, "\n[Stop] Token budget exhausted.")
                    break

                step    = current_plan[step_idx]
                advance = True

                self._log(1,
                    f"\n[Step {step_idx + 1}/{len(current_plan)}]  "
                    + textwrap.shorten(step.get("goal", ""), 70, placeholder="…")
                )

                for attempt in range(1, self.max_attempts + 1):

                    if tokens_used >= self.max_tokens:
                        break

                    if attempt > 1:
                        self._log(1, f"  ↺  Retry attempt {attempt}/{self.max_attempts}")

                    context = self.memory.all()

                    # ── Determine action ─────────────────────────────────
                    # For a named tool step, pass only that tool's schema so
                    # the prompt stays focused.  All schemas are passed when
                    # the step has no named tool (e.g. after a replan).
                    step_tool_name = step.get("tool_name")
                    if step_tool_name and step.get("type") == "tool":
                        primary = [
                            t.schema() for t in self.tools
                            if t.name == step_tool_name
                        ]
                        tool_schemas_for_action = (
                            primary if primary else [t.schema() for t in self.tools]
                        )
                    else:
                        tool_schemas_for_action = [t.schema() for t in self.tools]

                    action_msgs = prompts.action_messages(
                        task                 = task,
                        step                 = step,
                        step_num             = step_idx + 1,
                        total_steps          = len(current_plan),
                        context              = context,
                        tool_schemas         = tool_schemas_for_action,
                        available_tool_names = [t.name for t in self.tools],
                        persona              = self.persona,
                    )
                    action_raw, action_tokens = self._llm_call(
                        self.orchestrator, action_msgs
                    )
                    tokens_used += action_tokens

                    action = self._parse_json(action_raw)

                    # Log action
                    self._log_action(action, action_tokens)

                    # Early exit: orchestrator already has the final answer
                    if action.get("type") == "final_answer":
                        self._log(1,
                            f"  ✓  Final answer emitted early · "
                            f"total {tokens_used:,} tokens"
                        )
                        return AgentResult(
                            success=True, output=action.get("answer"),
                            mode=self.mode, steps_taken=step_idx,
                            tokens_used=tokens_used, plan=current_plan,
                            history=history, memory=self.memory.all(),
                        )

                    # ── Execute action ────────────────────────────────────
                    exec_error:  str | None = None
                    exec_tokens: int        = 0
                    try:
                        result, exec_tokens = self._execute_action(action, context)
                    except Exception as exc:
                        result     = f"ERROR: {exc}"
                        exec_error = str(exc)

                    tokens_used += exec_tokens

                    # Log execution result
                    self._log_exec(result, exec_error, exec_tokens)

                    # ── Reflect ───────────────────────────────────────────
                    reflect_msgs = prompts.reflection_messages(
                        task             = task,
                        mode             = self.mode,
                        step             = step,
                        step_num         = step_idx + 1,
                        total_steps      = len(current_plan),
                        action           = action,
                        result           = str(result),
                        history_summary  = self._history_summary(history),
                        memory           = self.memory.all(),
                        remaining_tokens = max(0, self.max_tokens - tokens_used),
                        persona          = self.persona,
                    )
                    reflect_raw, reflect_tokens = self._llm_call(
                        self.orchestrator, reflect_msgs
                    )
                    tokens_used  += reflect_tokens
                    reflection    = self._parse_json(reflect_raw)
                    decision      = reflection.get("decision", "continue")

                    # Store result in memory using the orchestrator's suggested key
                    store_as = self._sanitise_key(reflection.get("store_as", ""))
                    if store_as and exec_error is None:
                        self.memory.set(store_as, result)

                    # Log reflection
                    self._log_reflection(reflection, store_as, reflect_tokens)

                    # Record this attempt in history
                    history.append({
                        "step":        step_idx,
                        "attempt":     attempt,
                        "step_goal":   step.get("goal", ""),
                        "action_type": action.get("type"),
                        "action":      action,
                        "output":      result,
                        "exec_error":  exec_error,
                        "stored_as":   store_as or None,
                        "reflection":  reflection,
                        "tokens":      action_tokens + exec_tokens + reflect_tokens,
                    })

                    # ── Handle decision ───────────────────────────────────

                    if decision == "final_answer":
                        self._log(1,
                            f"\n[Done] ✓ Final answer · "
                            f"{step_idx + 1} step(s) · "
                            f"{tokens_used:,} tokens total"
                        )
                        return AgentResult(
                            success=True,
                            output=reflection.get("final_answer", result),
                            mode=self.mode, steps_taken=step_idx + 1,
                            tokens_used=tokens_used, plan=current_plan,
                            history=history, memory=self.memory.all(),
                        )

                    if decision == "stop":
                        reason = reflection.get("reason", "Agent stopped.")
                        self._log(1, f"\n[Stop] ✗ {reason}")
                        return AgentResult(
                            success=False, output=None, mode=self.mode,
                            steps_taken=step_idx + 1, tokens_used=tokens_used,
                            plan=current_plan, history=history,
                            memory=self.memory.all(),
                            error=reason,
                        )

                    if decision == "replan" and self.mode == "agile":
                        revised = reflection.get("revised_plan")
                        if revised:
                            old_len      = len(current_plan)
                            current_plan = revised[:self.max_steps]
                            goto         = int(reflection.get("goto_step", step_idx))
                            step_idx     = max(0, min(goto, len(current_plan) - 1))
                            advance      = False
                            self._log(1,
                                f"  ⟳  Replanned: {old_len} → {len(current_plan)} step(s), "
                                f"resuming at step {step_idx + 1}"
                            )
                            if self.verbose >= 2:
                                for s in current_plan:
                                    icon   = "⚙" if s.get("type") == "tool" else "◆"
                                    target = s.get("tool_name") or s.get("model") or "default"
                                    goal   = textwrap.shorten(s.get("goal", ""), 55, placeholder="…")
                                    self._log(2, f"       {s['id']:>2}.  {icon}  {target:<20}  {goal}")
                        break

                    if decision == "continue":
                        break

                    if decision == "retry":
                        continue

                    break  # unknown decision → treat as continue

                if advance:
                    step_idx += 1

            # ── Phase 3: Compile result ───────────────────────────────────
            final_output = self._extract_final_output(history)
            self._log(1,
                f"\n[Done] ✓ All steps complete · "
                f"{step_idx} step(s) · "
                f"{tokens_used:,} tokens total"
            )
            self._log(2, f"       Memory keys: {sorted(self.memory.keys())}")
            return AgentResult(
                success=True, output=final_output, mode=self.mode,
                steps_taken=step_idx, tokens_used=tokens_used,
                plan=current_plan, history=history, memory=self.memory.all(),
            )

        except Exception as exc:
            self._log(1, f"\n[Error] ✗ Unexpected exception: {exc}")
            return AgentResult(
                success=False, output=None, mode=self.mode,
                steps_taken=len(history), tokens_used=tokens_used,
                plan=current_plan, history=history, memory=self.memory.all(),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # spawn() — dynamic child-agent creation
    # ------------------------------------------------------------------

    def spawn(
        self,
        task:       str,
        *,
        tools:      list | None = None,
        model                   = None,
        max_steps:  int | None  = None,
        max_tokens: int | None  = None,
    ) -> str:
        """
        Create and run a child agent for a focused sub-task.

        The child inherits the parent's configuration by default; any parameter
        you pass here overrides the inherited value.  When finished, the child's
        output is returned as a plain string so it can be stored in the parent's
        memory like any other tool result.

        Parameters
        ----------
        task : str
            The specific, self-contained task for the child to execute.

        tools : list[Tool] | None, optional
            Tool list for the child.  When omitted, the child inherits all of
            the parent's tools **excluding** ``spawn_agent`` itself (children
            do not automatically receive spawning capability).

        model : Model | None, optional
            Orchestrator model for the child.  When omitted, inherits the
            parent's orchestrator.

        max_steps : int | None, optional
            Step budget for the child.  Inherits from parent when omitted.

        max_tokens : int | None, optional
            Token budget for the child.  Inherits from parent when omitted.

        Returns
        -------
        str
            The child's output on success, or ``"[subagent failed] <reason>"``
            on failure.
        """
        if self._depth >= self._max_depth:
            return (
                f"[spawn blocked] Maximum agent nesting depth ({self._max_depth}) reached. "
                f"Cannot spawn further sub-agents."
            )

        child_tools = (
            tools if tools is not None
            else [t for t in self.tools if t.name != "spawn_agent"]
        )

        child = Agent(
            orchestrator = model or self.orchestrator,
            tools        = child_tools,
            executors    = self.executors,
            mode         = self.mode,
            max_steps    = max_steps or self.max_steps,
            max_attempts = self.max_attempts,
            max_tokens   = max_tokens or self.max_tokens,
            verbose      = self.verbose,
            _depth       = self._depth + 1,
            _max_depth   = self._max_depth,
        )
        result = child.run(task)
        return (
            str(result.output) if (result.success and result.output is not None)
            else "(child agent produced no output)"
            if result.success
            else f"[subagent failed] {result.error}"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, level: int, message: str = "") -> None:
        """Print *message* when ``self.verbose >= level``."""
        if self.verbose >= level:
            print(message, flush=True)

    def _log_action(self, action: dict, tokens: int) -> None:
        """Log the action chosen by the orchestrator for a step."""
        atype = action.get("type")

        if atype == "tool":
            kwargs_str = json.dumps(action.get("kwargs", {}), ensure_ascii=False)
            if len(kwargs_str) > 120:
                kwargs_str = kwargs_str[:117] + "…"
            self._log(1, f"  ⚙  {action.get('tool_name')}  {kwargs_str}")
            self._log(2, f"     action tokens: +{tokens:,}")

        elif atype == "skill":
            model_name = action.get("model") or "default"
            self._log(1, f"  ◆  skill  model={model_name}")
            if self.verbose >= 2:
                sys_p  = textwrap.shorten(action.get("system_prompt", ""), 100, placeholder="…")
                user_p = textwrap.shorten(action.get("user_prompt", ""),   100, placeholder="…")
                self._log(2, f"     system : {sys_p}")
                self._log(2, f"     user   : {user_p}")
                self._log(2, f"     action tokens: +{tokens:,}")

        elif atype == "final_answer":
            self._log(1, "  ✓  final_answer (early exit)")

    def _log_exec(self, result: Any, error: str | None, tokens: int) -> None:
        """Log the result of executing a step action."""
        if error:
            self._log(1, f"  ✗  exec error: {textwrap.shorten(error, 100, placeholder='…')}")
        else:
            preview = textwrap.shorten(str(result).replace("\n", " "), 110, placeholder="…")
            self._log(1, f"  →  {preview}")
            self._log(2, f"     exec tokens: +{tokens:,}")

    def _log_reflection(
        self,
        reflection: dict,
        store_as:   str,
        tokens:     int,
    ) -> None:
        """Log the orchestrator's reflection and decision."""
        assessment = reflection.get("assessment", "?")
        decision   = reflection.get("decision",   "?")

        _DECISION_ICONS = {
            "continue":     "→  next step",
            "retry":        "↺  retry",
            "replan":       "⟳  replan",
            "stop":         "✗  stop (fatal)",
            "final_answer": "✓  final answer",
        }
        decision_label = _DECISION_ICONS.get(decision, decision)
        stored_label   = f"stored as '{store_as}'" if store_as else "not stored"

        self._log(1,
            f"  ↳  {assessment}  ·  {decision_label}  ·  "
            f"{stored_label}  ·  +{tokens:,} tokens"
        )
        if self.verbose >= 2:
            reason = reflection.get("reason", "")
            if reason:
                self._log(2,
                    f"     reason : "
                    + textwrap.shorten(reason, 120, placeholder="…")
                )

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _llm_call(
        self,
        model,
        messages:      list[dict],
        output_format: str = "text",
    ) -> tuple[str, int]:
        """
        Make one LLM call and return ``(content, tokens_used)``.

        Follows the same path as :meth:`~skills.Skill.run`:
        ``to_request`` → ``client._post`` → ``from_response``.
        Token usage is extracted from the raw provider response.
        """
        output   = {"modalities": ["text"], "format": {"type": output_format}}
        path, body = model.to_request(messages, output)
        raw        = model.client._post(path, body, model.client._auth_headers())
        response   = json.loads(raw)
        content    = model.from_response(response, output)
        tokens     = self._extract_tokens(response)
        return content, tokens

    def _extract_tokens(self, response: dict) -> int:
        """Extract total token count from a raw provider response dict."""
        # Anthropic / OpenAI style
        if "usage" in response:
            usage = response["usage"]
            inp   = usage.get("input_tokens")  or usage.get("prompt_tokens")     or 0
            out   = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            return int(inp) + int(out)
        # Google style
        if "usageMetadata" in response:
            return int(response["usageMetadata"].get("totalTokenCount", 0))
        return 0

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self,
        action:  dict,
        context: dict,
    ) -> tuple[Any, int]:
        """
        Execute a tool or skill action.

        Returns ``(result, tokens_used)``.  Tool calls return 0 tokens.
        Raises on error — the caller catches and records the exception.
        """
        atype = action.get("type")

        # ── Tool call ────────────────────────────────────────────────────
        if atype == "tool":
            tool = self._find_tool(action.get("tool_name", ""))
            if tool is None:
                raise ValueError(
                    f"Tool not found: {action.get('tool_name')!r}. "
                    f"Available: {list(self._tool_map.keys())}"
                )
            raw_kwargs = action.get("kwargs", {})
            kwargs = {}
            for k, v in raw_kwargs.items():
                if isinstance(v, str):
                    kwargs[k] = v.format_map(_SafeFormatMap(context))
                else:
                    kwargs[k] = v

            result = tool.run(**kwargs)
            return result, 0

        # ── Skill call (dynamic LLM call) ─────────────────────────────
        if atype == "skill":
            model         = self._select_executor(action.get("model", ""))
            system_prompt = action.get("system_prompt", "")
            user_prompt   = action.get("user_prompt", "")
            out_format    = action.get("output_format", "text")

            system_prompt = system_prompt.format_map(_SafeFormatMap(context))
            user_prompt   = user_prompt.format_map(_SafeFormatMap(context))

            messages: list[dict] = []
            if system_prompt:
                messages.append({
                    "role":  "system",
                    "parts": [{"type": "text", "text": system_prompt}],
                })
            messages.append({
                "role":  "user",
                "parts": [{"type": "text", "text": user_prompt}],
            })

            result, tokens = self._llm_call(model, messages, output_format=out_format)
            return result, tokens

        raise ValueError(f"Unknown action type: {atype!r}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_tool(self, name: str):
        return self._tool_map.get(name)

    def _select_executor(self, hint: str):
        if hint and hint in self._executor_map:
            return self._executor_map[hint]
        return self.executors[0]

    def _parse_json(self, text: str) -> dict:
        """
        Robustly parse JSON from an LLM response.

        Tries three strategies in order:
        1. Direct ``json.loads`` of the full text.
        2. Extraction from a ```json ... ``` code fence.
        3. Extraction of the first ``{...}`` block in the text.
        """
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"Could not parse JSON from orchestrator response.\n"
            f"Raw text (first 400 chars): {text[:400]!r}"
        )

    @staticmethod
    def _sanitise_key(key: str) -> str:
        """Convert an arbitrary string into a valid snake_case memory key."""
        if not key:
            return ""
        key = re.sub(r"[^a-z0-9_]", "_", key.lower())
        key = re.sub(r"_+", "_", key).strip("_")
        return key

    @staticmethod
    def _is_important(rec: dict) -> bool:
        """
        Return True when a history record represents a significant event.

        A record is important when:
        - The step produced an execution error.
        - The reflection assessment is ``"failure"``, ``"fatal"``, or
          ``"partial"``.
        - The reflection decision is ``"replan"``, ``"stop"``, or
          ``"final_answer"``.
        """
        if rec.get("exec_error") is not None:
            return True
        reflection = rec.get("reflection", {})
        if reflection.get("assessment") in {"failure", "fatal", "partial"}:
            return True
        if reflection.get("decision") in {"replan", "stop", "final_answer"}:
            return True
        return False

    def _history_summary(self, history: list[dict]) -> str:
        """
        Concise text summary of the most relevant prior steps for reflection
        prompts.

        Selection algorithm:
        1. Collect all *important* records (failures, fatal errors, significant
           decisions) and always include the last 3 records.
        2. Merge the two sets in original order, deduplicate, and cap at 8.
        3. Records flagged as important are prefixed with ``[!]`` so the
           orchestrator can distinguish them from routine steps.

        Returns ``"(no prior steps)"`` when history is empty or the selected
        set would otherwise be empty.
        """
        if not history:
            return "(no prior steps)"

        important = [r for r in history if self._is_important(r)]
        recent    = history[-3:]

        seen:     set   = set()
        selected: list  = []
        for rec in important + recent:
            key = (rec["step"], rec["attempt"])
            if key not in seen:
                seen.add(key)
                selected.append(rec)

        selected = selected[-8:]

        if not selected:
            return "(no prior steps)"

        lines = []
        for rec in selected:
            assessment = rec["reflection"].get("assessment", "?")
            stored     = rec.get("stored_as") or "—"
            goal       = rec["step_goal"][:70]
            flag       = "[!] " if self._is_important(rec) else "    "
            lines.append(
                f"{flag}Step {rec['step']+1} attempt {rec['attempt']} "
                f"[{rec['action_type']}] {goal!r}: "
                f"{assessment}, stored as '{stored}'"
            )
        return "\n".join(lines)

    def _extract_final_output(self, history: list[dict]) -> Any:
        """Return the last successful step's output."""
        if not history:
            return None
        for rec in reversed(history):
            if rec.get("exec_error") is None and rec.get("output") is not None:
                return rec["output"]
        return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        tool_names = [t.name for t in self.tools] or ["(none)"]
        execs      = [m.name for m in self.executors]
        parts      = [
            f"mode={self.mode!r}",
            f"orchestrator={self.orchestrator.name!r}",
            f"executors={execs}",
            f"tools={tool_names}",
            f"max_steps={self.max_steps}",
            f"max_tokens={self.max_tokens:,}",
            f"verbose={self.verbose}",
        ]
        if self.allow_spawn:
            parts.append("allow_spawn=True")
        if self.persona:
            short = (
                self.persona[:60] + "…"
                if len(self.persona) > 60 else self.persona
            )
            parts.append(f"persona={short!r}")
        return f"Agent({', '.join(parts)})"
