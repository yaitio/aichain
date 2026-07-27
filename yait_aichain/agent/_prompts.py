"""
agent._prompts
==============

Orchestrator prompt builders.

Each function returns a ``messages`` list in the universal format used by
:meth:`~models.Model.to_request`.  Every call type produces JSON output so
the agent can reliably parse the orchestrator's decisions.

Call types
----------
planning        — given a task, produce an ordered step plan
action          — given a step, decide the exact tool kwargs or skill prompt
reflection      — given a step result, assess and decide what to do next

Each builder accepts an optional ``persona`` string that is prepended to the
system prompt, giving the orchestrator a configurable identity, domain
context, or behavioural guidelines.
"""

from __future__ import annotations
import json
import re


# ---------------------------------------------------------------------------
# Per-call-type system prompts
# ---------------------------------------------------------------------------

_SYSTEM_PLAN = (
    "You are an autonomous AI agent orchestrator. "
    "Your role at this stage is to produce a minimal, focused execution plan. "
    "Prefer fewer steps over many — each step must have a clear, achievable goal "
    "that builds directly on the previous one. "
    "Economy and clarity matter more than exhaustive coverage. "
    "You always respond with valid, parseable JSON and nothing else — "
    "no prose, no code fences, no explanations outside the JSON structure."
)

_SYSTEM_ACTION = (
    "You are an autonomous AI agent orchestrator. "
    "Your role at this stage is to determine the exact, executable action for the "
    "current step. "
    "Be precise with parameter names and values — use only the names listed in the "
    "tool signatures. "
    "When injecting context variables use {variable_name} single-brace syntax; "
    "for URL parameters read the URL directly from the context preview and write it "
    "as a literal string. "
    "You always respond with valid, parseable JSON and nothing else — "
    "no prose, no code fences, no explanations outside the JSON structure."
)

_SYSTEM_REFLECT = (
    "You are an autonomous AI agent orchestrator. "
    "Your role at this stage is to honestly assess the step result and choose the "
    "most conservative decision that still makes progress. "
    "Do not retry steps that failed due to unrecoverable errors such as missing "
    "credentials, invalid tool names, or impossible requests. "
    "Only use 'replan' when the current plan is structurally wrong, not merely "
    "because a step returned partial results. "
    "You always respond with valid, parseable JSON and nothing else — "
    "no prose, no code fences, no explanations outside the JSON structure."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memory_block(
    memory:   dict,
    hint:     str = "",
    max_keys: int = 12,
) -> str:
    """
    Render memory as a compact JSON block, truncating large values.

    When the memory contains more than *max_keys* entries and a *hint* is
    provided, the keys most relevant to the hint are selected (scored by
    word overlap between the key name and the hint text).  If *hint* is
    empty or the memory is small enough, all keys are included up to
    *max_keys*.

    Parameters
    ----------
    memory   : The full memory dict.
    hint     : Short context string (e.g. the current step goal) used to
               score key relevance.  Default ``""``.
    max_keys : Maximum number of keys to include.  Default 12.
    """
    if not memory:
        return "{}"

    if len(memory) <= max_keys or not hint:
        selected = memory
    else:
        hint_words = set(re.sub(r"[^a-z0-9]", " ", hint.lower()).split())

        def _score(key: str) -> int:
            key_words = set(re.sub(r"[^a-z0-9]", " ", key.lower()).split())
            return len(hint_words & key_words)

        items = list(memory.items())
        items.sort(key=lambda kv: -_score(kv[0]))
        selected = dict(items[:max_keys])

    preview = {}
    for k, v in selected.items():
        s = str(v)
        preview[k] = s if len(s) <= 500 else s[:497] + "..."
    return json.dumps(preview, indent=2, ensure_ascii=False)


def _vars_list(context: dict) -> str:
    """
    Render available variable names and value previews.

    Shows up to 500 characters per variable so the orchestrator can read
    URLs and other actionable content directly from search results without
    needing to inject the whole variable via {variable_name} syntax.
    """
    if not context:
        return "(none)"
    lines = []
    for k, v in context.items():
        s = str(v)
        if len(s) <= 500:
            preview = s
        else:
            # Say that this is a preview and how to get the rest. A truncation
            # the reader cannot see is indistinguishable from a short value.
            preview = (s[:497] + "…"
                       + f"\n      [preview — {len(s):,} characters total; "
                         f"read the rest with memory_read('{k}')]")
        # Show multi-line values indented
        if "\n" in preview:
            indented = preview.replace("\n", "\n      ")
            lines.append(f"  {k}:\n      {indented}")
        else:
            lines.append(f"  {k}: {preview!r}")
    return "\n".join(lines)


def _truncate_result(text: str, max_chars: int = 2000) -> str:
    """
    Middle-preserving truncation for tool and skill results.

    If *text* is within *max_chars*, it is returned unchanged.  Otherwise,
    the first ``head`` and last ``tail`` characters are kept
    (``head = tail = (max_chars // 2) - 30``) and joined with a one-line
    omission notice.  This preserves both the opening context (usually the
    query echo or document title) and the tail (often URLs, conclusions, or
    structured data).

    Parameters
    ----------
    text      : The string to truncate.
    max_chars : Maximum total character budget before truncation.
                Default 2000.
    """
    if len(text) <= max_chars:
        return text
    half    = (max_chars // 2) - 30
    head    = half
    tail    = half
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n… [{omitted} characters omitted] …\n"
        + text[-tail:]
    )


# ---------------------------------------------------------------------------
# 1. Planning
# ---------------------------------------------------------------------------

def planning_messages(
    task:           str,
    tools:          list,
    executor_names: list[str],
    memory:         dict,
    max_steps:      int,
    persona:        str | None = None,
) -> list[dict]:
    """
    Build the messages for the initial planning call.

    The orchestrator reads the task, available tools, executor models, and
    current memory, then returns a JSON plan.

    Parameters
    ----------
    task           : Natural-language task description.
    tools          : List of Tool instances available for tool steps.
    executor_names : Names of executor models available for skill steps.
    memory         : Current memory state (seeded before planning).
    max_steps      : Hard cap on number of plan steps.
    persona        : Optional agent persona / system context injected before
                     the planning system prompt.  When ``None``, only the
                     default planning instructions are used.
    """
    tools_desc = "\n".join(
        f"  • {t.name}: {t.description}"
        for t in tools
    ) or "  (no tools available)"

    executors_desc = ", ".join(executor_names) if executor_names else "default"

    system_text = (
        persona + "\n\n" + _SYSTEM_PLAN
        if persona else _SYSTEM_PLAN
    )

    user = f"""Create a step-by-step plan to accomplish the following task.

TASK:
{task}

AVAILABLE TOOLS:
{tools_desc}

AVAILABLE EXECUTOR MODELS (for LLM skill steps):
{executors_desc}

CURRENT MEMORY:
{_memory_block(memory, hint=task[:120])}

Respond with this exact JSON structure:
{{
  "reasoning": "Brief analysis of the task and your approach",
  "steps": [
    {{
      "id": 1,
      "goal": "Precise description of what this step achieves",
      "type": "tool",
      "tool_name": "exact tool name from the list above"
    }},
    {{
      "id": 2,
      "goal": "Precise description of what this step achieves",
      "type": "skill",
      "model": "preferred executor model name, or empty string for default"
    }}
  ]
}}

Rules:
  - Use type "tool" for: web search, fetching URLs, file operations, external APIs
  - Use type "skill" for: reasoning, writing, summarizing, translating, analyzing
  - Use type "skill" when no suitable tool exists for the step
  - Maximum {max_steps} steps — prefer fewer, more focused steps
  - Each step must build logically on the outputs of previous steps
  - Be specific in each step's "goal" — one clear sentence"""

    return [
        {"role": "system", "parts": [{"type": "text", "text": system_text}]},
        {"role": "user",   "parts": [{"type": "text", "text": user}]},
    ]


# ---------------------------------------------------------------------------
# 2. Action determination
# ---------------------------------------------------------------------------

def _tool_signature(schema: dict) -> str:
    """
    Render one tool as a compact human-readable signature line, e.g.:

        markitdown  →  source=<string> [required], output_path=<string> [optional]

    This is far more reliable than a raw JSON schema blob for preventing
    the model from substituting intuitive-but-wrong parameter names.
    """
    fn       = schema.get("function", {})
    name     = fn.get("name", "unknown")
    params   = fn.get("parameters", {})
    props    = params.get("properties", {})
    required = params.get("required", [])
    parts    = []
    for pname, spec in props.items():
        label = "required" if pname in required else "optional"
        desc  = spec.get("description", "")
        short = desc.split(".")[0][:55] if desc else ""
        parts.append(
            f"{pname}=<{spec.get('type', 'any')}> [{label}]"
            + (f" — {short}" if short else "")
        )
    sig = ", ".join(parts) or "(no parameters)"
    return f"  {name}  →  {sig}"


def action_messages(
    task:                 str,
    step:                 dict,
    step_num:             int,
    total_steps:          int,
    context:              dict,
    tool_schemas:         list | None = None,
    available_tool_names: list | None = None,
    persona:              str | None  = None,
    do_not_redo:          str         = "",
) -> list[dict]:
    """
    Build the messages that ask the orchestrator to decide the exact action
    for a single step.

    For a tool step, it returns kwargs.
    For a skill step, it returns the prompt to send to the executor model.
    It may also emit ``"final_answer"`` if the answer is already in context.

    Parameters
    ----------
    task                 : Overall task description.
    step                 : Current plan step dict.
    step_num             : 1-based step index.
    total_steps          : Total number of steps in the current plan.
    context              : Current memory state (all variables).
    tool_schemas         : Tool schemas to render in the prompt.  When
                           Improvement 4 is in effect the caller passes only
                           the primary tool's schema for tool steps (or all
                           schemas when the step has no named tool).
    available_tool_names : Full list of registered tool names shown as a
                           guard against hallucinated names.
    persona              : Optional agent persona prepended to the system
                           prompt.
    """
    # ── All tools block (names + exact parameter signatures) ──────────────
    if tool_schemas:
        sig_lines   = "\n".join(_tool_signature(s) for s in tool_schemas)
        tools_block = (
            f"\nAVAILABLE TOOLS — names and EXACT parameter names to use:\n"
            f"{sig_lines}\n"
            f"  ↑ Use ONLY these tool names and ONLY these parameter names.\n"
            f"    Never invent names like 'web_search', 'url', 'link', etc.\n"
        )
    else:
        tools_block = "\nAVAILABLE TOOLS: (none — use skill or final_answer)\n"

    # ── Step-level hint ───────────────────────────────────────────────────
    step_tool_line = ""
    if step.get("type") == "tool" and step.get("tool_name"):
        step_tool_line = f"  Tool : {step['tool_name']}  ← use this exact name\n"

    system_text = (
        persona + "\n\n" + _SYSTEM_ACTION
        if persona else _SYSTEM_ACTION
    )

    # Approaches already ruled out — omitted entirely when nothing was refuted.
    ruled_out_block = (
        f"\nALREADY RULED OUT — do NOT attempt these again:\n{do_not_redo}\n"
        if do_not_redo else ""
    )

    user = f"""Determine the exact action to take for the current step.

OVERALL TASK:
{task}
{tools_block}
CURRENT STEP ({step_num} of {total_steps}):
  Goal : {step.get('goal', '')}
  Type : {step.get('type', '')}
{step_tool_line}{ruled_out_block}
AVAILABLE CONTEXT VARIABLES:
{_vars_list(context)}

Respond with EXACTLY ONE of the following JSON formats:

── For a tool step ──────────────────────────────────────────────────────────
{{
  "type": "tool",
  "tool_name": "MUST be one of the names listed in AVAILABLE TOOL NAMES above",
  "kwargs": {{
    "param_name": "literal value, e.g. a URL you read from the context above"
  }}
}}

IMPORTANT — variable injection:
  Use {{variable_name}} (single braces) to inject a context variable into a value.
  The value is substituted at run time before the tool is called.
  Example: {{"source": "{{search_results}}"}} injects the search_results variable.
  Never use double braces {{{{...}}}} — that is wrong and will not substitute.
  For tool steps that need a URL: read the URL directly from the context variable
  preview above and write it as a literal string — do not use variable injection
  for URLs.

── For a skill (LLM) step ───────────────────────────────────────────────────
{{
  "type": "skill",
  "model": "model name, or empty string for default",
  "system_prompt": "the system prompt for the executor LLM",
  "user_prompt": "the user prompt — use {{variable_name}} (single braces) to inject context variables",
  "output_format": "text"
}}

── To provide the final answer immediately ──────────────────────────────────
(use this when the context already contains everything needed for a complete answer)
{{
  "type": "final_answer",
  "answer": "the complete final answer to the task"
}}"""

    return [
        {"role": "system", "parts": [{"type": "text", "text": system_text}]},
        {"role": "user",   "parts": [{"type": "text", "text": user}]},
    ]


# ---------------------------------------------------------------------------
# 3. Reflection
# ---------------------------------------------------------------------------

def reflection_messages(
    task:             str,
    mode:             str,
    step:             dict,
    step_num:         int,
    total_steps:      int,
    action:           dict,
    result:           str,
    history_summary:  str,
    memory:           dict,
    remaining_tokens: int,
    persona:          str | None = None,
) -> list[dict]:
    """
    Build the messages for the post-step reflection call.

    The orchestrator evaluates the step result and decides whether to
    continue, retry, replan (agile only), stop (waterfall only),
    or emit the final answer.

    Parameters
    ----------
    task             : Overall task description.
    mode             : Execution mode (``"waterfall"`` or ``"agile"``).
    step             : Current plan step dict.
    step_num         : 1-based step index.
    total_steps      : Total number of steps in the current plan.
    action           : The action dict that was executed.
    result           : String representation of the execution result.
    history_summary  : Condensed prior-step summary from
                       ``Agent._history_summary()``.
    memory           : Current memory state.
    remaining_tokens : Remaining token budget.
    persona          : Optional agent persona prepended to the system prompt.
    """
    result_preview = _truncate_result(str(result))

    action_summary = f"type={action.get('type')}"
    if action.get("type") == "tool":
        action_summary += f", tool={action.get('tool_name')}, kwargs={action.get('kwargs', {})}"
    elif action.get("type") == "skill":
        action_summary += f", model={action.get('model') or 'default'}"

    # Build mode-specific decision options
    if mode == "waterfall":
        decision_options = (
            '  - "continue"     : step succeeded, proceed to next planned step\n'
            '  - "retry"        : step result was poor, retry it\n'
            '  - "stop"         : fatal issue — the task cannot be completed (waterfall only)\n'
            '  - "final_answer" : context already contains a complete answer, no more steps needed'
        )
        revised_plan_note = ""
    else:  # agile
        decision_options = (
            '  - "continue"     : step succeeded, proceed to next planned step\n'
            '  - "retry"        : step result was poor, retry it\n'
            '  - "replan"       : plan needs adjustment — update revised_plan accordingly (agile only)\n'
            '  - "final_answer" : context already contains a complete answer, no more steps needed'
        )
        revised_plan_note = (
            '\n  "revised_plan": [{"id":1,"goal":"...","type":"tool|skill",...}]'
            '   ← required when decision is "replan"\n'
            '  "goto_step": 0   ← 0-based index in revised_plan to resume from (default: current)'
        )

    step_goal = step.get("goal", "")
    system_text = (
        persona + "\n\n" + _SYSTEM_REFLECT
        if persona else _SYSTEM_REFLECT
    )

    user = f"""Assess the completed step and decide how to proceed.

TASK:
{task}

MODE: {mode}
REMAINING TOKEN BUDGET: ~{remaining_tokens:,}

COMPLETED STEP ({step_num} of {total_steps}):
  Goal   : {step_goal}
  Action : {action_summary}

STEP RESULT:
{result_preview}

PRIOR STEPS SUMMARY:
{history_summary or '(this is the first step)'}

CURRENT MEMORY:
{_memory_block(memory, hint=step_goal[:120])}

Respond with this JSON:
{{
  "assessment": "success / partial / failure / fatal",
  "store_as": "snake_case_variable_name to store this result in memory",
  "decision": "continue / retry / replan / stop / final_answer",
  "reason": "brief explanation of your assessment and decision",
  "final_answer": "complete answer to the task"   ← only when decision is "final_answer"
{revised_plan_note}
}}

Assessment guide:
  - "success" : result fully achieves the step goal
  - "partial" : result is useful but incomplete
  - "failure" : result is wrong or empty, step should be retried
  - "fatal"   : step cannot succeed regardless of retries (e.g. tool not found, impossible request)

Decision options:
{decision_options}

Important:
  - Always provide "store_as" so the result can be referenced in later steps
  - In waterfall mode, only use "stop" for truly unrecoverable fatal failures
  - In agile mode, use "replan" sparingly — only when the current plan is structurally wrong"""

    return [
        {"role": "system", "parts": [{"type": "text", "text": system_text}]},
        {"role": "user",   "parts": [{"type": "text", "text": user}]},
    ]


# ---------------------------------------------------------------------------
# Goal loop — no upfront plan; the objective and the journal drive each step
# ---------------------------------------------------------------------------

_SYSTEM_GOAL_ACT = (
    "You are an autonomous agent pursuing one objective. There is no fixed plan: "
    "you decide the single next action from what has been achieved so far. "
    "Prefer the cheapest action that produces new information or moves toward the "
    "done condition. Never repeat an approach listed as ruled out. "
    "Respond with ONE JSON object and nothing else."
)

_SYSTEM_GOAL_ASSESS = (
    "You assess the result of one action against an objective. Be strict: only "
    "report the done condition as met when the evidence actually supports it. "
    "If an approach has been shown not to work, say so explicitly so it is never "
    "retried. Respond with ONE JSON object and nothing else."
)


def goal_action_messages(
    objective:            str,
    done_when:            str,
    progress:             str,
    ruled_out:            str,
    context:              dict,
    iteration:            int,
    max_iterations:       int,
    repeating:            bool = False,
    tool_schemas:         list | None = None,
    available_tool_names: list | None = None,
    persona:              str | None  = None,
) -> list[dict]:
    """
    Ask the orchestrator for the next action in a goal loop.

    Unlike :func:`action_messages` there is no step to execute — the model picks
    the action from the objective, what has already landed (*progress*) and what
    has been ruled out (*ruled_out*).
    """
    if tool_schemas:
        sig_lines   = "\n".join(_tool_signature(s) for s in tool_schemas)
        tools_block = (
            f"\nAVAILABLE TOOLS — names and EXACT parameter names to use:\n"
            f"{sig_lines}\n"
            f"  ↑ Use ONLY these tool names and ONLY these parameter names.\n"
        )
    else:
        tools_block = "\nAVAILABLE TOOLS: (none — use skill or final_answer)\n"

    ruled_block = (f"\nALREADY RULED OUT — never attempt these again:\n{ruled_out}\n"
                   if ruled_out else "")
    # Surfaced, not enforced: the loop reports the pattern and the model decides.
    repeat_block = ("\nSTOP AND RECONSIDER — your recent attempts have all been the "
                    "same move. Repeating it will not produce new information. Either "
                    "change approach substantially, or, if the objective cannot be "
                    "reached with the tools you have, emit final_answer stating "
                    "plainly what you established and what remains open.\n"
                    if repeating else "")
    progress_block = progress or "(nothing achieved yet)"

    system_text = (persona + "\n\n" + _SYSTEM_GOAL_ACT) if persona else _SYSTEM_GOAL_ACT

    user = f"""Decide the single next action.

OBJECTIVE:
{objective}

DONE WHEN:
{done_when}

PROGRESS SO FAR:
{progress_block}
{ruled_block}{repeat_block}{tools_block}
AVAILABLE CONTEXT VARIABLES:
{_vars_list(context)}

Iteration {iteration} of at most {max_iterations}.

Respond with EXACTLY ONE of the following JSON formats:

── To call a tool ───────────────────────────────────────────────────────────
{{
  "intent": "what this action is meant to achieve, in a few words",
  "type": "tool",
  "tool_name": "MUST be one of the names listed above",
  "kwargs": {{"param_name": "literal value"}}
}}

── To reason / generate with the model ──────────────────────────────────────
{{
  "intent": "what this action is meant to achieve",
  "type": "skill",
  "system_prompt": "role for the executor model",
  "user_prompt": "the actual request; use {{variable}} to inject context"
}}

── When the objective is already achieved ───────────────────────────────────
{{
  "intent": "finish",
  "type": "final_answer",
  "answer": "the complete result"
}}
"""
    return [
        {"role": "system", "parts": [{"type": "text", "text": system_text}]},
        {"role": "user",   "parts": [{"type": "text", "text": user}]},
    ]


def goal_assess_messages(
    objective:  str,
    done_when:  str,
    intent:     str,
    result:     str,
    progress:   str,
    remaining_tokens: int,
    persona:    str | None = None,
) -> list[dict]:
    """Assess one iteration: record the outcome, and judge the done condition."""
    system_text = (persona + "\n\n" + _SYSTEM_GOAL_ASSESS) if persona else _SYSTEM_GOAL_ASSESS

    user = f"""Assess the action just taken.

OBJECTIVE:
{objective}

DONE WHEN:
{done_when}

THIS ACTION AIMED TO:
{intent}

RESULT:
{_truncate_result(str(result))}

PROGRESS SO FAR:
{progress or "(nothing achieved yet)"}

Remaining budget: ~{remaining_tokens:,} tokens.

Respond with ONE JSON object:
{{
  "outcome": "done" | "failed" | "refuted",
  "assessment": "one short sentence on what happened",
  "reason": "REQUIRED when outcome is failed or refuted — why it did not work",
  "store_as": "snake_case key to store the result under, or empty to discard",
  "objective_met": true | false,
  "final_answer": "the complete result — REQUIRED when objective_met is true"
}}

Choosing the outcome — this is the part that matters. Ask what the action was
testing, then ask what the result did to it:

  "done"    — the action ran and returned information — even negative
              information — and nothing was ruled out. A probe answered
              "wrong, go higher" narrowed a search: that is progress, and the
              method it belongs to is still alive.
  "failed"  — the action could not run, or returned nothing usable. Transient;
              it may be worth retrying.
  "refuted" — the result DISPROVED the idea the action was testing. That
              hypothesis, rule or approach is now dead and must never be
              proposed again. Write the dead idea itself into "reason",
              precisely enough that a later step will recognise it.

The line falls between a candidate and the idea behind it. Guessing 437 and
being told "higher" kills the candidate, not the search → "done". Submitting a
rule and being told it does not fit kills the rule itself → "refuted", with the
rule written into "reason". Both actions ran perfectly; only one closed a door.

"refuted" is about the IDEA, never about the action's own execution. If the
action did not actually test what it set out to test — the wrong example was
chosen, the result was ambiguous — that is "failed": the intent was not
achieved and nothing was ruled out. Only write "refuted" when you can name the
idea that is now dead and say what killed it. If your "reason" would contain
"does not refute", "not yet" or "still possible", the outcome is NOT "refuted".
"""
    return [
        {"role": "system", "parts": [{"type": "text", "text": system_text}]},
        {"role": "user",   "parts": [{"type": "text", "text": user}]},
    ]
