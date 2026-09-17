"""
The names a program reaches for that this library does not have, and the one to use.

The tables are measured rather than imagined. `evals/agent_builds` ran a model
on twenty tasks with nothing but `SKILL.md` and `llms.txt`, and most of its
failures were not wrong ideas but familiar names: `messages` written with
`content`, `from yait_aichain.hooks import Tracer`, `result.variables` on a
paused chain, `max_steps=` and `orchestrator=` from the 1.x agent. Each of
those used to end in Python's own `TypeError` or `AttributeError`, which says
what is wrong and nothing about what is right. A model — or a person — then
guesses again.

Two rules hold every entry:

* **A wrong name is refused, never accepted.** An alias quietly turns one API
  into two, and in a year the docs, the examples and the programs agents write
  disagree about which one is real. The hint names the call that works; the
  caller writes it.
* **A hint never shadows a real name.** `refuse` checks at import that no
  keyword it hints about is a parameter of the function it guards, and the
  attribute hints only run when normal lookup has already failed.
"""

from __future__ import annotations

import functools
import importlib.abc
import inspect
import sys

# The model options by name. `models` imports this module, so it cannot import
# `models._options` back; tests/test_hints.py fails if the two lists differ.
MODEL_OPTIONS = ("temperature", "top_p", "top_k", "max_tokens", "reasoning",
                 "cache_control", "cache_ttl")

_STEP_LIMIT = ("an agent's limit is stop_when=[step_count(n)], with "
               "`from yait_aichain.agent import step_count`")
_SYSTEM_IN_SKILL = ("a Skill has no system argument: prompt= is one user message; for a "
                    "system message use input={'messages': [{'role': 'system', 'parts': "
                    "['...']}, {'role': 'user', 'parts': ['...']}]}")
_STRUCTURED = "output={'format': {'type': 'json_schema', 'schema': {...}}}"
_BUDGET = "a ceiling in money is max_cost= (dollars)"
_HOOKS = "observe a run with hooks=[Tracer()], `from yait_aichain import Tracer`"


def _model_option(owner: str) -> dict:
    where = {"Model": "Model(name, options={{{key!r}: ...}})",
             "Skill": "Skill(Model(name, options={{{key!r}: ...}}), ...)",
             "Agent": "Agent(Model(name, options={{{key!r}: ...}}), ...)"}[owner]
    return {key: "a sampling option belongs to the model: " + where.format(key=key)
            for key in MODEL_OPTIONS}


#: owner → keyword → what to write instead.
KEYWORDS: dict = {
    "Model": {
        **_model_option("Model"),
        "base_url": "a server address goes in client_options={'url': ...}, or PRIVATE_BASE_URL for 'private/...' models",
        "url": "a server address goes in client_options={'url': ...}",
        "endpoint": "a server address goes in client_options={'url': ...}",
        "provider": "the provider is read from the name: Model('claude-sonnet-4-6'), Model('private/<model-id>')",
        "model": "the model name is the first argument: Model('gpt-5.4-mini')",
        "model_name": "the model name is the first argument: Model('gpt-5.4-mini')",
        "key": "the key is api_key=, or the provider's environment variable",
    },
    "Skill": {
        **_model_option("Skill"),
        "system": _SYSTEM_IN_SKILL,
        "system_prompt": _SYSTEM_IN_SKILL,
        "instructions": _SYSTEM_IN_SKILL,
        "messages": "messages go in input={'messages': [{'role': 'user', 'parts': ['...']}]}",
        "template": "a template is prompt='... {name}', filled by run(variables={'name': ...})",
        "response_format": _STRUCTURED,
        "schema": _STRUCTURED,
        "output_schema": _STRUCTURED,
        "json_schema": _STRUCTURED,
        "tools": "a Skill makes one model call; to let a model call tools, use Agent(model, tools=[...])",
        "budget": _BUDGET,
    },
    "Agent": {
        **_model_option("Agent"),
        "max_steps": _STEP_LIMIT,
        "max_iterations": _STEP_LIMIT,
        "max_turns": _STEP_LIMIT,
        "max_attempts": _STEP_LIMIT,
        "step_limit": _STEP_LIMIT,
        "orchestrator": ("removed in 2.0: the Agent is the loop itself; give it tools=, "
                         "and team=[worker_agent, ...] to hand sub-tasks to other agents"),
        "executors": "workers are team=[worker_agent, ...]",
        "workers": "workers are team=[worker_agent, ...]",
        "sub_agents": "workers are team=[worker_agent, ...]",
        "agents": "workers are team=[worker_agent, ...]",
        "memory": ("removed in 2.0: an agent's state is its conversation; put what it "
                   "needs in the task, or behind a tool"),
        "system": "an agent's standing instructions are instructions='...'",
        "system_prompt": "an agent's standing instructions are instructions='...'",
        "prompt": "the task is passed to run: agent.run('...')",
        "task": "the task is passed to run: agent.run('...')",
        "llm": "the model is the first argument: Agent(Model('...'), tools=[...])",
        "approver": "a gate is approve=fn together with permissions=PermissionPolicy({...})",
        "on_approval": "a gate is approve=fn together with permissions=PermissionPolicy({...})",
        "callbacks": _HOOKS,
        "tracer": _HOOKS,
        "budget": _BUDGET,
    },
    "Chain": {
        "on_error": "a Chain's error policy is on_step_error= (a Pool's is on_error=)",
        "memory": "a pause that survives the process needs store=FileStore('dir'), `from yait_aichain.state import FileStore`",
        "checkpointer": "a pause that survives the process needs store=FileStore('dir'), `from yait_aichain.state import FileStore`",
        "state": "a pause that survives the process needs store=FileStore('dir'), `from yait_aichain.state import FileStore`",
        "callbacks": _HOOKS,
        "budget": _BUDGET,
    },
    "Pool": {
        "concurrency": "parallelism is max_flows=",
        "max_workers": "parallelism is max_flows=",
        "max_concurrency": "parallelism is max_flows=",
        "workers": "parallelism is max_flows=",
        "on_step_error": "a Pool's error policy is on_error= (a Chain's is on_step_error=)",
        "inputs": "the work is items=[{...}, ...], one dict of variables per item",
        "tasks": "the work is items=[{...}, ...]; for an Agent runner each item carries 'task'",
        "budget": _BUDGET,
    },
}

_RUN_AGENT = "agent.run(task) returns an AgentResult; the answer is its .output"
_RUN_SKILL = "skill.run(variables={...}) returns the value"
_OUTPUT = "the answer is .output"
_SO_FAR = "what the run computed so far is result.document['variables']"

#: owner → attribute → what to use instead.
ATTRIBUTES: dict = {
    "Agent": {
        "chat": _RUN_AGENT, "invoke": _RUN_AGENT, "ask": _RUN_AGENT, "query": _RUN_AGENT,
        "suspend": ("nothing suspends a run from outside it; a run parks when a tool "
                    "the model called asks the world for an answer — give the agent a "
                    "Wait or Gate tool and store=FileStore('dir'), then resume it with "
                    "agent.resume(run_id, signal={...})"),
        "arun": "there is no async API; run agents in parallel with Pool(agent, items=[{'task': ...}])",
        "ainvoke": "there is no async API; run agents in parallel with Pool(agent, items=[{'task': ...}])",
    },
    "Skill": {
        "invoke": _RUN_SKILL, "chat": _RUN_SKILL, "generate": _RUN_SKILL,
        "ask": _RUN_SKILL, "predict": _RUN_SKILL, "complete": _RUN_SKILL,
        "arun": "there is no async API; run many inputs in parallel with Pool(skill, items=[...])",
        "ainvoke": "there is no async API; run many inputs in parallel with Pool(skill, items=[...])",
    },
    "SuspendedResult": {
        "variables": _SO_FAR,
        "state": _SO_FAR,
        "output": ("the run paused and has no output yet: resume it with "
                   "chain.resume(result.run_id, signal={...}); " + _SO_FAR),
    },
    "AgentResult": {"text": _OUTPUT, "content": _OUTPUT, "answer": _OUTPUT,
                    "final_output": _OUTPUT, "response": _OUTPUT, "result": _OUTPUT},
    "ChainResult": {"text": _OUTPUT, "content": _OUTPUT, "final_output": _OUTPUT,
                    "response": _OUTPUT, "result": _OUTPUT,
                    "outputs": "a step's output is result['key']; the last one is result.output"},
}

_TRACER = "Tracer, Hook and LoggingTracer are imported from yait_aichain: `from yait_aichain import Tracer`"

#: module that does not exist → where its contents live.
MODULES: dict = {
    "yait_aichain.hooks": _TRACER,
    "yait_aichain.tracing": _TRACER,
    "yait_aichain.callbacks": _TRACER,
    "yait_aichain.events": "`from yait_aichain import Event, Tracer`",
    "yait_aichain.observability": _TRACER,
    "yait_aichain.llm": "`from yait_aichain import Model`",
    "yait_aichain.llms": "`from yait_aichain import Model`",
    "yait_aichain.prompts": "a prompt is Skill(model, prompt='... {name}'), `from yait_aichain import Skill`",
    "yait_aichain.skill": "`from yait_aichain import Skill` (the package is yait_aichain.skills)",
    "yait_aichain.agents": "`from yait_aichain import Agent` (the package is yait_aichain.agent)",
    "yait_aichain.tool": "`from yait_aichain.tools import Tool` (the package is yait_aichain.tools)",
    "yait_aichain.permissions": "`from yait_aichain.tools import PermissionPolicy, ApprovalDecision`",
    "yait_aichain.memory": ("an agent's state is its conversation; a Chain's pause is kept by "
                           "`from yait_aichain.state import FileStore`"),
    "yait_aichain.pipeline": "`from yait_aichain import Chain`",
    "yait_aichain.workflow": "`from yait_aichain import Chain`",
    "yait_aichain.budget": "`from yait_aichain import Budget`",
    "yait_aichain.results": "`from yait_aichain import ChainResult, PoolResult`; AgentResult comes from yait_aichain.agent",
}


#: What a model client reads from ``Model(client_options=...)``. Every family
#: passes ``url`` to the transport and ``timeout``, ``retries`` and ``proxy``
#: through; anything else used to be accepted and dropped in silence — a
#: ``base_url`` sent every request to the provider's default address.
CLIENT_OPTIONS = ("url", "timeout", "retries", "proxy")
FAMILY_CLIENT_OPTIONS = {"qwen": ("region",)}
CLIENT_OPTION_HINTS = {
    "base_url": "url", "api_base": "url", "endpoint": "url", "host": "url",
    "base": "url", "server": "url", "address": "url",
    "max_retries": "retries", "request_timeout": "timeout",
    "proxies": "proxy", "http_proxy": "proxy", "https_proxy": "proxy",
}


def check_client_options(family: str, options: dict) -> None:
    """Raise when ``client_options`` holds a key the client would not read."""
    allowed = CLIENT_OPTIONS + FAMILY_CLIENT_OPTIONS.get(family, ())
    for key in options:
        if key in allowed:
            continue
        if key == "api_key":
            hint = " — the key is Model(name, api_key=...), not a client option"
        elif key in CLIENT_OPTION_HINTS:
            hint = f" — use {CLIENT_OPTION_HINTS[key]!r}"
        else:
            hint = ""
        raise ValueError(f"client_options has an unknown key {key!r}{hint}; "
                         f"this client reads {', '.join(allowed)}")


def refuse(owner: str):
    """Guard *owner*'s ``__init__``: a keyword from ``KEYWORDS[owner]`` raises the
    ``TypeError`` Python would, with the right call named. The signature seen by
    ``inspect`` is unchanged."""
    table = KEYWORDS[owner]

    def wrap(init):
        clash = set(inspect.signature(init).parameters) & set(table)
        if clash:
            raise RuntimeError(f"a hint for {owner}() names a real parameter: {sorted(clash)}")

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            for key in kwargs:
                hint = table.get(key)
                if hint is not None:
                    raise TypeError(f"{owner}() got an unexpected keyword argument "
                                    f"{key!r} — {hint}")
            return init(self, *args, **kwargs)

        return __init__

    return wrap


def missing(obj, name: str) -> AttributeError:
    """The ``AttributeError`` for *name* on *obj*, with a hint when one is known.
    Called from ``__getattr__``, so only after normal lookup has failed."""
    owner = type(obj).__name__
    hint = ATTRIBUTES.get(owner, {}).get(name)
    return AttributeError(f"{owner!r} object has no attribute {name!r}"
                          + (f" — {hint}" if hint else ""))


class _WrongModule(importlib.abc.MetaPathFinder):
    """Last on ``sys.meta_path``: asked only after every real finder has come up
    empty, so it can never hide a module that exists."""

    def find_spec(self, fullname, path=None, target=None):
        hint = MODULES.get(fullname)
        if hint is None:
            return None
        raise ModuleNotFoundError(f"No module named {fullname!r} — {hint}", name=fullname)


def install_module_hints() -> None:
    if not any(isinstance(finder, _WrongModule) for finder in sys.meta_path):
        sys.meta_path.append(_WrongModule())
