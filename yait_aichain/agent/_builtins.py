"""
agent._builtins — the agent's own calls: `pool`, `delegate`, `write_plan`.

Their schemas, the team's shape, and the handlers, as a mixin of `Agent`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

from ..skills import Skill
from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the release notes)

from ._journal import MODEL_CLAIM
from ..pool    import Pool

_logger = logging.getLogger("yait_aichain.agent")
from ._helpers import _copy, _safe, _action, _interrupted  # noqa: F401


_POOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pool",
        "description": (
            "Repeat ONE tool over every entry of a list, in parallel, in one "
            "step. Use this whenever the same work applies to each item — "
            "calling the tool once per item spends a whole turn on each. "
            "Give either `tool` (run that tool per entry) or `prompt` (run "
            "the model per entry; {item} is the current entry)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # An array parameter must say what its elements are: Google
                # rejects the whole request when `items` is missing, before
                # the first step. Declared as objects because that is the
                # form that carries named arguments; a plain value still
                # works at runtime and fills `arg`.
                "items": {"type": "array",
                          "items": {"type": "object"},
                          "description": "one entry per call; an object entry "
                                         "is spread as the tool's named "
                                         "arguments, a plain value fills `arg`"},
                "tool": {"type": "string",
                         "description": "name of the tool to run per entry"},
                "arg": {"type": "string",
                        "description": "parameter a plain entry fills "
                                       "(defaults to the tool's first)"},
                "prompt": {"type": "string",
                           "description": "instead of `tool`: a prompt run "
                                          "per entry, {item} = the entry"},
                "max_flows": {"type": "integer"},
            },
            "required": ["items"],
        },
    },
}

_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_plan",
        "description": "Record the plan: a short ordered list of what this "
                       "task requires. Call once, first, then hold to it.",
        "parameters": {
            "type": "object",
            "properties": {"items": {"type": "array",
                                     "items": {"type": "string"}}},
            "required": ["items"],
        },
    },
}


#: When to hand work down, put where the model reads it: in the tool. A
#: delegation is a whole run of another agent, and small work handed down
#: costs more than doing it.
DELEGATE_WHEN = (
    "Delegate only when one of these holds: specialisation — a member has tools, "
    "knowledge or a model you lack; determinism — a member does exactly this job; "
    "isolation — more than about five calls or a lot of raw material whose details the "
    "answer does not need. Do not delegate what one to three of your own calls finish, "
    "or what needs this whole conversation.")


def _delegate_schema(team: dict) -> dict:
    props: dict = {"task": {"type": "string",
                            "description": "a self-contained task"}}
    required = ["task"]
    members = team.get("members") or []
    if members:
        names = [getattr(w, "name", None) or f"worker{i}"
                 for i, w in enumerate(members)]
        chains = {n: w for n, w in zip(names, members) if _is_chain(w)}
        if chains:
            props["inputs"] = {"type": "object",
                               "description": "for a chain member: its inputs by name, as its "
                                              "description lists them; `task` fills a single "
                                              "text input left out"}
        # What each worker is for, beside its name, or the lead chooses
        # between `worker0` and `worker1` by guessing — a whole worker run
        # lost on every wrong guess.
        roster = "; ".join(f"{n}: {getattr(w, 'description', None) or 'no description given'}"
                           + (f" (a chain; inputs: {_inputs_line(w)})" if _is_chain(w) else "")
                           for n, w in zip(names, members))
        props["worker"] = {"type": "string", "enum": names,
                           "description": f"which member takes it — {roster}"}
        if not team.get("extend"):
            required.append("worker")
    if team.get("extend"):
        props["instructions"] = {
            "type": "string",
            "description": ("who a new member should be, named as a role — 'translator', not "
                            "'helper_for_task_3'; the same instructions reuse the same member"
                            + ("; leave it out to give the task to a listed member" if members
                               else ""))}
    return {
        "type": "function",
        "function": {
            "name": "delegate",
            "description": (
                "Hand a scoped sub-task to a team member. It runs its own conversation "
                "and returns a conclusion — use it to keep bulk work out of this one. "
                + DELEGATE_WHEN),
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        },
    }


def _is_chain(worker) -> bool:
    from ..chain import Chain
    return isinstance(worker, Chain)


def _inputs_line(chain) -> str:
    """A chain's inputs as a model reads them: name, whether required, what it is."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # said once, when the chain was built
        schema = chain.input_schema()
    return ", ".join(f"{n} ({'required' if s.get('required') else 'optional'}"
                     + (f" — {s['description']}" if s.get("description") else "") + ")"
                     for n, s in schema.items()) or "none"


def _team_parts(team) -> dict | None:
    """``team=`` in its one form: ``{"members": [...], "extend": bool, "max": N}``.
    A list is the members; ``"auto"`` is ``{"extend": True}``, said."""
    if team is None:
        return None
    if team == "auto":
        warnings.warn("team='auto' goes in 5.0; say team={'extend': True} (and 'members': [...] "
                      "for the ones you already have)", FutureWarning, stacklevel=4)
        team = {"extend": True}
    elif isinstance(team, (list, tuple)):
        team = {"members": list(team)}
    if not isinstance(team, dict):
        raise ValueError(f"team is a dict {{'members': [...], 'extend': bool, 'max': N}}, a list "
                         f"of agents, or None; got {type(team).__name__}")
    if unknown := sorted(set(team) - {"members", "extend", "max", "max_delegations"}):
        raise ValueError(f"team keys are 'members', 'extend', 'max' and 'max_delegations'; "
                         f"got {unknown}")
    from ._agent import Agent                 # here, not above: it imports this module
    members = list(team.get("members") or [])
    if bad := [type(m).__name__ for m in members if not (isinstance(m, Agent) or _is_chain(m))]:
        raise ValueError(f"a team member is an Agent or a Chain; got {bad}")
    extend = bool(team.get("extend", False))
    cap = team.get("max")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError(f"team['max'] is a whole number of at least 1; got {cap!r}")
    if cap is not None and len(members) > cap:
        raise ValueError(f"team['max'] is {cap} and {len(members)} members are listed")
    calls = team.get("max_delegations")
    if calls is not None and (isinstance(calls, bool) or not isinstance(calls, int) or calls < 1):
        raise ValueError(f"team['max_delegations'] is a whole number of at least 1; got {calls!r}")
    if not members and not extend:
        return None
    return {"members": members, "extend": extend, "max": cap, "max_delegations": calls}


# ── Agent ────────────────────────────────────────────────────────────────────

#: The most entries one `pool` call fans out to, and the most at once. A
#: model sets both; unbounded, a thousand items would be a thousand threads.
MAX_POOL_ITEMS = 200
MAX_POOL_FLOWS = 8





_RESERVED = frozenset({"pool", "delegate", "write_plan"})


class _Builtin:
    """What the policy is asked about for a built-in that runs no other tool."""
    def __init__(self, name: str, risk: str) -> None:
        self.name, self.risk = name, risk


#: `delegate` hands work to a worker and `write_plan` records a plan: neither
#: changes the world by itself, which is what `draft` means.
_DRAFTING = {"delegate": _Builtin("delegate", "draft"),
             "write_plan": _Builtin("write_plan", "draft")}



class _Builtins:
    """The handlers of the agent's own calls."""

    def _do_plan(self, args: dict, state: dict) -> str:
        items = args.get("items") or []
        if not isinstance(items, list):
            items = [items]                  # one item, not a string split into characters
        if state["plan"] is not None:
            # "You may not rewrite it" — and a rewrite would also invalidate
            # the prefix the freeze exists to keep cacheable.
            return "The plan is already recorded and cannot be rewritten."
        state["plan"] = [str(i) for i in items]
        self._log(1, f"  ▣  plan: {len(items)} item(s)")
        return "Plan recorded. Begin."

    def _do_pool(self, args: dict, state: dict) -> Any:
        """One call repeated over a list, so "read every document" is not one read
        reported as all. No way around the policy: the tool it names was decided on
        before this ran, and every item is checked against its schema first."""
        items = args.get("items")
        if not isinstance(items, (list, tuple)) or not items:
            raise ValueError("pool needs a non-empty 'items' list")
        if len(items) > MAX_POOL_ITEMS:
            raise ValueError(f"pool takes at most {MAX_POOL_ITEMS} items in one call; "
                             f"got {len(items)} — split the list")

        max_flows = max(1, min(int(args.get("max_flows") or 4), len(items), MAX_POOL_FLOWS))

        if args.get("tool"):
            tool = self._tool_map.get(args["tool"])
            if tool is None:
                raise ValueError(f"pool tool {args['tool']!r} not found. "
                                 f"Available: {sorted(self._tool_map)}")
            arg = args.get("arg") or next(
                iter(tool.parameters.get("properties", {})), "input")
            # A dict entry is a bundle of named arguments; wrapping it under
            # one name would pass the whole dict as a single argument and fail
            # every call on what it never received.
            pool_items = [it if (isinstance(it, dict) and not args.get("arg"))
                          else {arg: it} for it in items]
            for i, item in enumerate(pool_items):
                if (problem := tool.check_args(item)) is not None:
                    raise ValueError(f"pool item {i}: {problem}")
            runner = tool
        elif args.get("prompt"):
            runner = Skill(self.model, prompt=args["prompt"], max_cost=self.max_cost,
                           hooks=self._stamping_hooks(), max_retries=2)
            pool_items = [{"item": it} for it in items]
        else:
            raise ValueError("pool needs either 'tool' or 'prompt'")

        self._log(1, f"  ⇉  pool ×{len(items)} (max_flows={max_flows})")
        from .._dispatch import calling
        fan_out = Pool(runner, items=pool_items, max_flows=max_flows, on_error="collect",
                       max_cost=self.max_cost, hooks=self._live.hooks)
        fan_out._parent_call = calling() or "pool"
        results = fan_out.run()
        if results.usage is not None:
            state["tokens"] += results.tokens_used
            if results.cost is not None:
                state["cost"] = (state["cost"] or 0.0) + results.cost
        errors = [e for e in results.errors if e is not None]
        if len(errors) == len(pool_items):
            raise RuntimeError(f"all {len(pool_items)} fanned-out calls failed. "
                               f"First: {errors[0]}")
        if errors:
            self._log(1, f"  ⚠  {len(errors)}/{len(pool_items)} entries failed; "
                         f"first: {errors[0]}")
        # A list, which is what a model reads: the result object's repr,
        # `PoolResult(ok, items=3)`, would tell it nothing it asked for.
        return [out if err is None else {"error": err}
                for out, err in zip(results.output, results.errors)]

    def _do_delegate(self, args: dict, state: dict) -> Any:
        """Hand a scoped sub-task down: the child spends its own conversation and returns
        one turn's worth of conclusion."""
        if not self.team:
            raise ValueError("this agent has no team; delegate is unavailable "
                             "unless team= is set")
        if self._live.depth >= self._live.max_depth:
            raise RuntimeError(f"maximum delegation depth {self._live.max_depth} reached")
        limit = self.team.get("max_delegations")
        if limit is not None and (state.get("delegations") or 0) >= limit:
            raise RuntimeError(f"this run has delegated {limit} time(s), its team's "
                               "max_delegations; finish the rest yourself")
        state["delegations"] = (state.get("delegations") or 0) + 1

        task = args.get("task") or ""
        if args.get("worker") or not self.team["extend"]:
            worker = self._worker_map.get(args.get("worker", ""))
            if worker is None:
                raise ValueError(f"no worker named {args.get('worker')!r}. "
                                 f"Available: {sorted(self._worker_map)}")
            # A run of its own, one level down (below), so the depth guard
            # holds for a listed worker too.
        elif (instructions := str(args.get("instructions") or "")) in self._extended:
            worker = self._extended[instructions]       # the same role, given work again
        else:
            cap = self.team["max"]
            if cap is not None and len(self._worker_map) + len(self._extended) >= cap:
                raise ValueError(f"the team is at its max of {cap}; give the task to a member: "
                                 f"{sorted(self._worker_map) + list(self._extended)}")
            from ._agent import Agent             # here, not above: it imports this module
            worker = Agent(
                # Without the tools that park a run: a worker's pause has
                # nowhere to go — `delegate` waits for an answer, not a resume.
                self.model, tools=[t for t in self.tools if not getattr(t, "suspends", False)],
                instructions=args.get("instructions", ""),
                stop_when=self.stop_when, hooks=self._live.hooks,
                # No rules of its own: the lead's govern it from around the
                # `delegate` call (`_dispatch.governing`), approver included.
                max_cost=self.max_cost,
                verbose=self.verbose,
            )
            worker._max_depth = self._live.max_depth
            self._extended[instructions] = worker

        self._log(1, f"  ▽  delegate → {getattr(worker, 'name', None) or 'worker'}")
        if _is_chain(worker):
            return self._delegate_to_chain(worker, task, args.get("inputs"), state)
        with worker._opened(depth=self._live.depth + 1, parent_id=self._live.id,
                            hooks=self._live.hooks):
            child = worker.run(task)
        state["tokens"] += child.tokens_used
        if child.cost is not None:
            state["cost"] = (state["cost"] or 0.0) + child.cost
        steps = getattr(child, "steps_taken", None)
        # What the child says it did is the child's word for it. Delegation
        # multiplies the places where "it says it did" stands in for "we
        # checked"; naming the class keeps them countable.
        return {"worker": getattr(worker, "name", None) or "worker",
                "stopped_by": child.stopped_by,
                "success": child.success,
                "output": child.output,
                "error": child.error,
                # The worker's board already forwarded these to ours, so a
                # waiting Pool above has seen them. They are repeated here
                # because the model that delegated is waiting too, and the
                # tool result is the only channel it reads.
                "beacons": [b.to_dict() for b in (child.beacons or [])],
                "evidence": MODEL_CLAIM,
                "steps": steps, "cost": child.cost,
                **({"note": f"{getattr(worker, 'name', None) or 'the member'} finished in "
                            f"{steps} step(s) with your model and your tools; work this small "
                            "is cheaper done with your own calls than handed down"}
                   if self._no_wider(worker) and steps is not None and steps <= 1 else {})}

    def _no_wider(self, worker) -> bool:
        """The member has nothing the lead lacks: the same model, no other tools."""
        same = getattr(getattr(worker, "model", None), "name", None) == getattr(self.model, "name", "")
        theirs = {t.name for t in getattr(worker, "tools", ()) or ()}
        return same and theirs <= set(self._tool_map)

    def _delegate_to_chain(self, chain, task: str, given, state: dict) -> Any:
        """A chain member takes its inputs by name; the lead's rules, budget and
        hooks hold inside it, as around any call the lead makes."""
        from .._events import listening
        from ..state import SuspendedResult
        values = dict(given) if isinstance(given, dict) else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            schema = chain.input_schema()
        missing = [n for n, s in schema.items() if s.get("required") and n not in values]
        if len(missing) == 1 and task and schema[missing[0]].get("type") in (None, "string"):
            values[missing[0]] = task
            missing = []
        if missing:
            raise ValueError(f"chain {chain.name or 'member'!r} needs the inputs {missing}; "
                             f"give them in `inputs` — it takes: {_inputs_line(chain)}")
        with listening(self._live.hooks):
            result = chain.run(variables=values)
        if isinstance(result, SuspendedResult):
            raise RuntimeError(f"chain {chain.name or 'member'!r} paused ({result.awaiting.get('reason')}); "
                               "a delegated chain cannot park — put its pause in the lead's tools")
        state["tokens"] += result.tokens_used or 0
        if result.cost is not None:
            state["cost"] = (state["cost"] or 0.0) + result.cost
        return {"worker": chain.name or "chain", "success": result.success,
                "output": result.output, "error": result.error, "evidence": result.evidence}

    # ── Result ───────────────────────────────────────────────────────────

