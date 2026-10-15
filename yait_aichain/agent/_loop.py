"""
agent._loop — the loop itself, one for `run()` and `stream()`, and the pause.

A mixin of `Agent`: a turn asked, its calls dispatched, a run parked and resumed.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid

from .._budget import start_run
from ..clients._errors import ContextWindowExceededError, TruncatedResponseError
from ..models._adaptation import Adaptation
from ..skills import Skill
from .._dispatch import Denied, NeedsApproval
from .._dispatch import MAX_ARGUMENT_DEPTH  # noqa: F401  (named in the 4.0.1 notes)

from ._swarm import acceptance_from
from ._journal import Journal, evidence as _evidence, CHECK, MODEL_CLAIM, \
                      DONE as _J_DONE, FAILED as _J_FAILED, SKIPPED as _J_SKIPPED
from ._result  import AgentResult
from .         import _prompts as prompts
from ..models._calls import (ToolCall, ToolCallRequest, tool_result_turn)
from ..state   import Suspend, SuspendedResult, current

_logger = logging.getLogger("yait_aichain.agent")
from ._helpers import _copy, _safe, _action, _interrupted  # noqa: F401
from ._stops import _EFFECTS
from ._builtins import (_POOL_SCHEMA, _PLAN_SCHEMA, _delegate_schema, _RESERVED,  # noqa: F401
                        MAX_POOL_ITEMS, MAX_POOL_FLOWS)


class _Loop:
    """The loop and the pause."""

    def _begin(self, task: str, variables: dict | None):
        """A run's first moments, for `run()` and `stream()` alike: a fresh
        journal, the opening messages, working state, a new run id — and no
        document of an earlier run, which a finishing run would otherwise
        delete or overwrite."""
        self._live.id = uuid.uuid4().hex
        self._live.doc_id = None
        start_run(self)
        return Journal(), self.opening(task, variables, ground_rules=True), self.new_state()

    def _loop(self, messages: list, state: dict, journal: Journal) -> AgentResult:
        """Drive the loop to its end and return the result.

        One loop, driven two ways. ``run()`` exhausts it and takes the value;
        ``stream()`` walks it and hands each event on as it appears. A second
        copy of this loop for the streaming case is exactly the mistake 2.0
        undid when `Agent` stopped hand-rolling its own call path — two paths
        do not stay the same, and the one nobody watches is where the defect
        lives.
        """
        walk = self._loop_events(messages, state, journal)
        try:
            while True:
                next(walk)
        except StopIteration as done:
            return done.value

    def _loop_events(self, messages: list, state: dict, journal: Journal,
                     sink: list | None = None):
        """The loop itself, yielding what it has emitted as it goes.

        Events are not produced twice. They go out through ``_emit`` exactly
        as they always have — every hook a caller installed still fires, in
        the same order — and a streaming caller adds one more hook that
        appends to *sink*, which is drained at each boundary. So the stream is
        a view of the existing event channel rather than a second vocabulary
        beside it.

        The granularity is the turn, not the token: everything a turn emitted
        is handed over when the turn is done. Within one model call there is
        nothing to interleave, because the loop is synchronous by design —
        no threads, no async, because the target is Lambda.
        """
        def drain():
            while sink:
                yield sink.pop(0)

        self._live.journal = journal        # where an interceptor's change is written
        while True:
            try:
                # Walked, not called: the text deltas of a streamed turn reach
                # the consumer as they arrive, where they were handed over in
                # one batch after the model finished.
                asking = self._ask_events(messages, state)
                while True:
                    try:
                        next(asking)
                    except StopIteration as done:
                        reply = done.value
                        break
                    yield from drain()
            except Exception as exc:
                yield from drain()
                # A reply cut at the token ceiling is its own stop: the fix is
                # a larger max_tokens, not a retry or a different model.
                cut = isinstance(exc, TruncatedResponseError)
                # Its own stop: the history is longer than the model can read,
                # and that is not the same failure as a revoked key.
                full = isinstance(exc, ContextWindowExceededError)
                return self._result(state, False, None,
                                    "truncated" if cut else "context_overflow" if full else "llm_error",
                                    error=f"{type(exc).__name__}: {exc}")
            yield from drain()

            # Text is the answer — one exit, not two spellings of it — unless
            # a `verified()` gate holds it once for a check.
            if not isinstance(reply, ToolCallRequest):
                if state.get("truncated"):
                    # Cut at max_tokens: kept as the output, because half an
                    # answer can be worth reading, but not called an answer.
                    # It ended the run as `answered`, `success=True`.
                    self._log(1, "\n[Done] ✗ the answer was cut at max_tokens")
                    return self._result(state, False, str(reply), "truncated",
                                        error="the answer was cut at max_tokens; raise max_tokens "
                                              "for the whole of it")
                held = self._verify_gate(state) or self._plan_gate(state)
                if held is None:
                    self._log(1, f"\n[Done] ✓ answered · {state['steps']} step(s)")
                    return self._result(state, True, str(reply), "answered")
                messages.append({"role": "assistant", "parts": [{"type": "text", "text": str(reply)}]})
                messages.append({"role": "user", "parts": [{"type": "text", "text": held}]})
                yield from drain()
                continue

            state["steps"] += 1
            messages.append(reply.as_turn())

            # Providers may request several calls in one turn; all of them run,
            # each with its own result turn and journal entry. Executing only
            # the first would silently narrow the channel.
            for call in reply.calls:
                if state.get("plan") is not None and call.name != "write_plan":
                    # What the plan gate reads: work done since the plan, a
                    # pool counted by its items, since each may be a goal.
                    items = (call.arguments or {}).get("items") if call.name == "pool" else None
                    state["plan_calls"] = (state.get("plan_calls") or 0) + (
                        len(items) if isinstance(items, list) and items else 1)
                # Everything a program needs to rebuild this call, rather
                # than a description of it. `id` is what tells two calls in
                # one turn apart — a provider may request several, and the
                # agent honours all of them — and the raw result is the only
                # form in which "no rows" and "the tool declined" are still
                # different things. `prompts.observation_text` is the model's
                # view and stays the model's view.
                # A copy: a hook that masks a token in the payload changed
                # the argument the tool then ran with.
                self._emit("tool_call.started", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "",
                                    "arguments": _copy(call.arguments or {}),
                                    "agent": self.name or ""})
                yield from drain()

                step = self._dispatch.prepare(call, state)
                if isinstance(step, NeedsApproval):
                    # The request reaches the consumer *here*, before anyone
                    # is asked — which is the whole point of splitting it.
                    yield from drain()
                    try:
                        step = self._dispatch.answer(step, state)
                    except Suspend as susp:
                        step = susp                  # parked below, as a tool's pause is
                    except Exception as exc:
                        messages.append(tool_result_turn(call.id, prompts.result_message(
                            None, f"approval failed: {type(exc).__name__}: {exc}")))
                        yield from drain()
                        return self._callback_failed(state, "the approver", exc)
                    yield from drain()

                try:
                    if isinstance(step, Suspend):
                        raise step
                    result, error = ((None, step.reason) if isinstance(step, Denied)
                                     else self._dispatch.execute(step.call or call, state))
                except Suspend as susp:
                    try:
                        parked = self._park(susp, call, reply, messages,
                                            state, journal)
                    except Exception as exc:
                        # The store refused the document — a DynamoDB item
                        # over 400 KB after twenty calls. run() raised it raw
                        # and the paid conversation was gone; it comes back
                        # as a result, history and cost with it.
                        yield from drain()
                        return self._result(state, False, None, "park_failed",
                                            error=f"the run could not be parked: "
                                                  f"{type(exc).__name__}: {exc}")
                    yield from drain()
                    return parked
                failure, self._live.failure = self._live.failure, None
                if not error:
                    # What a `verified()` gate reads: the last change not yet
                    # looked at. Any later call that succeeds clears it.
                    from ..tools._permissions import risk_of
                    acted = self._dispatch.gated_tool(call)
                    if acted is not None and risk_of(acted, dict(call.arguments or {})) in _EFFECTS:
                        state["unverified"] = call.name
                    elif state.get("unverified"):
                        state["unverified"] = None
                self._emit("tool_call.ended", name=call.name,
                           step=state["steps"],
                           payload={"id": call.id or "", "result": _copy(result),
                                    "agent": self.name or "",
                                    # The type and the traceback, which the
                                    # error string had dropped.
                                    **({"exception": failure} if failure else {})},
                           error=error)
                yield from drain()

                journal.append(
                    # No intent: an intent is what an attempt was *for* — a plan
                    # step's goal — and this loop has none. The tool's name
                    # here would make ten different searches read as one
                    # search repeated to `repeating()`; the action carries it.
                    "",
                    action      = _action(call),
                    outcome     = _J_FAILED if error else _J_DONE,
                    # A call that *ran* is a fact, but what it returned is
                    # only as good as its source. `delegate` hands back a
                    # worker's own account and marks it MODEL_CLAIM; this
                    # recorded the call as CHECK regardless, so an unverified
                    # report entered the parent's journal as a programmatic
                    # fact — the distinction existed and decided nothing. A
                    # failure stays CHECK: that the call failed is verified.
                    evidence    = _evidence(
                        MODEL_CLAIM if (not error and isinstance(result, dict)
                                        and result.get("evidence") == MODEL_CLAIM)
                        else CHECK,
                        error or "executed"),
                    reason      = error or "",
                    # The journal is a written record; an image cannot go in
                    # it, so media is named rather than embedded.
                    observation = prompts.observation_text(result, error),
                    step        = state["steps"],
                )
                messages.append(tool_result_turn(
                    call.id, prompts.result_message(result, error), is_error=bool(error)))

            state["journal"] = journal
            try:
                fired = self._fired(state)
            except Exception as exc:
                yield from drain()
                return self._callback_failed(state, "a stop condition", exc)
            if fired:
                kind = getattr(fired[1], "kind", "ceiling")
                if kind == "check":
                    self._log(1, f"\n[Done] ✓ {fired[0]} · {state['steps']} step(s)")
                    return self._result(state, True, None, fired[0])
                self._log(1, f"\n[Stop] ✗ {fired[0]} · {state['steps']} step(s)")
                self._note_unfinished(journal, state, fired[0])
                return self._result(
                    state, False, None, fired[0],
                    error=f"stopped by {fired[0]} after {state['steps']} step(s)")

            # After the terminal conditions, never before: a ceiling that has
            # been reached ends the run whatever a nudge would have said, so a
            # nudge listed first in `stop_when` cannot talk the loop past its
            # own budget.
            try:
                notes = self._nudges(state)
            except Exception as exc:
                yield from drain()
                return self._callback_failed(state, "a nudge", exc)
            for note in notes:
                messages.append({"role": "user",
                                 "parts": [{"type": "text", "text": note}]})
                yield from drain()

    def _callback_failed(self, state: dict, where: str, exc: BaseException) -> AgentResult:
        """One handling for every caller-supplied callback that raises — a check,
        a ceiling, a nudge, an approver. They were handled three ways: a check
        swallowed it, a ceiling and a nudge raised out of run(), and so did an
        approver, with no result and no `run.finished`."""
        _logger.warning("%s raised; the run stops", where, exc_info=exc)
        return self._result(state, False, None, "condition_error",
                            error=f"{where} raised {type(exc).__name__}: {exc}")

    def _fired(self, state: dict):
        """The first terminal condition that fired. Nudges are not terminal."""
        for cond in self.stop_when:
            if getattr(cond, "kind", "ceiling") in ("nudge", "verify"):
                continue
            if fired := cond(state):
                return fired, cond
        return None

    def _verify_gate(self, state: dict) -> str | None:
        """The message a `verified()` gate holds an answer for, or None."""
        latched = state.get("nudged")
        if not isinstance(latched, set):
            latched = state["nudged"] = set(latched or ())
        for i, cond in enumerate(self.stop_when):
            if getattr(cond, "kind", None) != "verify":
                continue
            key = f"{i}:{cond.name}"
            if key in latched or not cond(state):
                continue
            latched.add(key)
            message = cond.message(state) if callable(cond.message) else cond.message
            self._emit("nudge.fired", step=state.get("steps"),
                       payload={"name": cond.name, "message": message})
            return message
        return None

    def _plan_gate(self, state: dict) -> str | None:
        """Hold an answer once when the plan has more goals than work done.

        A waterfall run that wrote five goals, made two calls and answered
        ended as a clean success: "plan not completed" was recorded only when
        a ceiling stopped the run. Nothing marks a goal done, so the count of
        calls since the plan is the fact there is — fewer calls than goals
        means some goal had none. Said once; the model may still answer, now
        knowing which goals it has not touched.
        """
        plan = state.get("plan")
        done = state.get("plan_calls") or 0
        if not plan or done >= len(plan):
            return None
        latched = state.get("nudged")
        if not isinstance(latched, set):
            latched = state["nudged"] = set(latched or ())
        if "plan" in latched:
            return None
        latched.add("plan")
        goals = "\n".join(f"{i}. {g}" for i, g in enumerate(plan, 1))
        message = (f"Before you answer: your plan has {len(plan)} goal(s) and the run has made "
                   f"{done} call(s) since it was written.\n{goals}\n"
                   "Do any goal that is not done now, or say in your answer which were not done "
                   "and why.")
        self._emit("nudge.fired", step=state.get("steps"),
                   payload={"name": "plan", "message": message})
        return message

    def _nudges(self, state: dict) -> list:
        """Messages from nudges that fired this turn — once per streak."""
        # Keyed by place, not name: two nudges of one name — every nudge is
        # "nudge" unless named, `stalled(3)` beside `stalled(8)` — shared one
        # latch, and the second never spoke. A list after a resume.
        latched = state.get("nudged")
        if not isinstance(latched, set):
            latched = state["nudged"] = set(latched or ())
        notes = []
        for i, cond in enumerate(self.stop_when):
            if getattr(cond, "kind", None) != "nudge":
                continue
            key = f"{i}:{cond.name}"
            if not cond(state):
                latched.discard(key)              # cleared: re-arm
                continue
            if key in latched:
                continue                          # said already, still true
            latched.add(key)
            message = (cond.message(state) if callable(cond.message)
                       else cond.message)
            self._emit("nudge.fired", step=state.get("steps"),
                       payload={"name": cond.name, "message": message})
            notes.append(message)
        return notes

    # ── One model call ───────────────────────────────────────────────────

    def _tool_schemas(self) -> list:
        """Real tools plus the synthetic ones this configuration earns."""
        schemas = [t.schema() for t in self.tools]
        if self.tools:
            schemas.append(_POOL_SCHEMA)
        if self.team:
            schemas.append(_delegate_schema(self.team))
        if self.mode == "waterfall":
            schemas.append(_PLAN_SCHEMA)
        return schemas

    def _ask(self, messages: list, state: dict):
        """One turn, to its end — for ``step()``, which has nobody to stream to."""
        asking = self._ask_events(messages, state)
        while True:
            try:
                next(asking)
            except StopIteration as done:
                return done.value

    def _ask_events(self, messages: list, state: dict):
        """
        One turn of the conversation, through ``Skill``, yielding after each
        event it emits so a stream can hand it on.

        Delegating instead of hand-rolling the call is what makes cost,
        caching and native tool calling arrive at all: they were added to the
        primitive and would never reach a private copy of it.
        """
        _t0 = time.monotonic()
        # The planning phase, when it exists and has its own model, deserves
        # deliberation; execution turns do not. The phase boundary is the
        # recorded plan.
        model = (self.planner_model
                 if self.planner_model is not None and state.get("plan") is None
                 else self.model)
        skill = Skill(
            model  = model,
            input  = {"messages": messages},
            # Stamped, not passed through: a Skill emits `llm_call.*` and
            # knows nothing of the run it is inside, so those events reached
            # the stream with no run_id and a consumer could not tell them
            # from another invocation's. Identity belongs to whoever owns the
            # run, and that is the agent.
            hooks  = self._stamping_hooks(),
            max_cost = self.max_cost,
            _tools = self._tool_schemas(),
            # Transient provider failures (rate limit / 5xx / network) retry
            # inside the call instead of costing the whole turn. The reference
            # agent this design was measured against retries every call; a
            # loop that doesn't pays for each blip with a lost decision.
            max_retries = 2,
        )
        reply = yield from self._reply(skill)
        # In the state, where both exits read it: `run()` ends on it, and a
        # driver of `step()` gets a string back and needs to know it is half.
        state["truncated"] = bool(getattr(skill, "last_truncated", False))
        # Kept per run, deduplicated: a loop asks the same thing of the same
        # model every turn, so without this the list grows by a copy a step.
        for a in skill.last_adaptations:
            # As plain data: the state of the step() seam is documented as
            # JSON-serialisable, and a dataclass in it broke `json.dumps` on
            # the first adaptation — and with `default=str` the `not in`
            # check never matched, so the list grew every step.
            entry = dataclasses.asdict(a) if dataclasses.is_dataclass(a) else a
            if entry not in state["adaptations"]:
                state["adaptations"].append(entry)
        usage = skill.last_usage
        state["tokens"] += getattr(usage, "total_tokens", 0) or 0
        if (c := getattr(usage, "cost", None)) is not None:
            state["cost"] = (state["cost"] or 0.0) + c
        self._log(2, f"     +{getattr(usage, 'total_tokens', 0):,} tokens "
                     f"({time.monotonic() - _t0:.1f}s)")
        return reply

    # ── Executing one call ───────────────────────────────────────────────

    def _result(self, state, success, output, stopped_by, error=None) -> AgentResult:
        return AgentResult(
            success=success, output=output, mode=self.mode,
            steps_taken=state["steps"], tokens_used=state["tokens"],
            cost=state["cost"], stopped_by=stopped_by, error=error,
            plan=[{"goal": g} for g in (state["plan"] or [])],
            adaptations=[Adaptation(**a) if isinstance(a, dict) else a
                         for a in state.get("adaptations") or []],
        )

    def _finished(self, result, journal, board):
        """Attach the run's records and announce the end — unless it parked.

        A :class:`~..state.SuspendedResult` is not an end: ``run.suspended``
        went out when the run parked, the document is in the store, and none
        of the closing records apply to a run that is still owed an answer.
        """
        if isinstance(result, SuspendedResult):
            return result
        result.journal = journal.to_list()
        result.beacons = board.all()
        result.acceptance = acceptance_from(result.journal)
        self._emit("run.finished", payload={"success": result.success,
                                            "stopped_by": result.stopped_by},
                   usage=result.tokens_used, error=result.error)
        if self._live.doc_id is not None:
            # The parked run reached its end; the store holds suspended runs
            # only, and a document left behind is a run a trigger can wake a
            # second time.
            self.store.delete(self._live.doc_id)
            self._live.doc_id = None
        return result

    def _park(self, susp, call, reply, messages, state, journal):
        """Write the parked run to the store and return a SuspendedResult.

        Two things are settled here rather than left to the resume. The other
        calls of the same turn — a provider may ask for several — are answered
        as not executed, because a history with calls nobody answered is the
        one shape every provider handles differently. And the parked call is
        deliberately left **unanswered**: that gap is what the run is waiting
        for, and `resume()` fills exactly it.
        """
        at = next(i for i, c in enumerate(reply.calls) if c is call)
        for other in reply.calls[at + 1:]:
            messages.append(tool_result_turn(
                other.id,
                prompts.result_message(None, f"not executed: the run paused at "
                                             f"{call.name!r} before this call")))

        # The question with what it is about: a person asked to approve saw a
        # reason and not the call it was for.
        awaiting = {"reason": susp.reason, "resume_with": dict(susp.resume_with),
                    "tool": call.name, "arguments": _copy(dict(call.arguments or {}))}
        if getattr(susp, "approval", False):
            # The question is a person's yes or no on this call (policy
            # approve="park"), so a UI can render it as one.
            awaiting["approval"] = True
        doc = self._save_parked(
            self._live.parked(messages, state, journal),
            {**awaiting, "hint": susp.hint, "call_id": call.id,
             "tool": call.name, "arguments": dict(call.arguments or {})},
            run_id=self._live.doc_id)            # parked again: one identifier
        self._live.doc_id = doc.run_id

        self._emit("run.suspended", name=call.name, step=state.get("steps"),
                   payload={"run_id": doc.run_id, "tool": call.name,
                            "hint": susp.hint, "agent": self.name or "",
                            **awaiting})
        self._log(1, f"\n[Paused] … {susp.reason or call.name} · {doc.run_id}")
        return SuspendedResult(run_id=doc.run_id, awaiting=awaiting,
                               document=doc.to_dict())

    def _save_parked(self, variables: dict, suspend: dict, run_id=None):
        """Store a run document whose one step is suspended on *suspend*."""
        from ..state import RunDocument, StepStatus

        doc = RunDocument.new("agent", [self.name or "agent"], variables=variables)
        if run_id is not None:
            doc.run_id = run_id
        if (context := current()) is not None:
            doc.context = context.to_dict()
        doc.steps[0]["status"]  = StepStatus.SUSPENDED
        doc.steps[0]["suspend"] = suspend
        self.store.save(doc.run_id, doc.to_dict())
        return doc

    @staticmethod
    def _task_from_signal(signal, parked: dict, doc):
        """What an armed run was waiting for: its task, and its variables."""
        seeded = dict(doc.variables.get("variables") or {})
        if isinstance(signal, str) and signal.strip():
            return signal, seeded
        if isinstance(signal, dict) and str(signal.get("task") or "").strip():
            seeded.update({k: v for k, v in signal.items() if k != "task"})
            return str(signal["task"]), seeded
        raise ValueError(
            "an armed run is started by its task: pass resume(run_id, "
            "signal=\"...\") or signal={\"task\": \"...\", ...}, where the "
            "other keys seed the conversation. This run was armed asking for "
            f"{parked.get('resume_with') or '{}'}"
            + (f" — {parked['reason']}" if parked.get("reason") else "") + ".")

    def _complete(self, parked: dict, signal):
        """Run the parked tool with the signal. Returns ``(result, error)``.

        A tool that parks and a tool that resumes are the same object called
        twice — `_signal=None` the first time, the world's answer the second.
        A second `Suspend` from here would be a tool that cannot be completed
        by the signal it asked for, so it is reported rather than parked
        again: the run would otherwise wait for the same answer forever.
        """
        tool = self._tool_map.get(parked.get("tool"))
        if tool is None:
            return None, (f"no tool named {parked.get('tool')!r} on this agent "
                          "any more; the run was parked by a different "
                          "configuration")
        call = ToolCall(id=parked.get("call_id") or "", name=tool.name,
                        arguments=dict(parked.get("arguments") or {}))
        dispatcher = self._dispatch
        if (parked.get("hint") or {}).get("approval"):
            # Parked for a person's answer, under policy approve="park": the
            # answer decides, and the call runs as any approved call does.
            answer = signal if isinstance(signal, dict) else {"approved": bool(signal)}
            self._emit("approval.decided", name=tool.name,
                       payload={"id": call.id, "tool": tool.name, "granted": bool(answer.get("approved")),
                                "reason": answer.get("reason") or "", "agent": self.name or ""})
            if not answer.get("approved"):
                return None, (f"tool {tool.name!r} was not approved: "
                              f"{answer.get('reason') or 'not approved'}")
            return dispatcher.execute(call, {})
        try:
            # Through the dispatcher, as every tool runs: the budget, the
            # interceptors, the rules around. It ran the tool directly.
            result = dispatcher.run_tool(call, extra={"_signal": signal}, check=False)
            return dispatcher._after(call, result), None
        except Suspend as susp:
            return None, (f"tool {parked.get('tool')!r} asked to pause again on "
                          f"resume ({susp.reason or 'no reason given'}); a "
                          "signal must complete the tool that asked for it")
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _note_unfinished(self, journal, state, why: str) -> None:
        """A ceiling reached with a plan outstanding is a plan not finished."""
        if not state.get("plan"):
            return
        journal.append(
            "plan not completed",
            outcome  = _J_SKIPPED,
            evidence = _evidence(MODEL_CLAIM, f"run ended via {why}"),
            reason   = (f"{len(state['plan'])} planned item(s) were outstanding "
                        f"when the run stopped."),
        )

    # ── Plumbing ─────────────────────────────────────────────────────────

    def _reply(self, skill):
        """One model turn — streamed when somebody is watching it.

        `run()` buffers, because it has nobody to show prose to and a
        buffered call keeps the fallback chain. `stream()` streams, because a
        reader is in front of the last turn and today it arrives in one piece
        after a silence as long as the model takes.

        The text goes out on the **same** channel as everything else, bracketed
        so a consumer can open a block, append to it and close it:
        ``text.started`` on the first piece, a ``text.delta`` per piece, and
        ``text.ended`` when the turn is done — all carrying one ``id``. A turn
        that asks for a tool and says nothing produces none of the three,
        rather than an empty pair a consumer has to filter.
        """
        if not self._live.streaming:
            return skill.run()

        message_id = uuid.uuid4().hex[:12]
        opened = False
        try:
            for piece in skill.stream():
                if not opened:
                    self._emit("text.started", payload={"id": message_id})
                    opened = True
                self._emit("text.delta",
                           payload={"id": message_id, "text": piece})
                yield
        finally:
            # A block opened is closed, whatever ended it: a consumer left with
            # `text.started` and no `text.ended` keeps a block open for ever.
            if opened:
                self._emit("text.ended", payload={"id": message_id})
        # `stream()` assembles as well as yields, so the decision this turn
        # reached is the same object `run()` would have returned — a
        # ToolCallRequest when the model asked to act, the text otherwise.
        return skill.last_result

